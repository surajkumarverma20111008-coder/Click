"""
Mr. Lawaris - Transcript Correction Pipeline v2
Fixes: download path bug, upload API, correct library (google-genai)
"""

import os
import re
import time
import logging
import random
from pathlib import Path

from google import genai
from google.genai import types
from huggingface_hub import (
    HfApi,
    snapshot_download,
    list_repo_files,
    CommitOperationAdd,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
HF_REPO_ID        = "Kumarverma11/PocketFM_Audio"
HF_INPUT_FOLDER   = "Generated_Transcripts"
HF_OUTPUT_FOLDER  = "Corrected_Transcripts"
DOWNLOAD_DIR      = Path("./hf_download")
CORRECTED_DIR     = Path("./corrected")

GEMMA_MODELS      = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]
RPM_PER_MODEL     = 14       # safe under 15 limit
BATCH_UPLOAD      = 50       # HF commit every N files
WORD_THRESHOLD    = 3000     # files above this → split

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── EPISODE BOUNDARY PATTERNS ────────────────────────────────────────────────
_PATTERNS = [
    r"जानने के लिए सुनिए[^\n]*(?:अगला|भाग|एपिसोड)[^\n]*",
    r"कहानी का अगला भाग[^\n]*",
    r"(?:सिर्फ|केवल)[^\n]*(?:पॉकेट|Pocket)[^\n]*(?:एफएम|FM)[^\n]*",
    r"अभी सुनें[^\n]*(?:अगला|भाग)[^\n]*",
    r"जवाब जानने के लिए[^\n]*सुन[^\n]*",
    r"(?:क्या|कब|कैसे|कौन)[^\n]+\?[^\n]*(?:जानने|सुनिए|सुनें)[^\n]*",
    r"अगला एपिसोड[^\n]*(?:पॉकेट|Pocket)[^\n]*",
]
BOUNDARY_RE = re.compile(
    "|".join(f"({p})" for p in _PATTERNS),
    re.IGNORECASE | re.UNICODE,
)

# ─── PROMPTS ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """तुम एक Hindi ASR transcript correction expert हो।

नियम:
1. केवल ASR errors सुधारो (गलत पहचाने गए शब्द)
2. कहानी की content, plot, dialogue बिल्कुल नहीं बदलनी
3. नए sentences नहीं जोड़ने
4. Original paragraph structure preserve करो

Common corrections (context देखकर करो):
- अबे / अभी-अभी → अभय (character name)
- Broken words: "ब न ना" → "बनाना"
- Punctuation: sentences properly end करो

Output: केवल corrected Hindi text। कोई explanation नहीं।"""


# ─── RATE LIMITER ─────────────────────────────────────────────────────────────
class RateLimiter:
    """Round-robin across 2 Gemma models, each capped at RPM_PER_MODEL/min."""

    def __init__(self):
        self.timestamps = {m: [] for m in GEMMA_MODELS}

    def _cleanup(self):
        now = time.time()
        for m in GEMMA_MODELS:
            self.timestamps[m] = [t for t in self.timestamps[m] if now - t < 60]

    def _best_model(self):
        self._cleanup()
        best_model, best_remaining = None, -1
        for m in GEMMA_MODELS:
            remaining = RPM_PER_MODEL - len(self.timestamps[m])
            if remaining > best_remaining:
                best_remaining, best_model = remaining, m
        return best_model, best_remaining

    def _wait_for_slot(self):
        while True:
            _, remaining = self._best_model()
            if remaining > 0:
                return
            all_ts = sorted(t for ts in self.timestamps.values() for t in ts)
            wait = 61 - (time.time() - all_ts[0]) if all_ts else 5
            log.info(f"Rate limit: both models full. Waiting {wait:.1f}s...")
            time.sleep(max(wait, 1))

    def call(self, client: genai.Client, text: str) -> str:
        for attempt in range(6):
            self._wait_for_slot()
            model, _ = self._best_model()
            self.timestamps[model].append(time.time())
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=f"यह Mr. Lawaris Hindi audio drama का transcript है। ASR errors सुधारो।\n\nTRANSCRIPT:\n{text}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=8192,
                    ),
                )
                return resp.text.strip()
            except Exception as e:
                self.timestamps[model].pop()   # failed → don't count
                err = str(e).lower()
                if "429" in err or "quota" in err or "rate" in err:
                    wait = 65 + random.uniform(5, 20)
                    log.warning(f"429 on {model} (attempt {attempt+1}). Waiting {wait:.0f}s...")
                    time.sleep(wait)
                elif "500" in err or "503" in err:
                    wait = 30 + random.uniform(5, 15)
                    log.warning(f"Server error on {model} (attempt {attempt+1}). Waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    log.error(f"Unexpected API error: {e}")
                    raise
        raise RuntimeError("Max retries exceeded")


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def word_count(text: str) -> int:
    return len(text.split())


def split_into_episodes(text: str, base_name: str):
    """
    Split transcript at episode boundaries.
    Returns list of (label, episode_text) e.g. [('a', ...), ('b', ...), ...]
    """
    boundaries = [m.end() for m in BOUNDARY_RE.finditer(text)]
    if not boundaries:
        log.warning(f"  No boundaries found in {base_name} — keeping whole file")
        return [("", text)]

    segments = []
    prev = 0
    for end in boundaries:
        seg = text[prev:end].strip()
        if seg and word_count(seg) > 150:
            segments.append(seg)
        prev = end
    tail = text[prev:].strip()
    if tail and word_count(tail) > 150:
        segments.append(tail)

    if len(segments) <= 1:
        return [("", text)]

    log.info(f"  Split {base_name} → {len(segments)} episodes")
    labels = [chr(ord("a") + i) if i < 26 else f"_{i}" for i in range(len(segments))]
    return list(zip(labels, segments))


def already_corrected_on_hf(hf_api: HfApi, hf_token: str) -> set:
    """Return set of filenames already in HF_OUTPUT_FOLDER."""
    try:
        all_files = list(list_repo_files(
            HF_REPO_ID, repo_type="dataset", token=hf_token
        ))
        return {
            Path(f).name
            for f in all_files
            if f.startswith(HF_OUTPUT_FOLDER + "/")
        }
    except Exception as e:
        log.warning(f"Could not check already-corrected files: {e}")
        return set()


# ─── PROCESS ONE FILE ─────────────────────────────────────────────────────────
def process_file(
    file_path: Path,
    rate_limiter: RateLimiter,
    client: genai.Client,
    done_set: set,
) -> list:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    stem = file_path.stem
    wc   = word_count(text)
    log.info(f"  {file_path.name} ({wc} words)")

    episodes = split_into_episodes(text, stem) if wc > WORD_THRESHOLD else [("", text)]

    results = []
    for label, ep_text in episodes:
        out_name = f"{stem}{label}.txt" if label else f"{stem}.txt"
        out_path = CORRECTED_DIR / out_name

        # Skip if already on HF or already in local corrected dir
        if out_name in done_set or out_path.exists():
            log.info(f"  Skip (done): {out_name}")
            results.append(out_path)
            continue

        ep_wc = word_count(ep_text)
        if ep_wc < 100:
            log.warning(f"  Skipping tiny segment {out_name} ({ep_wc} words)")
            continue

        log.info(f"  Correcting: {out_name} ({ep_wc} words)...")
        corrected = rate_limiter.call(client, ep_text)
        out_path.write_text(corrected, encoding="utf-8")
        log.info(f"  ✓ Saved: {out_name}")
        results.append(out_path)

    return results


# ─── BATCH UPLOAD ─────────────────────────────────────────────────────────────
def upload_batch(files: list, hf_api: HfApi, batch_num: int):
    if not files:
        return
    log.info(f"Uploading batch {batch_num} ({len(files)} files) to HuggingFace...")
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{HF_OUTPUT_FOLDER}/{f.name}",
            path_or_fileobj=str(f),
        )
        for f in files
    ]
    hf_api.create_commit(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Corrected transcripts — batch {batch_num} ({len(files)} files)",
    )
    log.info(f"✓ Batch {batch_num} uploaded successfully")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    api_key  = os.environ.get("GEMINI_API_KEY") or exit("❌ GEMINI_API_KEY not set")
    hf_token = os.environ.get("HF_TOKEN")       or exit("❌ HF_TOKEN not set")

    CORRECTED_DIR.mkdir(exist_ok=True)
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    client       = genai.Client(api_key=api_key)
    hf_api       = HfApi(token=hf_token)
    rate_limiter = RateLimiter()

    # ── Step 1: Download ALL transcripts in one shot ──────────────────────────
    # Files land at: ./hf_download/Generated_Transcripts/Episode_XXXX.txt
    log.info("Downloading all transcripts from HuggingFace (snapshot)...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(DOWNLOAD_DIR),
        allow_patterns=[f"{HF_INPUT_FOLDER}/*.txt"],
        token=hf_token,
    )
    input_dir   = DOWNLOAD_DIR / HF_INPUT_FOLDER
    input_files = sorted(input_dir.glob("*.txt"))
    log.info(f"Found {len(input_files)} transcript files to process")

    # ── Step 2: Get already-corrected files from HF ───────────────────────────
    done_set = already_corrected_on_hf(hf_api, hf_token)
    log.info(f"Already corrected on HuggingFace: {len(done_set)} files")

    pending_upload = []
    batch_num      = 1
    total_processed = 0

    # ── Step 3: Process each file ─────────────────────────────────────────────
    for i, file_path in enumerate(input_files):
        stem = file_path.stem

        # Skip if base name or split version already done
        if f"{stem}.txt" in done_set or f"{stem}a.txt" in done_set:
            log.info(f"[{i+1}/{len(input_files)}] Already done: {file_path.name}")
            continue

        log.info(f"[{i+1}/{len(input_files)}] ── {file_path.name}")
        try:
            output_files = process_file(file_path, rate_limiter, client, done_set)
            pending_upload.extend(output_files)
            total_processed += 1
        except Exception as e:
            log.error(f"FAILED: {file_path.name} → {e}")
            continue

        # Upload in batches of 50
        if len(pending_upload) >= BATCH_UPLOAD:
            upload_batch(pending_upload[:BATCH_UPLOAD], hf_api, batch_num)
            batch_num    += 1
            pending_upload = pending_upload[BATCH_UPLOAD:]

    # ── Step 4: Upload any remaining files ────────────────────────────────────
    if pending_upload:
        upload_batch(pending_upload, hf_api, batch_num)

    log.info(f"✅ Pipeline complete! Processed: {total_processed} files")


if __name__ == "__main__":
    main()
