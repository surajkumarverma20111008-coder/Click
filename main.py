"""
Mr. Lawaris – Transcript Correction Pipeline  v4  (FINAL)

Root-cause fix:  resp.text property itself throws
                 'NoneType has no attribute strip' internally.
                 We now wrap every resp.text access in try/except.

Extra fixes:
  - Safety settings set to BLOCK_ONLY_HIGH  (drama content was getting blocked)
  - Fallback: if ALL retries fail, save original text so no episode is ever lost
  - Cleaner rate-limit logic
  - All errors caught per-episode; pipeline never crashes mid-run
"""

import os
import re
import time
import logging
import random
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from huggingface_hub import (
    HfApi,
    snapshot_download,
    list_repo_files,
    CommitOperationAdd,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
HF_REPO_ID       = "Kumarverma11/PocketFM_Audio"
HF_INPUT_FOLDER  = "Generated_Transcripts"
HF_OUTPUT_FOLDER = "Corrected_Transcripts"
DOWNLOAD_DIR     = Path("./hf_download")
CORRECTED_DIR    = Path("./corrected")

GEMMA_MODELS     = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]
RPM_PER_MODEL    = 14      # safe under 15-RPM free-tier limit
BATCH_UPLOAD     = 50      # HuggingFace commit every N files
WORD_THRESHOLD   = 3000    # files larger than this get split

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── SAFETY SETTINGS (BLOCK_ONLY_HIGH keeps drama content unblocked) ───────────
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",        threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",  threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",         threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_ONLY_HIGH"),
]

# ── EPISODE BOUNDARY PATTERNS ─────────────────────────────────────────────────
_BOUNDARY_PATTERNS = [
    r"जानने के लिए सुनिए[^\n]*(?:अगला|भाग|एपिसोड)[^\n]*",
    r"कहानी का अगला भाग[^\n]*",
    r"(?:सिर्फ|केवल)[^\n]*(?:पॉकेट|Pocket)[^\n]*(?:एफएम|FM)[^\n]*",
    r"अभी सुनें[^\n]*(?:अगला|भाग)[^\n]*",
    r"जवाब जानने के लिए[^\n]*सुन[^\n]*",
    r"(?:क्या|कब|कैसे|कौन)[^\n]+\?[^\n]*(?:जानने|सुनिए|सुनें)[^\n]*",
    r"अगला एपिसोड[^\n]*(?:पॉकेट|Pocket)[^\n]*",
]
BOUNDARY_RE = re.compile(
    "|".join(f"({p})" for p in _BOUNDARY_PATTERNS),
    re.IGNORECASE | re.UNICODE,
)

# ── PREAMBLE/POSTAMBLE CLEANER ────────────────────────────────────────────────
_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"(?:here\s+is|here'?s|यहाँ\s+है|यहाँ|यहां)[^\n]*\n+"
    r"|(?:corrected\s+(?:transcript|text|story|version|hindi))[^\n]*\n+"
    r"|(?:सुधरा\s+हुआ|सुधारा\s+गया|corrected)[^\n]*\n+"
    r"|[-=*#]{3,}\s*\n+"
    r")",
    re.IGNORECASE,
)
_POSTAMBLE_RE = re.compile(
    r"\n+(?:"
    r"(?:note|नोट|changes\s+made|corrections?\s+made|explanation|i\s+(?:have|'ve)\s+corrected)[^\n]*"
    r"|[-=*#]{3,}"
    r")\s*$",
    re.IGNORECASE,
)

# ── PROMPTS ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """तुम एक expert Hindi ASR transcript editor हो।
तुम्हारा काम Mr. Lawaris audio drama की transcripts को speech-to-text errors से ठीक करना है।

सख्त नियम:
1. केवल ASR (speech recognition) errors सुधारो।
2. कहानी की content, plot, dialogue, meaning बिल्कुल नहीं बदलनी।
3. कोई नया sentence, word, scene नहीं जोड़ना।
4. Original paragraph structure और line breaks preserve करो।
5. Response में सिर्फ corrected story text होना चाहिए।
   कोई introduction नहीं, कोई explanation नहीं, कोई notes नहीं।

Common ASR corrections (context देखकर):
- Character names: अबे/अभी-अभी → अभय  |  आर्यन → अर्जुन (context देखो)
- Broken words: "ब न ना" → "बनाना"
- Punctuation: sentences properly end करो (। ? !)

Output: केवल corrected Hindi story text।"""


# ── SAFE TEXT EXTRACTOR ───────────────────────────────────────────────────────
def safe_get_text(resp) -> Optional[str]:
    """
    THE critical fix.

    google-genai's resp.text property calls .strip() internally on the raw
    text. If the model returns a part with text=None, the property throws:
        AttributeError: 'NoneType' object has no attribute 'strip'

    We catch that, then fall back to manually walking candidates → parts.
    Returns None if no usable text found.
    """
    # ── Path 1: standard .text property ──────────────────────────────────────
    try:
        t = resp.text
        if t and str(t).strip():
            return str(t).strip()
    except Exception:
        pass   # property itself threw – fall through to manual extraction

    # ── Path 2: walk candidates → content → parts ─────────────────────────────
    try:
        candidates = resp.candidates or []
        for candidate in candidates:
            try:
                parts = (candidate.content or object()).parts or []
                collected = []
                for part in parts:
                    try:
                        raw = part.text
                        if raw and str(raw).strip():
                            collected.append(str(raw).strip())
                    except Exception:
                        continue
                if collected:
                    return "\n".join(collected)
            except Exception:
                continue
    except Exception:
        pass

    return None   # nothing found


def clean_output(text: str) -> str:
    """Strip AI-added preamble / postamble. Return only story text."""
    text = _PREAMBLE_RE.sub("", text)
    text = _POSTAMBLE_RE.sub("", text)
    return text.strip()


# ── RATE LIMITER ──────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self):
        self.ts = {m: [] for m in GEMMA_MODELS}

    def _purge(self):
        now = time.time()
        for m in GEMMA_MODELS:
            self.ts[m] = [t for t in self.ts[m] if now - t < 60]

    def _best(self):
        self._purge()
        best_m, best_r = None, -1
        for m in GEMMA_MODELS:
            r = RPM_PER_MODEL - len(self.ts[m])
            if r > best_r:
                best_r, best_m = r, m
        return best_m, best_r

    def _wait(self):
        while True:
            _, r = self._best()
            if r > 0:
                return
            all_ts = sorted(t for v in self.ts.values() for t in v)
            wait = 61 - (time.time() - all_ts[0]) if all_ts else 5
            log.info(f"Rate limit – waiting {wait:.0f}s...")
            time.sleep(max(wait, 1))

    def call(self, client: genai.Client, text: str) -> Optional[str]:
        """
        Returns corrected text, or None if all retries exhausted.
        Never raises – caller decides what to do with None.
        """
        for attempt in range(8):
            self._wait()
            model, _ = self._best()
            self.ts[model].append(time.time())
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=f"नीचे दिया transcript सुधारो:\n\n{text}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=8192,
                        safety_settings=SAFETY_SETTINGS,
                    ),
                )
                result = safe_get_text(resp)
                if result:
                    return clean_output(result)

                # Empty result (blocked/filtered) – remove timestamp, try other model
                if self.ts[model]:
                    self.ts[model].pop()
                log.warning(f"  [{attempt+1}/8] Empty response from {model} – trying other model")
                # Temporarily saturate this model so the next attempt picks the other
                self.ts[model].extend([time.time()] * RPM_PER_MODEL)
                time.sleep(3)

            except Exception as e:
                if self.ts[model]:
                    self.ts[model].pop()
                err = str(e).lower()
                if "429" in err or "quota" in err or "rate" in err:
                    wait = 65 + random.uniform(5, 20)
                    log.warning(f"  [{attempt+1}/8] 429 on {model}. Waiting {wait:.0f}s...")
                    time.sleep(wait)
                elif "500" in err or "503" in err or "unavailable" in err:
                    wait = 30 + random.uniform(5, 15)
                    log.warning(f"  [{attempt+1}/8] Server error on {model}. Waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    log.warning(f"  [{attempt+1}/8] Error on {model}: {e}")
                    time.sleep(5)

        log.error("  All 8 attempts failed – returning None")
        return None


# ── HELPERS ───────────────────────────────────────────────────────────────────
def word_count(text: str) -> int:
    return len(text.split())


def split_episodes(text: str, name: str):
    """
    Split large transcript at episode-boundary cliffhangers.
    Returns list of (label, text): [('a', ...), ('b', ...), ...]
    or [('', text)] if no boundaries found.
    """
    boundaries = [m.end() for m in BOUNDARY_RE.finditer(text)]
    if not boundaries:
        log.warning(f"  No episode boundaries in {name} – keeping whole")
        return [("", text)]

    segs, prev = [], 0
    for end in boundaries:
        seg = text[prev:end].strip()
        if seg and word_count(seg) > 100:
            segs.append(seg)
        prev = end
    tail = text[prev:].strip()
    if tail and word_count(tail) > 100:
        segs.append(tail)

    if len(segs) <= 1:
        return [("", text)]

    log.info(f"  Split {name} → {len(segs)} episodes")
    labels = [chr(ord("a") + i) if i < 26 else f"_{i}" for i in range(len(segs))]
    return list(zip(labels, segs))


def fetch_done_set(hf_api: HfApi, token: str) -> set:
    """Names of files already in HF_OUTPUT_FOLDER."""
    try:
        return {
            Path(f).name
            for f in list_repo_files(HF_REPO_ID, repo_type="dataset", token=token)
            if f.startswith(HF_OUTPUT_FOLDER + "/")
        }
    except Exception as e:
        log.warning(f"Could not fetch done-set from HF: {e}")
        return set()


# ── PROCESS ONE FILE ──────────────────────────────────────────────────────────
def process_file(
    fp: Path,
    rl: RateLimiter,
    client: genai.Client,
    done: set,
) -> list:
    text = fp.read_text(encoding="utf-8", errors="replace")
    stem = fp.stem
    wc   = word_count(text)
    log.info(f"  {fp.name} ({wc} words)")

    episodes = split_episodes(text, stem) if wc > WORD_THRESHOLD else [("", text)]
    results  = []

    for label, ep_text in episodes:
        out_name = f"{stem}{label}.txt" if label else f"{stem}.txt"
        out_path = CORRECTED_DIR / out_name

        # Already done?
        if out_name in done or out_path.exists():
            log.info(f"  Skip (done): {out_name}")
            results.append(out_path)
            continue

        ep_wc = word_count(ep_text)
        if ep_wc < 80:
            log.warning(f"  Skipping tiny segment {out_name} ({ep_wc} words)")
            continue

        log.info(f"  Correcting: {out_name} ({ep_wc} words)...")
        corrected = rl.call(client, ep_text)

        if corrected:
            # ✅ Model returned corrected text
            out_path.write_text(corrected, encoding="utf-8")
            log.info(f"  ✓ Saved (corrected): {out_name}")
        else:
            # ⚠ All retries failed – save ORIGINAL so episode is not lost
            out_path.write_text(ep_text, encoding="utf-8")
            log.warning(f"  ⚠ Saved (ORIGINAL – correction failed): {out_name}")

        results.append(out_path)

    return results


# ── BATCH UPLOAD ──────────────────────────────────────────────────────────────
def upload_batch(files: list, hf_api: HfApi, batch_num: int):
    files = [f for f in files if f.exists()]
    if not files:
        return
    log.info(f"Uploading batch {batch_num} ({len(files)} files)...")
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{HF_OUTPUT_FOLDER}/{f.name}",
            path_or_fileobj=str(f),
        )
        for f in files
    ]
    hf_api.create_commit(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Corrected transcripts – batch {batch_num} ({len(files)} files)",
    )
    log.info(f"✓ Batch {batch_num} uploaded")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    api_key  = os.environ.get("GEMINI_API_KEY") or exit("❌ GEMINI_API_KEY not set")
    hf_token = os.environ.get("HF_TOKEN")       or exit("❌ HF_TOKEN not set")

    CORRECTED_DIR.mkdir(exist_ok=True)
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    client = genai.Client(api_key=api_key)
    hf_api = HfApi(token=hf_token)
    rl     = RateLimiter()

    # ── 1. Download all transcripts ───────────────────────────────────────────
    log.info("Downloading transcripts from HuggingFace (snapshot)...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(DOWNLOAD_DIR),
        allow_patterns=[f"{HF_INPUT_FOLDER}/*.txt"],
        token=hf_token,
    )
    input_files = sorted((DOWNLOAD_DIR / HF_INPUT_FOLDER).glob("*.txt"))
    log.info(f"Total transcript files: {len(input_files)}")

    # ── 2. Fetch already-corrected list ───────────────────────────────────────
    done = fetch_done_set(hf_api, hf_token)
    log.info(f"Already corrected on HuggingFace: {len(done)} files")

    pending   = []
    batch_num = 1
    processed = 0

    # ── 3. Process ────────────────────────────────────────────────────────────
    for i, fp in enumerate(input_files):
        stem = fp.stem
        if f"{stem}.txt" in done or f"{stem}a.txt" in done:
            log.info(f"[{i+1}/{len(input_files)}] Already done: {fp.name}")
            continue

        log.info(f"[{i+1}/{len(input_files)}] ── {fp.name}")
        try:
            out = process_file(fp, rl, client, done)
            pending.extend(out)
            processed += 1
        except Exception as e:
            log.error(f"Unexpected failure for {fp.name}: {e}")
            continue

        # Upload every 50 files
        if len(pending) >= BATCH_UPLOAD:
            upload_batch(pending[:BATCH_UPLOAD], hf_api, batch_num)
            batch_num += 1
            pending    = pending[BATCH_UPLOAD:]

    # ── 4. Upload remaining ───────────────────────────────────────────────────
    if pending:
        upload_batch(pending, hf_api, batch_num)

    log.info(f"✅ Done! Processed this run: {processed} files")


if __name__ == "__main__":
    main()
