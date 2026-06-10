"""
Mr. Lawaris - Transcript Correction Pipeline
Uses Gemma 4 via Gemini API (2 models, 14 req/min each = 28 req/min total)
- Downloads from HuggingFace
- Splits large files by episode boundaries
- Corrects ASR errors using context-aware Gemma 4
- Uploads corrected files in batches of 50
"""

import os
import re
import time
import json
import math
import random
import logging
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
import google.generativeai as genai

# ─── CONFIG ───────────────────────────────────────────────────────────────────
HF_REPO_ID      = "Kumarverma11/PocketFM_Audio"
HF_FOLDER       = "Generated_Transcripts"
OUTPUT_FOLDER   = "Corrected_Transcripts"
WORK_DIR        = Path("./work")
CORRECTED_DIR   = Path("./corrected")

# Two Gemma 4 models - each gets 14 RPM (safe under 15 limit)
GEMMA_MODELS = [
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
]
RPM_PER_MODEL   = 14          # safe limit per model
BATCH_UPLOAD    = 50          # HF commit every N files
WORD_THRESHOLD  = 3000        # files > this need splitting
SMALL_THRESHOLD = 2000        # files <= this = single episode

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─── EPISODE BOUNDARY PATTERNS ────────────────────────────────────────────────
BOUNDARY_PATTERNS = [
    r"जानने के लिए सुनिए.*?(?:अगला एपिसोड|कहानी का अगला भाग|अगला भाग)[^\n]*",
    r"(?:कहानी का|इस कहानी का) अगला भाग[^\n]*(?:पॉकेट एफएम|Pocket\s*FM)[^\n]*",
    r"(?:सिर्फ|केवल) (?:पॉकेट एफएम|Pocket\s*FM) पर[^\n]*",
    r"अभी सुनें[^\n]*(?:अगला भाग|अगला एपिसोड)[^\n]*",
    r"(?:जवाब जानने के लिए|जानने के लिए)[^\n]*(?:सुनिए|सुनें)[^\n]*",
    r"(?:क्या|कब|कैसे|कौन)[^\n]*\?[^\n]*(?:जानने के लिए|सुनिए|सुनें)[^\n]*",
    r"(?:तकदीर|किस्मत|भगवान)[^\n]*(?:के लिए|सोच)[^\n]*(?:सुनिए|सुनें|जानिए)[^\n]*",
]
COMBINED_PATTERN = re.compile(
    "|".join(f"({p})" for p in BOUNDARY_PATTERNS),
    re.IGNORECASE | re.UNICODE
)

# ─── RATE LIMITER ─────────────────────────────────────────────────────────────
class DualModelRateLimiter:
    """Round-robin across 2 models, each capped at RPM_PER_MODEL calls/min."""
    def __init__(self):
        self.idx        = 0
        self.timestamps = {m: [] for m in GEMMA_MODELS}

    def get_model(self) -> str:
        # Pick the model with most remaining quota this minute
        now = time.time()
        best = None
        best_remaining = -1
        for m in GEMMA_MODELS:
            # Purge timestamps older than 60s
            self.timestamps[m] = [t for t in self.timestamps[m] if now - t < 60]
            remaining = RPM_PER_MODEL - len(self.timestamps[m])
            if remaining > best_remaining:
                best_remaining = remaining
                best = m
        return best, best_remaining

    def call(self, fn, *args, **kwargs):
        """Execute fn with rate-limiting. Retries on 429."""
        max_retries = 5
        for attempt in range(max_retries):
            model, remaining = self.get_model()
            if remaining <= 0:
                # Both models exhausted - wait until oldest slot frees
                all_ts = sorted([
                    t for ts in self.timestamps.values() for t in ts
                ])
                wait = 61 - (time.time() - all_ts[0]) if all_ts else 5
                log.info(f"Rate limit reached. Waiting {wait:.1f}s...")
                time.sleep(max(wait, 1))
                continue
            try:
                self.timestamps[model].append(time.time())
                result = fn(model, *args, **kwargs)
                return result
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "rate" in err:
                    wait = 60 + random.uniform(5, 15)
                    log.warning(f"429 on {model}. Waiting {wait:.0f}s... (attempt {attempt+1})")
                    # Remove the failed timestamp (didn't count)
                    if self.timestamps[model]:
                        self.timestamps[model].pop()
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded on rate limiter")

# ─── WORD COUNT ───────────────────────────────────────────────────────────────
def word_count(text: str) -> int:
    return len(text.split())

# ─── EPISODE SPLITTER ─────────────────────────────────────────────────────────
def find_boundaries(text: str) -> list[int]:
    """Return list of character positions where episode boundaries occur."""
    boundaries = []
    for m in COMBINED_PATTERN.finditer(text):
        boundaries.append(m.end())
    return boundaries

def split_episodes(text: str, base_name: str) -> list[tuple[str, str]]:
    """
    Split large transcript into episodes.
    Returns list of (filename_suffix, episode_text).
    e.g. [("111a", text1), ("111b", text2), ...]
    """
    boundaries = find_boundaries(text)
    if not boundaries:
        log.warning(f"No boundaries found in {base_name} - keeping as single file")
        return [("", text)]

    # Build segments
    segments = []
    prev = 0
    for end in boundaries:
        seg = text[prev:end].strip()
        if seg:
            segments.append(seg)
        prev = end
    # Remaining text after last boundary
    tail = text[prev:].strip()
    if tail and word_count(tail) > 200:
        segments.append(tail)

    if len(segments) <= 1:
        return [("", text)]

    # Label as a, b, c, ...
    labels = []
    for i in range(len(segments)):
        suffix = chr(ord('a') + i) if i < 26 else f"_{i}"
        labels.append(suffix)
    
    log.info(f"Split {base_name} into {len(segments)} episodes")
    return list(zip(labels, segments))

# ─── ASR CORRECTION PROMPT ────────────────────────────────────────────────────
SYSTEM_PROMPT = """तुम एक Hindi ASR transcript correction expert हो।

तुम्हारा काम:
- केवल ASR (speech-to-text) errors सुधारना है
- कहानी की content, plot, या dialogue नहीं बदलनी
- नए sentences नहीं जोड़ने
- केवल गलत पहचाने गए शब्दों को context के आधार पर सुधारना

Common ASR errors जो सुधारने हैं:
- Character names: अबे→अभय, अभी-अभी→अभय, राज→Raj (context देखो)
- Broken words: ब न ना→बनाना, क ह ना→कहना
- Missing spaces या extra spaces
- Punctuation fixes (sentences properly end करना)
- Similar sounding words जो context में गलत हों

Output में केवल corrected Hindi story text दो।
कोई explanation, notes, या extra text नहीं।
Original formatting और paragraph breaks preserve करो।"""

def correction_prompt(episode_text: str) -> str:
    return f"""यह एक Hindi audio drama (Mr. Lawaris) का episode transcript है।

ASR errors सुधारो और corrected text वापस दो।

TRANSCRIPT:
{episode_text}"""

# ─── GEMINI API CALL ──────────────────────────────────────────────────────────
def call_gemini(model: str, text: str) -> str:
    """Call Gemini API with given model and return corrected text."""
    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
    )
    response = client.generate_content(
        correction_prompt(text),
        generation_config=genai.GenerationConfig(
            temperature=0.1,    # Low temp for correction task
            max_output_tokens=8192,
        )
    )
    return response.text.strip()

# ─── PROCESS ONE FILE ─────────────────────────────────────────────────────────
def process_file(
    file_path: Path,
    rate_limiter: DualModelRateLimiter,
    output_dir: Path
) -> list[Path]:
    """
    Process one transcript file.
    Returns list of output file paths created.
    """
    text = file_path.read_text(encoding="utf-8")
    stem = file_path.stem      # e.g. "episode_111"
    suffix = file_path.suffix  # e.g. ".txt"
    wc = word_count(text)
    
    log.info(f"Processing: {file_path.name} ({wc} words)")

    # Decide: split or keep whole
    if wc <= SMALL_THRESHOLD:
        # Small file - single episode, correct as-is
        episodes = [("", text)]
        log.info(f"  → Single episode (≤{SMALL_THRESHOLD} words)")
    elif wc <= WORD_THRESHOLD:
        # Medium file - still process as-is (may be 1-2 episodes)
        episodes = [("", text)]
        log.info(f"  → Processing whole ({wc} words, under threshold)")
    else:
        # Large file - split by episode boundaries
        log.info(f"  → Large file ({wc} words), detecting boundaries...")
        episodes = split_episodes(text, stem)

    output_paths = []
    for label, ep_text in episodes:
        ep_wc = word_count(ep_text)
        if ep_wc < 100:
            log.warning(f"  Skipping tiny segment ({ep_wc} words)")
            continue

        # Build output filename
        if label:
            out_name = f"{stem}{label}{suffix}"   # episode_111a.txt
        else:
            out_name = f"{stem}{suffix}"           # episode_111.txt
        out_path = output_dir / out_name

        if out_path.exists():
            log.info(f"  Already done: {out_name} - skipping")
            output_paths.append(out_path)
            continue

        log.info(f"  Correcting: {out_name} ({ep_wc} words)...")
        
        # Call Gemma via rate limiter
        corrected = rate_limiter.call(call_gemini, ep_text)
        
        out_path.write_text(corrected, encoding="utf-8")
        output_paths.append(out_path)
        log.info(f"  ✓ Saved: {out_name}")

    return output_paths

# ─── HUGGINGFACE UPLOAD ───────────────────────────────────────────────────────
def upload_batch(files: list[Path], hf_api: HfApi, batch_num: int):
    """Upload a batch of corrected files to HuggingFace."""
    if not files:
        return
    log.info(f"Uploading batch {batch_num} ({len(files)} files)...")
    
    # Build list of (local_path, repo_path) pairs
    path_in_repo_list = []
    local_paths = []
    for f in files:
        path_in_repo_list.append(f"{OUTPUT_FOLDER}/{f.name}")
        local_paths.append(str(f))
    
    hf_api.upload_files(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        path_or_fileobj=[open(p, "rb") for p in local_paths],
        path_in_repo=path_in_repo_list,
        commit_message=f"Add corrected transcripts - batch {batch_num} ({len(files)} files)",
    )
    log.info(f"✓ Batch {batch_num} uploaded successfully")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Setup API
    api_key = os.environ.get("GEMINI_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    if not hf_token:
        raise ValueError("HF_TOKEN not set")
    
    genai.configure(api_key=api_key)
    hf_api = HfApi(token=hf_token)
    
    WORK_DIR.mkdir(exist_ok=True)
    CORRECTED_DIR.mkdir(exist_ok=True)
    
    # ── Step 1: List all transcript files from HuggingFace ──
    log.info("Fetching file list from HuggingFace...")
    all_files = list(list_repo_files(
        HF_REPO_ID,
        repo_type="dataset",
        token=hf_token
    ))
    transcript_files = [
        f for f in all_files
        if f.startswith(HF_FOLDER + "/") and f.endswith(".txt")
    ]
    log.info(f"Found {len(transcript_files)} transcript files")
    
    # ── Step 2: Check which are already corrected ──
    done_files = set(list(list_repo_files(
        HF_REPO_ID, repo_type="dataset", token=hf_token
    )))
    
    rate_limiter = DualModelRateLimiter()
    pending_upload = []
    batch_num = 1
    
    for i, hf_path in enumerate(sorted(transcript_files)):
        filename = Path(hf_path).name
        stem = Path(hf_path).stem
        
        # Check if already corrected (skip if found in output folder)
        corrected_path_check = f"{OUTPUT_FOLDER}/{stem}.txt"
        if corrected_path_check in done_files:
            log.info(f"[{i+1}/{len(transcript_files)}] Already corrected: {filename}")
            continue
        
        # Download file
        local_path = WORK_DIR / filename
        if not local_path.exists():
            log.info(f"Downloading: {filename}")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_path,
                repo_type="dataset",
                local_dir=str(WORK_DIR),
                token=hf_token,
            )
        
        # Process (split if needed + correct)
        try:
            output_files = process_file(local_path, rate_limiter, CORRECTED_DIR)
            pending_upload.extend(output_files)
        except Exception as e:
            log.error(f"Failed to process {filename}: {e}")
            continue
        
        # Batch upload every 50 files
        if len(pending_upload) >= BATCH_UPLOAD:
            upload_batch(pending_upload[:BATCH_UPLOAD], hf_api, batch_num)
            batch_num += 1
            pending_upload = pending_upload[BATCH_UPLOAD:]
    
    # Upload remaining files
    if pending_upload:
        upload_batch(pending_upload, hf_api, batch_num)
    
    log.info("✅ Pipeline complete!")

if __name__ == "__main__":
    main()
