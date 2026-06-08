import os
import re
import asyncio
import aiohttp
import time
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

# --- CONFIGURATION & SECRETS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
HF_TOKEN = os.environ.get("HF_TOKEN")

if not GEMINI_API_KEY or not HF_TOKEN:
    raise ValueError("❌ ERROR: API Keys missing hain! GitHub Secrets check karein.")

REPO_ID = "Kumarverma11/PocketFM_Audio"
INPUT_FOLDER = "Generated_Transcripts"   
OUTPUT_FOLDER = "Generated_Transcripts"  
CONTINUITY_FILE = "Continuity_Checker.txt"

LOCAL_DOWNLOAD_DIR = Path("./text_download")
LOCAL_FINAL_DIR = Path("./text_clean_final")
LOCAL_DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)
LOCAL_FINAL_DIR.mkdir(exist_ok=True, parents=True)

# 🔴 28 RPM TARGET (1 Request every 4.3s per model)
DELAY_BETWEEN_CALLS = 4.3 
LAST_CALL_TIME = {"gemma-4-31b-it": 0.0, "gemma-4-26b-a4b-it": 0.0}
MODEL_LOCKS = {"gemma-4-31b-it": asyncio.Lock(), "gemma-4-26b-a4b-it": asyncio.Lock()}

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# 🛡️ LAYER 1: OFFLINE BACKUP (Zero Crash Guarantee)
def offline_local_fix(text):
    text = re.sub(r'[॥\|]', '', text)
    corrections = {
        r'\bअबे\b': 'अभय', r'\bअब है\b': 'अभय', r'\bकि आंश\b': 'कियांश',
        r'\bरी मदद\b': 'मेरी मदद', r'\bतुम री\b': 'तुम मेरी', r'\bवह री\b': 'मेरी',
        r'\bवह री क्या\b': 'मेरी क्या', r'\bशांति 2\b': 'शांति दो', r'\bरे घर\b': 'मेरे घर',
        r'\bरी खैर\b': 'मेरी खैर', r'\bरी बात\b': 'मेरी बात', r'\bसौरी\b': 'सॉरी',
        r'\bऑबेरॉय\b': 'ओबेरॉय'
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r'\s+', ' ', text).strip()

# 🛡️ LAYER 2: AI OUTPUT SAFETY CHECKER
def is_safe_output(text):
    if not text or text.strip() == "": return False
    bad_words = ["Segment", "Professional Hindi", "Self-Correction", "Correct spelling"]
    if any(word.lower() in text.lower() for word in bad_words): return False
    if len(re.findall(r'[a-zA-Z]', text)) > 50: return False
    return True

# 🔴 SMART SPLITTING RULE (Words Based)
def split_merged_episodes(file_path, filename):
    with open(file_path, 'r', encoding='utf-8') as f: 
        content = f.read()
    
    word_count = len(content.split())
    if word_count <= 2000 and "_to_" not in filename.lower():
        return [(filename, content)]
    
    split_pattern = r"(जानने के लिए सुनिए कहानी का अगला भाग सिर्फ पॉकेट पर।|सिर्फ पॉकेट एफएम पर।|कहानी का अगला भाग सिर्फ पॉकेट|पॉकेट एफएम पर सुनिए)"
    parts = re.split(split_pattern, content)
    
    processed_chunks = []
    current_chunk = ""
    
    for i in range(len(parts)):
        text_part = parts[i].strip()
        if not text_part: continue
        
        current_chunk += " " + text_part
        if any(kw in text_part for kw in ["अगला भाग", "पॉकेट एफएम", "पॉकेट पर"]):
            if len(current_chunk.split()) > 200: 
                processed_chunks.append(current_chunk.strip())
                current_chunk = ""
                
    if current_chunk.strip(): 
        processed_chunks.append(current_chunk.strip())

    base_name = filename.replace(".txt", "").split("_to_")[0]
    split_files = []
    
    if len(processed_chunks) > 1:
        for idx, text_block in enumerate(processed_chunks):
            suffix = chr(97 + idx) 
            split_files.append((f"{base_name}{suffix}.txt", text_block))
    else:
        split_files.append((filename, content))
        
    return split_files

async def get_rate_limited_model(index):
    model_name = "gemma-4-31b-it" if index % 2 == 0 else "gemma-4-26b-a4b-it"
    async with MODEL_LOCKS[model_name]:
        now = time.monotonic()
        elapsed = now - LAST_CALL_TIME[model_name]
        if elapsed < DELAY_BETWEEN_CALLS:
            await asyncio.sleep(DELAY_BETWEEN_CALLS - elapsed)
        LAST_CALL_TIME[model_name] = time.monotonic()
    return model_name

# 🛡️ LAYER 3: BULLETPROOF OFFICIAL API CALL
async def call_gemma_api(session, text, index):
    model_name = await get_rate_limited_model(index)
    
    # FIX: Using Official Native Google API Endpoint to avoid KeyError
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    
    sys_instruct = (
        "तुम एक टेक्स्ट एडिटर हो। केवल स्पेलिंग सुधारो: "
        "'अबे/अब है'->'अभय', 'कि आंश'->'कियांश', 'री मदद/रे घर'->'मेरी मदद/मेरे घर'। श्लोक चिह्न (॥ या |) हटाओ। "
        "चेतावनी: कोई स्पष्टीकरण, टिप्पणी या अंग्रेजी का शब्द आउटपुट में नहीं आना चाहिए।"
    )
    
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "systemInstruction": {"parts": [{"text": sys_instruct}]},
        "generationConfig": {"temperature": 0.05, "maxOutputTokens": 4096}
    }
    
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(1, 4):
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status in [429, 500, 503]:
                    await asyncio.sleep(2.0 ** attempt)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                
                # Official Safe JSON Parsing
                ai_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if ai_text: return ai_text, model_name
                
        except Exception as e:
            await asyncio.sleep(1.5 ** attempt)
            
    return None, model_name 

async def handle_single_episode(filename, text_content, index, semaphore, session):
    output_path = LOCAL_FINAL_DIR / filename
    if output_path.exists(): 
        return filename, True
        
    async with semaphore:
        fixed_text, used_model = await call_gemma_api(session, text_content, index)
        
        if fixed_text and is_safe_output(fixed_text):
            final_text = fixed_text
            msg = f"✅ AI Success: {filename} ({used_model})"
        else:
            final_text = offline_local_fix(text_content)
            msg = f"⚡ Fallback Success: {filename} (Offline Fix Applied)"
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_text.strip())
        print(msg)
        return filename, True

def extract_summary(text):
    sentences = [s.strip() for s in re.split(r'[।\.!?]', text) if s.strip()]
    if len(sentences) <= 8: return f"[SHORT]\n{text}\n"
    return f"[START] {'। '.join(sentences[:4])}।\n[END] {'। '.join(sentences[-4:])}।"

async def main():
    print("📥 1. Downloading ONLY Text Files (No Audio)...")
    try:
        # FIX: Added allow_patterns to PREVENT downloading 20GB audio files and fixing 429 error!
        snapshot_download(
            repo_id=REPO_ID, 
            repo_type="dataset", 
            allow_patterns=[f"{INPUT_FOLDER}/*.txt"], 
            ignore_patterns=["*.mp3", "*.wav", "*.m4a"],
            local_dir=str(LOCAL_DOWNLOAD_DIR), 
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"❌ HF Download Error: {e}")
        return
    
    raw_folder = LOCAL_DOWNLOAD_DIR / INPUT_FOLDER
    raw_files = sorted([f for f in os.listdir(raw_folder) if f.endswith('.txt') and f != CONTINUITY_FILE], key=natural_sort_key)
    
    final_pool = [] 
    for f in raw_files: 
        final_pool.extend(split_merged_episodes(os.path.join(raw_folder, f), f))
        
    print(f"🚀 2. Processing {len(final_pool)} files (Parallel Fast Mode)...")
    
    semaphore = asyncio.Semaphore(20) 
    api = HfApi()
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300), connector=aiohttp.TCPConnector(limit=20)) as session:
        PARALLEL_BATCH = 10 
        
        for i in range(0, len(final_pool), PARALLEL_BATCH):
            batch = final_pool[i:i+PARALLEL_BATCH]
            print(f"\n▶️ Processing Files {i+1} to {min(i+PARALLEL_BATCH, len(final_pool))}...")
            
            tasks = [handle_single_episode(fn, content, i+j, semaphore, session) for j, (fn, content) in enumerate(batch)]
            await asyncio.gather(*tasks)
            
            # Batch Upload Tracker
            if (i + PARALLEL_BATCH) % 50 == 0 or (i + PARALLEL_BATCH) >= len(final_pool):
                checker_data = []
                ready_files = sorted([f for f in os.listdir(LOCAL_FINAL_DIR) if f.endswith('.txt') and f != CONTINUITY_FILE], key=natural_sort_key)
                for f_name in ready_files:
                    with open(LOCAL_FINAL_DIR / f_name, 'r', encoding='utf-8') as f:
                        checker_data.append(f"\n{'='*40}\n🎬 {f_name}\n{'='*40}\n{extract_summary(f.read())}\n")
                
                with open(LOCAL_FINAL_DIR / CONTINUITY_FILE, 'w', encoding='utf-8') as f: 
                    f.writelines(checker_data)

                try:
                    # FIX: upload_folder is much faster than uploading files one by one
                    api.upload_folder(
                        folder_path=str(LOCAL_FINAL_DIR), 
                        path_in_repo=OUTPUT_FOLDER, 
                        repo_id=REPO_ID, 
                        repo_type="dataset", 
                        token=HF_TOKEN,
                        commit_message=f"Auto-Fix (Final Fast Mode): Up to file {min(i+PARALLEL_BATCH, len(final_pool))}"
                    )
                    print("☁️ Batch Uploaded to Hugging Face successfully!")
                except Exception as e: 
                    print(f"⚠️ Upload Error (will retry next batch): {e}")

if __name__ == "__main__":
    asyncio.run(main())
            
