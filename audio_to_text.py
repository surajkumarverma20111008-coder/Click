import os
import re
import requests
import time
import io
from huggingface_hub import HfApi
from pydub import AudioSegment

# Setup Variables
REPO_ID = "Kumarverma11/PocketFM_Audio"
HF_TOKEN = os.environ.get("HF_TOKEN")

# MULTI-ENGINE SETUP
ENGINES = [
    {
        "name": "DeepInfra",
        "url": "https://api.deepinfra.com/v1/openai/audio/transcriptions",
        "key": os.environ.get("DEEPINFRA_KEY"),
        "model": "openai/whisper-large-v3"
    },
    {
        "name": "Telnyx",
        "url": "https://api.telnyx.com/v2/ai/audio/transcriptions",
        "key": os.environ.get("TELNYX_KEY"),
        "model": "whisper-1" # Telnyx uses standard whisper-1 routing
    }
]

active_engine_idx = 0
MAX_FILE_SIZE_BYTES = 24 * 1024 * 1024 # 24 MB limit

api = HfApi()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def transcribe_audio_multi_engine(audio_bytes, retry_count=0):
    global active_engine_idx
    
    if active_engine_idx >= len(ENGINES):
        print("❌ Saare AI Engines ka balance khatam ho chuka hai!")
        return None

    engine = ENGINES[active_engine_idx]
    
    # Agar key set nahi ki gayi hai toh agle engine par skip karein
    if not engine["key"]:
        active_engine_idx += 1
        return transcribe_audio_multi_engine(audio_bytes, 0)

    headers = {"Authorization": f"Bearer {engine['key']}"}
    files = {"file": ("audio.mp3", audio_bytes, "audio/mpeg")}
    data = {
        "model": engine["model"],
        "language": "hi",
        "temperature": "0", # Urdu aur Hallucination Blocker
        "prompt": "यह एक शुद्ध हिंदी कहानी है। इसे केवल देवनागरी (Hindi) और Hinglish शब्दों में ही लिखें। किसी भी उर्दू लिपि (Nastaliq) का उपयोग न करें।"
    }
    
    response = requests.post(engine["url"], headers=headers, files=files, data=data)
    
    if response.status_code == 200:
        return response.json().get("text", "")
        
    elif response.status_code in [401, 402, 403, 429]: # Balance limit ya Rate limit error
        print(f"⚠️ {engine['name']} ka balance/limit khatam ho gaya hai (Error {response.status_code}).")
        active_engine_idx += 1
        
        if active_engine_idx < len(ENGINES):
            new_engine = ENGINES[active_engine_idx]["name"]
            print(f"🔄 Automatic Switching to {new_engine}... (Resuming process)")
            time.sleep(5)
            if isinstance(audio_bytes, io.BytesIO):
                audio_bytes.seek(0)
            return transcribe_audio_multi_engine(audio_bytes, 0)
        else:
            print("❌ Saari API Keys exhaust ho chuki hain.")
            return None
    else:
        print(f"⚠️ {engine['name']} API Error: {response.text}")
        if retry_count < 2:
            time.sleep(10)
            if isinstance(audio_bytes, io.BytesIO):
                audio_bytes.seek(0)
            return transcribe_audio_multi_engine(audio_bytes, retry_count + 1)
        return None

def main():
    print("🚀 VEDA AI Multi-Engine (DeepInfra + Telnyx) Transcriber Start...")
    
    try:
        hf_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
        audio_files = [f for f in hf_files if f.endswith(('.mp3', '.m4a'))]
        txt_files = [f for f in hf_files if f.endswith('.txt')]
        print(f"📊 Total Audio Files: {len(audio_files)} | Text Ban Chuki Files: {len(txt_files)}")
    except Exception as e:
        print(f"❌ HF Repo check fail: {e}")
        return

    audio_files_sorted = sorted(audio_files, key=natural_sort_key)

    for file_name in audio_files_sorted:
        base_name = os.path.splitext(file_name)[0]
        txt_file_name = f"{base_name}.txt"
        
        if txt_file_name in txt_files:
            continue
            
        print(f"\n📥 Sequence Processing -> {file_name}")
        
        try:
            file_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{file_name}"
            file_response = requests.get(file_url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
            
            if file_response.status_code != 200:
                print(f"❌ File download fail: {file_name}")
                continue
                
            temp_audio_file = "temp_audio_file"
            with open(temp_audio_file, "wb") as f:
                f.write(file_response.content)
                
            print(f"🗜️ Quality Drop Kar Rahe Hain (Mono, 32kbps)...")
            audio = AudioSegment.from_file(temp_audio_file)
            audio = audio.set_channels(1)
            
            buffer = io.BytesIO()
            audio.export(buffer, format="mp3", bitrate="32k")
            file_size = buffer.getbuffer().nbytes
            
            full_text = ""
            
            if file_size <= MAX_FILE_SIZE_BYTES:
                print(f"✅ File choti hai ({file_size / (1024*1024):.2f} MB). Bina kaate ek sath bhej rahe hain...")
                buffer.seek(0)
                text = transcribe_audio_multi_engine(buffer)
                if text:
                    full_text = text
                else:
                    print(f"❌ Process fail ho gaya: {file_name}")
                    
            else:
                print(f"✂️ File badi hai ({file_size / (1024*1024):.2f} MB). Sequence mein kaat kar bhejenge...")
                chunk_length_ms = 60 * 60 * 1000 # 60 Min chunk
                chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
                chunk_success = True
                
                for idx, chunk in enumerate(chunks):
                    print(f"   -> Part {idx+1}/{len(chunks)} bhej rahe hain...")
                    chunk_buffer = io.BytesIO()
                    chunk.export(chunk_buffer, format="mp3", bitrate="32k")
                    chunk_buffer.seek(0)
                    
                    text = transcribe_audio_multi_engine(chunk_buffer)
                    if text:
                        full_text += text + " "
                    else:
                        print(f"❌ Part {idx+1} fail. File skip kar rahe hain.")
                        chunk_success = False
                        break
                        
                if not chunk_success:
                    full_text = ""

            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
                
            if full_text.strip():
                with open(txt_file_name, "w", encoding="utf-8") as f:
                    f.write(full_text.strip())
                
                print(f"📤 Uploading {txt_file_name} to Hugging Face...")
                api.upload_file(
                    path_or_fileobj=txt_file_name,
                    path_in_repo=txt_file_name,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
                print(f"✅ FINAL SUCCESS: {txt_file_name} safe ho gayi.")
                
                print(f"⏳ Safe Mode: {ENGINES[active_engine_idx]['name']} server ko protect karne ke liye 10 seconds ruk rahe hain...")
                time.sleep(10)
            else:
                print(f"⚠️ {file_name} skip ho gaya.")
                
            if active_engine_idx >= len(ENGINES):
                print("🛑 Process ko band kar rahe hain kyunki saari keys expire ho chuki hain.")
                break
            
        except Exception as e:
            print(f"❌ Error while processing {file_name}: {e}")

    print("🎉 Auto-Switch Process Complete!")

if __name__ == "__main__":
    main()
          
