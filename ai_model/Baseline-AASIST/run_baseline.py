import os
import torch
import json
import soundfile as sf
from tqdm import tqdm
from models.AASIST import Model

print("--- ASVspoof LA Baseline Evaluation ---")

# 1. Setup paths based on your local machine
PROTOCOL_FILE = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
AUDIO_DIR = "LA/ASVspoof2019_LA_eval/flac"
OUTPUT_FILE = "my_baseline_scores.txt"

# 2. Build the Model
print("🧠 Waking up AASIST brain...")
device = torch.device("cpu")
with open("config/AASIST_ASVspoof5.conf", "r") as f:
    config = json.load(f)

model = Model(config["model_config"]).to(device)
model.load_state_dict(torch.load("models/weights/AASIST/best.pth", map_location=device))
model.eval() # Lock it for inference!

# Helper function to ensure audio is exactly 4 seconds (looping, not silencing!)
def pad_crop(waveform, max_len=64600):
    num_samples = waveform.shape[1]
    
    # If it's too long, chop the end off
    if num_samples >= max_len:
        return waveform[:, :max_len]
    
    # If it's too short, loop (repeat) the audio until it's long enough
    num_repeats = int(max_len / num_samples) + 1
    repeated_waveform = waveform.repeat(1, num_repeats)
    
    # Trim the final looped version to exactly max_len
    return repeated_waveform[:, :max_len]

# 3. Read the master answer key
print("📜 Loading protocol file...")
with open(PROTOCOL_FILE, "r") as f:
    lines = f.readlines()

print(f"🎯 Found {len(lines)} files to process. Starting the engine...\n")

# Check what we've already processed
processed_files = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            processed_files.add(line.split()[0]) # Grab the audio_id
print(f"⏭️ Found {len(processed_files)} already processed. Resuming...")

# 4. The Assembly Line
with open(OUTPUT_FILE, "a") as out_file:
    # tqdm wraps around our list to give us a sweet progress bar
    with torch.no_grad():
        for line in tqdm(lines, desc="Processing Audio", unit="file"):
            pieces = line.strip().split()
            speaker_id = pieces[0]
            audio_id = pieces[1]

            if audio_id in processed_files:
                            continue

            environment = pieces[2]
            attack_type = pieces[3]
            true_label = pieces[4] # 'bonafide' or 'spoof'
            
            audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.flac")
            
            try:
                # Read using our bulletproof soundfile bypass
                audio_data, sr = sf.read(audio_path)
                waveform = torch.tensor(audio_data).float()
                
                # Squeeze to Mono if needed, add batch dimension
                if len(waveform.shape) > 1 and waveform.shape[1] > 1:
                    waveform = waveform.mean(dim=1)
                waveform = waveform.unsqueeze(0)
                
                # Standardize length
                waveform = pad_crop(waveform)
                
                # Run Inference
                outputs = model(waveform)
                final_scores = outputs[1] if isinstance(outputs, tuple) else outputs
                
                # Grab the score for Class 1 (Bonafide/Real)
                bonafide_score = final_scores[0][1].item()
                
                # Write the result to our output file (Audio_ID, True_Label, AI_Score)
                out_file.write(f"{audio_id} {attack_type} {true_label} {bonafide_score:.6f}\n")
                
            except Exception as e:
                print(f"\n❌ Error on file {audio_id}: {e}")

print(f"\n✅ Bloody legend! Evaluation complete. Scores saved to '{OUTPUT_FILE}'.")