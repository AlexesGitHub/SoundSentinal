import os
import torch
import json
import soundfile as sf
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
from models.AASIST import Model
import random

print("--- 2-Minute Diagnostic Run ---")

PROTOCOL_FILE = "mini_protocol.txt"
AUDIO_DIR =  "flac_E_aa/flac_E_eval"

# Wake up the AI
device = torch.device("cpu")
with open("config/AASIST_ASVspoof5.conf", "r") as f:
    config = json.load(f)

model = Model(config["model_config"]).to(device)
model.load_state_dict(torch.load("models/weights/AASIST/best.pth", map_location=device))
model.eval()

def pad_crop(waveform, max_len=64600):
    num_samples = waveform.shape[1]
    if num_samples >= max_len: return waveform[:, :max_len]
    num_repeats = int(max_len / num_samples) + 1
    return waveform.repeat(1, num_repeats)[:, :max_len]

# Read only the first 1000 files for a fast test
with open(PROTOCOL_FILE, "r") as f:
    lines = f.readlines()

random.seed(42) # Keeps the shuffle consistent just in case
random.shuffle(lines)
lines = lines[:1000]

print(f"🎯 Testing 1000 files to diagnose the 34% EER issue...\n")

y_true = []
scores_class_0 = []
scores_class_1 = []

with torch.no_grad():
    for line in tqdm(lines, desc="Diagnostic Run", unit="file"):
        pieces = line.strip().split()
        audio_id = pieces[1] 
        
        # THE FIX: Search the whole line instead of guessing the column!
        is_bonafide = 'bonafide' in line.lower()
        
        audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.flac")
        
        try:
            audio_data, _ = sf.read(audio_path)
            waveform = torch.tensor(audio_data).float()
            if len(waveform.shape) > 1 and waveform.shape[1] > 1:
                waveform = waveform.mean(dim=1)
            waveform = pad_crop(waveform.unsqueeze(0))
            
            outputs = model(waveform)
            final_scores = outputs[1] if isinstance(outputs, tuple) else outputs
            
            probs = torch.nn.functional.softmax(final_scores[0], dim=0)
            
            # Store everything using our new bulletproof check
            y_true.append(1 if is_bonafide else 0)
            scores_class_0.append(probs[0].item())
            scores_class_1.append(probs[1].item())
            
        except Exception as e:
            print(f"Error on {audio_id}: {e}")

print("\n🧮 Crunching Diagnostic EERs...")
y_true = np.array(y_true)

# Function to calculate EER
def get_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    return brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

# Test if Class 1 is Bonafide
eer_1 = get_eer(y_true, np.array(scores_class_1))
# Test if Class 0 is Bonafide
eer_0 = get_eer(y_true, np.array(scores_class_0))

print("\n🚨 --- DIAGNOSTIC RESULTS --- 🚨")
print(f"EER if Class 1 is Bonafide: {eer_1 * 100:.3f}%")
print(f"EER if Class 0 is Bonafide: {eer_0 * 100:.3f}%")