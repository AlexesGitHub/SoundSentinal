import torch
import soundfile as sf
import time
import json
import os

print("--- Deepfake CPU Inference Test ---\n")

# 1. Import the AASIST architecture blueprint
try:
    from models.AASIST import Model
except ImportError:
    print("❌ Error: Could not find the 'models/AASIST.py' file. Let me know if your folder structure is different!")
    exit()

# 2. Setup CPU and load the config
device = torch.device("cpu")
config_path = "config/AASIST_ASVspoof5.conf" # Standard ASVspoof config path
audio_path = "LA_E_1000147.flac"     # CHANGE THIS if your file is a .wav!

try:
    with open(config_path, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: Could not find config file at {config_path}")
    exit()

# 3. Build the model and load the trained weights
print("🧠 Building AASIST model and loading best.pth...")
model = Model(config["model_config"]).to(device)
weights = torch.load("models/weights/AASIST/best.pth", map_location=device)
model.load_state_dict(weights)
model.eval() # Locks the model so it doesn't try to learn from the test file

# 4. Process the Audio File
if not os.path.exists(audio_path):
    print(f"❌ Error: Please put an audio file named '{audio_path}' in the folder!")
    exit()

print(f"🎤 Loading audio file: {audio_path}")

# Read the raw audio data
audio_data, sample_rate = sf.read(audio_path)
waveform = torch.tensor(audio_data).float()

# If the audio is stereo (2 channels), average them down to mono
if len(waveform.shape) > 1 and waveform.shape[1] > 1:
    waveform = waveform.mean(dim=1)

# Add the batch dimension so the AI sees exactly (1, samples)
waveform = waveform.unsqueeze(0)

# 5. The Race: Run Inference and Time It
print("\n⏳ Starting inference on CPU...")
start_time = time.time()

with torch.no_grad(): # Tells PyTorch not to waste memory tracking gradients
    # AASIST usually expects a specific shape, typically (batch_size, samples)
    # We add a batch dimension of 1 for our single file
    outputs = model(waveform)
    
end_time = time.time()

# 6. Results
cpu_time = round(end_time - start_time, 3)
print(f"\n✅ Done! CPU Processing Time: {cpu_time} seconds")

# Grab the final scores (handling whether the model returns a tuple or just the tensor)
import torch.nn.functional as F
final_scores = outputs[1] if isinstance(outputs, tuple) else outputs

# Print the raw tensor numbers
print(f"📊 Raw Logits: {final_scores.tolist()[0]}")

# Convert the raw numbers into readable percentages
probs = F.softmax(final_scores, dim=1).squeeze()
print(f"\n🔍 Final Verdict:")
print(f"Class 0 Probability: {probs[0].item():.2%}")
print(f"Class 1 Probability: {probs[1].item():.2%}")
print("\n(Note: In standard ASVspoof, Class 0 is usually 'Spoof' and Class 1 is 'Bonafide'!)")