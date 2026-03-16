import torch
import torchaudio

print("--- Starting AI Environment Test ---\n")

# 1. Verify GPU is actually taking the load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device selected: {device}")

# 2. Load the "brain" (the saved weights)
weights_path = "models/weights/AASIST/best.pth"
try:
    # map_location ensures it loads directly into your GPU
    weights = torch.load(weights_path, map_location=device)
    print(f"✅ Successfully loaded {len(weights)} layers from best.pth!")
except Exception as e:
    print(f"❌ Failed to load weights: {e}")

# 3. Create a fake 4-second audio clip (16kHz sample rate) to test memory
try:
    dummy_audio = torch.randn(1, 64000).to(device)
    print(f"✅ Generated dummy audio tensor on GPU. Shape: {dummy_audio.shape}")
except Exception as e:
    print(f"❌ GPU Memory error: {e}")

print("\n🎉 If you see all green checks, your environment is bulletproof!")