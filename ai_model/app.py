import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from pathlib import Path
from flask_cors import CORS
import os
import librosa
import numpy as np

# ===================================================================
# 1. MODEL ARCHITECTURE
# ===================================================================
class AudioClassifierCNN(nn.Module):
    def __init__(self):
        super(AudioClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 128, 126)
            dummy_output = self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(dummy_input))))))
            self.flattened_size = dummy_output.numel()

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ===================================================================
# 2. LOAD MODEL & UTILS
# ===================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "deepfake_audio_detector.pth"
model = AudioClassifierCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

class_names = ["Real Audio", "Deepfake Audio"]

def preprocess_audio(audio_path, max_len=64000):
    # librosa handles MP3/WAV decoding better on Windows than torchaudio
    waveform_np, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    if len(waveform_np) > max_len:
        waveform_np = waveform_np[:max_len]
    else:
        padding = max_len - len(waveform_np)
        waveform_np = np.pad(waveform_np, (0, padding))
        
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
    )
    spectrogram = transform(waveform)
    return spectrogram.unsqueeze(0) 

# ===================================================================
# 3. FLASK SERVER
# ===================================================================
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    temp_path = "temp_prediction_file.wav"
    
    try:
        audio_file.save(temp_path)
        tensor = preprocess_audio(temp_path)
        
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted_idx = torch.max(outputs.data, 1)
            probabilities = F.softmax(outputs, dim=1)
            
            confidence = probabilities[0][predicted_idx.item()].item()
            predicted_label = class_names[predicted_idx.item()]

        return jsonify({
            "prediction": predicted_label,
            "confidence": f"{confidence:.2%}"
        })
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True, port=5000)