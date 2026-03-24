import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

print("--- EER Calculation ---")
SCORE_FILE = "my_baseline_scores.txt"

y_true = []
y_scores = []

print(f"📊 Reading 71k scores from {SCORE_FILE}...")
with open(SCORE_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split()
        # Our format was: audio_id attack_type true_label bonafide_score
        label = parts[2]
        score = float(parts[3])
        
        # In ASVspoof, Bonafide (Real) = 1, Spoof (Deepfake) = 0
        if label == 'bonafide':
            y_true.append(1)
        elif label == 'spoof':
            y_true.append(0)
        
        y_scores.append(score)

y_true = np.array(y_true)
y_scores = np.array(y_scores)

print("🧮 Finding the exact intersection of FAR and FRR...")
# Calculate the ROC curve points
fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

# Use scipy to perfectly interpolate the exact Equal Error Rate crossing point
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
optimal_threshold = interp1d(fpr, thresholds)(eer)

print("\n🎉 --- FINAL THESIS BASELINE RESULTS --- 🎉")
print(f"Total Files Evaluated: {len(y_true)}")
print(f"Equal Error Rate (EER): {eer * 100:.3f}%")
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print("-----------------------------------------")