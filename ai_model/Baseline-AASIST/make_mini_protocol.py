import os
from tqdm import tqdm

# --- UPDATE THESE TWO PATHS AFTER YOU EXTRACT ---
AUDIO_DIR = "flac_E_aa/flac_E_eval" # The folder containing your extracted .flac files
MASTER_PROTOCOL = "./ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv" # The master text file from the protocols extraction
# ------------------------------------------------

MINI_PROTOCOL = "mini_protocol.txt"

print("🔍 Scanning your extracted audio folder...")
# Grab all the file names (minus the .flac part)
available_audio = set([f.replace(".flac", "") for f in os.listdir(AUDIO_DIR) if f.endswith(".flac")])
print(f"✅ Found {len(available_audio)} audio files ready to go.")

print("📜 Trimming the master protocol...")
matched_count = 0

with open(MASTER_PROTOCOL, "r") as master, open(MINI_PROTOCOL, "w") as mini:
    # Read the master file line by line
    for line in master:
        pieces = line.strip().split()
        if len(pieces) < 2: continue
        
        audio_id = pieces[1] # ASVspoof always puts the audio ID in the second column
        
        # If we have the audio file, save the answer key line!
        if audio_id in available_audio:
            mini.write(line)
            matched_count += 1

print(f"🎉 Done! Wrote {matched_count} matching lines to {MINI_PROTOCOL}.")