import numpy as np
from tqdm import tqdm

from data import AudioAugDataset
import json
import torch
import torch.nn.functional as F

def load_config(config_path):
    with open(config_path, 'r') as f:
        s = f.read()
    config = json.loads(s)
    return config

# Load config file
config = load_config('/path/to/data_preprocess/configs/libri_get_spec.json')
dataset = AudioAugDataset(config)

# Initialize dictionaries and variables
amps_mode_dict = dict()
amps_mode = dict()

amp_max = -100
amp_min = 100

for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
    x = dataset.__getitem__(idx)
    durations = x['durations']
    amp = x['clean']  # Assume 'clean' is amplitude spectrogram
    amp_max = max(amp_max, np.max(amp))
    amp_min = min(amp_min, np.min(amp))
    phonemes = x['phonemes']
    print(len(durations), amp.shape, amp_max, amp_min, len(phonemes))
    start_frame = 0
    for i in range(len(phonemes)):
        phoneme = phonemes[i]
        end_frame = start_frame + durations[i]
        # Extract amplitude spectrogram frames for current phoneme
        phoneme_amp = np.round(np.median(amp[:, start_frame:end_frame], axis=1), 1)
        if phoneme not in amps_mode_dict:
            amps_mode_dict[phoneme] = [phoneme_amp]
        else:
            amps_mode_dict[phoneme].append(phoneme_amp)
        start_frame = end_frame

# Calculate average amplitude spectrogram for each phoneme
for p in amps_mode_dict:
    print(f"Processing phoneme: {p}")
    amps_mode[p] = np.mean(np.asarray(amps_mode_dict[p]), axis=0)
    print(amps_mode[p])

# Save average amplitude spectrogram to file
with open("/path/to/data/librispeech_metadata/amps_avg_libri_spec.txt", "w+") as outfile:
    for p in amps_mode:
        msg = p + '\t' + str(amps_mode[p].tolist()) + '\n'
        outfile.write(msg)

print("amps_avg saved")
print("amp max and min:", amp_max, amp_min)
