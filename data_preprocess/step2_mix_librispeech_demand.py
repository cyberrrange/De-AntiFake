import os
import numpy as np
import soundfile as sf
import random
from pathlib import Path

def mix_signals(clean, noise, snr):
    # Calculate required noise length
    if snr is None:
        # If SNR is None, return original clean signal
        return clean, clean
    clean_length = len(clean)
    noise_length = len(noise)
    
    # If noise is longer than clean, randomly select a segment
    if noise_length > clean_length:
        start = random.randint(0, noise_length - clean_length)
        noise = noise[start:start + clean_length]
    # If noise is shorter than clean, repeat cyclically
    else:
        repetitions = clean_length // noise_length + 1
        noise = np.tile(noise, repetitions)[:clean_length]
    
    # Calculate power of clean and noise signals
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate noise scaling factor based on SNR
    K = np.sqrt(clean_power / (noise_power * 10 ** (snr / 10)))
    
    # Mix signals
    noisy = clean + K * noise
    
    # Normalize to prevent overflow
    max_val = np.max(np.abs(noisy))
    if max_val > 1:
        noisy = noisy / max_val
        clean = clean / max_val  # Scale clean signal accordingly
    
    return clean, noisy

def get_noise_types(demand_base_path):
    # Get all folder names ending with _16k
    noise_types = []
    for path in Path(demand_base_path).glob('*_16k'):
        noise_type = path.name.replace('_16k', '')
        noise_types.append(noise_type)
    return noise_types

def process_dataset(libri_base_path, demand_base_path, output_base_path, split):
    # Automatically get noise types
    noise_types = get_noise_types(demand_base_path)
    channels = range(1, 17)  # DEMAND has 16 channels per scenario
    snr_levels = [15, 10, 5, 0, None] 
    
    # Create output directories
    output_dir = Path(output_base_path) / split
    clean_dir = output_dir / 'clean'
    noisy_dir = output_dir / 'noisy'
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    
    # File to record mixing information
    mix_info_file = open(output_dir.parent / f'{split}_mix_info.txt', 'w')
    
    # Iterate through Librispeech files
    for wav_path in Path(libri_base_path).glob('**/*.wav'):
        # Read clean speech
        clean_signal, sr = sf.read(str(wav_path))
        
        # Randomly select noise type, channel and SNR
        noise_type = random.choice(noise_types)
        #channel = random.choice(channels)
        channel = "01"
        snr = random.choice(snr_levels)
        
        # Build noise file path
        noise_path = Path(demand_base_path) / f'{noise_type}_16k' / noise_type / f'ch{channel}.wav'
        
        # Read noise signal
        noise_signal, _ = sf.read(str(noise_path))
        
        # Mix signals, get scaled clean and noisy signals
        scaled_clean, noisy_signal = mix_signals(clean_signal, noise_signal, snr)
        
        # Save scaled clean signal and mixed noisy signal
        clean_file = clean_dir / wav_path.name
        noisy_file = noisy_dir / wav_path.name
        sf.write(str(clean_file), scaled_clean, sr)
        sf.write(str(noisy_file), noisy_signal, sr)
        
        # Record mixing information
        if snr == None:
            mix_info = f'{wav_path.stem} clean ch{channel} 0\n'
        else:
            mix_info = f'{wav_path.stem} {noise_type} ch{channel} {snr}\n'
        mix_info_file.write(mix_info)
    
    mix_info_file.close()

def main():
    # Set paths
    libri_base = '/path/to/LibriSpeech_wav'
    demand_base = '/path/to/DEMAND'
    output_base = '/path/to/mixed_dataset'
    
    # Process train, validation and test sets
    for split in ['train', 'valid', 'test']:
        libri_split_path = os.path.join(libri_base, split)
        process_dataset(libri_split_path, demand_base, output_base, split)

if __name__ == '__main__':
    main()