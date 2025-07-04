import os
import re
from pathlib import Path

def extract_audio_id(filename):
    """Extract audio ID from filename
    Example: librispeech_p1089-134686-0000.wav -> 1089-134686-0000
    """
    # Remove prefix "librispeech_" and suffix ".wav"
    if filename.startswith("librispeech_p"):
        audio_id = filename[len("librispeech_p"):]
        if audio_id.endswith(".wav"):
            audio_id = audio_id[:-4]
        return audio_id
    return None

def load_transcriptions(folder1):
    """Load all transcriptions from folder1 (LibriSpeech/test-clean or similar)
    Returns dictionary: {audio_id: transcription}
    """
    transcriptions = {}
    
    for root, dirs, files in os.walk(folder1):
        for file in files:
            if file.endswith('.trans.txt'):
                trans_file_path = os.path.join(root, file)
                
                with open(trans_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Split line: [speaker]-[chapter]-[uttr] [transcription]
                            parts = line.split(' ', 1)
                            if len(parts) >= 2:
                                audio_id = parts[0]
                                transcription = parts[1]
                                transcriptions[audio_id] = transcription
    
    return transcriptions

def find_and_save_transcriptions(folder1, folder2, folder3):
    """Main function: find matching transcriptions for .wav files in folder2 and save .txt to folder3"""
    
    # Create output folder
    os.makedirs(folder3, exist_ok=True)
    
    # Load all transcriptions
    print("Loading transcriptions...")
    transcriptions = load_transcriptions(folder1)
    print(f"Loaded {len(transcriptions)} transcriptions")
    
    # Process audio files in folder2
    matched_count = 0
    total_count = 0
    
    for file in os.listdir(folder2):
        if file.endswith('.wav') and file.startswith('librispeech_p'):
            total_count += 1
            
            # Extract audio ID
            audio_id = extract_audio_id(file)
            if not audio_id:
                print(f"Warning: Cannot extract audio ID from filename: {file}")
                continue
            
            # Find matching transcription
            if audio_id in transcriptions:
                transcription = transcriptions[audio_id]
                
                # Create output filename (remove .wav, add .txt)
                output_filename = file[:-4] + '.txt'
                output_path = os.path.join(folder3, output_filename)
                
                # Save transcription
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                
                matched_count += 1
                print(f"Matched: {file} -> {output_filename}")
            else:
                print(f"No matching transcription found: {file} (Audio ID: {audio_id})")
    
    print(f"\nProcessing completed!")
    print(f"Total files: {total_count}")
    print(f"Successfully matched: {matched_count}")
    print(f"Failed to match: {total_count - matched_count}")

if __name__ == "__main__":
    # Set folder paths
    folder1 = "/path/to/LibriSpeech/test-clean"  # Folder containing LibriSpeech transcription files (LibriSpeech/test-clean or similar)
    folder2 = "/path/to/test_set_example/audio"  # Folder containing audio files (sth like librispeech_p1089-134686-0000.wav)
    folder3 = "/path/to/test_set_example/text"  # Output folder for transcription files
    
    # You can pass paths via command line arguments
    import sys
    if len(sys.argv) == 4:
        folder1 = sys.argv[1]
        folder2 = sys.argv[2]
        folder3 = sys.argv[3]
    
    print(f"Folder1 (transcription source): {folder1}")
    print(f"Folder2 (audio files): {folder2}")
    print(f"Folder3 (output): {folder3}")
    
    # Check if input folders exist
    if not os.path.exists(folder1):
        print(f"Error: Folder1 does not exist: {folder1}")
        sys.exit(1)
    
    if not os.path.exists(folder2):
        print(f"Error: Folder2 does not exist: {folder2}")
        sys.exit(1)
    
    # Execute matching and saving
    find_and_save_transcriptions(folder1, folder2, folder3)