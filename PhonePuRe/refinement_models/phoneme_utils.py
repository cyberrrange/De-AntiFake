import numpy as np
import glob

from tensorboard import summary
from tqdm import tqdm
from torchaudio import load, save
import torch
import os
import time
from pypapi import events, papi_high as high


from sgmse.util.other import *

import matplotlib.pyplot as plt
import sys
import logging
import subprocess
from praatio import tgio
import math
import shutil
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def init_avgspec_dict(avg_spec_path="../data/librispeech_metadata/amps_avg_libri_spec.txt"):
	
	avgspec_dict = dict()
	with open(avg_spec_path) as f:
		lines = [line.strip() for line in f.readlines()]
	for line in lines:
		p, value = line.split('\t')
		value = value[1:-1].split(',')
		value = np.round(np.array([float(x) for x in value]), 1)
		avgspec_dict[p] = value
	return avgspec_dict
def create_metadata(noisy_files, f_path):
	uttinfo_fpath_subsets = {"test": ""} 						
	uttinfo_fpath_subsets["test"] = f_path
	for subset in uttinfo_fpath_subsets.keys():
		# uttinfo
		with open(uttinfo_fpath_subsets[subset], 'r') as f:
			all_metadata = [line.strip() for line in f.readlines()]
			# Create a dictionary with clean_path as key and metadata as value
			metadata_dict = {}
			for meta in all_metadata:
				parts = meta.split('|')
				if len(parts) < 5:
					raise ValueError(f"Metadata line does not have enough parts: {meta}")
				_, phonemes_duration, phonemes_code, clean_path, _ = parts
				metadata_dict[os.path.basename(clean_path).split('_p')[-1]] = meta
	# Reorder metadata according to the order of noisy_files
	_metadata = []
	for noisy_file in noisy_files:
		# Get filename
		filename = os.path.basename(noisy_file)
		
		# Try to find filename match in metadata_dict
		meta = metadata_dict.get(filename)
		
		if meta is None:
			# If full path doesn't match, try matching only by the latter part of [dataset]_p[spkid]-[uttr].wav
			meta = metadata_dict.get(filename.split('_p')[-1])
		
		if meta is None:
			logger.warning(f"Warning: Metadata for noisy file {noisy_file} not found.")
			meta = "|".join(["", "", "", filename.split('_p')[-1], filename.split('_p')[-1].split('-')[0]])
			#raise ValueError(f"Metadata for clean file {clean_path} not found.")
		
		_metadata.append(meta)
	logger.info(f"metadata_length:{str(len(_metadata))}")
	return _metadata
# Tags
def get_text_spec_from_metadata(item_meta, filename, avgspec_dict):
	try:
		start_frames, phonemes_duration, phonemes_code, clean_path, speaker_id = item_meta.split(
			'|')
		try:
			phonemes_code = phonemes_code.split()
			phonemes_duration = [int(dur) for dur in phonemes_duration.split()]
			text_spec = []
			for ph, dur in zip(phonemes_code, phonemes_duration):
				text_spec.append(np.repeat(
					np.expand_dims(avgspec_dict[ph], 1), dur, 1))
			text_spec = np.concatenate(text_spec, axis=1)
		except:
			text_spec = None
			logger.warning(f"Warning: Phonemes for noisy file {filename} not found.")

	except:
		text_spec = None
		logger.warning(f"Warning: Metadata of {filename} has wrong format: {item_meta}.")
	return text_spec

def get_speaker_from_file_name(noisy_align_file):
	if noisy_align_file.endswith(".wav"):
		speaker_chapter_uttr = noisy_align_file.split("/")[-1].split(".")[0].split('_p')[-1].split("-")
		if len(speaker_chapter_uttr) != 3:
			logger.warning(f"Warning: Noisy align file {noisy_align_file} does not have correct format.")
			return None
		speaker, chapter, uttr = speaker_chapter_uttr
		return speaker

def get_text_spec_from_file_name(file_name, y, _metadata, avg_spec_dict, noisy_files, stft):
	Y = stft(y)
	try:
		item_meta = _metadata[noisy_files.index(file_name)]
		text_spec = get_text_spec_from_metadata(item_meta, file_name, avg_spec_dict)
		text_spec = torch.tensor(text_spec).unsqueeze(0).float()
		if text_spec.shape[-1] / Y.shape[-1] > 1.3 or text_spec.shape[-1] / Y.shape[-1] < 0.8:
			logger.warning(f"Warning: {file_name} text_spec length: {text_spec.shape[-1]} and audio_spec length: {Y.shape[-1]} may not match.")
		if text_spec.shape[-1] < Y.shape[-1]:
			pad_width = Y.shape[-1] - text_spec.shape[-1]
			text_spec = torch.nn.functional.pad(text_spec, (0, pad_width), "constant", 0)
		elif text_spec.shape[-1] > Y.shape[-1]:
			text_spec = text_spec[..., :Y.shape[-1]]
	except:
		text_spec = torch.zeros_like(Y).float()
		logger.warning(f"Warning: Metadata for noisy file {file_name} not found, using zero text spec instead.")
	return text_spec

def run_command_in_conda_env(env_name, command):
    # Get current conda environment
    current_env = os.environ.get('CONDA_PREFIX')

    
    full_command = f'conda run -n {env_name} {command}'
    logger.info(full_command)
    try:
        result = subprocess.run(full_command, shell=True, executable='/bin/bash', check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e}")
        return None
	
    env = os.environ.get('CONDA_PREFIX')
    logger.info(f"Current environment: {env}")
	


    return result

def create_metadata_from_textgrid(noisy_dir, textgrid_dir, sampling_rate=16000, hop_length=128):

    _metadata = []
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    for noisy_path in noisy_files:
        base_name = os.path.splitext(os.path.basename(noisy_path))[0]
        speaker = base_name.split('_p')[-1].split('-')[0]
        # .TextGrid will be in the output directory after alignment
        textgrid_path = os.path.join(textgrid_dir, speaker, base_name + '.TextGrid')
        if not os.path.exists(textgrid_path):
            logger.warning(f"TextGrid not found: {textgrid_path}")
			# stamps_str, durations_str, phones_str, noisy_basename, speaker
            result_str = "|".join(["", "", "", noisy_path.split('/')[-1].split('_p')[-1], speaker])
            _metadata.append(result_str)
            continue
        # Open the TextGrid files
        textgrid = tgio.openTextgrid(textgrid_path)
        phones_tier = textgrid.tierDict.get("phones")
        if not phones_tier:
            logger.warning(f"No 'phones' found in TextGrid {textgrid_path}.")
            result_str = "|".join(["", "", "", noisy_path.split('/')[-1].split('_p')[-1], speaker])
            _metadata.append(result_str)
            continue

        # Define a function to convert time to frame index
        def time2frame(t):
            return math.floor((t + 1e-9) * sampling_rate / hop_length)

        sil_phones = ["", "sp", "spn", "sil", '', "''", '""']

        phones = []
        stamps = []
        durations = []
        previous_end = 0.0  # Initialize previous end time

        # Iterate through each entry in the phoneme tier
        for entry in phones_tier.entryList:
            s, e, p = entry.start, entry.end, entry.label
            
            # Check if there's a gap (silence)
            if s > previous_end:
                phones.append('sil')
                start_frame = time2frame(previous_end)
                end_frame = time2frame(s)
                stamps.append(start_frame)
                durations.append(end_frame - start_frame)
            
            # Add current phoneme
            if p not in sil_phones:
                phones.append(p)  # Regular phoneme
            else:
                phones.append('sil')  # Silence phoneme
            
            start_frame = time2frame(s)
            end_frame = max(time2frame(e), start_frame + 1)
            stamps.append(start_frame)
            durations.append(end_frame - start_frame)
            
            previous_end = e  # Update previous end time

        # If there's still a gap after the last phoneme, add silence
        audio_duration = textgrid.maxTimestamp
        if previous_end < audio_duration:
            phones.append('sil')
            start_frame = time2frame(previous_end)
            end_frame = time2frame(audio_duration)
            stamps.append(start_frame)
            durations.append(end_frame - start_frame)

        # Convert frame number list to string
        stamps_str = " ".join(map(str, stamps))
        durations_str = " ".join(map(str, durations))
        phones_str = " ".join(phones)

        # Construct final result string
        # Here we assume speaker information is obtained from filename or other sources, using 'speaker' as example
        speaker = noisy_path.split('/')[-1].split('_p')[-1].split('-')[0]
        noisy_basename = noisy_path.split('/')[-1].split('_p')[-1]
        result_str = "|".join([stamps_str, durations_str, phones_str, noisy_basename, speaker])
        _metadata.append(result_str)
    return _metadata




def create_metadata_from_audio_text_mfa(noisy_dir, 
										text_dir,
										dict_path='english_us_arpa', 
										model_path='english_us_arpa', 
										output_dir=None,):
    if output_dir is None:
        output_dir_name = f"{os.path.basename(noisy_dir)}-mfa"
        output_dir = os.path.join(os.path.dirname(noisy_dir), output_dir_name)
    textgrid_dir_name = f"{os.path.basename(noisy_dir)}-textgrid"
    textgrid_dir = os.path.join(os.path.dirname(noisy_dir), textgrid_dir_name)
    noisy_audio_dir = noisy_dir
    if os.path.exists(f"{textgrid_dir}/metadata.txt"):
        try:
            noisy_files = sorted(glob.glob(os.path.join(noisy_audio_dir, "*.wav")))
            _metadata = create_metadata(noisy_files, f_path=f"{textgrid_dir}/metadata.txt")
            return _metadata
        except:
            pass
		
			
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(textgrid_dir, exist_ok=True)
	# noisy_dir/audio/*.wav
	# noisy_dir/text/*.txt
    noisy_files = sorted(glob.glob(os.path.join(noisy_audio_dir, "*.wav")))
    for noisy_path in noisy_files:
        base_name = os.path.splitext(os.path.basename(noisy_path))[0]
        speaker = get_speaker_from_file_name(noisy_path)
        # save the TextGrid files in the output directory
        os.makedirs(os.path.join(output_dir, speaker), exist_ok=True)
        temp_text_path = os.path.join(output_dir, speaker, base_name + '.txt')
		# copy the audio file to the output directory for MFA
        temp_audio_path = os.path.join(output_dir, speaker, os.path.basename(noisy_path))
        if not os.path.exists(temp_audio_path):
            shutil.copy(noisy_path, temp_audio_path)
            shutil.copy(os.path.join(text_dir, base_name + '.txt'), temp_text_path)

    # run MFA alignment
	# suppose the MFA command line tool is installed and configured in the environment
    logger.info(f"Running MFA on {output_dir}...")
    command = [
        'mfa', 'align',
        output_dir,           # the directory containing audio and text files to align
        dict_path,            # the dictionary file
        model_path,           # the acoustic model path
        textgrid_dir,           # the output directory for TextGrid files
		"--clean"
    ]

    try:
        run_command_in_conda_env('aligner', ' '.join(command)) # 'aligner' is the conda environment name for MFA
        logger.info("Alignment completed.")
    except subprocess.CalledProcessError as e:
        logger.info(f"Alignment filed: {e}")
        return None
    _metadata = create_metadata_from_textgrid(noisy_audio_dir, textgrid_dir)
    with open(f"{textgrid_dir}/metadata.txt", 'w') as f:
        for meta in _metadata:
            f.write(meta + '\n')
    return _metadata
