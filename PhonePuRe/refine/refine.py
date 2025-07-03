import numpy as np
import glob

from tensorboard import summary
from tqdm import tqdm
from torchaudio import load, save
import torch
import os
from argparse import ArgumentParser
import time
from pypapi import events, papi_high as high

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import StochasticRegenerationModel, ScoreModel, DiscriminativeModel, NoScoreRegenerationModel

from sgmse.util.other import *

import matplotlib.pyplot as plt
import sys
import logging
from phoneme_utils import create_metadata_from_audio_text_mfa, get_text_spec_from_file_name, init_avgspec_dict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


EPS_LOG = 1e-10


base_parser = ArgumentParser(add_help=False)
parser = ArgumentParser()
for parser_ in (base_parser, parser):
	# used for phoneme conditioning
	parser_.add_argument("--noisy_audio_dir", type=str, default=None, help="Audio files before purification (for alignment).")
	parser_.add_argument("--text_dir", type=str, required=True, help="Text files of corresponding noisy audio files.")
	parser_.add_argument("--phoneme_avg_spec_dict", type=str, required=True, help="Path to the phoneme average spectrogram dictionary file. This is used to condition the model on phonemes.")

	# input directories
	parser_.add_argument("--purified_dir", type=str, required=True, help="Input purified audio files.")
	# output directories
	parser_.add_argument("--refined_dir", type=str, required=True, help="Where to write your refined files.")

	# model checkpoint, and mode/params
	parser_.add_argument("--ckpt", type=str, required=True)
	parser_.add_argument("--mode", required=True, choices=["score-only", "denoiser-only", "storm", "ref-diff"])

	parser_.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
	parser_.add_argument("--corrector-steps", type=int, default=1, help="Number of corrector steps")
	parser_.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynamics.")
	parser_.add_argument("--N", type=int, default=50, help="Number of reverse steps")

args = parser.parse_args()

os.makedirs(args.refined_dir, exist_ok=True)

#Checkpoint
checkpoint_file = args.ckpt

# Settings
model_sr = 16000

# Load score model 
if args.mode == "storm":
	model_cls = StochasticRegenerationModel
elif args.mode == "score-only":
	model_cls = ScoreModel
elif args.mode == "denoiser-only":
	model_cls = DiscriminativeModel
elif args.mode =="ref-diff":
	model_cls = NoScoreRegenerationModel

model = model_cls.load_from_checkpoint(
	checkpoint_file, base_dir="",
	batch_size=1, num_workers=0, kwargs=dict(gpu=False)
)
model.eval(no_ema=False)
model.cuda()

import datetime
time_cost = 0
total_audio_time = 0


purified_files = sorted(glob.glob(os.path.join(args.purified_dir, "*.wav")))

if args.noisy_audio_dir is None:
	# If no noisy audio directory is provided, use the purified directory for alignment
	align_folder = args.purified_dir
else:
	align_folder = args.noisy_audio_dir

for f in purified_files:
	y, sample_sr = torchaudio.load(f)
	total_audio_time += y.size(1) / model_sr

logger.info(f"Total audio time: {total_audio_time} seconds")


if model.use_text:

	phoneme_avg_spec_dict = args.phoneme_avg_spec_dict
	enhance_metadata = create_metadata_from_audio_text_mfa(align_folder, text_dir=args.text_dir)

	trainset_avg_spec_dict = init_avgspec_dict(phoneme_avg_spec_dict)
	#assert len(purified_files) == len(enhance_metadata), "You need to make sure the number of files and metadata match."
# Loop on files

for f in tqdm.tqdm(purified_files):
	y, sample_sr = torchaudio.load(f)
	if sample_sr != model_sr:
		y = torchaudio.transforms.Resample(orig_freq=sample_sr, new_freq=model_sr)(y)
	#assert sample_sr == model_sr, "You need to make sure sample_sr matches model_sr --> resample to 16kHz"
	
	if model.use_text:
		text_spec = get_text_spec_from_file_name(f, y, enhance_metadata, trainset_avg_spec_dict, purified_files, model._stft)
	else:
		text_spec = None

	y_list = []
	audio, sr = torchaudio.load(os.path.join(args.purified_dir, os.path.basename(f)))
	if sr != model_sr:
		audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)(audio)
	y_list.append(audio)
	current_time = datetime.datetime.now()
	

	x_hat, _ = model.enhance(y_list, corrector=args.corrector, N=args.N, corrector_steps=args.corrector_steps, snr=args.snr, text_spec=text_spec, predictor='reverse_diffusion', sampler_type="pc")

	save(f'{args.refined_dir}/{os.path.basename(f)}', x_hat.type(torch.float32).cpu().squeeze().unsqueeze(0), model_sr)
	time_cost += (datetime.datetime.now() - current_time).total_seconds()

logger.info(f"Time cost: {time_cost} seconds")