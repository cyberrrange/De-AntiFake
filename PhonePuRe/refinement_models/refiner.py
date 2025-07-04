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

import sys
from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import StochasticRegenerationModel
from sgmse.util.other import *
from phoneme_utils import create_metadata_from_audio_text_mfa, get_text_spec_from_file_name, init_avgspec_dict

import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


EPS_LOG = 1e-10



class Refiner(torch.nn.Module):
    def __init__(self, align_from_folder, 
                 text_dir, phoneme_avg_spec_dict,
                 corrector="ald", corrector_steps=1, snr=0.4, N=30, model_sr=16000,
                 checkpoint_file=".../checkpoints/refinement.ckpt"):
        super().__init__()
        model_cls = StochasticRegenerationModel
        model = model_cls.load_from_checkpoint(
            checkpoint_file, base_dir="",
            batch_size=1, num_workers=0, kwargs=dict(gpu=False)
        )
        model.eval(no_ema=False)
        model.cuda()
        self.model = model
        self.model_sr = model_sr
        self.corrector = corrector
        self.corrector_steps = corrector_steps
        self.snr = snr
        self.N = N
        if self.model.use_text:
            self.align_folder = align_from_folder
            self.noisy_files = sorted(glob.glob(os.path.join(align_from_folder, "*.wav")))
            self.enhance_metadata = create_metadata_from_audio_text_mfa(self.align_folder, text_dir=text_dir)
            self.trainset_avg_spec_dict = init_avgspec_dict(phoneme_avg_spec_dict)
        self.noisy_files = [f for f in self.noisy_files]
        logger.info(f"Metadata created, length: {len(self.enhance_metadata)}, example: {self.enhance_metadata[0]}, noisy files {self.noisy_files[0]}")

    def forward(self, y, file_name=None):
        f = file_name
        if y.ndim == 1:
            y = y.unsqueeze(0)
        elif y.ndim == 3:
            y = y.squeeze(0)
        if self.model.use_text:
            text_spec = get_text_spec_from_file_name(f, y, self.enhance_metadata, self.trainset_avg_spec_dict, self.noisy_files, self.model._stft)
        else:
            text_spec = None

        y_list = [y]
        

        x_hat, _ = self.model.enhance(y_list, corrector=self.corrector, N=self.N, corrector_steps=self.corrector_steps, snr=self.snr, text_spec=text_spec, predictor='reverse_diffusion', sampler_type="pc")
        return x_hat