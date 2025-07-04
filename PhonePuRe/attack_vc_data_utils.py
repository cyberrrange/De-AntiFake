import copy
import os
import pickle
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.signal import lfilter
import sys
import io


from pathlib import Path
import torchaudio





#rtvc
sys.path.insert(0, "./encoder_models/rtvc")
from encoder import inference as encoder
from encoder import audio
from encoder.params_data import *

RTVC_DEFAULT_MODEL_PATH = "./encoder_models/speaker_encoder_ckpts/rtvc"
OPENVOICE_MODEL_PATH = "./encoder_models/speaker_encoder_ckpts/openvoice"


# Load additional modules

import os


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename="config.yml"):
    from huggingface_hub import hf_hub_download
    os.makedirs("./checkpoints", exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir="./checkpoints", local_files_only=True)
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir="./checkpoints", local_files_only=True)

    return model_path, config_path

def inv_mel_matrix(sample_rate: int, n_fft: int, n_mels: int) -> np.array:
    m = librosa.filters.mel(sample_rate, n_fft, n_mels)
    p = np.matmul(m, m.T)
    d = [1.0 / x if np.abs(x) > 1e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m.T, np.diag(d))


def normalize(mel: np.array, attr: Dict) -> np.array:
    mean, std = attr["mean"], attr["std"]
    mel = (mel - mean) / std
    return mel


def denormalize(mel: np.array, attr: Dict) -> np.array:
    mean, std = attr["mean"], attr["std"]
    mel = mel * std + mean
    return mel


def file2mel(
    audio_path: str,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float,
) -> np.array:
    wav, _ = librosa.load(audio_path, sr=sample_rate)
    wav, _ = librosa.effects.trim(wav, top_db=top_db)
    wav = np.append(wav[0], wav[1:] - preemph * wav[:-1])
    linear = librosa.stft(
        y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    mag = np.abs(linear)

    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels)
    mel = np.dot(mel_basis, mag)

    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mel = mel.T.astype(np.float32)

    return mel


def mel2wav(
    mel: np.array,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float,
) -> np.array:
    mel = mel.T
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db
    mel = np.power(10.0, mel * 0.05)
    inv_mat = inv_mel_matrix(sample_rate, n_fft, n_mels)
    mag = np.dot(inv_mat, mel)
    wav = griffin_lim(mag, hop_length, win_length, n_fft)
    wav = lfilter([1], [1, -preemph], wav)

    return wav.astype(np.float32)


def griffin_lim(
    spect: np.array,
    hop_length: int,
    win_length: int,
    n_fft: int,
    n_iter: Optional[int] = 100,
) -> np.array:
    X_best = copy.deepcopy(spect)
    for _ in range(n_iter):
        X_t = librosa.istft(X_best, hop_length, win_length, window="hann")
        est = librosa.stft(X_t, n_fft, hop_length, win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spect * phase
    X_t = librosa.istft(X_best, hop_length, win_length, window="hann")
    y = np.real(X_t)
    return y




def load_coqui_model() -> Tuple[nn.Module, Dict, Dict, str]:
    #coqui
    sys.path.insert(0, "./encoder_models/TTS")
    from TTS.api import TTS
    COQUI_YOURTTS_PATH = "tts_models/multilingual/multi-dataset/your_tts"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coqui_tts = TTS(model_name=COQUI_YOURTTS_PATH, progress_bar=True, gpu = True)
    model = coqui_tts.synthesizer.tts_model.speaker_manager
    return model, None, None, device

def load_rtvc_model() -> Tuple[nn.Module, Dict, Dict, str]:
    #ensure_default_models(Path(RTVC_DEFAULT_MODEL_PATH))
    model, device = encoder.load_model(Path(RTVC_DEFAULT_MODEL_PATH + '/encoder.pt'))
    model.train()
    return model, None, None, device

def load_tortoise_model() -> Tuple[nn.Module, Dict, Dict, str]:
    #tortoise
    sys.path.insert(0, "./encoder_models")
    from tortoise.tortoise_backward import load_voice_path, TextToSpeech, format_conditioning, pad_or_truncate, wav_to_univnet_mel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tortoise_tts = TextToSpeech()
    model = tortoise_tts.autoregressive.to(device)
    # model = tortoise_tts.diffusion.to(device)
    return model, None, None, device


def load_openvoice_model() -> Tuple[nn.Module, Dict, Dict, str]:
    # os.environ['LD_LIBRARY_PATH'] = '/path/to/conda/envs/openvoice/lib/python3.9/site-packages/nvidia/cudnn/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    from encoder_models.openvoice.api import ToneColorConverter
    # export LD_LIBRARY_PATH=/path/to/conda/envs/openvoice/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
    # Initialization
    ckpt_converter = OPENVOICE_MODEL_PATH
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    tone_color_converter.model.ref_enc.train()
    return tone_color_converter, None, None, device




def coqui_preprocess(model, input_path):
    null_stream = io.StringIO() 
    sys.stdout = null_stream
    input_wav = model.encoder_ap.load_wav(input_path, sr=model.encoder_ap.sample_rate)
    sys.stdout = sys.__stdout__
    input_tensor = torch.from_numpy(input_wav).cuda().unsqueeze(0)
    return input_tensor, None

def rtvc_preprocess(input_path, device):
    preprocessed_wav = encoder.preprocess_wav(input_path, 16000)

    wav, _, mel_slices, _, _ = encoder.embed_utterance_preprocess(preprocessed_wav, using_partials=True)
    wav_tensor_initial = torch.from_numpy(wav).unsqueeze(0).to(device)
    return wav_tensor_initial, mel_slices



def openvoice_preprocess(input_path, device):
    audio_ref, sr = librosa.load(input_path, sr=22050)
    y = torch.FloatTensor(audio_ref)
    y = y.to(device)
    y = y.unsqueeze(0)
    return y, None

def tortoise_auto_preprocess(input_path, device):
    from tortoise.tortoise_backward import load_voice_path
    [voice_samples] = load_voice_path(input_path)
    voice_sample = voice_samples.to(device)
    return voice_sample, None

def tortoise_diff_preprocess(input_path, device):
    from tortoise.tortoise_backward import load_voice_path
    [voice_samples] = load_voice_path(input_path)
    sample = torchaudio.functional.resample(voice_samples, 22050, 24000)
    sample = sample.to(device)
    return sample, None




def universal_postprocess(adv_inp, sr):
    adv_inp_wav = adv_inp.squeeze(0).detach().cpu().numpy()
    return adv_inp_wav, sr


def get_rtvc_embedding(model, wav_tensor_initial, mel_slices, device):
    frame_tensor_list = []
    frames_tensor = audio.wav_to_mel_spectrogram_torch(wav_tensor_initial).to(device)
    # Get the mel slices and cat them
    for s in mel_slices:
        frame_tensor = frames_tensor[s].unsqueeze(0).to(device)
        frame_tensor_list.append(frame_tensor)
    frames_tensor = torch.cat(frame_tensor_list, dim=0)
    embedding = model.forward(frames_tensor)
    embedding = torch.mean(embedding, dim=0, keepdim=True)
    return embedding

def get_tortoise_auto_embedding(model, input, device):
    from tortoise.tortoise_backward import load_voice_path, TextToSpeech, format_conditioning, pad_or_truncate, wav_to_univnet_mel
    voice_sample = format_conditioning(input, device=device)
    voice_sample = voice_sample.unsqueeze(0)

    embedding = model.get_conditioning(voice_sample)
    return embedding

def get_tortoise_diff_embedding(model, input, device):
    from tortoise.tortoise_backward import load_voice_path, TextToSpeech, format_conditioning, pad_or_truncate, wav_to_univnet_mel
    sample = pad_or_truncate(input, 102400)
    cond_mel = wav_to_univnet_mel(sample.to(
        device), do_normalization=False, device=device)  # tacotron mel spectrum encoder
    cond_mel = cond_mel.unsqueeze(0)
    embedding = model.get_conditioning(cond_mel)
    return embedding


def get_openvoice_embedding(tone_color_converter, tgt_wav):

    target_se = tone_color_converter.extract_se_torch(tgt_wav).squeeze(2)

    # Load source audio and extract features
    #src_wav, sr = torchaudio.load(src_wav_path)
    #src_wav = src_wav.to(device)


    #print("target_se: ", target_se.shape)
    return target_se

class ModelManager():
    def __init__(self, syn_type):
        self.syn_type = syn_type
        self.sample_rate = {"coqui": 16000, "rtvc": 16000, "tortoise": 22050, "openvoice": 22050}[syn_type]
        self.model, self.config, self.attr, self.device = self.load_model()
    def load_model(self) -> Tuple[nn.Module, Dict, Dict, str]:

        if self.syn_type == "coqui":
            return load_coqui_model()
        elif self.syn_type == "rtvc":
            return load_rtvc_model()
        elif self.syn_type == "tortoise":
            return load_tortoise_model()
        elif self.syn_type == "openvoice":
            return load_openvoice_model()
        else:
            raise NotImplementedError("Unsupported synthesis type.")
    def preprocess_to_tensor(self, input_path):
        if self.syn_type == "coqui":
            return coqui_preprocess(self.model, input_path)
        elif self.syn_type == "rtvc":
            return rtvc_preprocess(input_path, self.device)
        elif self.syn_type == "tortoise":
            return tortoise_auto_preprocess(input_path, self.device)
            # return tortoise_diff_preprocess(input_path, device)
        elif self.syn_type == "openvoice":
            return openvoice_preprocess(input_path, self.device)
        else:
            raise NotImplementedError("Unsupported synthesis type.")
    def postprocess_to_wav(self, adv_inp):
        return universal_postprocess(adv_inp, self.sample_rate)
        
    def speaker_encoder(self, input, mel_slices=None):
        if self.syn_type == "coqui":
            embedding = self.model.encoder.compute_embedding(input)
        elif self.syn_type == "rtvc":
            embedding = get_rtvc_embedding(self.model, input, mel_slices, self.device)
        elif self.syn_type == "tortoise":
            embedding = get_tortoise_auto_embedding(self.model, input, self.device)
            # embedding = get_tortoise_diff_embedding(self.model, input, self.device)
        elif self.syn_type == "openvoice":
            embedding = get_openvoice_embedding(self.model, input)
        else:
            raise NotImplementedError("Unsupported synthesis type.")
        return embedding