import torch
import torchaudio
import numpy as np

import librosa.display
import matplotlib.pyplot as plt
from typing import Union
import os

def spec_save(x: Union[np.ndarray, torch.Tensor], path=None, name=None):
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.squeeze()
    # assert x.shape == (32, 32)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(data=x, 
                                   x_axis='ms', y_axis='mel', 
                                   sr=16000, n_fft=2048, 
                                   fmin=0, fmax=8000, 
                                   ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    
    if path is None:
        path = './_Spec_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'spec.png'
    fig.savefig(os.path.join(path, name))

def audio_save(x: Union[np.ndarray, torch.Tensor], path=None, name=None):

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.detach().cpu()
    assert x.ndim == 2 and x.shape[0] == 1

    if path is None:
        path = './_Audio_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'audio.wav'

    torchaudio.save(os.path.join(path,name), x, 16000) # default sample rate = 16000

def audio_save_as_img(x: Union[np.ndarray, torch.Tensor], path=None, name=None, color=None):
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.squeeze()
    assert x.ndim == 1

    fig = plt.figure(figsize=(21, 9), dpi=100)

    from scipy.interpolate import make_interp_spline

    # x_smooth = make_interp_spline(np.arange(0, len(x)), x)(np.linspace(0, len(x), 1000))
    # plt.ylim(-1.5*max(abs(x.max()), abs(x.min())),1.5*max(abs(x.max()), abs(x.min())))
    # plt.plot((np.linspace(0, len(x), 1000)), x_smooth,'-')
    # plt.ylim(-1,1)
    plt.plot(x,'-',color=color if color is not None else 'steelblue', transparent=True)

    if path is None:
        path = './_Audio_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'waveform.png'

    fig.savefig(os.path.join(path, name))

## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40

## Audio
sampling_rate = 16000

def wav_to_mel_spectrogram_torch(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    # wav = wav.unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels,
        norm = "slaney",
        mel_scale = "slaney"
    ).to(wav.device)
    frames = mel_transform(wav)
    return frames#.transpose(0, 1).float()
    
def mel_spectrogram_to_wav(mel_spectrogram, n_fft, hop_length, sample_rate, win_length):
    # turn dB into linear scale
    mel_spectrogram = 10.0 ** (mel_spectrogram / 20.0)
    mel_spectrogram = mel_spectrogram.squeeze().detach().cpu().numpy()
    #mel_spectrogram = mel_spectrogram.T
    reconstructed_wav = librosa.feature.inverse.mel_to_audio(mel_spectrogram,
                                               n_fft=n_fft,
                                               hop_length=hop_length,
                                               sr=sample_rate,
                                               n_iter=300,
                                               win_length=win_length,)
    
    return np.expand_dims(np.expand_dims(reconstructed_wav, 0), 0)