import numpy as np
import scipy.stats
import torch
import csv
import os
import glob
import tqdm
import torchaudio
import matplotlib.pyplot as plt
import time
from pydub import AudioSegment
import scipy.signal as ss

stft_kwargs = {"n_fft": 510, "hop_length": 128, "window": torch.hann_window(510), "return_complex": True}

def lsd(s_hat, s, eps=1e-10):
    S_hat, S = torch.stft(torch.from_numpy(s_hat), **stft_kwargs), torch.stft(torch.from_numpy(s), **stft_kwargs)
    logPowerS_hat, logPowerS = 2*torch.log(eps + torch.abs(S_hat)), 2*torch.log(eps + torch.abs(S))
    return torch.mean( torch.sqrt(torch.mean(torch.abs( logPowerS_hat - logPowerS ))) ).item()

def si_sdr_components(s_hat, s, n, eps=1e-10):
    # s_target
    alpha_s = np.dot(s_hat, s) / (eps + np.linalg.norm(s)**2)
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / (eps + np.linalg.norm(n)**2)
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise
    
    return s_target, e_noise, e_art

def energy_ratios(s_hat, s, n, eps=1e-10):
    """
    """
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_noise + e_art)**2))
    si_sir = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_noise)**2))
    si_sar = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_art)**2))

    return si_sdr, si_sir, si_sar

def mean_conf_int(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

class Method():
    def __init__(self, name, base_dir, metrics):
        self.name = name
        self.base_dir = base_dir
        self.metrics = {} 
        
        for i in range(len(metrics)):
            metric = metrics[i]
            value = []
            self.metrics[metric] = value 
            
    def append(self, matric, value):
        self.metrics[matric].append(value)

    def get_mean_ci(self, metric):
        return mean_conf_int(np.array(self.metrics[metric]))

def hp_filter(signal, cut_off=80, order=10, sr=16000):
    factor = cut_off /sr * 2
    sos = ss.butter(order, factor, 'hp', output='sos')
    filtered = ss.sosfilt(sos, signal)
    return filtered

def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr

def si_sdr_torch(s, s_hat):
    min_len = min(s.size(-1), s_hat.size(-1))
    s, s_hat = s[..., : min_len], s_hat[..., : min_len]
    alpha = torch.dot(s_hat, s)/torch.norm(s)**2   
    sdr = 10*torch.log10(1e-10 + torch.norm(alpha*s)**2/(1e-10 + torch.norm(
        alpha*s - s_hat)**2))
    return sdr

class EmbeddingLoss(torch.nn.Module):
    def __init__(self, model_source="speechbrain/spkrec-ecapa-voxceleb", device=None, model_type = "xv"):
        super(EmbeddingLoss, self).__init__()
        self.model_type = model_type
        if model_type == 'xv':
            from speechbrain.inference.speaker import SpeakerRecognition
            self.emb_model = SpeakerRecognition.from_hparams(
                source="/data/fanwei/De-AntiFake/pretrained_models/spkrec-ecapa-voxceleb-local",
                run_opts={"device":"cuda" if torch.cuda.is_available() else "cpu"},
                freeze_params=False,
                hparams_file="hyperparams.yaml"
            )
        elif model_type == "dv":
            sampling_rate = 16000
            mel_window_length = 25  # in milliseconds
            mel_window_step = 10  # in milliseconds
            mel_n_channels = 40
            self.mel_transform =  torchaudio.transforms.MelSpectrogram(
                sample_rate=sampling_rate,
                n_fft=int(sampling_rate * mel_window_length / 1000),
                hop_length=int(sampling_rate * mel_window_step / 1000),
                n_mels=mel_n_channels
            ).to(self.device)
            from resemblyzer import VoiceEncoder, preprocess_wav
            self.emb_model = VoiceEncoder()
        else:
            raise ValueError("Model type not recognized")
        for param in self.emb_model.parameters():
            param.requires_grad = False
        print("Embedding model loaded:", self.emb_model.device)

    def forward(self, x_wav, y_denoised_wav, loss_type = "mse"):
        if self.model_type == "xv":
            x_emb = self.emb_model.encode_batch(x_wav.squeeze(1))
            y_denoised_emb = self.emb_model.encode_batch(y_denoised_wav.squeeze(1))
        elif self.model_type == "dv":
            x_emb = self.emb_model(self.preprocess_batch(x_wav))
            
            y_denoised_emb = self.emb_model(self.preprocess_batch(y_denoised_wav))
        else:
            raise ValueError("Model type not recognized")
        # normalize the embeddings
        # get max number in x_emb
        norm_factor = torch.max(torch.abs(x_emb))
        x_emb = x_emb / norm_factor
        y_denoised_emb = y_denoised_emb / norm_factor
        if loss_type == "cos":
            cos_sim = self.emb_model.similarity(x_emb, y_denoised_emb)
            emb_loss =  1 - cos_sim.mean()
            return emb_loss
        elif loss_type == "mse":
            return torch.nn.functional.mse_loss(x_emb, y_denoised_emb)
        else:
            raise ValueError("Loss type not recognized")
        
    def preprocess_batch(self, batch_audio_tensor):
        """
        Preprocesses a batch of audio tensors to generate mel spectrograms.

        :param batch_audio_tensor: Tensor of shape [batch, channel, time]
        :return: Tensor of shape (batch_size, n_frames, n_channels) suitable for the forward function
        """
        batch_size, channels, time = batch_audio_tensor.shape
        assert channels == 1, "Expected mono channel audio"
        # Assuming these values are globally defined or passed as parameters


        # Apply mel spectrogram transform to the batch
        batch_audio_tensor = batch_audio_tensor.squeeze(1)  # Shape: [batch, time]
        mel_spectrograms = self.mel_transform(batch_audio_tensor)  # Shape: [batch, n_mels, n_frames]
        # Convert to log mel spectrogram to match librosa's output
        mel_spectrograms = torch.clamp(mel_spectrograms, min=1e-10)  # Avoid log of zero
        mel_spectrograms = 10 * torch.log10(mel_spectrograms)  # Convert to dB scale
        mel_spectrograms = mel_spectrograms.transpose(1, 2)  # Shape: [batch, n_frames, n_mels]
        return mel_spectrograms

def snr_dB(s,n):
    s_power = 1/len(s)*np.sum(s**2)
    n_power = 1/len(n)*np.sum(n**2)
    snr_dB = 10*np.log10(s_power/n_power)
    return snr_dB

def pad_spec(Y):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    return pad2d(Y)

# def pad_time(Y):
#     padding_target = 8320
#     T = Y.size(2)
#     if T%padding_target !=0:
#         num_pad = padding_target-T%padding_target
#     else:
#         num_pad = 0
#     pad2d = torch.nn.ZeroPad2d((0, num_pad, 0, 0))
#     return pad2d(Y)

def pad_time(Y, length):
    # (B, T)
    T = Y.size(1)
    if T != length:
        num_pad = length-T
    if T > length:
        # 如果序列过长，裁剪多余部分
        # if the sequence is longer than the target length, slice it
        return Y[:, :length]

    # if the sequence is shorter than the target length, pad it
    num_pad = length - T
    return torch.nn.functional.pad(Y, (0, num_pad), "constant", 0)

def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std



def init_exp_csv_samples(output_path, tag_metric):
    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        fieldnames = ["Filename", "Length", "T60", "iSNR"] + tag_metric
        writer.writerow(fieldnames)
        csv_file.close()

def snr_scale_factor(speech, noise, snr):
    noise_var = np.var(noise)
    speech_var = np.var(speech)

    factor = np.sqrt(speech_var / (noise_var * 10. ** (snr / 10.)))

    return factor

def pydub_read(path, sr=16000):
    y = AudioSegment.from_file(path)
    y = y.set_frame_rate(sr)
    channel_sounds = y.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    return fp_arr

def align(y, ref):
    l = np.argmax(ss.fftconvolve(ref.squeeze(), np.flip(y.squeeze()))) - (ref.shape[0] - 1)
    if l:
        y = torch.from_numpy(np.roll(y, l, axis=-1))
    return y

def wer(r, h):
    '''
    by zszyellow
    https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return float(d[len(r)][len(h)]) / len(r)