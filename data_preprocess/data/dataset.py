import torch
import torch.utils.data as data
import os
import numpy as np
import librosa
import os
import random
from .audiolib import audioread, snr_mixer, add_clipping, add_pyreverb, unify_normalize, is_clipped
from .lowpass import lowpass, highpass_filter, bandpass_filter


from librosa.filters import mel as librosa_mel_fn

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('WOWOWO min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('WOWOWO max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def spectrogram(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('WOWOWO min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('WOWOWO max value is ', torch.max(y))
    
    hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    # spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

EPS = np.finfo(float).eps


def make_dataset(data_flist):
    assert os.path.isfile(
        data_flist), "{} is not a valid file".format(data_flist)
    wavs = []
    with open(data_flist) as f:
        lines = f.readlines()
    wavs = [line.strip() for line in lines]
    return wavs


def uniform_sample(value_range):
    lower, upper = value_range
    sample = (upper-lower)*torch.rand(1)+lower
    return sample.float().item()

def get_mel_80(audio):
    audio = torch.tensor(audio).unsqueeze(0).float()
    melspec = mel_spectrogram(
        audio, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000)
    melspec = torch.squeeze(melspec, 0).numpy()
    # melshape = 80*T
    return melspec

def get_mel_128(audio):
    audio = torch.tensor(audio).unsqueeze(0).float()
    melspec = mel_spectrogram(
        audio, n_fft=1024, num_mels=128, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000)
    melspec = torch.squeeze(melspec, 0).numpy()
    # print("melspec", melspec.shape)
    return melspec

def get_mel(audio):
    return get_mel_128(audio)


def get_spec(audio):
    audio = torch.tensor(audio).unsqueeze(0).float()
    spec = spectrogram(
        audio, n_fft=1024, sampling_rate=22050, hop_size=256, win_size=1024)
    spec = torch.squeeze(spec, 0).numpy()
    # melshape = 80*T
    return spec

def get_amp(audio):
    """
    Compute amplitude spectrogram.

    Args:
        audio (numpy.ndarray): Input audio signal.

    Returns:
        numpy.ndarray: Amplitude spectrogram, shape (frequency, time frames).
    """
    # Convert audio to PyTorch tensor and add batch dimension
    audio_tensor = torch.tensor(audio).unsqueeze(0).float()
    
    # Define STFT parameters
    n_fft = 510
    hop_length = 128
    win_length = 510
    window = torch.hann_window(win_length)
    
    # Compute STFT, return real and imaginary parts
    stft_result = torch.stft(
        audio_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=False
    )
    
    # Compute amplitude spectrogram
    spec_exp = 0.5
    spec_factor = 0.33
    magnitude = torch.sqrt(stft_result[..., 0]**2 + stft_result[..., 1]**2) ** spec_exp
    magnitude = magnitude * spec_factor
    
    # Remove batch dimension and convert to numpy array
    magnitude = magnitude.squeeze(0).numpy()
    
    return magnitude

class AudioAugDataset(data.Dataset):
    '''Used for training. Producing augmented audio on the fly when calling'''
    def __init__(self, config):
        self.config = config
        uttinfo_fpath = config["path"]["dataset_file_path"]
        # uttinfo
        with open(uttinfo_fpath, 'r') as f:
            self._metadata = [line.replace('\n', '') for line in f.readlines()]
        # noise + rir

        # augmentation
        self.aug_params = config["augmentation"]
        self.aug = AudioAug(self.aug_params)
        # textinfo
        self.avgmel_dict = None
        self.use_text = config['model']['use_text']
        if self.use_text > 0:
            self.avgmel_dict = self.init_avgmel_dict()

    def init_avgmel_dict(self):
        avgmel_dict = dict()
        with open(self.config["path"]["avg_mel_path"]) as f:
            lines = [line.strip() for line in f.readlines()]
        for line in lines:
            p, value = line.split('\t')
            value = value[1:-1].split(',')
            value = np.round(np.array([float(x) for x in value]), 1)
            avgmel_dict[p] = value
        return avgmel_dict

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, index):
        ret = {}
        item_meta = self._metadata[index]
        _, phonemes_duration, phonemes_code, clean_path, _ = item_meta.split(
            '|')

        phonemes_code = phonemes_code.split()
        phonemes_duration = [int(dur) for dur in phonemes_duration.split()]

        clean_snr, noisy_snr = self.aug.apply_effect(clean_path)

        ret["clean_wav_16k"] = clean_snr
        ret["noisy_wav_16k"] = noisy_snr

        '''clean_snr = librosa.resample(
            clean_snr, orig_sr=16000, target_sr=self.config["preprocessing"]["audio"]["sampling_rate"])

        noisy_snr = librosa.resample(
            noisy_snr, orig_sr=16000, target_sr=self.config["preprocessing"]["audio"]["sampling_rate"])'''

        if is_clipped(clean_snr) or is_clipped(noisy_snr):
            clean_snr, noisy_snr = unify_normalize(clean_snr, noisy_snr)

        clean_mel = get_amp(clean_snr)  # get_mel(clean_snr)
        noisy_mel = get_amp(noisy_snr)  # get_mel(noisy_snr)
        mel_len = min(
            [clean_mel.shape[1], noisy_mel.shape[1]])

        if self.use_text:
            text_mel = []
            for ph, dur in zip(phonemes_code, phonemes_duration):
                text_mel.append(np.repeat(
                    np.expand_dims(self.avgmel_dict[ph], 1), dur, 1))
            text_mel = np.concatenate(text_mel, axis=1)
            mel_len = min([mel_len, text_mel.shape[1]])

        ret["clean"] = clean_mel[:, :mel_len]
        ret["noisy"] = noisy_mel[:, :mel_len]
        ret["index"] = clean_path.split('/')[-1][:-4]
        ret['durations'] = phonemes_duration
        ret['phonemes'] = phonemes_code
        if self.use_text:
            ret["text"] = text_mel[:, :mel_len]

        return ret


class AudioAug:
    def __init__(self, aug_params):
        self.aug_params = aug_params
        self.sample_rate = 16000


    def apply_lowpass(self, clean_aug):
        highcut = uniform_sample(
            self.aug_params["lowpass"]["low_pass_range"])
        order = int(uniform_sample(
            self.aug_params["lowpass"]["filter_order_range"]))
        filter_type = np.random.choice(
            self.aug_params["lowpass"]["filter_type"])
        clean_aug = lowpass(
            clean_aug, highcut=highcut,
            fs=self.sample_rate, order=order, _type=filter_type)
        return clean_aug.copy()

    def apply_highpass(self, clean_aug):
        lowcut = uniform_sample([100, 300])
        order = int(uniform_sample([2, 8]))
        filter_type = np.random.choice(
            self.aug_params["lowpass"]["filter_type"])
        clean_aug = highpass_filter(
            clean_aug, lowcut, self.sample_rate, order, filter_type)
        return clean_aug.copy()

    def apply_bandpass(self, clean_aug):
        lowcut = uniform_sample([100, 300])
        highcut = uniform_sample([4000, 7900])
        order = int(uniform_sample([2, 8]))
        filter_type = np.random.choice(
            self.aug_params["lowpass"]["filter_type"])
        clean_aug = bandpass_filter(
            clean_aug, [lowcut, highcut], self.sample_rate, order, filter_type)
        return clean_aug.copy()

    def apply_clipping(self, clean_aug):
        clip_level = uniform_sample(
            self.aug_params["clipping"]["max_thresh_perc"])
        clean_aug = add_clipping(clean_aug, clip_level)
        return clean_aug

    def apply_effect(self, clean_path):
        clean, _ = librosa.load(clean_path, sr=self.sample_rate)
        clean_aug = clean.copy()
        clean, clean_aug = unify_normalize(clean, clean_aug)



        # clipping
        apply_clipping = torch.rand(
            1).item() < self.aug_params["clipping"]["prob"]
        if apply_clipping:
            clean_aug = self.apply_clipping(clean_aug)

        # lowpass
        apply_lowpass = torch.rand(
            1).item() < self.aug_params["lowpass"]["prob"]
        if apply_lowpass:
            clean_aug = self.apply_lowpass(clean_aug)
            clean, clean_aug = unify_normalize(clean, clean_aug)
        elif torch.rand(1).item() < 0.5:
            clean_aug = self.apply_highpass(clean_aug)
            clean, clean_aug = unify_normalize(clean, clean_aug)
        else:
            clean_aug = self.apply_bandpass(clean_aug)
            clean, clean_aug = unify_normalize(clean, clean_aug)

        return clean, clean_aug


class BatchCollate(object):
    """
    Collates batch objects with padding, decreasing sort by input length, etc.
    """

    def __call__(self, batch):
        B = len(batch)
        # Sorting batch by length of inputs
        mel_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x['clean'].shape[1] for x in batch]), dim=0, descending=True
        )
        max_mel_len = mel_lengths[0]
        n_mel = batch[0]['clean'].shape[0]

        clean_padded = torch.zeros(
            B, n_mel, max_mel_len, dtype=torch.float32)
        noisy_padded = torch.zeros(
            B, n_mel, max_mel_len, dtype=torch.float32)
        text_padded = torch.zeros(
            B, n_mel, max_mel_len, dtype=torch.float32)

        for index, i in enumerate(ids_sorted_decreasing):
            x = batch[i]
            clean = torch.FloatTensor(x['clean'])
            noisy = torch.FloatTensor(x['noisy'])
            if 'text' in x:
                text = torch.FloatTensor(x['text'])
                text_padded[index, :, :text.shape[1]] = text
            # B, 80, T
            clean_padded[index, :, :clean.shape[1]] = clean
            noisy_padded[index, :, :noisy.shape[1]] = noisy

        outputs = {
            'clean': clean_padded,
            'noisy': noisy_padded,
            'mel_lengths': mel_lengths,
        }
        
        if 'text' in batch[0]:
            outputs['text'] = text_padded

        return outputs

