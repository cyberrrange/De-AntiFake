import torch
#from robustness_eval.white_box_attack import *


import torchaudio
from torchvision.transforms import *
import os
n_mels = 32
n_fft = 2048
hop_length = 512
sample_rate = 16000  # make sure this matches your purification model's sample rate

# Define inverse Mel spectrogram transformation
inverse_mel_transform = torchaudio.transforms.InverseMelScale(
    n_stft=n_fft//2 + 1,
    n_mels=n_mels,
    sample_rate=sample_rate
)

# Define Griffin-Lim transformation
griffin_lim_transform = torchaudio.transforms.GriffinLim(
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=n_fft
)
inverse_mel_transform = inverse_mel_transform.cuda()
griffin_lim_transform = griffin_lim_transform.cuda()
def spectrogram_to_waveform(spectrogram):
    # dB to amplitude
    spectrogram = 10.0 ** (spectrogram / 20.0)
    # reconstruct linear frequency spectrogram from mel spectrogram
    linear_spectrogram = inverse_mel_transform(spectrogram)

    # Use Griffin-Lim to reconstruct waveform from linear spectrogram
    waveform = griffin_lim_transform(linear_spectrogram)

    return waveform

'''if args.classifier_input == 'mel40':
    n_mels = 40
'''
MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels,
                                                    norm='slaney', pad_mode='constant', mel_scale='slaney')
Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])

def chunk_audio(waveform, chunk_size, overlap=0):
    """
    Split audio into fixed-size chunks with optional overlap.
    """
    length = waveform.shape[-1]
    stride = chunk_size - overlap
    chunks = []
    for i in range(0, length, stride):
        chunk = waveform[..., i:i+chunk_size]
        if chunk.shape[-1] < chunk_size:
            # pad the last chunk if it is shorter than chunk_size
            padding = chunk_size - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, padding))
        chunks.append(chunk)
    return chunks

def merge_chunks(chunks, original_length, overlap=0):
    """
    Merge processed chunks back into audio of original length.
    """
    chunk_size = chunks[0].shape[-1]
    stride = chunk_size - overlap
    result = torch.zeros(chunks[0].shape[:-1] + (original_length,), device=chunks[0].device)
    weights = torch.zeros_like(result)
    
    for i, chunk in enumerate(chunks):
        start = i * stride
        end = start + chunk_size
        result[..., start:end] += chunk
        weights[..., start:end] += 1
    
    # process the last chunk if it is shorter than chunk_size
    result = result / weights.clamp(min=1)
    return result[..., :original_length]

class AcousticSystem_robust(torch.nn.Module):

    def __init__(self, 
                 spk_encoder: torch.nn.Module=None, 
                 defender_wav: torch.nn.Module=None,
                 defender_spec: torch.nn.Module=None,
                 defense_type: str='wave',
                 defense_method: str='DualPure'
                 ):
        super().__init__()

        '''
            the whole audio system: audio -> prediction probability distribution
            
            *defender: audio -> audio or spectrogram -> spectrogram
            *transform: audio -> spectrogram
            *spk_encoder: spectrogram -> prediction probability distribution or 
                            audio -> prediction probability distribution
        '''

        self.spk_encoder = spk_encoder
        if self.spk_encoder.sample_rate != 16000:
            self.resampler = torchaudio.transforms.Resample(orig_freq=self.spk_encoder.sample_rate, new_freq=16000).to(self.spk_encoder.device)
            self.inv_resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=self.spk_encoder.sample_rate).to(self.spk_encoder.device)
        if defense_method in ['DualPure', 'DiffSpec']:
            self.transform = Wave2Spect
        else:
            self.transform = None
        self.defender_wav = defender_wav
        self.defender_spec = defender_spec
        self.defense_type = defense_type
        self.defense_method = defense_method

    def forward(self, x, mel_slices, defend=True, file_name = None):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        # resample to defender sr
        if self.spk_encoder.sample_rate != 16000 and defend == True and (self.defender_wav is not None or self.defender_spec is not None):
            x = self.resampler(x)
        # wave defender by chunk
        if defend == True and self.defender_wav is not None and (self.defense_type == 'wave' or self.defense_method in ['DualPure', 'PhonePuRe']):
            output = self.defender_wav_variable_input(x)
        else:
            output = x
        # transform to spec if [DualPure, DiffSpec]
        if self.transform is not None:
            output = self.transform(output)
        # spec defender by chunk
        if defend == True and self.defender_spec is not None and self.defense_method in ['DualPure', 'DiffSpec']:
            output = self.defender_spec_variable_input(output)
        elif defend == True and self.defender_spec is not None and self.defense_method in ['PhonePuRe']:
            output = self.defender_spec(output, file_name = file_name)
        else:
            output = output
        output = output.squeeze(0)
        # resample to encoder model sr
        if self.spk_encoder.sample_rate != 16000 and defend == True and (self.defender_wav is not None or self.defender_spec is not None):
            output = self.inv_resampler(output)
        
        output = self.speaker_encoder(output, mel_slices)

        return output
    def speaker_encoder(self, x, mel_slices):
        return self.spk_encoder.speaker_encoder(x, mel_slices)
    
    def defense(self, x, mel_slices = None, file_name = None, ddpm = False):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        # resample to defender sr
        if self.spk_encoder is not None and self.spk_encoder.sample_rate != 16000 and (self.defender_wav is not None or self.defender_spec is not None):
            x = self.resampler(x)
        if self.defender_wav is not None:
            # wave defender by chunk
            if (self.defense_type == 'wave' or self.defense_method in ['DualPure', 'PhonePuRe']):
                output = self.defender_wav_variable_input(x, ddpm=ddpm)
        else:
            output = x  
        # transform to spec if [DualPure, DiffSpec]
        if self.transform is not None:
            output = self.transform(output)
        # spec defender by chunk
        if self.defender_spec is not None:
            if self.defense_method in ['DualPure', 'DiffSpec']:
                output = self.defender_spec_variable_input(output)
            elif self.defense_method in ['PhonePuRe']:
                output = self.defender_spec(output, file_name = file_name)
        else:
            output = output
        if x.ndim == 3:
            output = output.squeeze(0)
        # resample to encoder model sr
        if self.spk_encoder is not None and self.spk_encoder.sample_rate != 16000 and (self.defender_wav is not None or self.defender_spec is not None):
            output = self.inv_resampler(output)
        return output
    
    def split_audio(self, x, chunk_size, over_lap):
        """
        split the audio into fixed-size chunks, with optional overlap, and pad the last chunk if necessary.
        """
        chunks = []
        step = chunk_size - over_lap
        for start in range(0, x.size(-1) - chunk_size + 1, step):
            chunk = x[..., start:start + chunk_size]
            chunks.append(chunk)
        # if the last chunk is smaller than chunk_size, pad it
        if (x.size(-1) - chunk_size) % step != 0:
            last_chunk = x[..., step * len(chunks):]
            if last_chunk.size(-1) < chunk_size:
                padding = chunk_size - last_chunk.size(-1)
                last_chunk = torch.nn.functional.pad(last_chunk, (0, padding))
            chunks.append(last_chunk)
        return chunks

    def combine_chunks(self, chunks, over_lap, original_length):
        """
        Concatenate the processed chunks back into a single audio tensor,
        considering overlap regions for smooth transitions and restoring to the original length.
        """
        if len(chunks) == 0:
            return None

        combined_audio = chunks[0]
        if over_lap > 0:
            overlap_weight = torch.linspace(0, 1, steps=over_lap).unsqueeze(0)
            for i in range(1, len(chunks)):
                # Mix the overlapping regions
                overlap_start = combined_audio[..., -over_lap:] * (1 - overlap_weight) + chunks[i][..., :over_lap] * overlap_weight
                combined_audio = torch.cat([combined_audio[..., :-over_lap], overlap_start, chunks[i][..., over_lap:]], dim=-1)
        else:
            for i in range(1, len(chunks)):
                combined_audio = torch.cat([combined_audio, chunks[i]], dim=-1)

        # Truncate to the original length
        combined_audio = combined_audio[..., :original_length]
        return combined_audio

    def defender_wav_variable_input(self, x, chunk_size=16000, over_lap=0, ddpm=False):
        # For methods that require chunk processing
        if self.defense_method in ['DualPure', 'DDPM', 'DiffNoise', 'DiffRev', 'OneShot', 'PhonePuRe']:
            original_length = x.size(-1)
            chunks = self.split_audio(x, chunk_size, over_lap)
            defended_chunks = [self.defender_wav(chunk, ddpm=ddpm) if self.defender_wav is not None else chunk for chunk in chunks]
            # Concatenate the processed audio chunks back into a single waveform tensor
            waveform_defended = self.combine_chunks(defended_chunks, over_lap, original_length)
        else:
            waveform_defended = self.defender_wav(x)
        return waveform_defended

    def defender_spec_variable_input(self, x, chunk_size=32, over_lap=0):
        # For methods that require chunk processing
        if self.defense_method in ['DualPure', 'DiffSpec']:
            original_length = x.size(-1)
            chunks = self.split_audio(x, chunk_size, over_lap)
            defended_chunks = [self.defender_spec(chunk) if self.defender_spec is not None else chunk for chunk in chunks]
            # Concatenate the processed spectrogram chunks back into a single spectrogram tensor
            # and convert it back to waveform
            spectrogram_defended = self.combine_chunks(defended_chunks, over_lap, original_length)
            wave_defended = spectrogram_to_waveform(spectrogram_defended)
        else:
            spectrogram_defended = self.defender_spec(x)
        return wave_defended

class AcousticSystem_purify(torch.nn.Module):
    def __init__(self, 
                 #spk_encoder: torch.nn.Module=None, 
                 defender_wav: torch.nn.Module=None,
                 defender_spec: torch.nn.Module=None,
                 defense_type: str='wave',
                 defense_method: str='DualPure'
                 ):
        super().__init__()

        '''
            same as AcousticSystem_robust, but without speaker encoder
            the whole audio system: audio -> prediction probability distribution
            
            *defender: audio -> audio or spectrogram -> spectrogram
            *transform: audio -> spectrogram

        '''

        
        if defense_method in ['DualPure', 'DiffSpec']:
            self.transform = Wave2Spect
        else:
            self.transform = None
        self.defender_wav = defender_wav
        self.defender_spec = defender_spec
        self.defense_type = defense_type
        self.defense_method = defense_method

    def forward(self, x, defend=True, file_name = None):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        # resample to defender sr

        # wave defender by chunk
        if defend == True and self.defender_wav is not None and (self.defense_type == 'wave' or self.defense_method in ['DualPure', 'PhonePuRe']):
            output = self.defender_wav_variable_input(x)
        else:
            output = x
        # transform to spec if [DualPure, DiffSpec]
        if self.transform is not None:
            output = self.transform(output)
        # spec defender by chunk
        if defend == True and self.defender_spec is not None and self.defense_method in ['DualPure', 'DiffSpec']:
            output = self.defender_spec_variable_input(output)
        elif defend == True and self.defender_spec is not None and self.defense_method in ['PhonePuRe']:
            output = self.defender_spec(output, file_name = file_name)
        else:
            output = output
        output = output.squeeze(0)


        return output

    
    def defense(self, x, file_name = None, ddpm = False):
        if x.ndim == 2:
            x = x.unsqueeze(0)

        if self.defender_wav is not None:
            # wave defender by chunk
            if (self.defense_type == 'wave' or self.defense_method in ['DualPure', 'PhonePuRe']):
                output = self.defender_wav_variable_input(x, ddpm=ddpm)
                
        else:
            output = x  
        # transform to spec if [DualPure, DiffSpec]
        if self.transform is not None:
            output = self.transform(output)
        # spec defender by chunk
        if self.defender_spec is not None:
            if self.defense_method in ['DualPure', 'DiffSpec']:
                output = self.defender_spec_variable_input(output)
            elif self.defense_method in ['PhonePuRe']:
                output = self.defender_spec(output, file_name = file_name)
        else:
            output = output
        if output.ndim == 1:
            output = output.unsqueeze(0)
        if output.ndim == 3:
            output = output.squeeze(0)

        return output
    
    def split_audio(self, x, chunk_size, over_lap):
        """
        split the audio into fixed-size chunks, with optional overlap, and pad the last chunk if necessary.
        """
        chunks = []
        step = chunk_size - over_lap
        for start in range(0, x.size(-1) - chunk_size + 1, step):
            chunk = x[..., start:start + chunk_size]
            chunks.append(chunk)
        # if the last chunk is smaller than chunk_size, pad it
        if (x.size(-1) - chunk_size) % step != 0:
            last_chunk = x[..., step * len(chunks):]
            if last_chunk.size(-1) < chunk_size:
                padding = chunk_size - last_chunk.size(-1)
                last_chunk = torch.nn.functional.pad(last_chunk, (0, padding))
            chunks.append(last_chunk)
        return chunks

    def combine_chunks(self, chunks, over_lap, original_length):
        """
        merge the processed chunks back into a single audio tensor, 
        considering overlap regions for smooth transitions and restoring to the original length.
        """
        if len(chunks) == 0:
            return None

        combined_audio = chunks[0]
        if over_lap > 0:
            overlap_weight = torch.linspace(0, 1, steps=over_lap).unsqueeze(0)
            for i in range(1, len(chunks)):
                # Mix the overlapping regions
                overlap_start = combined_audio[..., -over_lap:] * (1 - overlap_weight) + chunks[i][..., :over_lap] * overlap_weight
                combined_audio = torch.cat([combined_audio[..., :-over_lap], overlap_start, chunks[i][..., over_lap:]], dim=-1)
        else:
            for i in range(1, len(chunks)):
                combined_audio = torch.cat([combined_audio, chunks[i]], dim=-1)

        # Truncate to the original length
        combined_audio = combined_audio[..., :original_length]
        return combined_audio

    def defender_wav_variable_input(self, x, chunk_size=16000, over_lap=0, ddpm=False):
        # For methods that require chunk processing
        if self.defense_method in ['DualPure', 'DDPM', 'DiffNoise', 'DiffRev', 'OneShot', 'PhonePuRe']:
            original_length = x.size(-1)
            chunks = self.split_audio(x, chunk_size, over_lap)
            defended_chunks = [self.defender_wav(chunk, ddpm=ddpm) if self.defender_wav is not None else chunk for chunk in chunks]
            # Concatenate the processed audio chunks back into a single waveform tensor
            waveform_defended = self.combine_chunks(defended_chunks, over_lap, original_length)
        else:
            waveform_defended = self.defender_wav(x)
        return waveform_defended

    def defender_spec_variable_input(self, x, chunk_size=32, over_lap=0):
        # For methods that require chunk processing
        if self.defense_method in ['DualPure', 'DiffSpec']:
            original_length = x.size(-1)
            chunks = self.split_audio(x, chunk_size, over_lap)
            defended_chunks = [self.defender_spec(chunk) if self.defender_spec is not None else chunk for chunk in chunks]
            # Concatenate the processed spectrogram chunks back into a single spectrogram tensor
            # and convert it back to waveform
            spectrogram_defended = self.combine_chunks(defended_chunks, over_lap, original_length)
            wave_defended = spectrogram_to_waveform(spectrogram_defended)
        else:
            spectrogram_defended = self.defender_spec(x)
        return wave_defended