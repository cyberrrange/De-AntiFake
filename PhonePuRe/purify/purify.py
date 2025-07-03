import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio

import utils

# torch.manual_seed(42)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    '''SC09 classifier arguments'''
    parser.add_argument("--input_dir", help='Input folder path containing multiple .wav files')
    parser.add_argument("--output_dir", help='Output folder path')
    #parser.add_argument("--classifier_path", help='Saved classifier model path')
    #parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='NN input')
    #parser.add_argument("--num_per_class", type=int, default=10)

    '''DiffWave-VPSDE arguments'''
    parser.add_argument('--ddpm_config', type=str, default='configs/config.json', help='Configuration JSON file')
    parser.add_argument('--diffwav_path', type=str, default=".../checkpoints/purification.pkl", help='Diffusion model checkpoint path')
    parser.add_argument('--diffspec_path', type=str, default=".../checkpoints/diffspec_model.pt", help='Diffusion model checkpoint path')
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=3, help='Diffusion steps, controls sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=0, help='Perturbation range for sampling noise scale; default is 0')
    parser.add_argument('--rand_t', action='store_true', default=False, help='Whether to randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='sde', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
    parser.add_argument('--use_bm', action='store_true', default=True, help='Whether to use Brownian motion')

    '''Defense method parameters'''
    parser.add_argument('--defense_methods', nargs='+', choices=['DiffSpec', 'DDPM', 'DualPure', 'AS', 'MS', 'DS', 'LPF', 'BPF', 'AudioPure', 'DiffNoise', 'DiffRev', 'OneShot'], default=['DDPM'], help='Select defense methods')

    '''Device parameters'''
    parser.add_argument("--dataload_workers_nums", type=int, default=8, help='Number of dataloader worker threads')
    parser.add_argument("--batch_size", type=int, default=1, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    '''Device setup'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    print('gpu id: {}'.format(args.gpu))

    '''SC09 classifier setup'''
    from transforms import *
    #from datasets.sc_dataset import *
    #from audio_models.create_model import *


    n_mels = 32
    #n_mels = 256
    #n_fft = 1024
    n_fft = 2048
    #hop_length = 256
    hop_length = 512
    #sample_rate = 16000  
    
    #win_length = 1024
    win_length = 2048
    sample_rate = 16000

    # Define inverse Mel spectrogram transform
    inverse_mel_transform = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft//2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate
    )

    # Define Griffin-Lim transform
    griffin_lim_transform = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft
    )
    inverse_mel_transform = inverse_mel_transform.cuda()
    griffin_lim_transform = griffin_lim_transform.cuda()
    def spectrogram_to_waveform(spectrogram):
        # Convert dB to amplitude
        spectrogram = 10.0 ** (spectrogram / 20.0)
        # Reconstruct linear spectrogram from Mel spectrogram
        linear_spectrogram = inverse_mel_transform(spectrogram)

        # Reconstruct waveform from linear spectrogram using Griffin-Lim
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
        Split audio into fixed-size chunks, with optional overlap
        """
        length = waveform.shape[-1]
        stride = chunk_size - overlap
        chunks = []
        for i in range(0, length, stride):
            chunk = waveform[..., i:i+chunk_size]
            if chunk.shape[-1] < chunk_size:
                # Pad the last chunk
                padding = chunk_size - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, padding))
            chunks.append(chunk)
        return chunks

    def merge_chunks(chunks, original_length, overlap=0):
        """
        Merge processed chunks back to the original audio length
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
        
        # Handle overlap regions
        result = result / weights.clamp(min=1)
        return result[..., :original_length]
    '''Create dataset and dataloader'''
    class WavDataset(torch.utils.data.Dataset):
        def __init__(self, folder, transform=None):
            self.folder = folder
            self.file_paths = []
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith('.wav'):
                        self.file_paths.append(os.path.join(root, file))
            self.transform = transform

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            path = self.file_paths[idx]
            waveform, sample_rate = torchaudio.load(path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            sample = {'samples': waveform, 'path': path}
            if self.transform:
                sample = self.transform(sample)
            return sample
    '''Define AcousticSystem_purify class'''
    class AcousticSystem_purify(torch.nn.Module):

        def __init__(self,
                     #classifier: torch.nn.Module,
                     transform,
                     defender_wav: torch.nn.Module = None,
                     defender_spec: torch.nn.Module = None,
                     defense_type: str = 'wave',
                     defense_method: str = 'DDPM'
                     ):
            super().__init__()

            # self.classifier = classifier
            self.transform = transform
            self.defender_wav = defender_wav
            self.defender_spec = defender_spec
            self.defense_type = defense_type
            self.defense_method = defense_method
            self.defense_spec = None
            if self.defense_type not in ['wave', 'spec']:
                raise NotImplementedError('argument defense_type should be \'wave\' or \'spec\'!')

        def forward(self, x, defend=True):

            if defend == True and self.defender_wav is not None and (self.defense_type == 'wave' or self.defense_method == 'DualPure'):
                output = self.defender_wav_variable_input(x)
            else:
                output = x

            if self.transform is not None:
                output = self.transform(output)

            if defend == True and self.defender_spec is not None and self.defense_spec == 'spec':
                output = self.defender_spec_variable_input(output)
            else:
                output = output

            #output = self.classifier(output)

            return output
        
        def split_audio(self, x, chunk_size, over_lap):
            """
            Split audio into chunks, pad the last chunk to keep the same size.
            """
            chunks = []
            step = chunk_size - over_lap
            for start in range(0, x.size(-1) - chunk_size + 1, step):
                chunk = x[..., start:start + chunk_size]
                chunks.append(chunk)
            # Pad the last chunk if it is less than chunk_size
            if (x.size(-1) - chunk_size) % step != 0:
                last_chunk = x[..., step * len(chunks):]
                if last_chunk.size(-1) < chunk_size:
                    padding = chunk_size - last_chunk.size(-1)
                    last_chunk = torch.nn.functional.pad(last_chunk, (0, padding))
                chunks.append(last_chunk)
            return chunks

        def combine_chunks(self, chunks, over_lap, original_length):
            """
            Concatenate processed audio chunks, handle overlap smoothing, and restore original length.
            """
            if len(chunks) == 0:
                return None

            combined_audio = chunks[0]
            if over_lap > 0:
                overlap_weight = torch.linspace(0, 1, steps=over_lap).unsqueeze(0)
                for i in range(1, len(chunks)):
                    # Mix overlap region
                    overlap_start = combined_audio[..., -over_lap:] * (1 - overlap_weight) + chunks[i][..., :over_lap] * overlap_weight
                    combined_audio = torch.cat([combined_audio[..., :-over_lap], overlap_start, chunks[i][..., over_lap:]], dim=-1)
            else:
                for i in range(1, len(chunks)):
                    combined_audio = torch.cat([combined_audio, chunks[i]], dim=-1)

            # Truncate to original length
            combined_audio = combined_audio[..., :original_length]
            return combined_audio

        def defender_wav_variable_input(self, x, chunk_size=16000, over_lap=0):
            # Chunk processing
            if self.defense_method in ['DualPure', 'DDPM', 'DiffNoise', 'DiffRev', 'OneShot', 'ScorePure', 'AudioPure']:
                original_length = x.size(-1)
                chunks = self.split_audio(x, chunk_size, over_lap)
                defended_chunks = [self.defender_wav(chunk) if self.defender_wav is not None else chunk for chunk in chunks]
                # Concatenate processed audio
                waveform_defended = self.combine_chunks(defended_chunks, over_lap, original_length)
            else:
                waveform_defended = self.defender_wav(x)
            return waveform_defended

        def defender_spec_variable_input(self, x, chunk_size=32, over_lap=0):
            # Chunk processing
            if self.defense_method in ['DualPure', 'DiffSpec']:
                original_length = x.size(-1)
                chunks = self.split_audio(x, chunk_size, over_lap)
                defended_chunks = [self.defender_spec(chunk) if self.defender_spec is not None else chunk for chunk in chunks]
                # Concatenate processed spectrogram
                spectrogram_defended = self.combine_chunks(defended_chunks, over_lap, original_length)
            else:
                spectrogram_defended = self.defender_spec(x)
            return spectrogram_defended
        
    transform = None
    test_dataset = WavDataset(folder=args.input_dir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.dataload_workers_nums)

    # defense_method_all = ['DualPure', 'AudioPure', 'DDPM', 'DiffNoise', 'DiffRev', 'OneShot', 'DiffSpec']
    defense_method_all = ['DiffSpec', 'DualPure', \
                          'AS', 'MS', 'DS', 'LPF', 'BPF', \
                          'AudioPure', 'DiffNoise', 'DiffRev', 'OneShot', 'DDPM', \
                          'OPUS', 'SPEEX', 'AMR', 'ACC_V', 'ACC_C', 'MP3_V', 'MP3_C']

    '''Process each defense method'''
    for defense_method in args.defense_methods:
        print('Processing defense method: {}'.format(defense_method))
        args.defense = defense_method
        if args.defense == 'None':
            AS_MODEL = AcousticSystem_purify(transform=None, defender_wav=None, defender_spec=None, defense_method=defense_method)

            '''Defense setup'''
        else:
            if args.defense in ['DDPM', 'AudioPure', 'DiffNoise', 'DiffRev', 'OneShot']:
                from diffusion_models.diffwave_sde import *
                Defender_wav = RevDiffWave(args)
                defense_type = 'wave'
            elif args.defense == 'DiffSpec':
                Defender_wav = None
                from diffusion_models.improved_diffusion_sde import *
                Defender_spec = RevImprovedDiffusion(args)
                defense_type = 'spec'
            elif args.defense == 'DualPure':
                from diffusion_models.diffwave_sde import *
                Defender_wav = RevDiffWave(args)
                from diffusion_models.improved_diffusion_sde import *
                Defender_spec = RevImprovedDiffusion(args)
                # infact, defense_type is 'dual'
                defense_type = 'spec'
            elif args.defense in ['AS', 'MS']:
                from transforms.time_defense import *
                Defender_wav = TimeDomainDefense(defense_type=args.defense)
                defense_type = 'wave'
            elif args.defense in ['DS', 'LPF', 'BPF']:
                from transforms.frequency_defense import *
                Defender_wav = FreqDomainDefense(defense_type=args.defense)
                defense_type = 'wave'
            elif args.defense in ['OPUS', 'SPEEX', 'AMR', 'ACC_V', 'ACC_C', 'MP3_V', 'MP3_C']:
                from transforms.speech_compression import *
                Defender_wav = CompressionDefense(defense_type=args.defense)
                defense_type = 'wave'
            else:
                raise NotImplementedError(f'Unknown defense: {args.defense}!')

            if defense_type == 'wave':
                AS_MODEL = AcousticSystem_purify(transform=None, defender_wav=Defender_wav, defense_method=defense_method)
            # for DualPure and DiffSpec
            else:
                AS_MODEL = AcousticSystem_purify(transform=Wave2Spect, defender_wav=Defender_wav, defender_spec=Defender_spec, defense_method=defense_method, defense_type=defense_type)
            print('Defense method: {}'.format(args.defense))
            print('Defense config: {}'.format(args.ddpm_config))
            AS_MODEL.eval()

            '''Process each file'''
            from tqdm import tqdm
            pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

            for batch in pbar:
                waveforms = batch['samples']
                paths = batch['path']

                waveforms = waveforms.cuda()

                with torch.no_grad():
                    '''Apply defense method'''
                    if AS_MODEL.defense_type == 'wave':
                        waveforms_defended = AS_MODEL.defender_wav_variable_input(waveforms)
                        # Save output
                        for i in range(waveforms_defended.shape[0]):
                            input_file_name = os.path.basename(paths[i])
                            input_dir_name = os.path.basename(os.path.dirname(paths[i]))
                            output_dir = args.output_dir

                            os.makedirs(output_dir, exist_ok=True)
                            # Save defended waveform
                            utils.audio_save(waveforms_defended[i], path=output_dir, name=input_file_name)
                    elif AS_MODEL.defense_type == 'spec':
                        if defense_method == 'DualPure':
                            waveforms = AS_MODEL.defender_wav_variable_input(waveforms)
                        spectrogram = AS_MODEL.transform(waveforms)
                        spectrogram_defended = AS_MODEL.defender_spec_variable_input(spectrogram)
                        # Optionally save defended spectrogram or convert back to waveform
                        with torch.set_grad_enabled(True):
                            waveforms_defended = utils.mel_spectrogram_to_wav(spectrogram_defended, n_fft=n_fft, hop_length=hop_length, sample_rate=sample_rate, win_length=win_length)
                            #waveforms_original = utils.mel_spectrogram_to_wav(spectrogram, n_fft=n_fft, hop_length=hop_length, sample_rate=sample_rate, win_length=win_length)
                        # Save output
                        for i in range(waveforms_defended.shape[0]):
                            input_file_name = os.path.basename(paths[i])
                            input_dir_name = os.path.basename(os.path.dirname(paths[i]))
                            output_dir = args.output_dir

                            os.makedirs(output_dir, exist_ok=True)
                            # Save defended spectrogram and audio
                            utils.spec_save(spectrogram_defended[i], path=output_dir, name=input_file_name.replace('.wav', '.png'))
                            utils.audio_save(waveforms_defended[i], path=output_dir, name=input_file_name)
                        pass

                pbar.update(1)


