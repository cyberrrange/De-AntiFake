import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio
from acoustic_system import AcousticSystem_purify

import utils
import shutil
import sys


# torch.manual_seed(42)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    '''input/output arguments'''
    parser.add_argument("--input_dir", help='input folder path with multiple .wav files')
    parser.add_argument("--output_dir", help='output folder path to save purified-refined audios')
    parser.add_argument("--text_dir", type=str, help="Text files of corresponding noisy audio files.")
    parser.add_argument("--phoneme_avg_spec_dict", type=str, help="Path to the phoneme average spectrogram dictionary file. This is used to condition the model on phonemes.")



    '''DiffWave-VPSDE arguments'''
    parser.add_argument('--ddpm_config', type=str, default='configs/config.json', help='json config file for Purification model')
    parser.add_argument('--diffwav_path', type=str, default="", help="path to Purification model checkpoint")
    parser.add_argument('--score_path', type=str, help="path to Refinement model checkpoint")
    parser.add_argument('--diffspec_path', type=str, default="", help="used by DualPure/DiffSpec method")
    parser.add_argument('--sample_step', type=int, default=1, help='total sampling steps for Purification model')
    parser.add_argument('--t', type=int, default=3, help='noise scale for Purification model; default is 3')
    parser.add_argument('--t_delta', type=int, default=0, help='noise scale delta for Purification model; default is 0')
    parser.add_argument('--rand_t', action='store_true', default=False, help='whether to use random noise scale for Purification model; default is False')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
    parser.add_argument('--use_bm', action='store_true', default=True, help='whether to use brownian motion for Purification model; default is True')
    parser.add_argument('--score_N', type=int, default=30, help='Refinement model sample steps')
    parser.add_argument('--snr', type=float, default=0.4, help='Refinement model SNR')

    '''purification methods'''
    parser.add_argument('--purification_methods', nargs='+', choices=['AudioPure', 'DDPM', 'DiffNoise', 'DiffRev', 'OneShot', #wave
                                                                 'DiffSpec', 'DualPure', 'PhonePuRe', #spec
                                                                 'AS', 'MS', 'DS', 'LPF', 'BPF', #signal processing
                                                                 'OPUS', 'SPEEX', 'AMR', 'ACC_V', 'ACC_C', 'MP3_V', 'MP3_C' #compression
                                                                ], default=['PhonePuRe'], help='select purification methods')

    '''device and dataloader settings'''
    parser.add_argument("--dataload_workers_nums", type=int, default=8, help='dataloader workers nums')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size for dataloader')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    '''device settings'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    print('gpu id: {}'.format(args.gpu))


    from transforms import *
    #from datasets.sc_dataset import *


    n_mels = 32
    #n_mels = 256
    #n_fft = 1024
    n_fft = 2048
    #hop_length = 256
    hop_length = 512
    #sample_rate = 16000  # mel spectrogram params for DualPure and DiffSpec methods
    
    #win_length = 1024
    win_length = 2048
    sample_rate = 16000


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

        
    transform = None
    input_audio_dir = os.path.join(args.input_dir)
    test_dataset = WavDataset(folder=input_audio_dir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.dataload_workers_nums)

    # defense_method_all = ['DualPure', 'AudioPure', 'DDPM', 'DiffNoise', 'DiffRev', 'OneShot', 'DiffSpec']
    defense_method_all = ['DiffSpec', 'DualPure', \
                          'AS', 'MS', 'DS', 'LPF', 'BPF', \
                          'AudioPure', 'DiffNoise', 'DiffRev', 'OneShot', 'DDPM', 'PhonePuRe' \
                          'OPUS', 'SPEEX', 'AMR', 'ACC_V', 'ACC_C', 'MP3_V', 'MP3_C']

    '''process each defense method'''
    for defense_method in args.purification_methods:
        print('Processing defense method: {}'.format(defense_method))
        args.defense = defense_method
        if args.defense == 'None':
            AS_MODEL = AcousticSystem_purify(defender_wav=None, defender_spec=None, defense_method=defense_method)

            '''Defense settings'''
        else:
            if args.defense in ['DDPM', 'AudioPure', 'DiffNoise', 'DiffRev', 'OneShot']:
                from purification_models.diffwave_sde import *
                Defender_wav = RevDiffWave(args)
                defense_type = 'wave'
            elif args.defense == 'PhonePuRe':
                from purification_models.diffwave_sde import *
                Defender_wav = RevDiffWave(args)
                sys.path.append('./refinement_models')
                from refiner import Refiner
                from datetime import datetime
                #for alignment by mfa
                '''
                tmp_dir = f'./.tmp/tmp_{datetime.now().strftime("%Y%m%d%H%M%S")}'
                os.makedirs(tmp_dir, exist_ok=True)
                input_files = os.listdir(args.input_dir)for file in input_files:
                    if file.endswith('.wav'):
                        tmp_in_path = os.path.join(args.input_dir, file)
                        tmp_out_path = os.path.join(tmp_dir, file)
                        #tmp_tensor, _ = model_manager.preprocess_to_tensor(tmp_in_path)
                        #tmp_wav, tmp_sr = model_manager.postprocess_to_wav(tmp_tensor)
                        #sf.write(tmp_out_path, tmp_wav, tmp_sr)
                        shutil.copy(tmp_in_path, tmp_out_path)'''
                Defender_spec = Refiner(align_from_folder=args.input_dir, 
                                          text_dir=args.text_dir,
                                          phoneme_avg_spec_dict=args.phoneme_avg_spec_dict,
                                          corrector="ald", 
                                          corrector_steps=1, 
                                          snr=args.snr, 
                                          N=args.score_N, 
                                          checkpoint_file=args.score_path)
                #os.system(f'rm -rf {tmp_dir}')
                defense_type = 'spec'
            elif args.defense == 'DiffSpec':
                Defender_wav = None
                from purification_models.improved_diffusion_sde import *
                Defender_spec = RevImprovedDiffusion(args)
                defense_type = 'spec'
            elif args.defense == 'DualPure':
                from purification_models.diffwave_sde import *
                Defender_wav = RevDiffWave(args)
                from purification_models.improved_diffusion_sde import *
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
                AS_MODEL = AcousticSystem_purify(defender_wav=Defender_wav, defense_method=defense_method, defense_type=defense_type)
            # for DualPure and DiffSpec and ScorePure
            else:
                AS_MODEL = AcousticSystem_purify(defender_wav=Defender_wav, defender_spec=Defender_spec, defense_method=defense_method, defense_type=defense_type)
            print('Defense Method: {}'.format(args.defense))
            print('Defense Params: {}'.format(args.ddpm_config))
            AS_MODEL.eval()

            '''Process each audio file'''
            from tqdm import tqdm
            pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

            for batch in pbar:
                waveforms = batch['samples']
                paths = batch['path']

                waveforms = waveforms.cuda()

                with torch.no_grad():
                    for i in range(waveforms.shape[0]):
                        waveform = waveforms[i].unsqueeze(0)
                        waveform_defended = AS_MODEL.defense(waveform, file_name=paths[i], ddpm=True)
                        input_dir_name = os.path.basename(os.path.dirname(paths[i]))
                        input_file_name = os.path.basename(paths[i])
                        output_dir = os.path.join(args.output_dir, f'{input_dir_name}-purified-{defense_method}')
                        
                        if defense_method in ['DDPM', 'DiffNoise', 'DiffRev', 'OneShot', 'AudioPure', 'PhonePuRe', 'DiffSpec', 'DualPure']:
                            output_dir = output_dir + '_t' + str(args.t) + '_step' + str(args.sample_step)
                        os.makedirs(output_dir, exist_ok=True)
                        utils.audio_save(waveform_defended, path=output_dir, name=input_file_name)

                pbar.update(1)

