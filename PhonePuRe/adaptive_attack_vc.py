import argparse
import soundfile as sf
import torch
import os
import random
#from attack_vc_utils import e2e_attack, emb_attack, fb_attack
from attack_vc_data_utils import ModelManager
from attack_vc_robust import EmbAttack
from acoustic_system import AcousticSystem_robust
import sys



def get_speaker_id(file_path):
    """Get speaker ID from the file name."""
    speaker_id = file_path.split("-")[0].split("_p")[-1]
    #print(f"Speaker ID: {speaker_id}")
    return speaker_id

def select_different_voice(file_path, folder_path):
    """Select a different file from the folder."""
    files = [f for f in os.listdir(folder_path) if f != file_path and f.endswith(".wav")]
    return os.path.join(folder_path, random.choice(files)) if files else None

def select_adv_target(file_path, folder_path, gender, speaker_info):
    """Select a voice file of a speaker with a different gender."""
    print(f"Selecting adversarial target for {file_path}...")
    different_gender_files = []
    for f in os.listdir(folder_path):
        if f.endswith(".wav") and f != file_path:
            speaker_id = get_speaker_id(f)
            if speaker_info.get(speaker_id) and speaker_info[speaker_id]["gender"] != gender:
                different_gender_files.append(f)
    return os.path.join(folder_path, random.choice(different_gender_files)) if different_gender_files else None

def load_speaker_info(speaker_file):
    """Load speaker info from SPEAKERS.TXT."""
    speaker_info = {}
    with open(speaker_file, 'r') as f:
        for line in f:
            if line.startswith(";") or line.strip() == "":
                continue
            fields = line.split("|")
            if len(fields) >= 2:
                speaker_id = fields[0].strip()
                gender = fields[1].strip()
                speaker_info[speaker_id] = {"gender": gender}
                #print(f"Speaker info: {speaker_info}")
    return speaker_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Folder containing target voices for conversion.")
    parser.add_argument("output_folder", type=str, help="Folder to save adversarial samples.")

    parser.add_argument("syn_type", type=str, choices=["coqui", "rtvc", "tortoise", "openvoice"], default="rtvc", help="Type of attack.")

    parser.add_argument("--text_dir", type=str, help="Text files of corresponding noisy audio files.")
    parser.add_argument("--phoneme_avg_spec_dict", type=str, help="Path to the phoneme average spectrogram dictionary file. This is used to condition the Refinement model on phonemes.")
    parser.add_argument("--speaker_file", type=str, help="Speaker information file.", default="/path/to/LibriSpeech/SPEAKERS.TXT")

    parser.add_argument("--eps", type=float, default=0.1, help="Max amplitude of the perturbation.")
    parser.add_argument("--n_iters", type=int, default=1500, help="Iterations for updating the perturbation.")
    parser.add_argument("--attack_type", type=str, choices=["e2e", "emb", "fb"], default="emb", help="Type of attack.")
    parser.add_argument('--eot_attack_size', type=int, default=1, help='EOT size of attack')
    parser.add_argument('--eot_defense_size', type=int, default=1)
    parser.add_argument('--save_iters', type=int, nargs='+', default=None, help='Save adversarial samples at these iterations')
    parser.add_argument('--defense', type=str, choices=['DualPure', 'AudioPure', 'DDPM', 'DiffNoise', 'DiffRev', 'OneShot', 'DiffSpec', 'PhonePuRe', 'None'], default='None')
    '''DiffWave-VPSDE arguments'''
    parser.add_argument('--ddpm_config', type=str, default='configs/config.json', help='JSON file for configuration')
    parser.add_argument('--diffwav_path', type=str, default=None, help='dir of Purification model checkpoint')
    parser.add_argument('--diffspec_path', type=str, default=None, help='dir of diffspec model checkpoint')
    parser.add_argument('--score_path', type=str, help="path to Refinement model checkpoint")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=3, help='diffusion steps, control the sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=0, help='perturbation range of sampling noise scale; set to 0 by default')
    parser.add_argument('--rand_t', action='store_true', default=False, help='decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='sde', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
    parser.add_argument('--use_bm', action='store_true', default=True, help='whether to use brownian motion')
    parser.add_argument('--bpda', type=bool, default=False, help='whether to use BPDA adaptive strategy')
    


    args = parser.parse_args()
    '''Create dataset and dataloader'''

    speaker_file = args.speaker_file
    input_folder = args.input_folder
    output_folder = args.output_folder
    syn_type = args.syn_type
    eps = args.eps
    n_iters = args.n_iters
    attack_type = args.attack_type
    defense_method = args.defense
    save_iters = args.save_iters
    save_folders = []

    os.makedirs(output_folder, exist_ok=True)
    if save_iters is not None:
        for i in range(len(save_iters)):
            save_iters[i] = int(save_iters[i])
            save_folders.append(f"{output_folder}_iter_{save_iters[i]}")
            os.makedirs(save_folders[i], exist_ok=True)
    else:
        save_folders = [output_folder]


    speaker_info = load_speaker_info(speaker_file)
    model_manager = ModelManager(syn_type)
    torch.backends.cudnn.benchmark = True
    if args.defense == 'None':
        AS_MODEL = AcousticSystem_robust(model_manager, defender_wav=None, defender_spec=None, defense_method=defense_method)

        '''purification settings'''
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
            # for alignment by mfa
            '''tmp_dir = f'.tmp/tmp_{datetime.now().strftime("%Y%m%d%H%M%S")}'
            os.makedirs(tmp_dir, exist_ok=True)
            input_files = os.listdir(input_folder)
            for file in input_files:
                if file.endswith('.wav'):
                    tmp_in_path = os.path.join(input_folder, file)
                    tmp_out_path = os.path.join(tmp_dir, file)
                    tmp_tensor, _ = model_manager.preprocess_to_tensor(tmp_in_path)
                    tmp_wav, tmp_sr = model_manager.postprocess_to_wav(tmp_tensor)
                    sf.write(tmp_out_path, tmp_wav, tmp_sr)'''
            Defender_spec = Refiner(align_from_folder=args.input_folder, 
                                          text_dir=args.text_dir,
                                          phoneme_avg_spec_dict=args.phoneme_avg_spec_dict,
                                          corrector="ald", 
                                          corrector_steps=1, 
                                          snr=0.4, 
                                          N=30, 
                                          checkpoint_file=args.score_path)
            '''os.system(f'rm -rf {tmp_dir}')'''
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
            AS_MODEL = AcousticSystem_robust(model_manager, defender_wav=Defender_wav, defense_method=defense_method)
        # for DualPure and DiffSpec
        else:
            AS_MODEL = AcousticSystem_robust(model_manager, defender_wav=Defender_wav, defender_spec=Defender_spec, defense_method=defense_method, defense_type=defense_type)
    attacker = EmbAttack(AS_MODEL, eps, n_iters, eot_attack_size=args.eot_attack_size, eot_defense_size=args.eot_defense_size, bpda=args.bpda)
    
    for vc_tgt_file in os.listdir(input_folder):
        vc_tgt_path = os.path.join(input_folder, vc_tgt_file)
        if not vc_tgt_file.endswith(".wav") or not os.path.isfile(vc_tgt_path):
            continue
        if os.path.exists(os.path.join(output_folder, vc_tgt_file)):
            print(f"File {vc_tgt_file} already exists in the output folder, skipping...")
            continue
        # Get speaker information
        speaker_id = get_speaker_id(vc_tgt_file)
        if speaker_id not in speaker_info:
            continue
        gender = speaker_info[speaker_id]["gender"]
        #print(f"Processing {vc_tgt_file}, gender: {gender}")

        # Select vc_src and adv_tgt
        # vc_src: a different voice from the same folder as content, no need for emb attack
        # adv_tgt: a voice of a speaker with different gender as adversarial target
        vc_src = select_different_voice(vc_tgt_file, input_folder)
        adv_tgt = select_adv_target(vc_tgt_file, input_folder, gender, speaker_info)
        #print(f"vc_src: {vc_src}, adv_tgt: {adv_tgt}")
        print("Selected vc_src: ", vc_src, "adv_tgt: ", adv_tgt)
        
        if vc_src and adv_tgt:
            vc_tgt_tensor, vc_tgt_mel_slices = model_manager.preprocess_to_tensor(vc_tgt_path)
            print("vc_tgt_tensor: ", vc_tgt_tensor.shape)
            print("vc_tgt_max: ", torch.max(vc_tgt_tensor), "vc_tgt_min: ", torch.min(vc_tgt_tensor))
            adv_tgt_tensor, adv_tgt_mel_slices = model_manager.preprocess_to_tensor(adv_tgt)
            print("adv_tgt_tensor: ", adv_tgt_tensor.shape)
            
            if attack_type == "emb":
                adv_inp = attacker.generate(vc_tgt_tensor, adv_tgt_tensor, vc_tgt_mel_slices, adv_tgt_mel_slices, file_name=vc_tgt_path)

            else:
                raise NotImplementedError("Unsupported attack type.")
            
            # Convert and save adversarial sample

            adv_inp_wav, sr = model_manager.postprocess_to_wav(adv_inp)

            output_path = os.path.join(output_folder, vc_tgt_file)
            sf.write(output_path, adv_inp_wav, sr)