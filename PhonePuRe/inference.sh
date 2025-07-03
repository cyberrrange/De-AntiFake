DEANTIFAKE_ROOT=/path/to/De-AntiFake

PHONEPURE_ROOT=$DEANTIFAKE_ROOT/PhonePuRe
DATA_ROOT=$DEANTIFAKE_ROOT/data
CKPT_ROOT=$DEANTIFAKE_ROOT/checkpoints

# input noisy audio directory, text directory, and phoneme average spectrogram dictionary
input_audio_dir=$DATA_ROOT/test_set_example/audio
input_text_dir=$DATA_ROOT/test_set_example/text
phoneme_avg_spec_dict=$DATA_ROOT/phoneme_avg_spec_dict/amps_avg_libri_spec.txt

purified_dir=$DATA_ROOT/test_set_example/audio-purified
output_dir=$DATA_ROOT/test_set_example/audio-purified-refined

diffspec_model_path=$CKPT_ROOT/diffspec_model.pt

diffwav_model_path=$CKPT_ROOT/purification.pkl
refine_model_path=$CKPT_ROOT/refinement.ckpt

defense_method="DDPM"
# defense_methods=("DDPM" "OneShot" "AS" "MS" "DS" "LPF" "BPF" "AudioPure")

GPU_ID=2

purify_step=3

cd $PHONEPURE_ROOT/purify
export CUDA_VISIBLE_DEVICES=$GPU_ID && python purify.py --input_dir $input_audio_dir \
                  --output_dir $purified_dir \
                  --t $purify_step \
                  --diffusion_type ddpm \
                  --sample_step 1 \
                  --defense_methods $defense_method \
                  --diffwav_path $diffwav_model_path \
                  --diffspec_path $diffspec_model_path \
                  --gpu $GPU_ID



# source activate storm
cd $PHONEPURE_ROOT/refine
export CUDA_VISIBLE_DEVICES=$GPU_ID && python refine.py \
        --mode storm \
        --noisy_audio_dir $input_audio_dir \
        --text_dir $input_text_dir \
        --phoneme_avg_spec_dict $phoneme_avg_spec_dict \
        --purified_dir $purified_dir \
        --refined_dir $output_dir \
        --ckpt $refine_model_path \
        --snr 0.4 \
        --N 30