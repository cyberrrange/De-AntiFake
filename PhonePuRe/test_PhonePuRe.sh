DEANTIFAKE_ROOT=/path/to/De-AntiFake

PHONEPURE_ROOT=$DEANTIFAKE_ROOT/PhonePuRe
DATA_ROOT=$DEANTIFAKE_ROOT/data
CKPT_ROOT=$DEANTIFAKE_ROOT/checkpoints

input_audio_dir=$DATA_ROOT/test_set_example_protected/audio
input_text_dir=$DATA_ROOT/test_set_example_protected/text
phoneme_avg_spec_dict=$DATA_ROOT/librispeech_metadata/amps_avg_libri_spec.txt


output_dir=$DATA_ROOT/test_set_example_protected


purification_model_path=$CKPT_ROOT/purification.pkl
refinement_model_path=$CKPT_ROOT/refinement.ckpt

purification_method="PhonePuRe"
# purification_methods=("DDPM" "AS" "MS" "DS" "LPF" "BPF" "AudioPure")

GPU_ID=2

purification_t=3
purification_sample_step=1
refinement_step=30
refinement_snr=0.4

cd $PHONEPURE_ROOT
export CUDA_VISIBLE_DEVICES=$GPU_ID && python purify.py --input_dir $input_audio_dir \
                  --output_dir $output_dir \
                  --text_dir $input_text_dir \
                  --phoneme_avg_spec_dict $phoneme_avg_spec_dict \
                  --diffusion_type ddpm \
                  --t $purification_t \
                  --sample_step $purification_sample_step \
                  --score_N $refinement_step \
                  --snr $refinement_snr \
                  --purification_methods $purification_method \
                  --diffwav_path $purification_model_path \
                  --score_path $refinement_model_path \
                  --gpu $GPU_ID