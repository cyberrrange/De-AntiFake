DEANTIFAKE_ROOT=/path/to/De-AntiFake

PHONEPURE_ROOT=$DEANTIFAKE_ROOT/PhonePuRe
DATA_ROOT=$DEANTIFAKE_ROOT/data
CKPT_ROOT=$DEANTIFAKE_ROOT/checkpoints

# Purification Params
purification_method="DDPM"
# Set purify_step=5 when generating the clean-purified pairs
purify_step=5
sample_step=1

# Input and Output Directories
input_trainset_dir=/path/to/mixed_dataset/train/noisy
input_validset_dir=/path/to/mixed_dataset/valid/noisy
input_testset_dir=/path/to/mixed_dataset/test/noisy

output_trainset_dir=/path/to/mixed_dataset/train/noisy-purified-${defense_method}_t${purify_step}_step${sample_step}
output_validset_dir=/path/to/mixed_dataset/valid/noisy-purified-${defense_method}_t${purify_step}_step${sample_step}
output_testset_dir=/path/to/mixed_dataset/test/noisy-purified-${defense_method}_t${purify_step}_step${sample_step}

# Purification Model Path
diffwav_model_path=$CKPT_ROOT/purification.pkl

GPU_ID=0

cd $PHONEPURE_ROOT
# Generate Purified Train Set
input_audio_dir=$input_trainset_dir
purified_dir=$output_trainset_dir
export CUDA_VISIBLE_DEVICES=$GPU_ID && python purify.py --input_dir $input_audio_dir \
                  --output_dir $purified_dir \
                  --t $purify_step \
                  --diffusion_type ddpm \
                  --sample_step 1 \
                  --purification_methods $purification_method \
                  --diffwav_path $diffwav_model_path \
                  --gpu $GPU_ID

# Generate Purified Validation Set
input_audio_dir=$input_validset_dir
purified_dir=$output_validset_dir
export CUDA_VISIBLE_DEVICES=$GPU_ID && python purify.py --input_dir $input_audio_dir \
                  --output_dir $purified_dir \
                  --t $purify_step \
                  --diffusion_type ddpm \
                  --sample_step 1 \
                  --purification_methods $purification_method \
                  --diffwav_path $diffwav_model_path \
                  --gpu $GPU_ID
# Generate Purified Test Set
input_audio_dir=$input_testset_dir
purified_dir=$output_testset_dir
export CUDA_VISIBLE_DEVICES=$GPU_ID && python purify.py --input_dir $input_audio_dir \
                  --output_dir $purified_dir \
                  --t $purify_step \
                  --diffusion_type ddpm \
                  --sample_step 1 \
                  --purification_methods $purification_method \
                  --diffwav_path $diffwav_model_path \
                  --gpu $GPU_ID