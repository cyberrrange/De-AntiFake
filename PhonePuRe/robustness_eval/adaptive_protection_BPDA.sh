gpu_id=2

DEANTIFAKE_ROOT=/path/to/De-AntiFake

PHONEPURE_ROOT=$DEANTIFAKE_ROOT/PhonePuRe
DATA_ROOT=$DEANTIFAKE_ROOT/data
CKPT_ROOT=$DEANTIFAKE_ROOT/checkpoints

clean_dir=$DATA_ROOT/test_set_example_clean/audio
output_dir=$DATA_ROOT/test_set_example_clean/audio-adaptive_protected_BPDA

# for PhonePuRe to generate phoneme representations
input_text_dir=$DATA_ROOT/test_set_example_clean/text
phoneme_avg_spec_dict=$DATA_ROOT/librispeech_metadata/amps_avg_libri_spec.txt

# Used for selecting the target speaker in the AttackVC protection method
speaker_file=$DATA_ROOT/librispeech_metadata/SPEAKERS.TXT
# The target voice cloning method for the protection
target_vc_method=rtvc


# The model and params for purification and refinement
purification_model_path=$CKPT_ROOT/purification.pkl
refinement_model_path=$CKPT_ROOT/refinement.ckpt
defense_method=PhonePuRe
defense_t=3
defense_step=1

# AttackVC protection parameters
eps=0.01
n_iters=150

# Adaptive Protection parameters
eot_size=5
diffusion_type=ddpm # diffusion_type=sde and defense_method=DDPM for "Adjoint" adaptive strategy in the paper
BPDA=True # True for "BPDA" adaptive strategy in the paper


cd $PHONEPURE_ROOT


# For adaptive protection, we only test on DiffVC (target_vc_method=rtvc) in the paper.


# Note that we directly adapt the AttackVC method into time domain, 
# for many voice cloning methods use different mel parameters in speaker embedding extraction and vocoder,
# which may lead to the failure of the protection.
# The direct adaptation into time domain also satisfies the `emb` method in the paper.

# rtvc-> Real-Time Voice Cloning, DiffVC (They share the same speaker encoder)

# Adaptive Protection
export CUDA_VISIBLE_DEVICES=$gpu_id && python adaptive_attack_vc.py $clean_dir \
                                                                    $output_dir \
                                                                    $target_vc_method \
                                                                    --text_dir $input_text_dir \
                                                                    --phoneme_avg_spec_dict $phoneme_avg_spec_dict \
                                                                    --speaker_file $speaker_file \
                                                                    --eps $eps \
                                                                    --t $defense_t\
                                                                    --sample_step $defense_step\
                                                                    --defense $defense_method\
                                                                    --diffwav_path $purification_model_path \
                                                                    --score_path $refinement_model_path \
                                                                    --n_iters $n_iters \
                                                                    --diffusion_type $diffusion_type \
                                                                    --eot_attack_size $eot_size \
                                                                    --bpda $BPDA