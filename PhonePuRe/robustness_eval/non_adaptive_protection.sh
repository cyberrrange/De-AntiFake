gpu_id=0

DEANTIFAKE_ROOT=/path/to/De-AntiFake

PHONEPURE_ROOT=$DEANTIFAKE_ROOT/PhonePuRe
DATA_ROOT=$DEANTIFAKE_ROOT/data

clean_dir=$DATA_ROOT/test_set_example_clean/audio
output_dir=$DATA_ROOT/test_set_example_clean/audio-non_adaptive_protected

# Used for selecting the target speaker in the AttackVC protection method
speaker_file=$DATA_ROOT/librispeech_metadata/SPEAKERS.TXT
# The target voice cloning method for the protection
target_vc_method=rtvc



# AttackVC protection parameters
n_iters=150
eps=0.01




cd $PHONEPURE_ROOT

# For non-adaptive protection, you can select target_vc_method from [rtvc, openvoice, coqui, tortoise].
# If you select from [openvoice, coqui, tortoise], you may need to install the corresponding TTS library.
# Create a new conda environment and install the required libraries in `env_for_protection.yaml` is recommended.

# Note that we directly adapt the AttackVC method into time domain, 
# for many voice cloning methods use different mel parameters in speaker embedding extraction and vocoder,
# which may lead to the failure of the protection.
# The direct adaptation into time domain also satisfies the `emb` method in the paper.

# rtvc-> Real-Time Voice Cloning, DiffVC (They share the same speaker encoder)
# openvoice-> OpenVoice V2
# coqui-> YourTTS
# tortoise-> Tortoise TTS

# Non-Adaptive Protection
export CUDA_VISIBLE_DEVICES=$gpu_id && python adaptive_attack_vc.py $clean_dir \
                                                                    $output_dir \
                                                                    $target_vc_method \
                                                                    --speaker_file $speaker_file \
                                                                    --eps $eps \
                                                                    --n_iters $n_iters