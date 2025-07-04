DEANTIFAKE_ROOT=/path/to/De-AntiFake
PHONEPURE_ROOT=$DEANTIFAKE_ROOT/PhonePuRe
cd $PHONEPURE_ROOT/refinement_models
#export CUDA_HOME=/app/cuda/cuda-11.3
gpu_id=0
gpu_num=1
export CUDA_VISIBLE_DEVICES=$gpu_id && python train.py --gpus $gpu_num \
                                                    --format librispeech-demand-diffwave-t5 \
                                                    --base_dir /path/to/mixed_dataset \
                                                    --backbone_denoiser none \
                                                    --backbone_score ncsnpp \
                                                    --loss_type_score TF-mse \
                                                    --mode regen-freeze-denoiser \
                                                    --lr 1e-4 \
                                                    --condition post_denoiser \
                                                    --return_time \
                                                    --use_text