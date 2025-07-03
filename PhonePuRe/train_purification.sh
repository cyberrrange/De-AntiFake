cd purify/diffusion_models/DiffWave_Unconditional

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id && python distributed_train_libri.py -c config_libri.json