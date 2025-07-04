DEANTIFAKE_ROOT=/path/to/De-AntiFake
PHONEPURE_ROOT=$DEANTIFAKE_ROOT/PhonePuRe
cd $PHONEPURE_ROOT/purification_models/DiffWave_Unconditional

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id && python distributed_train_libri.py -c config_libri.json