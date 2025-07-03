# Prepare Train Data

## Step1: Get LibriSpeech Dataset and DEMAND Dataset
- Download the LibriSpeech dataset from [LibriSpeech](http://www.openslr.org/12/).
- Unzip the dataset to a directory, e.g., `/data/LibriSpeech`.
- Extract the dataset to a directory in **wav** format, e.g., `/data/LibriSpeech_wav`. This can be used as **Purification Model** training data. See `PhonePuRe/data_preprocess/step1_libri_flac2wav.py`.
- Ensure the directory structure is as follows:
```
/LibriSpeech_wav
├── train
│   ├── 108-134-0000.wav
│   ├── 108-134-0001.wav
│   └── ...
├── valid
│   ├── 108-134-0002.wav
│   ├── 108-134-0003.wav
│   └── ...
└── test
    ├── 108-134-0004.wav
    ├── 108-134-0005.wav
    └── ...
```
- Download the DEMAND dataset from [DEMAND](https://zenodo.org/records/1227121).
- Extract the DEMAND dataset to a directory, e.g., `/data/DEMAND`.
- Ensure the directory structure is as follows:
```
/DEMAND
├── TMETRO_16k
│   ├── TMETRO
│   │   ├── ch01.wav
│   │   ├── ch02.wav
│   │   └── ...
├── TCAR_16k
│   ├── TCAR
│   │   ├── ch01.wav
│   │   ├── ch02.wav
│   │   └── ...
└── ...
```
## Step2: Mix LibriSpeech and DEMAND for Data Augmentation
- Use the provided script `PhonePuRe/data_preprocess/step2_mix_librispeech_demand.py` to mix the LibriSpeech and DEMAND datasets.
- The script will create a mixed dataset in the specified output directory.
- The output directory structure will be as follows:
```
/mixed_dataset
├── train
│   ├── clean
│   │   ├── 108-134-0000.wav
│   │   ├── 108-134-0001.wav
│   │   └── ...
│   ├── noisy
│   │   ├── 108-134-0000.wav
│   │   ├── 108-134-0001.wav
│   │   └── ...
├── valid
│   ├── clean
│   │   └── ...
│   ├── noisy
│   │   └── ...
├── test
│   ├── clean
│   │   └── ...
│   └── noisy
│       └── ...
├── train_mix_info.txt
├── val_mix_info.txt
└── test_mix_info.txt
```
The `train_mix_info.txt`, `valid_mix_info.txt`, and `test_mix_info.txt` files contain information about the mixed audio files, the content will be in the format:
```
1246-124548-0043 PCAFETER ch01 10
4481-17499-0016 DKITCHEN ch01 0
7511-102419-0007 SPSQUARE ch01 10
```

## Step3: Generate Clean-Purified Audio Pairs as Refinement Model Training Data
- Use the provided script `PhonePuRe/data_preprocess/step3_clean_purified_pairs.sh` to generate clean-purified audio pairs. The input directory should contain the mixed dataset created in Step2: `/mixed_dataset/train/noisy`, `/mixed_dataset/valid/noisy`, and `/mixed_dataset/test/noisy`.
- The script will create a directory structure for the purified audio pairs.
- The `/mixed_dataset` directory in step 2 structure will be as follows:
```
/mixed_dataset
├── train
│   ├── clean
│   │   ├── 108-134-0000.wav
│   │   ├── 108-134-0001.wav
│   │   └── ...
│   ├── noisy
│   │   ├── 108-134-0000.wav
│   │   ├── 108-134-0001.wav
│   │   └── ...
│   ├── noisy-purified-DDPM_t5_step1
│   │   ├── 108-134-0000.wav
│   │   ├── 108-134-0001.wav
│   │   └── ...
├── valid
│   ├── clean
│   │   └── ...
│   ├── noisy
│   │   └── ...
│   ├── noisy-purified-DDPM_t5_step1
│   │   └── ...
├── test
│   ├── clean
│   │   └── ...
│   ├── noisy
│   │   └── ...
│   └── noisy-purified-DDPM_t5_step1
│       └── ...
├── train_mix_info.txt
├── val_mix_info.txt
└── test_mix_info.txt
```

## Step4: Generate TextGrid Files for the LibriSpeech Dataset
You can download the TextGrid files from the [Google Drive Link](https://drive.google.com/file/d/1OgfXbTYhgp8NW5fRTt_TXwViraOyVEyY/view?usp=sharing) in [LibriSpeech Alignments](https://github.com/CorentinJ/librispeech-alignments).

Or you can generate them follow the instructions in the [MFA documentation](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/example.html).

## Step5: Generate Phoneme Average Spec Dictionary
### Step5.1: If you want to use the provided phoneme average spec dictionary:
You can directly use the provided files `data/phoneme_avg_spec_dict/amps_avg_libri_spec.txt`, `data/phoneme_avg_spec_dict/libri_get_spec_dataset_train.txt` and `data/phoneme_avg_spec_dict/libri_get_spec_dataset_valid.txt` for training the Refinement Model.

### Step5.2: If you want to generate the phoneme average spec dictionary by yourself:
If you want to generate the phoneme average spec dictionary by yourself, you can follow these steps:
- First preprocess the TextGrid files and audio files to generate the metadata for the train files. See `PhonePuRe/data_preprocess/step5_1_preprocess.py`. 
  - Modify the `PhonePuRe/data_preprocess/configs/libri_get_spec.json` to specify the correct paths for the `clean_flist`, `textgrid_path`, and `dataset_file_path`. 
  - The `clean_flist` should point to the file containing the list of clean audio files.
  - The `textgrid_path` should point to the directory containing the TextGrid files. 
  - The `dataset_file_path` should be the output file where the metadata will be saved.
- Then run the script `PhonePuRe/data_preprocess/step5_2_get_avg_spec.py` to generate the phoneme average spec dictionary. 
  - The `PhonePuRe/data_preprocess/configs/libri_get_spec.json` should also be modified to specify the correct paths for the `avg_mel_path` to save the average spectrograms dictionary.
- The output will be saved in a file like `data/phoneme_avg_spec_dict/amps_avg_libri_spec.txt` in the specified directory.