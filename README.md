# 🔓De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks

Source code for paper *De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks*

by _Wei Fan, Kejiang Chen, Chang Liu, Weiming Zhang, and Nenghai Yu_ 

In [International Conference on Machine Learning (ICML) 2025](https://icml.cc/Conferences/2025).

Visit our [website](https://de-antifake.github.io/) for audio samples.

## TODO List
- [x] Upload code (training, test and data preprocessing)
- [ ] Upload robustness evaluation code
- [ ] Upload evaluation data
- [x] Upload model checkpoints
- [x] Write introduction (data preparation, environments, etc.)

## 🔶Introduction

In this repository, we provide the complete code for training and testing the Purification and Refinement model. 

- [x] **Our Proposed PhonePuRe**: see [PhonePuRe](https://github.com/cyberrrange/de-antifake/tree/main/PhonePuRe) for details.


- [ ] **Robustness Evaluation**: see [robustness_eval](https://github.com/cyberrrange/de-antifake/tree/main/robustness_eval) for details. (**Note**: We will update it soon.)

## 🧊Installation
To run the code, you need to set up the environment and install the required dependencies:
- **Clone the repository**:
  ```bash
  git clone https://github.com/cyberrrange/De-AntiFake.git
  cd De-AntiFake
  ```
- **Create Env and Install dependencies**:
  You can use the provided `environment.yaml` file to create the environment:
  ```bash
  conda env create -f environment.yaml
  conda activate phonepure
  ```
- **Install MFA** to generate the phoneme alignment files for inference. You can **create a new conda environment named `aligner`** and install MFA in it (follow the instructions in the [MFA documentation](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)). The environment name `aligner` here **will be used in the inference script**.


## 🍷Use Our Pre-trained Model


If you just want to **test** our Purification and Refinement model:
- **Prepare your test data**. You can use our example data from `data/test_set_example`, or prepare your own dataset. Make sure the audio files are in `.wav` format and named in the format `[dataset]_p[speaker_id]-[utterance_id].wav`. And the transcript files should be in `.txt` format with the same naming convention. An example of the test data structure is as follows:
  ```
  test_data/
  ├── audio/
  │   ├── [dataset]_p[speaker1]-[utterance1].wav
  │   ├── [dataset]_p[speaker1]-[utterance2].wav
  └── text/
      ├── [dataset]_p[speaker1]-[utterance1].txt
      └── [dataset]_p[speaker1]-[utterance2].txt
  ```

- **Download checkpoints**. The parameter files used in our work are available at [Google Drive](https://drive.google.com/drive/folders/1jr6D96cVTS9qOQAUQHkHKdNuIGYVwf3X?usp=sharing). Download the checkpoints and place them in the `checkpoints` directory. The directory structure should look like this:
  ```
  checkpoints/
  ├── diffspec_model.pth
  ├── purification.pth
  └── refinement.pth
  ```

- **Run the inference script**. You can run the inference script to test the model:
  ```bash
  bash PhonePuRe/inference.sh
  ```
  Remember to set the `DEANTIFAKE_ROOT` variable in the `inference.sh` script to the root directory of the `De-AntiFake` repository. 

## 🍵Train Your Own Model

If you want to **train** the Purification and Refinement model:
- **Download the LibriSpeech dataset** from [LibriSpeech](http://www.openslr.org/12/).
- **Download the DEMAND dataset** from [DEMAND](https://zenodo.org/records/1227121) for data augmentation.
- **Follow the instructions in [data_preprocess](https://github.com/cyberrrange/de-antifake/tree/main/PhonePuRe/data_preprocess)** to form the phoneme dictionary and the phoneme alignment files.
- **Run the training script**. 
  You can run the training script to train the purification model:
  ```bash
  bash PhonePuRe/train_purification.sh
  ```
  Due to the purification model and refinement model is cascaded, you need to train the purification model first, and use the trained purification model to generate the purified audio files for the refinement model training.
  And you can run the training script to train the refinement model:
  ```bash
  bash PhonePuRe/train_refinement.sh
  ```
  Some parameters and paths in the training scripts need to be set according to your environment. You can search for `/path/to/` in the scripts and replace them with the actual paths in your environment.

You can also prepare your own dataset, but make sure to follow the similar data structure as the instruction. 



## Acknowledgments

Part of our code were based on several open-source repositories, including [DiffWave](https://github.com/philsyn/DiffWave-unconditional), [AudioPure](https://github.com/cychomatica/AudioPure), [DualPure](https://github.com/Sec4ai/DualPure), [StoRM](https://github.com/sp-uhh/storm) and [DMSE4TTS](https://github.com/dmse4tts/DMSE4TTS). Their code served as a foundation for portions of our experiments.

## Citation
If you find this work useful, please consider citing our paper:
```
@inproceedings{de-antifake-icml2025,
  title = {De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks},
  author = {Fan, Wei and Chen, Kejiang and Liu, Chang and Zhang, Weiming and Yu, Nenghai},
  booktitle = {International Conference on Machine Learning},
  year = {2025},
}
```