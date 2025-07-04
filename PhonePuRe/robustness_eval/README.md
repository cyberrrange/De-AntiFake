## Adaptive Protection

This directory contains the code to evaluate the robustness of PhonePuRe and other purification methods against adaptive voice cloning defenses. It supports both non-adaptive and adaptive protection scenarios.



### 1. Download Speaker Encoder Checkpoints

The evaluation requires pre-trained speaker encoder  as the target models for adaptive protection.

- **Download checkpoints**: The necessary speaker encoder models are available at [Google Drive](https://drive.google.com/drive/folders/13RroDAtI0MAXx--0fGitpY5N4o_MeY1u?usp=sharing).

- **Place checkpoints**: After downloading, place the files in `PhonePuRe/encoder_models/speaker_encoder_ckpts/`. The directory structure should look like this:
  ```
  PhonePuRe/encoder_models/speaker_encoder_ckpts/
  ├── rtvc/
  │   └── encoder.pt
  ├── tortoise/
  │   ├── autoregressive.pth
  │   ├── diffusion_decoder.pth
  │   └── mel_norms.pth
  └── openvoice/
    └── ...
  ```

### 3. Usage

Before running, please modify the paths (e.g., `DEANTIFAKE_ROOT`) inside the shell scripts to match your local environment.

We implement the adaptive protection based on the [AttackVC](https://github.com/cyhuang-tw/attack-vc) method.
Note that we directly adapt the AttackVC protection method into the time domain. This is because many voice cloning methods use different mel parameters for speaker embedding extraction and the vocoder, which can lead to mismatch and protection failure. This direct time-domain adaptation also satisfies the `emb` method described in the [paper](https://arxiv.org/abs/2005.08781).


#### Non-Adaptive Protection

This scenario generate protective perturbations when the purification method is unknown to the protector.

```bash
bash non_adaptive_protection.sh
```

Key parameters in `non_adaptive_protection.sh`:
- `target_vc_method`: The target voice cloning model. Supported options are:
    - `rtvc`: [Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) and [DiffVC](https://github.com/agoyr/DiffVC/tree/feature/uniform/DiffVC) (they share the same speaker encoder).
    - `openvoice`: [OpenVoice V2](https://github.com/myshell-ai/OpenVoice)
    - `coqui`: [YourTTS](https://github.com/Edresson/YourTTS)
    - `tortoise`: [TorToiSe](https://github.com/neonbjb/tortoise-tts)
- `eps`: The perturbation budget for the protection.
- `n_iters`: The number of protection iterations.

If you select `target_vc_method` from `[openvoice, coqui, tortoise]`, you may need to install the corresponding TTS library. Create a new conda environment and install the required libraries in `env_for_protection.yaml` is recommended: 
```bash
conda env create -f env_for_protection.yaml
conda activate protection
```
This will create a conda environment named `protection`.

#### Adaptive Protection

This scenario generate adaptive protective perturbations where the protector has full knowledge of the purification methods. We use `target_vc_method=rtvc` (DiffVC) in the paper.

For the "BPDA" adaptive strategy, run:
```bash
bash adaptive_protection_BPDA.sh
```
For the "Adjoint" adaptive strategy, run:
```bash
bash adaptive_protection_Adjoint.sh
```

Key parameters in `adaptive_protection.sh`:
- `defense_method`: The purification method to be evaluated (e.g., `PhonePuRe`).
- `target_vc_method`: The target voice cloning model (e.g., `rtvc`).
- `BPDA`: Set to `True` for the "BPDA" adaptive strategy from the paper.
- `diffusion_type`: Use `sde` (with `defense_method=DDPM`) for the "Adjoint" adaptive strategy from the paper.
- `eot_size`: The number of random transformations for the EOT attack.
