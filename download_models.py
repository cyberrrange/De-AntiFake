from huggingface_hub import hf_hub_download
from pathlib import Path 
import os


SCRIPT_DIR = Path(__file__).resolve().parent


target_dir = SCRIPT_DIR / "checkpoints"  

print(f"Target download directory will be: {target_dir}")
os.makedirs(target_dir, exist_ok=True) # Create the directory if it doesn't exist

# --- Download the first file ---
purification_weights_path = hf_hub_download(
    repo_id="cyberrrange/De-AntiFake",
    filename="purification.pkl",
    local_dir=target_dir,
    local_dir_use_symlinks=False
)

print(f"Purification model weights downloaded to: {purification_weights_path}")


# --- Download the second file ---
refinement_weights_path = hf_hub_download(
    repo_id="cyberrrange/De-AntiFake",
    filename="refinement.ckpt",
    local_dir=target_dir,
    local_dir_use_symlinks=False
)

print(f"Refinement model weights downloaded to: {refinement_weights_path}")

print("\nBoth files are now in your specified directory:")
print(os.listdir(target_dir))
