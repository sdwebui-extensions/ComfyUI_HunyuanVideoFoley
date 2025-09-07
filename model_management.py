import os
import torch
import comfy.utils
from loguru import logger
import folder_paths
from huggingface_hub import hf_hub_download

# --- Constants ---
FOLEY_MODEL_NAMES = ["hunyuanvideo_foley.pth", "vae_128d_48k.pth", "synchformer_state_dict.pth"]
SIGLIP_MODEL_REPO = "google/siglip-base-patch16-512"
CLAP_MODEL_REPO = "laion/clap-htsat-unfused"

# --- Path Management ---
def get_model_dir(subfolder=""):
    """Returns the primary Foley models directory."""
    return os.path.join(folder_paths.get_folder_paths("foley")[0], subfolder)

def get_full_model_path(model_name, subfolder=""):
    """Returns the full path for a given model name."""
    return os.path.join(get_model_dir(subfolder), model_name)

# --- Core Functionality ---
def find_or_download(model_name, repo_id, subfolder="", subfolder_in_repo=""):
    """
    Finds a model file, downloading it if it's not found in standard locations.
    - Checks the main ComfyUI foley models directory first.
    - Falls back to downloading from Hugging Face.
    """
    local_path = get_full_model_path(model_name, subfolder)
    
    if os.path.exists(local_path):
        logger.info(f"Found local model: {local_path}")
        return local_path
    
    logger.warning(f"Could not find {model_name} locally. Attempting to download from {repo_id}...")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_name,
            subfolder=subfolder_in_repo,
            local_dir=get_model_dir(subfolder),
            local_dir_use_symlinks=False
        )
        logger.info(f"Successfully downloaded model to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Failed to download {model_name} from {repo_id}: {e}")
        raise FileNotFoundError(f"Could not find or download {model_name}. Please check your connection or download it manually.")

def get_siglip_path():
    """Special handling for the SigLIP model which is a directory."""
    return find_or_download_directory(repo_id=SIGLIP_MODEL_REPO, local_dir_name="siglip-base-patch16-512")

def get_clap_path():
    """Special handling for the CLAP model which is a directory."""
    return find_or_download_directory(repo_id=CLAP_MODEL_REPO, local_dir_name="clap-htsat-unfused")

def find_or_download_directory(repo_id, local_dir_name):
    """
    Finds a model directory, downloading it if it's not found.
    This is for models like SigLIP that are not single files.
    """
    local_path = get_model_dir(local_dir_name)

    if os.path.exists(local_path) and os.listdir(local_path):
        logger.info(f"Found local model directory: {local_path}")
        return local_path
    
    logger.warning(f"Could not find {local_dir_name} directory locally. Attempting to download from {repo_id}...")

    # We can't use hf_hub_download for a whole directory in the same way,
    # but the transformers library will handle this caching for us automatically
    # when `from_pretrained` is called. We just need to return the repo_id.
    # The actual "download" is implicit.
    return repo_id
