"""
Utility functions for ComfyUI HunyuanVideo-Foley custom node
"""

import os
import tempfile
import torch
import numpy as np
from typing import Union, Optional, Tuple
from loguru import logger
import decord
from PIL import Image
from einops import rearrange

# We need to import the original library functions that our safe wrappers will call.
from hunyuanvideo_foley.utils.feature_utils import encode_text_feat, encode_video_with_siglip2, encode_video_with_sync
from hunyuanvideo_foley.utils.config_utils import AttributeDict


def tensor_to_video(video_tensor: torch.Tensor, output_path: str, fps: int = 30) -> str:
    """
    Convert a video tensor to a video file
    """
    try:
        import cv2
        video_np = video_tensor.detach().cpu().numpy() if isinstance(video_tensor, torch.Tensor) else np.array(video_tensor)
        
        if video_np.ndim == 4 and video_np.shape[1] == 3:
            video_np = np.transpose(video_np, (0, 2, 3, 1))
        elif video_np.ndim == 5 and video_np.shape[1] == 3:
            video_np = np.transpose(video_np[0], (0, 2, 3, 1))
        
        video_np = (video_np * 255).astype(np.uint8) if video_np.max() <= 1.0 else video_np.astype(np.uint8)
        
        frames, height, width, channels = video_np.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i in range(frames):
            frame = video_np[i]
            if channels == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        logger.info(f"Video saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to convert tensor to video: {e}")
        raise

def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        cap.release()
        return info
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {}

def ensure_video_file(video_input: Union[str, torch.Tensor, np.ndarray]) -> str:
    """
    Ensure the video input is converted to a file path
    """
    if isinstance(video_input, str):
        if os.path.exists(video_input): return video_input
        else: raise FileNotFoundError(f"Video file not found: {video_input}")
    elif isinstance(video_input, (torch.Tensor, np.ndarray)):
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "input_video.mp4")
        return tensor_to_video(video_input, output_path)
    else:
        raise ValueError(f"Unsupported video input type: {type(video_input)}")

def validate_model_files(model_path: str) -> Tuple[bool, str]:
    """
    Validate that all required model files exist
    """
    required_files = ["hunyuanvideo_foley.pth", "vae_128d_48k.pth", "synchformer_state_dict.pth"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        return False, f"Missing model files: {', '.join(missing_files)}"
    return True, "All required model files found"

def get_optimal_device() -> torch.device:
    """
    Get the optimal device for model execution
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        props = torch.cuda.get_device_properties(device)
        logger.info(f"Using CUDA device: {props.name} with {props.total_memory / 1e9:.1f}GB memory")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Using MPS device (Apple Silicon)")
        return torch.device("mps")
    else:
        logger.info("Using CPU device")
        return torch.device("cpu")

def check_memory_requirements(device: torch.device, required_gb: float = 16.0) -> Tuple[bool, str]:
    """
    Check if the device has enough memory for model execution
    """
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        if total_memory < required_gb:
            return False, f"GPU has {total_memory:.1f}GB memory, but {required_gb}GB is recommended"
        else:
            return True, f"GPU has {total_memory:.1f}GB memory (sufficient)"
    return True, "Using non-CUDA device (memory check not applicable)"

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable format
    """
    seconds = float(seconds)
    if seconds < 60: return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60: return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"

# --- NEW MEMORY-SAFE AND UNIFIED FUNCTIONS ---

def extract_video_path(video):
    """Extract a file path from various potential video input types. Moved here to be a standalone utility."""
    if hasattr(video, '__class__') and 'VideoFromFile' in video.__class__.__name__:
        if hasattr(video, '_VideoFromFile__file'): return getattr(video, '_VideoFromFile__file')
        for attr in ['file', 'path', 'filename']:
            if hasattr(video, attr) and isinstance(getattr(video, attr), str): return getattr(video, attr)
    if isinstance(video, str): return video
    if isinstance(video, dict) and 'path' in video: return video['path']
    return None

@torch.inference_mode()
def _encode_visual_features_safely(frames_uint8, model_dict, fps_hint):
    """ Memory-safe visual feature extraction from a list of pre-loaded frames. """
    dev = model_dict.device
    
    # Offload other models before starting
    model_dict.clap_model.to("cpu")
    model_dict.foley_model.to("cpu")
    model_dict.dac_model.to("cpu")
    if dev.type == 'cuda': torch.cuda.empty_cache()
    
    visual_features = {}
    pil_list = [Image.fromarray(f).convert("RGB") for f in frames_uint8]
    
    try:
        logger.info("Moving SigLIP2 to device for feature extraction...")
        model_dict.siglip2_model.to(dev)
        siglip_list = [model_dict.siglip2_preprocess(im) for im in pil_list]
        clip_frames = torch.stack(siglip_list, dim=0).unsqueeze(0).to(dev)
        visual_features['siglip2_feat'] = encode_video_with_siglip2(clip_frames, model_dict).to(dev)
    finally:
        logger.info("Offloading SigLIP2 from device."); model_dict.siglip2_model.to("cpu")
        if dev.type == 'cuda': torch.cuda.empty_cache()

    try:
        logger.info("Moving Syncformer to device for feature extraction...")
        model_dict.syncformer_model.to(dev)
        # Correctly preprocess frames for syncformer as per original library logic
        images = torch.from_numpy(np.array(frames_uint8)).permute(0, 3, 1, 2)
        sync_frames = model_dict.syncformer_preprocess(images).unsqueeze(0).to(dev)
        visual_features['syncformer_feat'] = encode_video_with_sync(sync_frames, model_dict)
    finally:
        logger.info("Offloading Syncformer from device."); model_dict.syncformer_model.to("cpu")
        if dev.type == 'cuda': torch.cuda.empty_cache()

    audio_len_in_s = len(frames_uint8) / fps_hint
    return AttributeDict(visual_features), audio_len_in_s

def feature_process_unified(video_input, image_input, prompt, model_dict, cfg, fps_hint=24.0, max_frames=450):
    """ Unified, memory-safe feature processing for either a video path or an image tensor. """
    frames_uint8 = None; fps = fps_hint
    
    if image_input is not None:
        logger.info("Processing features from IMAGE tensor input.")
        frames_uint8 = (image_input.cpu().clip(0, 1) * 255).byte().numpy() if image_input.dtype != torch.uint8 else image_input.cpu().numpy()
    elif video_input is not None:
        logger.info("Processing features from VIDEO path input.")
        video_path = extract_video_path(video_input)
        if not video_path: raise ValueError("Invalid video path provided.")
        
        video_reader = decord.VideoReader(video_path)
        total_frames = len(video_reader)
        fps = video_reader.get_avg_fps()
        
        frame_indices = np.linspace(0, total_frames - 1, num=min(total_frames, max_frames), dtype=int)
        frames_uint8 = video_reader.get_batch(frame_indices).asnumpy()

    if frames_uint8 is None: raise ValueError("No valid video or image frames to process.")

    visual_feats, audio_len_in_s = _encode_visual_features_safely(frames_uint8, model_dict, fps)

    neg_prompt = cfg.get("neg_prompt", "noisy, harsh")
    prompts = [neg_prompt, prompt]
    clap_model = model_dict.clap_model
    try:
        clap_model.to(model_dict.device)
        text_feat_res, _ = encode_text_feat(prompts, model_dict)
    finally:
        clap_model.to("cpu")

    text_feat = text_feat_res[1:]; uncond_text_feat = text_feat_res[:1]
    if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
        text_seq_length = cfg.model_config.model_kwargs.text_length
        text_feat = text_feat[:, :text_seq_length]
        uncond_text_feat = uncond_text_feat[:, :text_seq_length]

    text_feats = AttributeDict({'text_feat': text_feat, 'uncond_text_feat': uncond_text_feat})
    return visual_feats, text_feats, audio_len_in_s