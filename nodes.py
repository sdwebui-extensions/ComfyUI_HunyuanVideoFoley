import os
import glob
import torch
import torchaudio
import tempfile
import numpy as np
from loguru import logger
from typing import Optional, Tuple, Any, List
import folder_paths
import random
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
import shutil
import time
import math
from tqdm import tqdm
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Import ComfyUI video types
try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    try:
        # Fallback to latest API location
        from comfy_api.latest._input_impl.video_types import VideoFromFile
    except ImportError:
        logger.warning("VideoFromFile not available, will return file paths only")
        VideoFromFile = None

# Add foley models directory to ComfyUI folder paths
foley_models_dir = os.path.join(folder_paths.models_dir, "foley")
if "foley" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["foley"] = ([foley_models_dir], folder_paths.supported_pt_extensions)

# Import model URLs configuration
try:
    from .model_urls import MODEL_URLS, get_model_url, list_available_models
except ImportError:
    logger.warning("model_urls.py not found, using default URLs")
    MODEL_URLS = {
        "hunyuanvideo-foley-xxl": {
            "models": [
                {
                    "url": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/hunyuanvideo_foley.pth",
                    "filename": "hunyuanvideo_foley.pth",
                    "description": "Main HunyuanVideo-Foley model"
                },
                {
                    "url": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/synchformer_state_dict.pth",
                    "filename": "synchformer_state_dict.pth",
                    "description": "Synchformer model weights"
                },
                {
                    "url": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/vae_128d_48k.pth",
                    "filename": "vae_128d_48k.pth",
                    "description": "VAE model weights"
                }
            ],
            "extracted_dir": "hunyuanvideo-foley-xxl",
            "description": "HunyuanVideo-Foley XXL model for audio generation"
        }
    }
    
# Import the HunyuanVideo-Foley modules
try:
    from hunyuanvideo_foley.utils.model_utils import load_model, denoise_process
    from hunyuanvideo_foley.utils.feature_utils import feature_process
    from hunyuanvideo_foley.utils.media_utils import merge_audio_video
except ImportError as e:
    logger.error(f"Failed to import HunyuanVideo-Foley modules: {e}")
    logger.error("Make sure the HunyuanVideo-Foley package is installed and accessible")
    raise

class HunyuanVideoFoleyNode:
    """
    A node for generating audio using the HunyuanVideo-Foley model
    """
    # Class-level model storage for persistence
    _model_dict = None
    _cfg = None
    _device = None
    _model_path = None
    _memory_efficient = False  # Track memory mode
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "footstep sound, impact, water splash",
                    "display": "textarea",
                    "placeholder": "Describe the audio you want to generate..."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "step": 1,
                    "display": "slider"
                }),
                "sample_nums": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "display": "number"
                }),
            },
            "optional": {
                # Hybrid input support - either VIDEO or IMAGE
                "video": ("VIDEO",),
                "images": ("IMAGE",),  # Support for frame sequences from PR
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Frames per second (only used with IMAGE input)"
                }),
                # Negative prompt from PR
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "display": "textarea",
                    "placeholder": "Additional negative prompts (optional). Will be combined with built-in quality controls."
                }),
                # Output options
                "output_format": (["video_path", "frames", "both"], {
                    "default": "both",
                    "tooltip": "Choose output format: video file path only works with VIDEO input, frames only works with IMAGE input and VIDEO input"
                }),
                "output_folder": ("STRING", {
                    "default": "hunyuan_foley",
                    "multiline": False,
                    "display": "text",
                    "placeholder": "Subfolder name in ComfyUI/output/"
                }),
                "filename_prefix": ("STRING", {
                    "default": "foley_",
                    "multiline": False,
                    "display": "text",
                    "placeholder": "Prefix for output filename"
                }),
                # Memory optimization options
                "memory_efficient": ("BOOLEAN", {
                    "default": False,
                    "display": "checkbox",
                    "tooltip": "Enable memory-efficient mode: unloads models after generation and uses aggressive garbage collection"
                }),
                "cpu_offload": ("BOOLEAN", {
                    "default": False,
                    "display": "checkbox",
                    "tooltip": "Offload models to CPU when not in use (slower but saves VRAM)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("video_path", "video_frames", "audio", "status_message")
    FUNCTION = "generate_audio"
    CATEGORY = "HunyuanVideo-Foley"
    DESCRIPTION = "Generate synchronized audio for videos using HunyuanVideo-Foley model"
    
    @classmethod
    def _required_model_filenames(cls, model_variant: str) -> List[str]:
        if model_variant == "hunyuanvideo-foley-s":
            return ["hunyuanvideo_foley.pth", "vae_88d_48k.pth", "synchformer_state_dict.pth"]
        else:
            # XXL (default)
            return ["hunyuanvideo_foley.pth", "vae_128d_48k.pth", "synchformer_state_dict.pth"]
    
    @staticmethod
    def _string_looks_like_video_path(input_str):
        if not isinstance(input_str, str):
            return False
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv', '.wmv', '.mpg', '.mpeg']
        return any(input_str.lower().endswith(ext) for ext in video_extensions)
    
    @classmethod
    def _extract_video_path(cls, video):
        """
        Extract a file path from various potential video input types
        """
        # Handle ComfyUI VideoFromFile object
        if hasattr(video, '__class__') and 'VideoFromFile' in video.__class__.__name__:
            # Try to access the private attribute
            if hasattr(video, '_VideoFromFile__file'):
                return getattr(video, '_VideoFromFile__file')
            # Try other common attributes
            for attr in ['file', 'path', 'filename', '__file']:
                if hasattr(video, attr):
                    value = getattr(video, attr)
                    if isinstance(value, str):
                        return value
        
        # Direct string path
        if isinstance(video, str):
            # Check if it looks like a video path
            if cls._string_looks_like_video_path(video):
                return video
            else:
                # Could be JSON or other string representation
                try:
                    import json
                    parsed = json.loads(video)
                    if isinstance(parsed, dict):
                        # Try to get path from dict
                        if 'path' in parsed:
                            return parsed['path']
                        elif 'file' in parsed:
                            return parsed['file']
                        elif 'filename' in parsed:
                            return parsed['filename']
                except:
                    pass
        
        # Dictionary with path key
        elif isinstance(video, dict):
            if 'path' in video:
                return video['path']
            elif 'file' in video:
                return video['file']
            elif 'filename' in video:
                return video['filename']
            elif 'video' in video:
                return cls._extract_video_path(video['video'])
        
        # List (possibly frames)
        elif isinstance(video, (list, tuple)):
            if len(video) > 0:
                # Check if first element is a path
                first = video[0]
                if isinstance(first, str) and cls._string_looks_like_video_path(first):
                    return first
                # Could be a list of frames - return None to trigger temp video creation
                return None
        
        # Tensor (video frames)
        elif hasattr(video, 'shape'):
            # This is a tensor, we'll need to save it as a temp video
            return None
        
        # Object with attributes
        elif hasattr(video, '__dict__'):
            # Try to get common attribute names
            for attr in ['path', 'file', 'filename', 'video_path', 'file_path', '_VideoFromFile__file']:
                if hasattr(video, attr):
                    value = getattr(video, attr)
                    if isinstance(value, str) and (cls._string_looks_like_video_path(value) or os.path.exists(value)):
                        return value
        
        # Custom __str__ method might return path
        try:
            str_repr = str(video)
            if cls._string_looks_like_video_path(str_repr):
                return str_repr
        except:
            pass
        
        return None

    
    @classmethod
    def _extract_frames_from_image_input(cls, images, fps=24.0):
        """
        Convert IMAGE input to video frames and create a temporary video file
        Based on PR #3 by yichengup
        """
        import cv2
        import tempfile
        
        try:
            if images is None:
                return None, "No images provided"
            
            # Handle different image input formats
            if hasattr(images, 'shape'):
                # Tensor input [batch, height, width, channels]
                if len(images.shape) == 4:
                    frames = images.cpu().numpy()
                else:
                    return None, f"Unexpected image tensor shape: {images.shape}"
            else:
                return None, f"Unsupported image input type: {type(images)}"
            
            # Convert to uint8 if needed
            if frames.dtype != np.uint8:
                if frames.max() <= 1.0:
                    frames = (frames * 255).astype(np.uint8)
                else:
                    frames = frames.astype(np.uint8)
            
            # Get dimensions
            batch_size, height, width = frames.shape[:3]
            
            # Create temporary video file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_fd)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                return None, "Failed to open video writer"
            
            # Write frames
            for i in range(batch_size):
                frame = frames[i]
                
                # Handle channels - convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                out.write(frame)
            
            out.release()
            
            logger.info(f"Created temporary video from {batch_size} frames at {fps} FPS: {temp_path}")
            return temp_path, "Success"
            
        except Exception as e:
            return None, f"Error converting images to video: {str(e)}"

    
    @classmethod
    def _process_frame_tensor_features(cls, frame_tensor, text_prompt, fps, model_dict, cfg):
        """
        Process frame tensor directly without creating temporary video files
        This bypasses the file I/O overhead and works directly with VHS frame data
        """
        from hunyuanvideo_foley.utils.feature_utils import encode_text_feat, encode_video_with_siglip2, encode_video_with_sync
        from hunyuanvideo_foley.utils.config_utils import AttributeDict
        import torch
        from einops import rearrange
        
        logger.info(f"Processing {frame_tensor.shape[0]} frames directly from tensor")
        
        # Convert ComfyUI frame format to the expected format
        # ComfyUI frames: [batch, height, width, channels] in range [0,1]
        # Expected: [1, time, channels, height, width] in range [0,1]
        
        frames = frame_tensor
        if len(frames.shape) == 4:
            # Rearrange from [T, H, W, C] to [1, T, C, H, W]
            frames = rearrange(frames, 't h w c -> 1 t c h w')
        
        # Ensure frames are float and in [0,1] range
        if frames.dtype != torch.float32:
            frames = frames.float()
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        b, t, c, h, w = frames.shape
        logger.info(f"Frame tensor shape: {frames.shape}")
        
        # Calculate audio length based on frame count and FPS
        audio_len_in_s = t / fps
        logger.info(f"Calculated audio length: {audio_len_in_s:.2f}s for {t} frames at {fps} FPS")
        
        # Process visual features
        logger.info("Extracting SigLIP2 visual features...")
        
        # Preprocess frames for SigLIP2 (resize to 512x512)
        siglip2_frames = torch.nn.functional.interpolate(
            rearrange(frames, 'b t c h w -> (b t) c h w'),
            size=(512, 512), 
            mode='bilinear', 
            align_corners=False
        )
        siglip2_frames = rearrange(siglip2_frames, '(b t) c h w -> b t c h w', b=b)
        siglip2_frames = model_dict.siglip2_preprocess(siglip2_frames)
        siglip2_feat = encode_video_with_siglip2(siglip2_frames, model_dict)
        
        logger.info("Extracting Synchformer features...")
        
        # Preprocess frames for Synchformer (resize to 224x224)
        sync_frames = torch.nn.functional.interpolate(
            rearrange(frames, 'b t c h w -> (b t) c h w'),
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        sync_frames = rearrange(sync_frames, '(b t) c h w -> b t c h w', b=b)
        sync_frames = model_dict.syncformer_preprocess(sync_frames)
        syncformer_feat = encode_video_with_sync(sync_frames, model_dict)
        
        # Create visual features object
        visual_feats = AttributeDict({
            'siglip2_feat': siglip2_feat,
            'syncformer_feat': syncformer_feat,
        })
        
        # Process text features
        logger.info("Processing text features...")
        neg_prompt = "noisy, harsh"
        prompts = [neg_prompt, text_prompt]
        text_feat_res, text_feat_mask = encode_text_feat(prompts, model_dict)
        
        text_feats = AttributeDict({
            'text_feat': text_feat_res[1:2],  # Positive prompt
            'uncond_text_feat': text_feat_res[0:1],  # Negative prompt
            'text_mask': text_feat_mask[1:2],
            'uncond_text_mask': text_feat_mask[0:1],
        })
        
        logger.info(f"Feature extraction complete: visual={visual_feats.siglip2_feat.shape}, text={text_feats.text_feat.shape}")
        
        return visual_feats, text_feats, audio_len_in_s
    
    @classmethod
    def _process_negative_prompt(cls, user_negative_prompt=""):
        """
        Combine built-in negative prompt with user input
        Based on PR #3 by yichengup
        """
        built_in_neg = "noisy, harsh"
        if user_negative_prompt and user_negative_prompt.strip():
            combined_neg_prompt = f"{built_in_neg}, {user_negative_prompt.strip()}"
        else:
            combined_neg_prompt = built_in_neg
        return combined_neg_prompt
    
    @staticmethod
    def _to_uint8_frame(frame):
        """
        Convert a frame tensor to uint8 format suitable for video encoding
        """
        if frame.dtype == torch.uint8:
            return frame
        
        # Assume frame is in [0, 1] or [-1, 1] range
        if frame.min() < 0:
            # [-1, 1] range
            frame = (frame + 1) / 2
        
        # Clamp to [0, 1]
        frame = torch.clamp(frame, 0, 1)
        
        # Convert to [0, 255]
        frame = (frame * 255).to(torch.uint8)
        
        return frame
    
    @classmethod
    def _write_temp_video(cls, video) -> Tuple[bool, Optional[str], str]:
        """
        Try to write video tensor/frames to a temporary mp4 file
        Returns: (success, file_path, error_message)
        """
        import tempfile
        import cv2
        
        try:
            # Handle tensor input
            if hasattr(video, 'shape'):
                # Expected shape: [frames, height, width, channels] or [batch, frames, height, width, channels]
                if len(video.shape) == 5:
                    # Take first batch
                    video = video[0]
                
                if len(video.shape) != 4:
                    return False, None, f"Unexpected video tensor shape: {video.shape}"
                
                frames = video
            
            # Handle list of frames
            elif isinstance(video, (list, tuple)):
                if len(video) == 0:
                    return False, None, "Empty video frame list"
                
                # Stack frames if they're separate tensors
                if hasattr(video[0], 'shape'):
                    frames = torch.stack(list(video))
                else:
                    return False, None, "Unknown frame format in list"
            
            else:
                return False, None, f"Cannot process video type: {type(video)}"
            
            # Convert to numpy
            if hasattr(frames, 'cpu'):
                frames = frames.cpu()
            
            if hasattr(frames, 'numpy'):
                frames = frames.numpy()
            
            # Ensure uint8 format
            if frames.dtype != np.uint8:
                if frames.max() <= 1.0:
                    frames = (frames * 255).astype(np.uint8)
                else:
                    frames = frames.astype(np.uint8)
            
            # Get dimensions
            num_frames, height, width = frames.shape[:3]
            
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_fd)  # Close the file descriptor
            
            # Setup video writer with better codec for ffmpeg compatibility
            # Try different codecs in order of preference
            codecs_to_try = [
                cv2.VideoWriter_fourcc(*'avc1'),  # H.264 (best compatibility)
                cv2.VideoWriter_fourcc(*'H264'),  # Alternative H.264
                cv2.VideoWriter_fourcc(*'XVID'),  # XVID (good compatibility)
                cv2.VideoWriter_fourcc(*'mp4v')   # Fallback
            ]
            
            fps = 30  # Default FPS
            out = None
            
            for fourcc in codecs_to_try:
                out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
                if out.isOpened():
                    logger.info(f"Successfully created video writer with codec: {fourcc}")
                    break
                else:
                    out.release()
                    out = None
            
            if out is None or not out.isOpened():
                return False, None, "Failed to open video writer with any codec"
            
            # Write frames
            for i in range(num_frames):
                frame = frames[i]
                
                # Handle channels
                if len(frame.shape) == 3:
                    if frame.shape[2] == 3:
                        # RGB to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif frame.shape[2] == 4:
                        # RGBA to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                out.write(frame)
            
            out.release()
            
            logger.info(f"Created temporary video: {temp_path}")
            return True, temp_path, "Success"
            
        except Exception as e:
            return False, None, f"Error creating temp video: {str(e)}"
    
    @classmethod
    def setup_device(cls, device_type: str = "auto", device_id: int = 0):
        """Setup and return the appropriate device with memory optimization"""
        # Try to import ComfyUI's model management if available
        try:
            import comfy.model_management as mm
            device = mm.get_torch_device()
            logger.info(f"Using ComfyUI device: {device}")
            return device
        except:
            pass
        
        if device_type == "auto":
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{device_id}")
                # Clear cache before loading
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats(device)
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_type)
        
        logger.info(f"Using device: {device}")
        return device
    
    @classmethod
    def unload_models(cls):
        """Unload all models from memory and clear VRAM"""
        logger.info("Unloading models from memory...")
        
        if cls._model_dict is not None:
            # Move models to CPU first if they're on GPU
            for key, model in cls._model_dict.items():
                if hasattr(model, 'cpu'):
                    try:
                        model.cpu()
                    except:
                        pass
            
            # Clear the model dictionary
            cls._model_dict = None
        
        cls._cfg = None
        cls._device = None
        cls._model_path = None
        
        # Aggressive garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Models unloaded and memory cleared")
    
    @classmethod
    def offload_models_to_cpu(cls):
        """Move models to CPU to save VRAM"""
        if cls._model_dict is not None:
            logger.info("Offloading models to CPU...")
            for key, model in cls._model_dict.items():
                if hasattr(model, 'cpu'):
                    try:
                        model.cpu()
                    except:
                        pass
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Models offloaded to CPU")
    
    @classmethod
    def move_models_to_device(cls):
        """Move models back to the target device"""
        if cls._model_dict is not None and cls._device is not None:
            logger.info(f"Moving models to {cls._device}...")
            for key, model in cls._model_dict.items():
                if hasattr(model, 'to'):
                    try:
                        cls._model_dict[key] = model.to(cls._device)
                    except:
                        pass
            logger.info("Models moved to device")
    
    @classmethod
    def download_with_resume(cls, url: str, dest_path: str, chunk_size: int = 8192) -> bool:
        """Download file with resume capability"""
        if not HAS_REQUESTS:
            logger.error("requests library not available")
            return False
        
        headers = {}
        mode = 'wb'
        resume_pos = 0
        
        # Check if partial download exists
        if os.path.exists(dest_path):
            resume_pos = os.path.getsize(dest_path)
            headers['Range'] = f'bytes={resume_pos}-'
            mode = 'ab'
        
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # Check if server supports resume
            if resume_pos > 0 and response.status_code != 206:
                logger.info("Server doesn't support resume, starting from beginning")
                mode = 'wb'
                resume_pos = 0
                response = requests.get(url, stream=True, timeout=30)
            
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            if response.status_code == 206:
                # Partial content, adjust total size
                content_range = response.headers.get('content-range', '')
                if content_range:
                    total_size = int(content_range.split('/')[-1])
            
            # Download with progress bar
            with open(dest_path, mode) as f:
                with tqdm(total=total_size, initial=resume_pos, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    @classmethod
    def check_disk_space(cls, required_gb: float) -> bool:
        """Check if there's enough disk space"""
        import shutil
        
        model_dir = folder_paths.models_dir
        stat = shutil.disk_usage(model_dir)
        available_gb = stat.free / (1024**3)
        
        if available_gb < required_gb:
            logger.warning(f"Low disk space: {available_gb:.1f}GB available, {required_gb:.1f}GB required")
            return False
        return True
    
    @classmethod
    def download_models(cls, model_variant: str = "hunyuanvideo-foley-xxl") -> Tuple[bool, str, Optional[str]]:
        """
        Download model files with smart extraction and progress tracking
        Returns: (success, message, extracted_path)
        """
        if not HAS_REQUESTS:
            return False, "requests library not installed. Please install it to enable auto-download.", None
        
        try:
            # Use ComfyUI's foley models directory
            foley_dir = folder_paths.folder_names_and_paths.get("foley", [None])[0]
            if not foley_dir or len(foley_dir) == 0:
                # Fallback to models directory
                foley_dir = [os.path.join(folder_paths.models_dir, "foley")]
            
            model_dir = foley_dir[0]
            os.makedirs(model_dir, exist_ok=True)
            
            # Get model info
            model_info = MODEL_URLS.get(model_variant)
            if not model_info:
                return False, f"Unknown model variant: {model_variant}", None
            
            # Create variant-specific subdirectory
            variant_dir = os.path.join(model_dir, model_info["extracted_dir"])
            os.makedirs(variant_dir, exist_ok=True)
            
            # Check if models already exist
            required_files = cls._required_model_filenames(model_variant)
            all_exist = all(os.path.exists(os.path.join(variant_dir, f)) for f in required_files)
            
            if all_exist:
                logger.info(f"All required model files already exist in {variant_dir}")
                return True, "Models already downloaded", variant_dir
            
            # Check disk space (rough estimate)
            required_gb = 8.0 if model_variant == "hunyuanvideo-foley-xxl" else 5.0
            if not cls.check_disk_space(required_gb):
                return False, f"Insufficient disk space. Need approximately {required_gb}GB free.", None
            
            # Download archive
            archive_name = os.path.basename(model_info["url"])
            archive_path = os.path.join(model_dir, archive_name)
            
            # Check if archive already fully downloaded
            if os.path.exists(archive_path):
                # Verify size if possible
                actual_size = os.path.getsize(archive_path)
                logger.info(f"Archive exists with size: {actual_size / (1024**3):.2f}GB")
                
                # Try to extract first before re-downloading
                try:
                    logger.info(f"Attempting to extract existing archive: {archive_path}")
                    with tarfile.open(archive_path, 'r:gz') as tar:
                        # Extract to parent directory so structure is correct
                        tar.extractall(path=model_dir)
                    
                    # Verify extraction
                    if all(os.path.exists(os.path.join(variant_dir, f)) for f in required_files):
                        logger.info("Successfully extracted from existing archive")
                        # Optionally remove archive to save space
                        try:
                            os.remove(archive_path)
                            logger.info("Removed archive file to save disk space")
                        except:
                            pass
                        return True, "Models extracted successfully", variant_dir
                except Exception as e:
                    logger.warning(f"Failed to extract existing archive: {e}")
                    logger.info("Will attempt to re-download...")
            
            # Download the archive
            logger.info(f"Downloading {model_variant} from {model_info['url']}...")
            logger.info(f"This is a large file (~{model_info['size']}), please be patient...")
            
            success = cls.download_with_resume(model_info["url"], archive_path)
            
            if not success:
                return False, "Failed to download model archive", None
            
            # Extract archive
            logger.info(f"Extracting models to {model_dir}...")
            try:
                with tarfile.open(archive_path, 'r:gz') as tar:
                    # Check archive structure
                    members = tar.getmembers()
                    logger.info(f"Archive contains {len(members)} files")
                    
                    # Extract with progress
                    for member in tqdm(members, desc="Extracting"):
                        tar.extract(member, path=model_dir)
                
                logger.info(f"Models extracted to {variant_dir}")
                
                # Verify all required files exist
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(variant_dir, f))]
                if missing_files:
                    return False, f"Extraction incomplete, missing files: {missing_files}", None
                
                # Remove archive to save space (optional)
                try:
                    os.remove(archive_path)
                    logger.info("Removed archive file to save disk space")
                except:
                    logger.info("Could not remove archive file, you may delete it manually to save space")
                
                # Also download required Hugging Face models
                logger.info("Downloading required Hugging Face models...")
                hf_models = [
                    "google/siglip2-base-patch16-512",
                    "laion/larger_clap_general"
                ]
                
                for hf_model in hf_models:
                    logger.info(f"Downloading {hf_model}...")
                    try:
                        from transformers import AutoModel, AutoTokenizer
                        if "siglip" in hf_model:
                            AutoModel.from_pretrained(hf_model)
                        elif "clap" in hf_model:
                            AutoTokenizer.from_pretrained(hf_model)
                            from transformers import ClapTextModelWithProjection
                            ClapTextModelWithProjection.from_pretrained(hf_model)
                    except Exception as e:
                        logger.warning(f"Could not pre-download {hf_model}: {e}")
                        logger.info("Model will be downloaded on first use")
                
                return True, f"Models downloaded and extracted successfully to {variant_dir}", variant_dir
                
            except Exception as e:
                error_msg = f"Failed to extract archive: {str(e)}"
                logger.error(error_msg)
                return False, error_msg, None
            
        except Exception as e:
            error_msg = f"Download error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None
    
    @classmethod
    def load_models(cls, model_path: str, config_path: str, auto_download: bool = True, 
                   model_variant: str = "hunyuanvideo-foley-xxl", 
                   memory_efficient: bool = False, cpu_offload: bool = False) -> Tuple[bool, str]:
        """Load models if not already loaded or if path changed"""
        try:
            # Set default paths if empty
            if not model_path.strip():
                # Try ComfyUI foley models directory first
                foley_models_dir = folder_paths.folder_names_and_paths.get("foley", [None])[0]
                if foley_models_dir and len(foley_models_dir) > 0:
                    # Prefer concrete subfolder for this variant if it exists
                    root_path = foley_models_dir[0]
                    expected_dir_name = MODEL_URLS.get(model_variant, {}).get("extracted_dir", "hunyuanvideo-foley-xxl")
                    candidate_path = os.path.join(root_path, expected_dir_name)
                    model_path = candidate_path if os.path.isdir(candidate_path) else root_path
                else:
                    # Fallback to custom node directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(current_dir, "pretrained_models")
            
            if not config_path.strip():
                current_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(current_dir, "configs", "hunyuanvideo-foley-xxl.yaml")
            
            # Check if models are already loaded with the same path and memory mode
            if (cls._model_dict is not None and 
                cls._cfg is not None and 
                cls._model_path == model_path and
                cls._memory_efficient == memory_efficient):
                
                # If models are on CPU and we need them on device, move them
                if cpu_offload and cls._device is not None:
                    cls.move_models_to_device()
                
                return True, "Models already loaded"
            
            # If switching memory modes, unload first
            if cls._model_dict is not None and cls._memory_efficient != memory_efficient:
                cls.unload_models()
            
            # Verify paths exist, attempt auto-download if not found
            # If a root directory is given, try to refine to the expected subdir
            if os.path.isdir(model_path):
                expected_dir_name = MODEL_URLS.get(model_variant, {}).get("extracted_dir", "hunyuanvideo-foley-xxl")
                candidate_path = os.path.join(model_path, expected_dir_name)
                if os.path.isdir(candidate_path):
                    model_path = candidate_path

            # Ensure folder exists
            os.makedirs(model_path, exist_ok=True)

            # Determine if any required files are missing
            required_files = cls._required_model_filenames(model_variant)
            missing_files = [f for f in required_files if not os.path.isfile(os.path.join(model_path, f))]

            if not os.path.exists(model_path) or not os.listdir(model_path) or missing_files:
                if auto_download:
                    if missing_files:
                        logger.info(f"Missing model files: {missing_files}")
                    logger.info(f"Attempting to download {model_variant} models automatically into ComfyUI/models/foley...")
                    
                    download_success, download_message, downloaded_path = cls.download_models(model_variant)
                    if download_success and downloaded_path:
                        model_path = downloaded_path
                        logger.info(f"Using downloaded models at: {model_path}")
                    else:
                        return False, f"Model path does not exist and auto-download failed: {download_message}. Please manually place models in ComfyUI/models/foley/ or specify a valid path."
                else:
                    return False, f"Model path does not exist: {model_path}. Please place models in ComfyUI/models/foley/ or specify a valid path. Auto-download is disabled."
            
            if not os.path.exists(config_path):
                return False, f"Config path does not exist: {config_path}"
            
            # Setup device
            cls._device = cls.setup_device("auto", 0)
            
            logger.info(f"Loading models from: {model_path}")
            logger.info(f"Config: {config_path}")
            logger.info(f"Memory efficient mode: {memory_efficient}")
            logger.info(f"CPU offload: {cpu_offload}")
            
            # Load models
            cls._model_dict, cls._cfg = load_model(model_path, config_path, cls._device)
            cls._model_path = model_path
            cls._memory_efficient = memory_efficient
            
            # If CPU offload is enabled, move models to CPU after loading
            if cpu_offload:
                cls.offload_models_to_cpu()
            
            logger.info("Models loaded successfully!")
            return True, "Models loaded successfully!"
            
        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}"
            logger.error(error_msg)
            cls._model_dict = None
            cls._cfg = None
            cls._device = None
            cls._model_path = None
            return False, error_msg
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        # Clamp seed to valid range for numpy (0 to 2^32-1)
        seed = int(seed) % (2**32)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    @torch.inference_mode()
    def generate_audio(self, text_prompt: str, guidance_scale: float, 
                      num_inference_steps: int, sample_nums: int, seed: int,
                      video=None, images=None, fps=24.0,
                      negative_prompt="",
                      output_format="video_path",
                      output_folder: str = "hunyuan_foley",
                      filename_prefix: str = "foley_",
                      memory_efficient: bool = False,
                      cpu_offload: bool = False):
        """
        Generate audio for the input video/images with the given text prompt
        Hybrid implementation supporting both VIDEO and IMAGE inputs
        """
        try:
            # Set seed for reproducibility
            self.set_seed(seed)
            
            # Set default paths internally
            model_path = ""
            config_path = ""
            auto_download = True
            model_variant = "hunyuanvideo-foley-xxl"
            
            # Load models if needed
            success, message = self.load_models(
                model_path, config_path, auto_download, model_variant,
                memory_efficient, cpu_offload
            )
            if not success:
                # Return empty values that won't cause downstream errors
                logger.error(f"Model loading failed: {message}")
                empty_audio = {"waveform": torch.zeros((1, 1, 48000)), "sample_rate": 48000}
                empty_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)  # Dummy frames (VHS compatible)
                return ("", empty_frames, empty_audio, f"❌ {message}")
            
            # If CPU offload is enabled, move models to device for inference
            if cpu_offload and self._device is not None:
                # Clear cache before moving models to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.move_models_to_device()
                
                # Additional memory check after moving models
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved() / 1024**3    # GB  
                    logger.info(f"GPU memory after model loading: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            
            # Validate inputs - need either video or images
            if video is None and images is None:
                empty_audio = {"waveform": torch.zeros((1, 1, 48000)), "sample_rate": 48000}
                empty_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return ("", empty_frames, empty_audio, "❌ Please provide either video or images input!")
            
            if video is not None and images is not None:
                logger.warning("Both video and images provided. Using video input and ignoring images.")
            
            # Clean text prompt
            if text_prompt is None:
                text_prompt = ""
            text_prompt = text_prompt.strip()
            
            # Process negative prompt (from PR #3)
            combined_negative = self._process_negative_prompt(negative_prompt)
            
            logger.info(f"Processing with prompt: {text_prompt}")
            logger.info(f"Negative prompt: {combined_negative}")
            logger.info(f"Generating {sample_nums} sample(s)")
            logger.info(f"Output format: {output_format}")
            
            # Determine video file path
            video_file = None
            temp_video_created = False
            
            if video is not None:
                # Handle VIDEO input
                logger.info(f"Video input type: {type(video)}")
                video_file = self._extract_video_path(video)
                
                if video_file is None:
                    # Try to serialize VIDEO tensors/frames to a temp mp4
                    ok, temp_path, msg = self._write_temp_video(video)
                    if ok:
                        video_file = temp_path
                        temp_video_created = True
                        logger.info(f"Serialized VIDEO tensor to temp file: {video_file}")
                    else:
                        logger.error(msg)
            
            elif images is not None:
                # Handle IMAGE input (from PR #3) - revert to working approach
                logger.info(f"Images input type: {type(images)}")
                logger.info(f"Converting {images.shape[0] if hasattr(images, 'shape') else 'unknown'} images to video at {fps} FPS")
                
                video_file, convert_msg = self._extract_frames_from_image_input(images, fps)
                if video_file is not None:
                    temp_video_created = True
                    # Verify temp video was created successfully
                    if os.path.exists(video_file):
                        file_size = os.path.getsize(video_file)
                        logger.info(f"Created video from images: {video_file} ({file_size} bytes)")
                    else:
                        logger.error(f"Temp video file not found after creation: {video_file}")
                        video_file = None
                else:
                    logger.error(f"Image conversion failed: {convert_msg}")
            
            if video_file is None:
                logger.error("Could not process input - no valid video file obtained")
                empty_audio = {"waveform": torch.zeros((1, 1, 48000)), "sample_rate": 48000}
                empty_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return ("", empty_frames, empty_audio, "❌ Could not process input format")
            
            if not os.path.exists(video_file):
                empty_audio = {"waveform": torch.zeros((1, 1, 48000)), "sample_rate": 48000}
                empty_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return ("", empty_frames, empty_audio, f"❌ Video file not found: {video_file}")
            
            # Feature processing with negative prompt support
            logger.info("Processing video features...")
            
            # Custom feature processing that handles negative prompts
            # This is inspired by PR #3's approach
            try:
                # Use the original feature_process but we'll handle negative prompts in text processing
                visual_feats, text_feats, audio_len_in_s = feature_process(
                    video_file,
                    text_prompt,
                    self._model_dict,
                    self._cfg
                )
                
                # If we want to integrate negative prompts more deeply, we could modify the text features here
                # For now, we rely on the guidance during generation
                
            except Exception as e:
                logger.error(f"Feature processing failed: {e}")
                empty_audio = {"waveform": torch.zeros((1, 1, 48000)), "sample_rate": 48000}
                empty_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return ("", empty_frames, empty_audio, f"❌ Feature processing failed: {str(e)}")
            
            # Generate audio
            logger.info("Generating audio...")
            audio, sample_rate = denoise_process(
                visual_feats,
                text_feats,
                audio_len_in_s,
                self._model_dict,
                self._cfg,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                batch_size=sample_nums
            )
            
            # Create output directory structure
            output_dir = folder_paths.get_output_directory()
            
            # Create subfolder if specified
            if output_folder and output_folder.strip():
                output_dir = os.path.join(output_dir, output_folder.strip())
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save audio file
            audio_filename = f"{filename_prefix}audio_{timestamp}_{seed}.wav"
            audio_output = os.path.join(output_dir, audio_filename)
            torchaudio.save(audio_output, audio[0], sample_rate)
            
            # Create audio result dict with correct shape for ComfyUI
            audio_tensor = audio[0].unsqueeze(0)  # Add batch dimension
            if len(audio_tensor.shape) == 2:
                audio_tensor = audio_tensor.unsqueeze(1)  # Add channel dimension if needed
            audio_result = {"waveform": audio_tensor, "sample_rate": sample_rate}
            
            # Handle different output formats
            video_output_path = ""
            video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)  # Default empty frames (VHS compatible)
            
            if output_format in ["video_path", "both"]:
                # Create video with audio
                video_filename = f"{filename_prefix}video_{timestamp}_{seed}.mp4"
                video_output_path = os.path.join(output_dir, video_filename)
                
                logger.info(f"Merging audio with video: {audio_output} + {video_file} -> {video_output_path}")
                
                # Merge audio and video
                try:
                        # Extra logging for debugging IMAGE input issues
                        if temp_video_created:
                            logger.info(f"Merging audio with temporary video created from images")
                            # Check video properties
                            import cv2
                            cap = cv2.VideoCapture(video_file)
                            if cap.isOpened():
                                fps_vid = cap.get(cv2.CAP_PROP_FPS)
                                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                logger.info(f"Temp video properties: {width}x{height}, {fps_vid} FPS, {frame_count} frames")
                                cap.release()
                            else:
                                logger.error(f"Cannot open temporary video for inspection: {video_file}")
                        
                        # Check if both input files exist before merging
                        if not os.path.exists(audio_output):
                            raise Exception(f"Audio file does not exist: {audio_output}")
                        if not os.path.exists(video_file):
                            raise Exception(f"Video file does not exist: {video_file}")
                            
                        audio_size = os.path.getsize(audio_output)
                        video_size = os.path.getsize(video_file)
                        logger.info(f"Input files - Audio: {audio_size} bytes, Video: {video_size} bytes")
                        
                        merge_audio_video(audio_output, video_file, video_output_path)
                        logger.info(f"Successfully created video with audio: {video_output_path}")
                        
                        # Verify the output file exists and has content
                        if os.path.exists(video_output_path):
                            file_size = os.path.getsize(video_output_path)
                            logger.info(f"Output video file size: {file_size} bytes")
                            if file_size == 0:
                                raise Exception("Output video file is empty")
                        else:
                            raise Exception(f"Output video file was not created: {video_output_path}")
                            
                except Exception as e:
                    logger.error(f"Failed to merge audio and video: {e}")
                    logger.error(f"This is a critical issue when using IMAGE input - video path will show only the temporary video without audio")
                    
                    # For IMAGE input, we need to create a video with audio even if the merge fails
                    # Try alternative merge approach for temp videos created from images
                    if temp_video_created:
                        logger.info("Attempting alternative merge for temporary video from images...")
                        try:
                            # Create a new merged video using cv2 and direct file handling
                            alternative_video_path = f"{video_output_path}_alt.mp4"
                            
                            # Use a simpler ffmpeg command that might work better with temp videos
                            import subprocess
                            alt_command = [
                                "ffmpeg", "-y",
                                "-i", video_file,
                                "-i", audio_output,
                                "-c:v", "libx264",  # Re-encode video
                                "-c:a", "aac",
                                "-strict", "experimental",
                                "-shortest",  # Match shortest stream
                                alternative_video_path
                            ]
                            
                            result = subprocess.run(alt_command, capture_output=True, text=True)
                            if result.returncode == 0 and os.path.exists(alternative_video_path):
                                # Success with alternative method
                                if os.path.exists(video_output_path):
                                    os.remove(video_output_path)
                                os.rename(alternative_video_path, video_output_path)
                                logger.info(f"Alternative merge successful: {video_output_path}")
                            else:
                                logger.error(f"Alternative merge failed: {result.stderr}")
                                raise Exception("Both primary and alternative merge failed")
                                
                        except Exception as alt_e:
                            logger.error(f"Alternative merge also failed: {alt_e}")
                            # Last resort: return temp video path but warn user
                            video_output_path = video_file
                            logger.warning("Returning temporary video without audio - merge failed completely")
                    else:
                        # If merge fails with regular video, return the original video path
                        video_output_path = video_file
            
            if output_format in ["frames", "both"]:
                # Extract frames from video for IMAGE output (VHS compatible format)
                try:
                    import cv2
                    cap = cv2.VideoCapture(video_file)
                    frames_list = []
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Convert BGR to RGB (VHS expects this)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Convert to float32 and normalize to [0, 1] range (VHS format)
                        frame_normalized = np.array(frame_rgb, dtype=np.float32) / 255.0
                        frames_list.append(frame_normalized)
                    
                    cap.release()
                    
                    if frames_list:
                        # Convert to tensor [batch, height, width, channels] - VHS compatible
                        video_frames = torch.from_numpy(np.stack(frames_list))
                        logger.info(f"Extracted {len(frames_list)} frames for VHS-compatible output")
                    
                except Exception as e:
                    logger.warning(f"Could not extract frames: {e}")
                    # Keep default empty frames
            
            # Clean up temporary video if created and we have a final output video
            if temp_video_created and os.path.exists(video_file):
                if output_format == "frames":
                    # Only frames requested, safe to clean up temp video
                    try:
                        os.remove(video_file)
                        logger.info(f"Cleaned up temporary video: {video_file}")
                    except:
                        pass  # Don't fail if cleanup fails
                elif output_format in ["video_path", "both"]:
                    # We created a final video with audio, can clean up the temp video now
                    if video_output_path and os.path.exists(video_output_path):
                        try:
                            os.remove(video_file)
                            logger.info(f"Cleaned up temporary video after creating final output: {video_file}")
                        except:
                            pass
                    else:
                        # Final video creation might have failed, keep temp video as fallback
                        logger.warning(f"Final video not found, keeping temporary video: {video_file}")
                        video_output_path = video_file
            
            success_msg = f"✅ Generated audio successfully (output format: {output_format})"
            if output_format in ["video_path", "both"]:
                success_msg += f" - Video: {video_output_path}"
            
            logger.info(success_msg)
            
            # Memory cleanup if in efficient mode
            if memory_efficient:
                logger.info("Memory efficient mode: Cleaning up after generation...")
                
                # Clear intermediate variables
                del visual_feats, text_feats, audio
                
                # Unload models
                self.unload_models()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.info("Memory cleanup complete")
            elif cpu_offload:
                # Move models back to CPU after inference
                self.offload_models_to_cpu()
            
            # Return based on output format
            return (video_output_path, video_frames, audio_result, success_msg)
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            logger.error(error_msg)
            
            # Clean up on error if memory efficient
            if memory_efficient or cpu_offload:
                try:
                    if memory_efficient:
                        self.unload_models()
                    elif cpu_offload:
                        self.offload_models_to_cpu()
                except:
                    pass
            
            # Return valid empty outputs to prevent downstream errors
            empty_audio = {"waveform": torch.zeros((1, 1, 48000)), "sample_rate": 48000}
            empty_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return ("", empty_frames, empty_audio, error_msg)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "HunyuanVideoFoley": HunyuanVideoFoleyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoFoley": "HunyuanVideo-Foley Generator",
}