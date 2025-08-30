"""Media utilities for audio/video processing."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger


class MediaProcessingError(Exception):
    """Exception raised for media processing errors."""
    pass


def merge_audio_video(
    audio_path: str, 
    video_path: str, 
    output_path: str,
    overwrite: bool = True,
    quality: str = "high"
) -> str:
    """
    Merge audio and video files using ffmpeg.
    
    Args:
        audio_path: Path to input audio file
        video_path: Path to input video file  
        output_path: Path for output video file
        overwrite: Whether to overwrite existing output file
        quality: Quality setting ('high', 'medium', 'low')
        
    Returns:
        Path to the output file
        
    Raises:
        MediaProcessingError: If input files don't exist or ffmpeg fails
        FileNotFoundError: If ffmpeg is not installed
    """
    # Validate input files
    if not os.path.exists(audio_path):
        raise MediaProcessingError(f"Audio file not found: {audio_path}")
    if not os.path.exists(video_path):
        raise MediaProcessingError(f"Video file not found: {video_path}")
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Quality settings
    quality_settings = {
        "high": ["-b:a", "192k"],
        "medium": ["-b:a", "128k"], 
        "low": ["-b:a", "96k"]
    }
    
    # Build ffmpeg command with more flexible stream handling
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-ac", "2", 
        "-shortest",  # Use shortest stream to avoid hanging
        *quality_settings.get(quality, quality_settings["high"]),
    ]
    
    if overwrite:
        ffmpeg_command.append("-y")
        
    ffmpeg_command.append(output_path)
    
    try:
        logger.info(f"Merging audio '{audio_path}' with video '{video_path}'")
        logger.info(f"FFmpeg command: {' '.join(ffmpeg_command)}")
        
        process = subprocess.Popen(
            ffmpeg_command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Primary merge failed, trying fallback method...")
            logger.error(f"FFmpeg stderr: {stderr}")
            
            # Try a more compatible fallback approach
            fallback_command = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "libx264",  # Re-encode video for compatibility
                "-c:a", "aac",
                "-b:a", "128k",
                "-preset", "fast",  # Faster encoding
                "-shortest",
                output_path
            ]
            
            logger.info(f"Fallback FFmpeg command: {' '.join(fallback_command)}")
            
            fallback_process = subprocess.Popen(
                fallback_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            fallback_stdout, fallback_stderr = fallback_process.communicate()
            
            if fallback_process.returncode != 0:
                error_msg = f"Both primary and fallback FFmpeg failed. Primary: {stderr}, Fallback: {fallback_stderr}"
                logger.error(error_msg)
                raise MediaProcessingError(error_msg)
            else:
                logger.info(f"Successfully merged video with fallback method: {output_path}")
        else:
            logger.info(f"Successfully merged video saved to: {output_path}")
            
    except FileNotFoundError:
        raise FileNotFoundError(
            "ffmpeg not found. Please install ffmpeg: "
            "https://ffmpeg.org/download.html"
        )
    except Exception as e:
        raise MediaProcessingError(f"Unexpected error during media processing: {e}")

    return output_path
