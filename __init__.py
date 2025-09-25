import os
import sys
from loguru import logger

# Add the current directory to Python path to import hunyuanvideo_foley modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the individual nodes (with FP8 quantization and torch.compile support)
logger.info("Loading HunyuanVideo-Foley nodes with FP8 quantization and torch.compile support")
from .hyvideofoley_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Export the mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']