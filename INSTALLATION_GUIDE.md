# Installation Guide for ComfyUI HunyuanVideo-Foley Custom Node

## Overview

This custom node wraps the HunyuanVideo-Foley model for use in ComfyUI, enabling text-video-to-audio synthesis directly within ComfyUI workflows.

## Prerequisites

- ComfyUI installation
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended, can run on less with memory optimization)
- At least 16GB system RAM

## Step-by-Step Installation

### 1. Clone the Custom Node

Navigate to your ComfyUI `custom_nodes` directory and clone the repository:
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/if-ai/ComfyUI_HunyuanVideoFoley.git
cd ComfyUI_HunyuanVideoFoley
```

### 2. Install Dependencies

Run the included installation script. This will check for and install any missing Python packages.
```bash
python install.py
```

### 3. Model Handling (Automatic)

**No manual download is required.**

The first time you use a generator node, the necessary models will be automatically downloaded and placed in the correct directory: `ComfyUI/models/foley/`.

The script will create this directory for you if it doesn't exist.

### 4. Restart ComfyUI

After the installation is complete, restart ComfyUI to load the new custom nodes.

## Expected Directory Structure

The installer will create a `foley` directory inside your main ComfyUI `models` folder for storing the downloaded models. The custom node directory will look like this:

```
ComfyUI/
├── models/
│   └── foley/
│       └── hunyuanvideo-foley-xxl/
│           ├── hunyuanvideo_foley.pth
│           ├── vae_128d_48k.pth
│           └── synchformer_state_dict.pth
└── custom_nodes/
    └── ComfyUI_HunyuanVideoFoley/
        ├── __init__.py
        ├── nodes.py
        ├── install.py
        └── ... (other node files)
```

## Usage

### Nodes Available

1.  **HunyuanVideo-Foley Generator**: The main, simplified node for audio generation.
2.  **HunyuanVideo-Foley Generator (Advanced)**: An advanced version that can accept pre-loaded models from loader nodes for optimized workflows.
3.  **HunyuanVideo-Foley Model Loader (FP8)**: Loads the model with optional memory-saving FP8 quantization.
4.  **HunyuanVideo-Foley Dependencies**: Pre-loads model dependencies like text encoders.
5.  **HunyuanVideo-Foley Torch Compile**: Optimizes the model with `torch.compile` for faster inference on compatible GPUs.

## Performance & Memory Optimization

The model includes several features to manage VRAM usage, allowing it to run on a wider range of hardware.

-   **VRAM Usage**: While 8GB of VRAM is recommended for a smooth experience, you can run the model on GPUs with less memory by enabling the following options in the generator node:
    -   **`memory_efficient`**: This checkbox aggressively unloads models from VRAM after each generation. This is the most effective way to save VRAM.
    -   **`cpu_offload`**: This option keeps the models on the CPU and only moves them to the GPU when needed. It is slower but significantly reduces VRAM usage.

-   **Generation Time**: Audio generation can take time depending on video length, settings, and hardware. Use the `HunyuanVideo-Foley Torch Compile` node for a potential speedup on subsequent runs.

## Troubleshooting

### Common Issues

1.  **"Failed to import..." errors**:
    Ensure the installation script completed successfully. You can run it again to be sure:
    ```bash
    python install.py
    ```

2.  **Model download issues**:
    If the automatic download fails, check your internet connection and the ComfyUI console for error messages. You can also manually download the models from [HuggingFace](https://huggingface.co/tencent/HunyuanVideo-Foley) and place them in `ComfyUI/models/foley/hunyuanvideo-foley-xxl/`.

3.  **CUDA out of memory**:
    -   Enable the `memory_efficient` checkbox in the node.
    -   Enable `cpu_offload` if you still have issues (at the cost of speed).
    -   Reduce `sample_nums` to 1.
    -   Use shorter videos for testing.