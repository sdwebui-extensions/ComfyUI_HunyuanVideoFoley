import torch
import os
from loguru import logger
from torchvision import transforms
from torchvision.transforms import v2
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, AutoModel, ClapTextModelWithProjection
from ..models.dac_vae.model.dac import DAC
from ..models.synchformer import Synchformer
from ..models.hifi_foley import HunyuanVideoFoley
from .config_utils import load_yaml, AttributeDict
from .schedulers import FlowMatchDiscreteScheduler
from tqdm import tqdm
import folder_paths

def load_state_dict(model, model_path):
    logger.info(f"Loading model state dict from: {model_path}")
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys in state dict ({len(missing_keys)} keys):")
        for key in missing_keys:
            logger.warning(f"  - {key}")
    else:
        logger.info("No missing keys found")
    
    if unexpected_keys:
        logger.warning(f"Unexpected keys in state dict ({len(unexpected_keys)} keys):")
        for key in unexpected_keys:
            logger.warning(f"  - {key}")
    else:
        logger.info("No unexpected keys found")
    
    logger.info("Model state dict loaded successfully")
    return model

def load_model(model_path, config_path, device):
    logger.info("Starting model loading process...")
    logger.info(f"Configuration file: {config_path}")
    logger.info(f"Model weights dir: {model_path}")
    logger.info(f"Target device: {device}")
    
    cfg = load_yaml(config_path)
    logger.info("Configuration loaded successfully")
    
    # HunyuanVideoFoley
    logger.info("Loading HunyuanVideoFoley main model...")
    foley_model = HunyuanVideoFoley(cfg, dtype=torch.bfloat16, device=device).to(device=device, dtype=torch.bfloat16)
    foley_model = load_state_dict(foley_model, os.path.join(model_path, "hunyuanvideo_foley.pth"))
    foley_model.eval()
    logger.info("HunyuanVideoFoley model loaded and set to evaluation mode")

    # DAC-VAE
    dac_path = os.path.join(model_path, "vae_128d_48k.pth")
    logger.info(f"Loading DAC VAE model from: {dac_path}")
    try:
        # Try loading with the standard DAC.load method
        dac_model = DAC.load(dac_path)
    except TypeError as e:
        if "map_location" in str(e):
            # Handle the map_location conflict by manually loading the state dict
            logger.warning(f"DAC.load() failed with map_location conflict: {e}")
            logger.info("Attempting manual DAC model loading...")
            
            # Create DAC model instance with appropriate parameters for vae_128d_48k
            # Based on filename, this appears to be 128-dimensional latent space, 48kHz sample rate
            dac_model = DAC(
                encoder_dim=64,
                latent_dim=128,  # 128d as indicated by filename
                decoder_dim=1536,
                sample_rate=48000,  # 48k as indicated by filename
                continuous=False
            )
            state_dict = torch.load(dac_path, map_location="cpu", weights_only=False)
            dac_model.load_state_dict(state_dict, strict=False)
        else:
            raise e
    
    dac_model = dac_model.to(device)
    dac_model.requires_grad_(False)
    dac_model.eval()
    logger.info("DAC VAE model loaded successfully")

    # Siglip2 visual-encoder
    logger.info("Loading SigLIP2 visual encoder...")
    siglip2_preprocess = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    
    # Try multiple approaches to load SigLIP2
    siglip2_model = None
    
    # Method 1: Try with standard transformers AutoModel
    try:
        siglip2_model = AutoModel.from_pretrained(os.path.join(folder_paths.cache_dir, "huggingface/google/siglip2-base-patch16-512"), trust_remote_code=True).to(device).eval()
        logger.info("SigLIP2 loaded using standard transformers")
    except Exception as e1:
        logger.warning(f"Standard transformers loading failed: {e1}")
        
        # Method 2: Try loading from local cache or downloaded weights
        try:
            from transformers import SiglipVisionModel
            siglip2_model = SiglipVisionModel.from_pretrained(os.path.join(folder_paths.cache_dir, "huggingface/google/siglip2-base-patch16-512")).to(device).eval()
            logger.info("SigLIP2 loaded using SiglipVisionModel (base variant)")
        except Exception as e2:
            logger.warning(f"SiglipVisionModel loading failed: {e2}")
            
            # Method 3: Try using a compatible CLIP model as fallback
            try:
                from transformers import CLIPVisionModel
                logger.warning("Falling back to CLIP vision model as SigLIP2 is not available")
                siglip2_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device).eval()
                logger.info("Using CLIP vision model as fallback")
            except Exception as e3:
                logger.error(f"All vision model loading attempts failed: {e3}")
                raise RuntimeError(
                    "Could not load SigLIP2 vision encoder. Please ensure you have a compatible "
                    "transformers version installed. You can try:\n"
                    "1. pip install transformers>=4.37.0\n"
                    "2. Or manually download the model weights"
                )
    
    logger.info("SigLIP2 model and preprocessing pipeline loaded successfully")

    # clap text-encoder
    logger.info("Loading CLAP text encoder...")
    clap_tokenizer = AutoTokenizer.from_pretrained(os.path.join(folder_paths.cache_dir, "huggingface/laion/larger_clap_general"))
    clap_model = ClapTextModelWithProjection.from_pretrained(os.path.join(folder_paths.cache_dir, "huggingface/laion/larger_clap_general")).to(device)
    logger.info("CLAP tokenizer and model loaded successfully")

    # syncformer
    syncformer_path = os.path.join(model_path, "synchformer_state_dict.pth")
    logger.info(f"Loading Synchformer model from: {syncformer_path}")
    syncformer_preprocess = v2.Compose(
        [
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    syncformer_model = Synchformer()
    syncformer_model.load_state_dict(torch.load(syncformer_path, weights_only=False, map_location="cpu"))
    syncformer_model = syncformer_model.to(device).eval()
    logger.info("Synchformer model and preprocessing pipeline loaded successfully")


    logger.info("Creating model dictionary with attribute access...")
    model_dict = AttributeDict({
        'foley_model': foley_model,
        'dac_model': dac_model,
        'siglip2_preprocess': siglip2_preprocess,
        'siglip2_model': siglip2_model,
        'clap_tokenizer': clap_tokenizer,
        'clap_model': clap_model,
        'syncformer_preprocess': syncformer_preprocess,
        'syncformer_model': syncformer_model,
        'device': device,
    })
    
    logger.info("All models loaded successfully!")
    logger.info("Available model components:")
    for key in model_dict.keys():
        logger.info(f"  - {key}")
    logger.info("Models can be accessed via attribute notation (e.g., models.foley_model)")

    return model_dict, cfg

def retrieve_timesteps(
    scheduler,
    num_inference_steps,
    device,
    **kwargs,
):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def prepare_latents(scheduler, batch_size, num_channels_latents, length, dtype, device):
    shape = (batch_size, num_channels_latents, int(length))
    latents = randn_tensor(shape, device=device, dtype=dtype)

    # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
    if hasattr(scheduler, "init_noise_sigma"):
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma

    return latents


@torch.no_grad()
def denoise_process(visual_feats, text_feats, audio_len_in_s, model_dict, cfg, guidance_scale=4.5, num_inference_steps=50, batch_size=1):

    target_dtype = model_dict.foley_model.dtype
    autocast_enabled = target_dtype != torch.float32
    device = model_dict.device

    scheduler = FlowMatchDiscreteScheduler(
        shift=cfg.diffusion_config.sample_flow_shift,
        reverse=cfg.diffusion_config.flow_reverse,
        solver=cfg.diffusion_config.flow_solver,
        use_flux_shift=cfg.diffusion_config.sample_use_flux_shift,
        flux_base_shift=cfg.diffusion_config.flux_base_shift,
        flux_max_shift=cfg.diffusion_config.flux_max_shift,
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
    )

    latents = prepare_latents(
        scheduler,
        batch_size=batch_size,
        num_channels_latents=cfg.model_config.model_kwargs.audio_vae_latent_dim,
        length=audio_len_in_s * cfg.model_config.model_kwargs.audio_frame_rate,
        dtype=target_dtype,
        device=device,
    )

    # Denoise loop
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising steps"):
        # noise latents
        latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_input = scheduler.scale_model_input(latent_input, t)

        t_expand = t.repeat(latent_input.shape[0])

        # siglip2 features
        siglip2_feat = visual_feats.siglip2_feat.repeat(batch_size, 1, 1)  # Repeat for batch_size
        uncond_siglip2_feat = model_dict.foley_model.get_empty_clip_sequence(
                bs=batch_size, len=siglip2_feat.shape[1]
        ).to(device)

        if guidance_scale is not None and guidance_scale > 1.0:
            siglip2_feat_input = torch.cat([uncond_siglip2_feat, siglip2_feat], dim=0)
        else:
            siglip2_feat_input = siglip2_feat

        # syncformer features
        syncformer_feat = visual_feats.syncformer_feat.repeat(batch_size, 1, 1)  # Repeat for batch_size
        uncond_syncformer_feat = model_dict.foley_model.get_empty_sync_sequence(
                bs=batch_size, len=syncformer_feat.shape[1]
        ).to(device)
        if guidance_scale is not None and guidance_scale > 1.0:
            syncformer_feat_input = torch.cat([uncond_syncformer_feat, syncformer_feat], dim=0)
        else:
            syncformer_feat_input = syncformer_feat

        # text features
        text_feat_repeated = text_feats.text_feat.repeat(batch_size, 1, 1)  # Repeat for batch_size
        uncond_text_feat_repeated = text_feats.uncond_text_feat.repeat(batch_size, 1, 1)  # Repeat for batch_size
        if guidance_scale is not None and guidance_scale > 1.0:
            text_feat_input = torch.cat([uncond_text_feat_repeated, text_feat_repeated], dim=0)
        else:
            text_feat_input = text_feat_repeated

        with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=target_dtype):
            # Predict the noise residual
            noise_pred = model_dict.foley_model(
                x=latent_input,
                t=t_expand,
                cond=text_feat_input,
                clip_feat=siglip2_feat_input,
                sync_feat=syncformer_feat_input,
                return_dict=True,
            )["x"]

        noise_pred = noise_pred.to(dtype=torch.float32)

        if guidance_scale is not None and guidance_scale > 1.0:
            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Post-process the latents to audio

    with torch.no_grad():
        audio = model_dict.dac_model.decode(latents)
        audio = audio.float().cpu()

    audio = audio[:, :int(audio_len_in_s*model_dict.dac_model.sample_rate)]

    return audio, model_dict.dac_model.sample_rate


