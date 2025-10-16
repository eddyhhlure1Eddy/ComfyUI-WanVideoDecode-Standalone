import os
import torch
import torch.nn.functional as F
import numpy as np

try:
    from comfy import model_management as mm
    from comfy.utils import ProgressBar, common_upscale
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    print("ComfyUI not available, using fallback implementations")

if COMFY_AVAILABLE:
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offload_device = torch.device("cpu")

try:
    from ComfyUI_WanVideoWrapper_main.taehv import TAEHV
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ComfyUI-WanVideoWrapper-main'))
        from taehv import TAEHV
    except ImportError:
        TAEHV = None
        print("TAEHV not available")

def soft_empty_cache():
    if COMFY_AVAILABLE:
        mm.soft_empty_cache()
    else:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class WanVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": (
                        "Drastically reduces memory use but will introduce seams at tile stride boundaries. "
                        "The location and number of seams is dictated by the tile stride size. "
                        "The visibility of seams can be controlled by increasing the tile size. "
                        "Seams become less obvious at 1.5x stride and are barely noticeable at 2x stride size. "
                        "Which is to say if you use a stride width of 160, the seams are barely noticeable with a tile width of 320."
                    )}),
                    "tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile width in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile height in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride width in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride height in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    },
                    "optional": {
                        "normalization": (["default", "minmax"], {"advanced": True}),
                    }
                }

    @classmethod
    def VALIDATE_INPUTS(s, tile_x, tile_y, tile_stride_x, tile_stride_y):
        if tile_x <= tile_stride_x:
            return "Tile width must be larger than the tile stride width."
        if tile_y <= tile_stride_y:
            return "Tile height must be larger than the tile stride height."
        return True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper/Standalone"

    def decode(self, vae, samples, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, normalization="default"):
        soft_empty_cache()
        video = samples.get("video", None)
        if video is not None:
            video.clamp_(-1.0, 1.0)
            video.add_(1.0).div_(2.0)
            return video.cpu().float(),
        latents = samples["samples"]
        end_image = samples.get("end_image", None)
        has_ref = samples.get("has_ref", False)
        drop_last = samples.get("drop_last", False)
        is_looped = samples.get("looped", False)

        vae.to(device)

        latents = latents.to(device = device, dtype = vae.dtype)

        soft_empty_cache()

        if has_ref:
            latents = latents[:, :, 1:]
        if drop_last:
            latents = latents[:, :, :-1]

        if TAEHV and type(vae).__name__ == "TAEHV":      
            images = vae.decode_video(latents.permute(0, 2, 1, 3, 4))[0].permute(1, 0, 2, 3)
            images = torch.clamp(images, 0.0, 1.0)
            images = images.permute(1, 2, 3, 0).cpu().float()
            return (images,)
        else:
            if end_image is not None:
                enable_vae_tiling = False
            images = vae.decode(latents, device=device, end_=(end_image is not None), tiled=enable_vae_tiling, tile_size=(tile_x//8, tile_y//8), tile_stride=(tile_stride_x//8, tile_stride_y//8))[0]
            
        
        images = images.cpu().float()

        if normalization == "minmax":
            images.sub_(images.min()).div_(images.max() - images.min())
        else:  
            images.clamp_(-1.0, 1.0)
            images.add_(1.0).div_(2.0)
        
        if is_looped:
            temp_latents = torch.cat([latents[:, :, -3:]] + [latents[:, :, :2]], dim=2)
            temp_images = vae.decode(temp_latents, device=device, end_=(end_image is not None), tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))[0]
            temp_images = temp_images.cpu().float()
            temp_images = (temp_images - temp_images.min()) / (temp_images.max() - temp_images.min())
            images = torch.cat([temp_images[:, 9:].to(images), images[:, 5:]], dim=1)

        if end_image is not None: 
            images = images[:, 0:-1]

        
        vae.to(offload_device)
        soft_empty_cache()

        images.clamp_(0.0, 1.0)

        return (images.permute(1, 2, 3, 0),)

NODE_CLASS_MAPPINGS = {
    "WanVideoDecodeStandalone": WanVideoDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoDecodeStandalone": "WanVideo Decode (Standalone)",
}
