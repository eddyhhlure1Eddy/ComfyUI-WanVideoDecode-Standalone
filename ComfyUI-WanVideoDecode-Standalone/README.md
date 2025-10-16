# ComfyUI WanVideo Decode Standalone

A standalone version of the WanVideo Decode node extracted from ComfyUI-WanVideoWrapper.

## Features

- Standalone WanVideo decode functionality
- VAE tiling support for memory efficiency
- Support for TAEHV decoder
- Configurable tile sizes and stride
- Multiple normalization options
- Fallback implementations when ComfyUI is not available

## Installation

1. Place this folder in your ComfyUI custom_nodes directory
2. Restart ComfyUI
3. The node will appear under "WanVideoWrapper/Standalone"

## Dependencies

- torch
- numpy
- ComfyUI (optional, fallback available)
- TAEHV (optional, for TAEHV decoder support)

## Usage

The node accepts:
- VAE model (WANVAE type)
- Latent samples
- Tiling configuration
- Normalization options

Outputs decoded video frames as images.

## Author

eddy
