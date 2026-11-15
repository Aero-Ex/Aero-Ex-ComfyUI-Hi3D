# ComfyUI-Hi3D

dependency-free [Hi3D v1.0](https://github.com/hi3d-official/Hi3Dv1.0) (SPARC3D) mesh refinement for ComfyUI: a coarse mesh and a reference image go in, a detailed mesh at 512, 1024 or 1536 comes out.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/Aero-Ex/Aero-Ex-ComfyUI-Hi3D
```

Restart ComfyUI. No extra packages are needed.

## Weights

Download from [Aero-Ex/ComfyUI-Hi3D](https://huggingface.co/Aero-Ex/ComfyUI-Hi3D):

| File | Folder |
| --- | --- |
| `hi3d_v1.safetensors` (fp16) or `hi3d_v1_int8.safetensors` (int8, faster, ~4.3 GB) | `models/checkpoints` |
| `dinov2-large.safetensors`, `clip-vit-large-patch14.safetensors` | `models/clip_vision` |

## Graph

```
Load Hi3D Checkpoint ─ MODEL, VAE
Load CLIP Vision x2 ─┐
Crop Image to Mask ──┴ Hi3D Conditioning ─ Hi3D Refine Stage (mesh, 512) ─ Hi3D Sampler ─ Hi3D VAE Decode ─ mesh
                                           └ Hi3D Refine Stage (decoded mesh, 1024) ─ ... ─ (1536) ─ Save GLB
```

`workflows/Hi3D Refine Direct.json` runs one pass at the resolution picked in Hi3D Refine Stage; `workflows/Hi3D Refine Chain.json` runs 512 → 1024 → 1536, feeding each decoded mesh into the next pass. Use Crop Image to Mask with pad_factor 1.2 and a white background for the image. Each pass voxelizes the mesh it is given, so the passes can start at any resolution and chain upwards; each decoded mesh is fitted to the bounds of its input mesh.

## Notes

- Sampling uses the release recipe: 25 Heun steps on the token-count-scaled schedule, guidance 4, then 10 t², then off below t = 0.7. It runs through ComfyUI's samplers and model patcher (dynamic VRAM, memory compiler).
- Measured on a 6 GB RTX 4050, 512 → 1024 → 1536: about 517 s with fp16 and 326 s with int8. The 1536 decode peaks around 5 GB.
- Hi3D is licensed under the Hi3D Open License Agreement Version 1.0, including its Acceptable Use Policy. Copyright © 2026 OCEANO HK HOLDING LIMITED.
