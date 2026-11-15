"""Pack the Hi3D v1.0 release directory into one ComfyUI checkpoint.

python custom_nodes/ComfyUI-Hi3D/convert_hi3d.py /path/to/Hi3Dv1.0 models/checkpoints/hi3d_v1.safetensors
python custom_nodes/ComfyUI-Hi3D/convert_hi3d.py --int8 /path/to/Hi3Dv1.0 models/checkpoints/hi3d_v1_int8.safetensors
"""

import argparse
import json
import os
import re

import torch
from comfy_kitchen.tensor import QuantizedTensor
from safetensors import safe_open
from safetensors.torch import save_file

# transformer block linears, quantized to per-row INT8 with Hadamard rotation by --int8
INT8_LINEAR = re.compile(r"^blocks\.\d+\.(self_attn\.(to_qkv|to_out)|cross_attn\.(to_q|to_kv|to_out)|mlp\.[02])\.weight$")

VAE_RENAMES = [
    (r"^decoder\.z_to_f\.conv\.0\.", "z_to_f."),
    (r"^decoder\.mid\.block_in\.", "mid.0."),
    (r"^decoder\.mid\.block0\.", "mid.1."),
    (r"^decoder\.mid\.block1\.", "mid.2."),
    (r"^decoder\.mid\.block_out\.", "mid.3."),
    (r"^decoder\.dec\.dec3\.", "dec.0."),
    (r"^decoder\.dec\.dec2\.", "dec.1."),
    (r"^decoder\.dec\.dec1\.", "dec.2."),
    (r"^decoder\.dec\.dec0\.", "dec.3."),
    (r"\.block(\d)\.", r".blocks.\1."),
    (r"\.block\.0\.", ".norm1."),
    (r"\.block\.2\.", ".conv1."),
    (r"\.block\.3\.", ".norm2."),
    (r"\.block\.5\.", ".conv2."),
    (r"\.res_conv\.", ".skip."),
    (r"\.cpe\.0\.", ".cpe_conv."),
    (r"\.cpe\.1\.", ".cpe_linear."),
    (r"\.cpe\.2\.", ".cpe_norm."),
    (r"\.norm([12])\.0\.", r".norm\1."),
    (r"\.attn\.", "."),
    (r"\.mlp\.0\.", "."),
    (r"\.up\.upsample\.cls_head\.fc\.0\.", ".up.cls_norm."),
    (r"\.up\.upsample\.cls_head\.fc\.2\.", ".up.cls_conv."),
    (r"\.up\.upsample\.", ".up."),
    (r"^decoder\.conv_out\.conv\.0\.", "conv_out.norm."),
    (r"^decoder\.conv_out\.conv\.2\.", "conv_out.conv."),
    (r"\.weights$", ".weight"),
]


def convert_vae(path, out):
    with safe_open(path, "pt") as f:
        for key in f.keys():
            if key.endswith("inv_grid_coord"):
                continue
            name = key
            for pattern, replacement in VAE_RENAMES:
                name = re.sub(pattern, replacement, name)
            tensor = f.get_tensor(key)
            if tensor.ndim == 2 and tensor.shape[0] == 1:  # PointGroupNorm affine [1, C]
                tensor = tensor[0]
            out["vae." + name] = tensor.contiguous()


def convert_transformer(path, name, out, int8):
    prefix = "model.diffusion_model.{}.".format(name)
    with safe_open(path, "pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key.endswith(".modulation"):
                tensor = tensor.flatten()
            key = key.replace(".conv1.conv.", ".conv1.").replace(".conv2.conv.", ".conv2.").replace(".mlp.mlp.", ".mlp.")
            if int8 and INT8_LINEAR.match(key):
                groupsize = 256 if tensor.shape[1] % 256 == 0 else 64
                weight = QuantizedTensor.from_float(tensor.cuda(), "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=groupsize)
                out.update({k: v.cpu().contiguous() for k, v in weight.state_dict(prefix + key).items()})
                config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize}
                out[prefix + key[:-len("weight")] + "comfy_quant"] = torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)
                continue
            out[prefix + key] = tensor.contiguous()
    if int8:
        attention = torch.tensor(list(json.dumps({"attention": "comfy_kitchen_int8"}).encode("utf-8")), dtype=torch.uint8)
        for key in list(out):
            if key.startswith(prefix) and key.endswith((".self_attn.to_qkv.weight", ".cross_attn.to_q.weight")):
                out[key.rsplit(".", 2)[0] + ".comfy_attention.config"] = attention.clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Hi3D v1.0 release directory")
    parser.add_argument("output", help="output .safetensors checkpoint")
    parser.add_argument("--int8", action="store_true", help="store the transformer block linears as INT8 ConvRot and prefer comfy-kitchen INT8 attention (needs CUDA)")
    args = parser.parse_args()
    out = {}
    for name in ("transformer_512", "transformer_1024"):
        convert_transformer(os.path.join(args.source, name, "model.safetensors"), name, out, args.int8)
    convert_vae(os.path.join(args.source, "vae", "model.safetensors"), out)
    save_file(out, args.output)


if __name__ == "__main__":
    main()
