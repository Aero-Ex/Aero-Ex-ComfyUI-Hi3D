import copy
import logging

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.storage
import comfy.supported_models_base
import comfy.utils

from .model import Hi3D
from .vae import Hi3DVae


class Hi3DLatent(comfy.latent_formats.LatentFormat):
    # sparse latent [B, 32, N, 1]: one column per voxel
    latent_channels = 32
    latents_mean = torch.tensor([
        -0.00155994, 0.03314649, -0.00307796, 0.00630751, -0.00835457, -0.03860179, 0.02528263, 0.0374062,
        -0.06209833, -0.00539356, -0.02110271, 0.05126001, 0.00753371, 0.02866821, 0.02772475, -0.02131735,
        -0.02799912, 0.01084617, 0.03433723, -0.009031, -0.00121899, -0.01944244, 0.01641873, -0.01063657,
        -0.01990618, 0.04856062, 0.01747232, -0.02387775, 0.03216169, -0.03732105, -0.03730354, -0.01948937
    ]).view(1, 32, 1, 1)
    latents_std = torch.tensor([
        0.61686146, 0.8011532, 0.854892, 0.96384716, 0.7700591, 0.6285985, 0.61541015, 0.7436831,
        0.6276261, 1.0184228, 0.77113867, 0.6262799, 0.68749195, 0.62698966, 0.7549124, 0.615774,
        0.6378066, 0.80406, 0.83846325, 0.6912039, 0.6922758, 0.79321617, 0.6347593, 0.638483,
        0.67194235, 0.7351938, 0.84781533, 0.9301291, 0.6269144, 0.6790608, 0.6519877, 0.7769204
    ]).view(1, 32, 1, 1)

    def process_in(self, latent):
        return (latent - self.latents_mean.to(latent)) / self.latents_std.to(latent)

    def process_out(self, latent):
        return latent * self.latents_std.to(latent) + self.latents_mean.to(latent)


class Hi3DModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device, unet_model=Hi3D)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        for k in ("hi3d_coords", "hi3d_coord_counts", "hi3d_resolution"):
            out[k] = comfy.conds.CONDConstant(kwargs[k])
        return out


class Hi3DConfig(comfy.supported_models_base.BASE):
    unet_config = {"image_model": "hi3d"}
    memory_usage_factor = 6
    latent_format = Hi3DLatent
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        return Hi3DModel(self, device=device)

    def clip_target(self, state_dict={}):
        return None


class Hi3DVAE(comfy.sd.VAE):
    # VAE.__init__ only detects formats from the state dict, so it is skipped and the fields decode uses are set here
    def __init__(self, sd, fast_disk):
        self.first_stage_model = Hi3DVae().eval()
        self.latent_channels = 32
        self.disable_offload = False
        self.size = None
        # ~80 surface voxels per token after three upsamples; ~600 values of features and extraction buffers per final voxel
        self.memory_used_decode = lambda shape, dtype: shape[2] * 80 * 600 * comfy.model_management.dtype_size(dtype)
        self.device = comfy.model_management.vae_device()
        self.vae_dtype = comfy.model_management.vae_dtype(self.device, [torch.float16, torch.float32])
        self.first_stage_model.to(self.vae_dtype)
        comfy.model_management.archive_model_dtypes(self.first_stage_model)
        self.output_device = comfy.model_management.intermediate_device()
        self.patcher = comfy.model_patcher.CoreModelPatcher(self.first_stage_model, load_device=self.device,
                                                            offload_device=comfy.model_management.vae_offload_device(), fast_disk=fast_disk)
        self.first_stage_model.current_patcher = self.patcher
        missing, _ = self.first_stage_model.load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())
        if not self.patcher.is_dynamic():
            self.first_stage_model.to(self.vae_dtype)
        if len(missing) > 0:
            logging.warning("Missing Hi3D VAE keys {}".format(missing))
        self.model_size()

    def clone(self):
        n = copy.copy(self)
        n.patcher = self.patcher.clone()
        n.first_stage_model.current_patcher = n.patcher
        return n


def load_checkpoint(path):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    prefix = "model.diffusion_model."
    if prefix + "transformer_512.t_embedder.mlp.0.weight" not in sd or "vae.dec.0.up.cls_conv.weight" not in sd:
        raise RuntimeError("Not a Hi3D checkpoint, pack the release with convert_hi3d.py.")
    fast_disk = comfy.storage.state_dict_fast_disk(sd)
    config = Hi3DConfig(dict(Hi3DConfig.unet_config))
    config.quant_config = comfy.utils.detect_layer_quantization(sd, prefix)
    load_device = comfy.model_management.get_torch_device()
    parameters = comfy.utils.calculate_parameters(sd, prefix)
    if config.quant_config is None:
        dtype = comfy.model_management.unet_dtype(model_params=parameters, supported_dtypes=config.supported_inference_dtypes, weight_dtype=comfy.utils.weight_dtype(sd, prefix))
        manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, load_device, config.supported_inference_dtypes)
    else:
        dtype = comfy.model_management.unet_dtype(model_params=parameters, supported_dtypes=config.supported_inference_dtypes)
        manual_cast_dtype = comfy.model_management.unet_manual_cast(None, load_device, config.supported_inference_dtypes)
    config.set_inference_dtype(dtype, manual_cast_dtype, device=load_device)

    model = config.get_model(sd, prefix, device=comfy.model_management.unet_inital_load_device(parameters, dtype))
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=comfy.model_management.unet_offload_device(), fast_disk=fast_disk)
    model.load_model_weights(sd, prefix, assign=patcher.is_dynamic())
    vae = Hi3DVAE(comfy.utils.state_dict_prefix_replace(sd, {"vae.": ""}, filter_keys=True), fast_disk)
    return patcher, vae
