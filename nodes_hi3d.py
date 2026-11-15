import torch
import torch.nn.functional as F
from tqdm.auto import trange
from typing_extensions import override

import comfy.model_management
import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
from comfy_api.latest import ComfyExtension, IO, Types
from comfy_extras.mesh3d.postprocess.qem_decimate import _connected_labels
from comfy_extras.nodes_save_3d import get_mesh_batch_item, pack_variable_mesh_batch
from .hi3d.loader import load_checkpoint
from .hi3d.voxelize import adaptive_voxelize

# output grid -> (structure grid, voxel budget) of each refinement pass
PASSES = {"512": (64, 80_000), "1024": (128, 90_000), "1536": (192, 110_000)}
SIGMA_MIN = 0.002


def _clip_vision_input(clip_vision, image, size, method):
    pixels = comfy.utils.common_upscale(image.movedim(-1, 1), size, size, method, "disabled").to(clip_vision.load_device)
    mean = torch.tensor(clip_vision.image_mean, device=pixels.device).view(1, 3, 1, 1)
    std = torch.tensor(clip_vision.image_std, device=pixels.device).view(1, 3, 1, 1)
    return ((pixels - mean) / std).to(clip_vision.dtype)


class Hi3DCheckpointLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hi3DCheckpointLoader",
            display_name="Load Hi3D Checkpoint",
            category="loaders",
            inputs=[IO.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints"))],
            outputs=[IO.Model.Output(), IO.Vae.Output()],
        )

    @classmethod
    def execute(cls, ckpt_name) -> IO.NodeOutput:
        return IO.NodeOutput(*load_checkpoint(folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)))


class Hi3DConditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hi3DConditioning",
            display_name="Hi3D Conditioning",
            category="model/conditioning/hi3d",
            inputs=[
                IO.ClipVision.Input("dino_vision", tooltip="DINOv2-large."),
                IO.ClipVision.Input("clip_vision", tooltip="CLIP ViT-L/14."),
                IO.Image.Input("image", tooltip="Subject on white, cropped square with 1.2x padding (Crop Image to Mask, pad_factor 1.2, white background)."),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, dino_vision, clip_vision, image) -> IO.NodeOutput:
        image = image[..., :3]
        comfy.model_management.load_model_gpu(dino_vision.patcher)
        tokens = dino_vision.model(pixel_values=_clip_vision_input(dino_vision, image, 756, "bilinear"))[0]
        comfy.model_management.load_model_gpu(clip_vision.patcher)
        pooled = clip_vision.model.vision_model(pixel_values=_clip_vision_input(clip_vision, image, 224, "lanczos"))[2]
        tokens = F.normalize(tokens.float(), dim=-1)
        pooled = F.normalize(pooled.float(), dim=-1)
        cond = torch.cat((tokens, pooled[:, None].expand(-1, tokens.shape[1], -1)), dim=-1).to(comfy.model_management.intermediate_device())
        return IO.NodeOutput([[cond, {}]], [[torch.zeros_like(cond), {}]])


class Hi3DRefineStage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hi3DRefineStage",
            display_name="Hi3D Refine Stage",
            category="model/conditioning/hi3d",
            description="Voxelize a mesh as the sparse support of one Hi3D refinement pass. Chain passes 512 -> 1024 -> 1536, feeding each decoded mesh into the next.",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Mesh.Input("mesh", tooltip="Coarse mesh, or the decoded mesh of the previous pass."),
                IO.Combo.Input("resolution", options=list(PASSES), default="512", tooltip="Extraction grid of this pass."),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, mesh, resolution) -> IO.NodeOutput:
        structure, budget = PASSES[resolution]
        device = comfy.model_management.get_torch_device()
        coords, bounds = [], []
        for i in range(mesh.vertices.shape[0]):
            vertices, faces = get_mesh_batch_item(mesh, i)[:2]
            vertices = vertices.to(device, torch.float32)
            c = adaptive_voxelize(vertices, faces.to(device), structure, budget).cpu()
            coords.append(torch.cat((torch.full_like(c[:, :1], i), c), dim=1))
            bounds.append(torch.stack((vertices.min(dim=0).values, vertices.max(dim=0).values)).cpu())
        counts = [c.shape[0] for c in coords]
        coords = torch.cat(coords)
        extras = {"hi3d_coords": coords, "hi3d_coord_counts": counts, "hi3d_resolution": structure}
        positive = [[c, {**opts, **extras}] for c, opts in positive]
        negative = [[c, {**opts, **extras}] for c, opts in negative]
        latent = {"samples": torch.zeros(len(counts), 32, max(counts), 1), "type": "hi3d", "coords": coords,
                  "coord_counts": counts, "resolution": structure, "bounds": torch.stack(bounds)}
        return IO.NodeOutput(positive, negative, latent)


def hi3d_sigmas(steps, tokens):
    """Release schedule: rational time warp scaled by the token count, plus the sigmas each step evaluates at (sigma * (1 + gamma))."""
    warp = min(5.0, max(1.0, (19.0 * tokens - 20000.0) / (3.0 * tokens + 60000.0)))
    warped = comfy.model_sampling.time_snr_shift(warp, comfy.model_sampling.time_snr_shift(2.0, torch.linspace(1.0, 0.0, steps + 1)))
    sigmas = SIGMA_MIN + (1.0 - SIGMA_MIN) * warped
    gammas = torch.where(warped[:-1] > 0.985, 0.04, 0.002)
    return sigmas, sigmas[:-1] * (1.0 + gammas)


def sample_hi3d(model, x, sigmas, extra_args=None, callback=None, disable=None, sigma_hats=None):
    """Heun from each inflated sigma down to the next sigma, without adding churn noise."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigma_hats[i].to(x.device)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = (x - denoised) / sigma_hat
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt
        d_2 = (x_2 - model(x_2, sigmas[i + 1] * s_in, **extra_args)) / sigmas[i + 1]
        x = x + (d + d_2) * (dt / 2)
    return x


class Hi3DGuider(comfy.samplers.CFGGuider):
    """Guidance 4 at t >= 1, 10 t^2 on [0.7, 1) and none below 0.7."""
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        t = float(timestep.max())
        cfg = 4.0 if t >= 1.0 else 10.0 * t * t if t >= 0.7 else 1.0
        return comfy.samplers.sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), cfg, model_options=model_options, seed=seed)


class Hi3DSampler(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hi3DSampler",
            display_name="Hi3D Sampler",
            category="sampling/hi3d",
            description="Sample one Hi3D refinement pass with the release schedule, guidance and Heun integrator.",
            inputs=[
                IO.Model.Input("model"),
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Latent.Input("latent", tooltip="Latent from Hi3D Refine Stage."),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                IO.Int.Input("steps", default=25, min=1, max=1000),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, positive, negative, latent, seed, steps) -> IO.NodeOutput:
        latent_image = latent["samples"]
        sigmas, sigma_hats = hi3d_sigmas(steps, max(latent["coord_counts"]))
        guider = Hi3DGuider(model)
        guider.set_conds(positive, negative)
        sampler = comfy.samplers.KSAMPLER(sample_hi3d, extra_options={"sigma_hats": sigma_hats})
        callback = latent_preview.prepare_callback(model, steps)
        samples = guider.sample(comfy.sample.prepare_noise(latent_image, seed), latent_image, sampler, sigmas, callback=callback,
                                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        out = latent.copy()
        out["samples"] = samples.to(comfy.model_management.intermediate_device())
        return IO.NodeOutput(out)


def _filter_components(vertices, faces, ratio=0.005):
    """Drop edge-connected face components smaller than ratio of all faces, keeping the largest."""
    face_count = faces.shape[0]
    first, second = faces, faces[:, [1, 2, 0]]
    keys, order = torch.sort((torch.minimum(first, second).long() * vertices.shape[0] + torch.maximum(first, second)).flatten())
    shared = keys[1:] == keys[:-1]
    face_ids = order // 3
    labels = _connected_labels(torch.stack((face_ids[:-1][shared], face_ids[1:][shared]), dim=1), face_count)
    sizes = torch.bincount(labels, minlength=face_count)
    keep = sizes >= face_count * ratio
    keep[sizes.argmax()] = True
    faces = faces[keep[labels]]
    used, remap = torch.unique(faces.flatten(), return_inverse=True)
    return vertices[used], remap.view(-1, 3)


def _fit_bounds(vertices, bounds):
    low, high = vertices.min(dim=0).values, vertices.max(dim=0).values
    scale = (bounds[1] - bounds[0]).max() / (high - low).max()
    return (vertices - (low + high) / 2) * scale + (bounds[0] + bounds[1]) / 2


class VAEDecodeHi3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEDecodeHi3D",
            display_name="Hi3D VAE Decode",
            category="latent/3d",
            description="Decode a Hi3D latent to a mesh in the frame of the mesh its stage voxelized.",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
            ],
            outputs=[IO.Mesh.Output()],
        )

    @classmethod
    def execute(cls, samples, vae) -> IO.NodeOutput:
        latent = samples["samples"]
        vae.prepare_decode(latent.shape)
        decoder = vae.first_stage_model
        coords = samples["coords"]
        vertices_out, faces_out = [], []
        start = 0
        for i, count in enumerate(samples["coord_counts"]):
            feats = latent[i, :, :count, 0].transpose(0, 1).to(vae.device, vae.vae_dtype)
            vertices, faces = decoder.decode(feats, coords[start:start + count, 1:].to(vae.device), samples["resolution"])
            start += count
            vertices, faces = _filter_components(vertices, faces)
            vertices = _fit_bounds(vertices, samples["bounds"][i].to(vertices))
            vertices_out.append(vertices.float().cpu())
            faces_out.append(faces.int().cpu())
        if len(vertices_out) == 1:
            return IO.NodeOutput(Types.MESH(vertices=vertices_out[0][None], faces=faces_out[0][None]))
        return IO.NodeOutput(pack_variable_mesh_batch(vertices_out, faces_out))


class Hi3DExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [Hi3DCheckpointLoader, Hi3DConditioning, Hi3DRefineStage, Hi3DSampler, VAEDecodeHi3D]


async def comfy_entrypoint():
    return Hi3DExtension()
