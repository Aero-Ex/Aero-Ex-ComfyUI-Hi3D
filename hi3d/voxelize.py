# Surface voxelization ported from SPARC3D (Zhihao Li).

import logging

import torch

DEFAULT_PAD = 3
MIN_RESOLUTION = 2 * DEFAULT_PAD + 1
_CHUNK = 1 << 20


def _axis_overlap(v0, v1, v2, axis):
    p0 = (v0 * axis).sum(-1)
    p1 = (v1 * axis).sum(-1)
    p2 = (v2 * axis).sum(-1)
    lower = torch.minimum(torch.minimum(p0, p1), p2)
    upper = torch.maximum(torch.maximum(p0, p1), p2)
    radius = 0.5 * axis.abs().sum(-1)
    return (lower <= radius) & (upper >= -radius)


def _triangle_hits(v0, v1, v2, x, y, z):
    center = torch.stack((x.float() + 0.5, y.float() + 0.5, z.float() + 0.5), dim=-1)
    a = v0 - center
    b = v1 - center
    c = v2 - center
    hit = (
        (torch.minimum(torch.minimum(a[:, 0], b[:, 0]), c[:, 0]) <= 0.5)
        & (torch.maximum(torch.maximum(a[:, 0], b[:, 0]), c[:, 0]) >= -0.5)
        & (torch.minimum(torch.minimum(a[:, 1], b[:, 1]), c[:, 1]) <= 0.5)
        & (torch.maximum(torch.maximum(a[:, 1], b[:, 1]), c[:, 1]) >= -0.5)
        & (torch.minimum(torch.minimum(a[:, 2], b[:, 2]), c[:, 2]) <= 0.5)
        & (torch.maximum(torch.maximum(a[:, 2], b[:, 2]), c[:, 2]) >= -0.5)
    )
    e0 = b - a
    e1 = c - b
    e2 = a - c
    zero = torch.zeros_like(a[:, 0])
    axes = (
        (zero, e0[:, 2], -e0[:, 1]), (-e0[:, 2], zero, e0[:, 0]), (e0[:, 1], -e0[:, 0], zero),
        (zero, e1[:, 2], -e1[:, 1]), (-e1[:, 2], zero, e1[:, 0]), (e1[:, 1], -e1[:, 0], zero),
        (zero, e2[:, 2], -e2[:, 1]), (-e2[:, 2], zero, e2[:, 0]), (e2[:, 1], -e2[:, 0], zero),
    )
    for ax, ay, az in axes:
        hit &= _axis_overlap(a, b, c, torch.stack((ax, ay, az), dim=-1))
    normal = torch.cross(e0, e1, dim=-1)
    offset = -(normal * a).sum(-1)
    radius = 0.5 * normal.abs().sum(-1)
    return hit & (offset.abs() <= radius)


def _candidate_pairs(vertices, faces, resolution):
    tri = vertices[faces]
    lo = torch.floor(tri.min(dim=1).values).long()
    hi = torch.floor(tri.max(dim=1).values).long()
    valid = ((hi >= 0).all(dim=1)) & ((lo <= resolution - 1).all(dim=1))
    lo = lo.clamp(0, resolution - 1)
    hi = hi.clamp(0, resolution - 1)
    sizes = hi - lo + 1
    counts = sizes.prod(dim=1)
    counts = torch.where(valid, counts, torch.zeros_like(counts))
    total = int(counts.sum().item())
    if total == 0:
        return None
    faces_per_pair = torch.repeat_interleave(torch.arange(faces.shape[0], device=faces.device), counts)
    starts = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(total, device=faces.device) - torch.repeat_interleave(starts, counts)
    sizes_p = sizes[faces_per_pair]
    nx = sizes_p[:, 0]
    ny = sizes_p[:, 1]
    z = rank // (nx * ny)
    rem = rank % (nx * ny)
    y = rem // nx
    x = rem % nx
    x = x + lo[faces_per_pair, 0]
    y = y + lo[faces_per_pair, 1]
    z = z + lo[faces_per_pair, 2]
    return faces_per_pair, x, y, z


def voxelize_surface(vertices, faces, resolution):
    """Return unique occupied voxel coordinates ``[N, 3]`` as int32; resolution must stay below 2048 for the key packing."""
    pairs = _candidate_pairs(vertices.float(), faces.long(), resolution)
    if pairs is None:
        return torch.zeros((0, 3), dtype=torch.int32, device=vertices.device)
    face_ids, x, y, z = pairs
    tri = vertices[faces[face_ids]]
    hits = []
    for start in range(0, face_ids.shape[0], _CHUNK):
        end = min(start + _CHUNK, face_ids.shape[0])
        hit = _triangle_hits(tri[start:end, 0], tri[start:end, 1], tri[start:end, 2],
                             x[start:end], y[start:end], z[start:end])
        if hit.any():
            keys = (z[start:end][hit].long() << 22) | (y[start:end][hit].long() << 11) | x[start:end][hit].long()
            hits.append(keys)
    if not hits:
        return torch.zeros((0, 3), dtype=torch.int32, device=vertices.device)
    keys = torch.unique(torch.cat(hits))
    coords = torch.stack((keys & 2047, (keys >> 11) & 2047, (keys >> 22) & 2047), dim=1)
    return coords.to(torch.int32)


def _normalize(vertices, resolution, pad=DEFAULT_PAD, eps=1e-6):
    lo = vertices.min(dim=0).values - eps
    hi = vertices.max(dim=0).values + eps
    center = (lo + hi) / 2
    span = (hi - lo).max()
    if span <= 0:
        raise ValueError("mesh must have non-zero extent")
    scale = (resolution - 2 * pad - 2 * eps) / span
    return (vertices - center) * scale + resolution / 2


def adaptive_voxelize(vertices, faces, resolution=64, target_num=80000):
    """Shrink the grid until the occupied set fits ``target_num``."""
    scale = 1.0
    while True:
        current = int(resolution * scale)
        if current < MIN_RESOLUTION:
            raise ValueError(f"target voxel budget {target_num} is unreachable for this mesh")
        normalized = _normalize(vertices.float(), current)
        coords = voxelize_surface(normalized, faces.long(), current)
        if coords.shape[0] <= target_num:
            break
        scale *= 0.95
    if current != resolution:
        logging.info("Hi3D voxelization settled at resolution %d to fit %d voxels", current, target_num)
    return coords
