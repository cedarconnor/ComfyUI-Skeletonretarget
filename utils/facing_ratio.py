# utils/facing_ratio.py
import torch
import math


def compute_facing_ratio(vertex_normals, facing_threshold=60.0, facing_feather=20.0):
    """Per-vertex facing ratio relative to a frontal camera at (0, 0, -1).

    Measures how much each vertex normal faces the original capture camera.
    Returns 1.0 for frontal, 0.0 for grazing/turned-away surfaces.

    Args:
        vertex_normals: (B, V, 3) unit normals of the reoriented mesh.
        facing_threshold: Angle in degrees where falloff begins.
        facing_feather: Width of falloff zone in degrees.

    Returns: (B, V) confidence values in [0, 1].
    """
    # Camera looks along -Z; normals facing camera point in -Z direction.
    # cos(angle) = dot(normal, -view_dir) where view_dir = [0,0,1]
    # So cos(angle) = dot(normal, [0,0,-1]) = -normal_z
    cos_angle = -vertex_normals[..., 2]  # (B, V)

    angle_deg = torch.acos(torch.clamp(cos_angle, -1.0, 1.0)) * (180.0 / math.pi)

    # Linear ramp then smoothstep
    linear = 1.0 - torch.clamp(
        (angle_deg - facing_threshold) / max(facing_feather, 1e-6),
        0.0, 1.0,
    )
    # Smoothstep (cubic hermite)
    confidence = linear * linear * (3.0 - 2.0 * linear)

    return confidence
