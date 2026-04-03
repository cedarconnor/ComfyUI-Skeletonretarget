# utils/rotation_utils.py
import torch
import math


def euler_to_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    """Pure-torch euler (YXZ order) to 3x3 rotation matrix.

    Args:
        pitch_deg: Rotation around X axis in degrees.
        yaw_deg: Rotation around Y axis in degrees.
        roll_deg: Rotation around Z axis in degrees.

    Returns: (3, 3) rotation matrix.
    """
    p = torch.tensor(math.radians(pitch_deg))
    y = torch.tensor(math.radians(yaw_deg))
    r = torch.tensor(math.radians(roll_deg))

    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    cr, sr = torch.cos(r), torch.sin(r)

    Rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    Rz = torch.tensor([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=torch.float32)

    return Ry @ Rx @ Rz


def compute_vertex_normals(vertices, triangles):
    """Compute area-weighted per-vertex normals.

    Args:
        vertices: (B, V, 3) vertex positions.
        triangles: (T, 3) int32 triangle indices.

    Returns: (B, V, 3) unit normals per vertex.
    """
    B, V, _ = vertices.shape
    T = triangles.shape[0]

    v0 = vertices[:, triangles[:, 0]]  # (B, T, 3)
    v1 = vertices[:, triangles[:, 1]]
    v2 = vertices[:, triangles[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)

    vertex_normals = torch.zeros_like(vertices)  # (B, V, 3)
    for i in range(3):
        idx = triangles[:, i].long().unsqueeze(0).unsqueeze(-1).expand(B, T, 3)
        vertex_normals.scatter_add_(1, idx, face_normals)

    return torch.nn.functional.normalize(vertex_normals, dim=-1)
