import torch
from utils.rotation_utils import euler_to_rotation_matrix, compute_vertex_normals
from utils.face_mesh_topology import get_face_triangles


class ReorientFaceMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "landmarks": ("FACE_LANDMARKS",),
                "face_matrix": ("FACE_TRANSFORM",),
            },
            "optional": {
                "pitch_offset": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.5}),
                "yaw_offset": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.5}),
                "roll_offset": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.5}),
                "mode": (["additive", "absolute"], {"default": "additive"}),
            },
        }

    RETURN_TYPES = ("FACE_LANDMARKS", "FACE_NORMALS")
    RETURN_NAMES = ("transformed_landmarks", "vertex_normals")
    FUNCTION = "reorient"
    CATEGORY = "FaceMeshRetarget"

    def reorient(self, landmarks, face_matrix, pitch_offset=0.0, yaw_offset=0.0,
                 roll_offset=0.0, mode="additive"):
        B = landmarks.shape[0]
        triangles = get_face_triangles()
        R_offset = euler_to_rotation_matrix(pitch_offset, yaw_offset, roll_offset)

        V_3d = self._landmarks_to_metric_3d(landmarks, face_matrix)

        centroid = V_3d.mean(dim=1, keepdim=True)
        V_centered = V_3d - centroid

        if mode == "additive":
            R = R_offset.unsqueeze(0).expand(B, -1, -1)
            V_rot = torch.bmm(V_centered, R.transpose(-1, -2))
        else:
            R_orig = face_matrix[:, :3, :3]
            R_orig_inv = R_orig.transpose(-1, -2)
            V_canonical = torch.bmm(V_centered, R_orig_inv.transpose(-1, -2))
            R_target = R_offset.unsqueeze(0).expand(B, -1, -1)
            V_rot = torch.bmm(V_canonical, R_target.transpose(-1, -2))

        V_final_3d = V_rot + centroid

        normals = compute_vertex_normals(V_final_3d, triangles)

        # Isolated vertices (not in any triangle) get zero normals from
        # compute_vertex_normals. Fall back to forward-facing (0, 0, 1).
        zero_mask = (torch.norm(normals, dim=-1, keepdim=True) < 1e-6)
        fallback = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype)
        normals = torch.where(zero_mask.expand_as(normals), fallback, normals)

        result = self._metric_3d_to_normalized(V_final_3d, face_matrix)

        return (result, normals)

    @staticmethod
    def _landmarks_to_metric_3d(landmarks_norm, face_matrix):
        """Convert normalized landmarks to metric 3D for rotation."""
        pts = landmarks_norm.clone()
        face_depth = face_matrix[:, 2, 3].unsqueeze(-1)  # (B, 1)
        scale = face_depth.abs().clamp(min=0.1)  # (B, 1)
        pts[..., 0] = (pts[..., 0] - 0.5) * scale
        pts[..., 1] = (pts[..., 1] - 0.5) * scale
        pts[..., 2] = pts[..., 2] * scale + face_depth
        return pts

    @staticmethod
    def _metric_3d_to_normalized(pts_3d, face_matrix):
        """Project metric 3D back to normalized image-space."""
        result = pts_3d.clone()
        face_depth = face_matrix[:, 2, 3].unsqueeze(-1)  # (B, 1)
        scale = face_depth.abs().clamp(min=0.1)  # (B, 1)
        result[..., 0] = result[..., 0] / scale + 0.5
        result[..., 1] = result[..., 1] / scale + 0.5
        result[..., 2] = (result[..., 2] - face_depth) / scale
        return result
