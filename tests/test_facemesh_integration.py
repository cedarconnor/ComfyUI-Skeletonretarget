# tests/test_facemesh_integration.py
import torch

def test_full_pipeline_shapes():
    """End-to-end data flow: landmarker output shapes -> reorient -> render."""
    from nodes.reorient_face_mesh import ReorientFaceMesh
    from nodes.render_textured_face import RenderTexturedFaceMesh

    B, H, W = 2, 128, 128

    # Simulate MediaPipeFaceLandmarker output
    landmarks = torch.rand(B, 478, 3)
    landmarks[..., 0:2] = landmarks[..., 0:2] * 0.4 + 0.3  # center cluster
    landmarks[..., 2] *= 0.05  # small z
    face_matrix = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()
    face_matrix[:, 2, 3] = -5.0
    uvs = landmarks[..., :2].clone()
    frames = torch.rand(B, H, W, 3)

    # ReorientFaceMesh
    reorient = ReorientFaceMesh()
    transformed, normals = reorient.reorient(
        landmarks, face_matrix,
        pitch_offset=0.0, yaw_offset=20.0, roll_offset=0.0,
    )
    assert transformed.shape == (B, 478, 3)
    assert normals.shape == (B, 478, 3)

    # RenderTexturedFaceMesh
    renderer = RenderTexturedFaceMesh()
    rendered, face_mask, facing_mask = renderer.render(
        transformed, normals, uvs, frames,
        output_width=W, output_height=H,
    )
    assert rendered.shape == (B, H, W, 3)
    assert face_mask.shape == (B, H, W)
    assert facing_mask.shape == (B, H, W)
    assert rendered.min() >= 0.0
    assert rendered.max() <= 1.0
    assert facing_mask.min() >= 0.0
    assert facing_mask.max() <= 1.0

def test_zero_rotation_uvs_match_landmarks():
    """With zero rotation, rendered pixels should sample from near the original
    landmark positions (UV = landmark xy)."""
    from nodes.reorient_face_mesh import ReorientFaceMesh

    B, H, W = 1, 64, 64
    landmarks = torch.rand(B, 478, 3)
    landmarks[..., 0:2] = landmarks[..., 0:2] * 0.4 + 0.3
    landmarks[..., 2] *= 0.01
    face_matrix = torch.eye(4).unsqueeze(0).clone()
    face_matrix[0, 2, 3] = -5.0
    uvs = landmarks[..., :2].clone()

    reorient = ReorientFaceMesh()
    transformed, normals = reorient.reorient(landmarks, face_matrix)

    # Transformed landmarks should be very close to original with zero rotation
    diff = (transformed[..., :2] - landmarks[..., :2]).abs().mean()
    assert diff < 0.02, f"Zero rotation moved landmarks by {diff:.4f}"
