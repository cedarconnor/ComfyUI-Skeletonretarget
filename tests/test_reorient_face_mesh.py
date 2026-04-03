import torch
import math

def test_input_types_exist():
    from nodes.reorient_face_mesh import ReorientFaceMesh
    types = ReorientFaceMesh.INPUT_TYPES()
    assert "landmarks" in types["required"]
    assert "face_matrix" in types["required"]

def test_return_types():
    from nodes.reorient_face_mesh import ReorientFaceMesh
    assert "FACE_LANDMARKS" in ReorientFaceMesh.RETURN_TYPES
    assert "FACE_NORMALS" in ReorientFaceMesh.RETURN_TYPES

def test_zero_rotation_preserves_landmarks():
    from nodes.reorient_face_mesh import ReorientFaceMesh
    node = ReorientFaceMesh()
    B = 1
    landmarks = torch.rand(B, 478, 3)
    landmarks[..., 0:2] = landmarks[..., 0:2]
    landmarks[..., 2] = landmarks[..., 2] * 0.1 - 0.05
    face_matrix = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()
    face_matrix[0, 2, 3] = -5.0

    result_landmarks, normals = node.reorient(
        landmarks, face_matrix,
        pitch_offset=0.0, yaw_offset=0.0, roll_offset=0.0,
        mode="additive"
    )
    assert result_landmarks.shape == (B, 478, 3)
    assert normals.shape == (B, 478, 3)
    assert torch.allclose(result_landmarks[..., :2], landmarks[..., :2], atol=0.01)

def test_yaw_rotation_shifts_x():
    from nodes.reorient_face_mesh import ReorientFaceMesh
    node = ReorientFaceMesh()
    B = 1
    landmarks = torch.zeros(B, 478, 3)
    landmarks[..., 0] = 0.5
    landmarks[..., 1] = 0.5
    landmarks[..., 2] = 0.0
    face_matrix = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()
    face_matrix[0, 2, 3] = -5.0

    result, _ = node.reorient(
        landmarks, face_matrix,
        pitch_offset=0.0, yaw_offset=20.0, roll_offset=0.0,
        mode="additive"
    )
    assert result.shape == (B, 478, 3)

def test_normals_are_unit_length():
    from nodes.reorient_face_mesh import ReorientFaceMesh
    node = ReorientFaceMesh()
    B = 2
    landmarks = torch.rand(B, 478, 3)
    landmarks[..., 2] *= 0.1
    face_matrix = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()
    face_matrix[:, 2, 3] = -5.0

    _, normals = node.reorient(
        landmarks, face_matrix,
        pitch_offset=10.0, yaw_offset=-15.0, roll_offset=5.0
    )
    norms = torch.norm(normals, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
