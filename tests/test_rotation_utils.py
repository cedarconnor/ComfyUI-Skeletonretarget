# tests/test_rotation_utils.py
import torch
import math

def test_identity_rotation():
    from utils.rotation_utils import euler_to_rotation_matrix
    R = euler_to_rotation_matrix(0.0, 0.0, 0.0)
    assert R.shape == (3, 3)
    assert torch.allclose(R, torch.eye(3), atol=1e-6)

def test_90_degree_yaw():
    from utils.rotation_utils import euler_to_rotation_matrix
    R = euler_to_rotation_matrix(0.0, 90.0, 0.0)
    # Yaw 90° around Y: x→z, z→-x
    point = torch.tensor([1.0, 0.0, 0.0])
    rotated = R @ point
    expected = torch.tensor([0.0, 0.0, -1.0])
    assert torch.allclose(rotated, expected, atol=1e-5)

def test_rotation_matrix_is_orthogonal():
    from utils.rotation_utils import euler_to_rotation_matrix
    R = euler_to_rotation_matrix(25.0, -30.0, 15.0)
    # R^T @ R should be identity
    assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-5)
    # det(R) should be 1
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)

def test_compute_vertex_normals_flat_triangle():
    from utils.rotation_utils import compute_vertex_normals
    # Single triangle lying in XY plane
    vertices = torch.tensor([[[0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]]])  # (1, 3, 3)
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32)  # (1, 3)
    normals = compute_vertex_normals(vertices, triangles)
    assert normals.shape == (1, 3, 3)
    # All normals should point in +Z (or -Z depending on winding)
    for i in range(3):
        n = normals[0, i]
        assert abs(n[2].item()) > 0.99, f"Normal {i} not along Z: {n.tolist()}"
        assert abs(n[0].item()) < 0.01
        assert abs(n[1].item()) < 0.01
