# tests/test_face_mesh_topology.py
import torch

def test_get_face_triangles_shape():
    from utils.face_mesh_topology import get_face_triangles
    triangles = get_face_triangles()
    assert isinstance(triangles, torch.Tensor)
    assert triangles.dtype == torch.int32
    assert triangles.shape == (852, 3)

def test_get_face_triangles_valid_indices():
    from utils.face_mesh_topology import get_face_triangles
    triangles = get_face_triangles()
    assert triangles.min() >= 0
    assert triangles.max() <= 477  # 478 vertices, 0-indexed

def test_get_face_triangles_no_degenerate():
    from utils.face_mesh_topology import get_face_triangles
    triangles = get_face_triangles()
    # No triangle should have duplicate vertices
    for i in range(triangles.shape[0]):
        tri = triangles[i]
        assert tri[0] != tri[1] and tri[1] != tri[2] and tri[0] != tri[2], \
            f"Degenerate triangle at index {i}: {tri.tolist()}"
