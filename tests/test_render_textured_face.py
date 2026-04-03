import torch

def test_input_types_exist():
    from nodes.render_textured_face import RenderTexturedFaceMesh
    types = RenderTexturedFaceMesh.INPUT_TYPES()
    assert "transformed_landmarks" in types["required"]
    assert "video_frames" in types["required"]

def test_return_types():
    from nodes.render_textured_face import RenderTexturedFaceMesh
    assert "IMAGE" in RenderTexturedFaceMesh.RETURN_TYPES
    assert "MASK" in RenderTexturedFaceMesh.RETURN_TYPES

def test_render_produces_correct_shapes():
    from nodes.render_textured_face import RenderTexturedFaceMesh
    node = RenderTexturedFaceMesh()
    B, H, W = 1, 128, 128
    landmarks = torch.rand(B, 478, 3) * 0.3 + 0.35
    normals = torch.zeros(B, 478, 3)
    normals[..., 2] = -1.0
    uvs = landmarks[..., :2].clone()
    frames = torch.rand(B, H, W, 3)

    rendered, face_mask, facing_mask = node.render(
        landmarks, normals, uvs, frames,
        output_width=W, output_height=H,
        facing_threshold=60.0, facing_feather=20.0,
    )
    assert rendered.shape == (B, H, W, 3)
    assert face_mask.shape == (B, H, W)
    assert facing_mask.shape == (B, H, W)

def test_facing_mask_values_in_range():
    from nodes.render_textured_face import RenderTexturedFaceMesh
    node = RenderTexturedFaceMesh()
    B, H, W = 1, 64, 64
    landmarks = torch.rand(B, 478, 3) * 0.4 + 0.3
    normals = torch.randn(B, 478, 3)
    normals = torch.nn.functional.normalize(normals, dim=-1)
    uvs = landmarks[..., :2].clone()
    frames = torch.rand(B, H, W, 3)

    _, _, facing_mask = node.render(
        landmarks, normals, uvs, frames,
        output_width=W, output_height=H,
    )
    assert facing_mask.min() >= 0.0
    assert facing_mask.max() <= 1.0
