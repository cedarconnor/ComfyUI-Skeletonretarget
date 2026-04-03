# tests/test_facing_ratio.py
import torch

def test_frontal_normals_full_confidence():
    from utils.facing_ratio import compute_facing_ratio
    B = 1
    V = 4
    # Normals pointing straight at camera (0, 0, -1)
    normals = torch.tensor([[[0, 0, -1.0]] * V])  # (1, 4, 3)
    confidence = compute_facing_ratio(normals, facing_threshold=60.0, facing_feather=20.0)
    assert confidence.shape == (B, V)
    assert torch.allclose(confidence, torch.ones(B, V), atol=1e-5)

def test_perpendicular_normals_zero_confidence():
    from utils.facing_ratio import compute_facing_ratio
    # Normals pointing sideways (1, 0, 0) — 90° from camera
    normals = torch.tensor([[[1.0, 0, 0]] * 4])  # (1, 4, 3)
    confidence = compute_facing_ratio(normals, facing_threshold=60.0, facing_feather=20.0)
    # 90° > 60° + 20° = 80°, so should be 0
    assert torch.all(confidence < 0.05)

def test_feather_zone():
    from utils.facing_ratio import compute_facing_ratio
    import math
    # Normal at 70° — inside the feather zone (60° to 80°)
    angle = math.radians(70.0)
    normal = torch.tensor([[[math.sin(angle), 0, -math.cos(angle)]]])  # (1, 1, 3)
    confidence = compute_facing_ratio(normal, facing_threshold=60.0, facing_feather=20.0)
    # Should be between 0 and 1 (in feather zone)
    assert 0.05 < confidence[0, 0].item() < 0.95

def test_batch_dimension():
    from utils.facing_ratio import compute_facing_ratio
    normals = torch.randn(5, 10, 3)
    normals = torch.nn.functional.normalize(normals, dim=-1)
    confidence = compute_facing_ratio(normals)
    assert confidence.shape == (5, 10)
    assert torch.all(confidence >= 0.0)
    assert torch.all(confidence <= 1.0)
