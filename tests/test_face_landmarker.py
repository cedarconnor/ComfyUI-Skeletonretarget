# tests/test_face_landmarker.py
import torch
import numpy as np

def test_input_types_exist():
    from nodes.face_landmarker import MediaPipeFaceLandmarker
    types = MediaPipeFaceLandmarker.INPUT_TYPES()
    assert "images" in types["required"]

def test_return_types():
    from nodes.face_landmarker import MediaPipeFaceLandmarker
    assert "FACE_LANDMARKS" in MediaPipeFaceLandmarker.RETURN_TYPES
    assert "FACE_TRANSFORM" in MediaPipeFaceLandmarker.RETURN_TYPES
    assert "FACE_UVS" in MediaPipeFaceLandmarker.RETURN_TYPES

def test_detect_synthetic_face():
    """Integration test: run on a blank image. MediaPipe may not detect a face,
    but the node should not crash and should return correctly shaped zero tensors."""
    from nodes.face_landmarker import MediaPipeFaceLandmarker
    node = MediaPipeFaceLandmarker()
    # 1 frame, 256x256, white image
    images = torch.ones(1, 256, 256, 3, dtype=torch.float32)
    landmarks, blendshapes, face_matrix, uvs = node.detect(
        images, output_blendshapes=False, output_transform_matrix=True, max_faces=1
    )
    # Should return tensors of the right shape even if no face found (zeros)
    assert landmarks.shape == (1, 478, 3)
    assert face_matrix.shape == (1, 4, 4)
    assert uvs.shape == (1, 478, 2)
