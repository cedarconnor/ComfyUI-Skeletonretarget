# nodes/face_landmarker.py
import os
import urllib.request
import numpy as np
import torch

try:
    from comfy.utils import ProgressBar
except ImportError:
    class ProgressBar:
        def __init__(self, total): pass
        def update(self, n): pass

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


def _ensure_model():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[FaceMeshRetarget] Downloading face_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[FaceMeshRetarget] Downloaded to {MODEL_PATH}")
    return MODEL_PATH


class MediaPipeFaceLandmarker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "output_blendshapes": ("BOOLEAN", {"default": False}),
                "output_transform_matrix": ("BOOLEAN", {"default": True}),
                "max_faces": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("FACE_LANDMARKS", "FACE_BLENDSHAPES", "FACE_TRANSFORM", "FACE_UVS")
    RETURN_NAMES = ("landmarks", "blendshapes", "face_matrix", "landmark_uvs")
    FUNCTION = "detect"
    CATEGORY = "FaceMeshRetarget"

    def detect(self, images, output_blendshapes=False, output_transform_matrix=True,
               max_faces=1):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        B, H, W, C = images.shape
        model_path = _ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=output_transform_matrix,
            num_faces=max_faces,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        all_landmarks = torch.zeros(B, 478, 3, dtype=torch.float32)
        all_blendshapes = torch.zeros(B, 52, dtype=torch.float32)
        all_matrices = torch.zeros(B, 4, 4, dtype=torch.float32)
        # Default matrix to identity so downstream math doesn't break
        for i in range(B):
            all_matrices[i] = torch.eye(4)
        all_uvs = torch.zeros(B, 478, 2, dtype=torch.float32)

        pbar = ProgressBar(B)
        for i in range(B):
            frame_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
            result = landmarker.detect(mp_image)

            if result.face_landmarks and len(result.face_landmarks) > 0:
                face_lms = result.face_landmarks[0]  # First face only
                for j, lm in enumerate(face_lms):
                    all_landmarks[i, j, 0] = lm.x
                    all_landmarks[i, j, 1] = lm.y
                    all_landmarks[i, j, 2] = lm.z
                    all_uvs[i, j, 0] = lm.x
                    all_uvs[i, j, 1] = lm.y

                if output_blendshapes and result.face_blendshapes:
                    bs = result.face_blendshapes[0]
                    for j, b in enumerate(bs[:52]):
                        all_blendshapes[i, j] = b.score

                if output_transform_matrix and result.facial_transformation_matrixes:
                    mat = result.facial_transformation_matrixes[0]
                    all_matrices[i] = torch.from_numpy(np.array(mat)).float()

            pbar.update(1)

        landmarker.close()
        return (all_landmarks, all_blendshapes, all_matrices, all_uvs)
