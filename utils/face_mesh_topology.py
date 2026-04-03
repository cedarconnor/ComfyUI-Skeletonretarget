# utils/face_mesh_topology.py
import torch

_TRIANGLES_CACHE = None

def get_face_triangles():
    """Extract 852 face mesh triangles from MediaPipe tesselation.

    Returns: (852, 3) int32 tensor of vertex indices.
    """
    global _TRIANGLES_CACHE
    if _TRIANGLES_CACHE is not None:
        return _TRIANGLES_CACHE

    from mediapipe.tasks.python.vision import FaceLandmarksConnections
    tess = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

    triangles = []
    for i in range(0, len(tess), 3):
        verts = {tess[i].start, tess[i].end,
                 tess[i + 1].start, tess[i + 1].end,
                 tess[i + 2].start, tess[i + 2].end}
        triangles.append(sorted(verts))

    _TRIANGLES_CACHE = torch.tensor(triangles, dtype=torch.int32)
    return _TRIANGLES_CACHE
