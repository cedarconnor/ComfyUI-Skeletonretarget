from .nodes.extract import ExtractSkeletonFromPose
from .nodes.transform import ComputeRetargetTransform, ApplyRetargetTransform
from .nodes.visualization import SkeletonToOpenPoseImage
from .nodes.face_landmarker import MediaPipeFaceLandmarker
from .nodes.reorient_face_mesh import ReorientFaceMesh
from .nodes.render_textured_face import RenderTexturedFaceMesh

NODE_CLASS_MAPPINGS = {
    "ExtractSkeletonFromPose": ExtractSkeletonFromPose,
    "ComputeRetargetTransform": ComputeRetargetTransform,
    "ApplyRetargetTransform": ApplyRetargetTransform,
    "SkeletonToOpenPoseImage": SkeletonToOpenPoseImage,
    "MediaPipeFaceLandmarker": MediaPipeFaceLandmarker,
    "ReorientFaceMesh": ReorientFaceMesh,
    "RenderTexturedFaceMesh": RenderTexturedFaceMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractSkeletonFromPose": "Extract Skeleton From Pose",
    "ComputeRetargetTransform": "Compute Retarget Transform",
    "ApplyRetargetTransform": "Apply Retarget Transform",
    "SkeletonToOpenPoseImage": "Skeleton To OpenPose Image",
    "MediaPipeFaceLandmarker": "MediaPipe Face Landmarker",
    "ReorientFaceMesh": "Reorient Face Mesh",
    "RenderTexturedFaceMesh": "Render Textured Face Mesh",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
