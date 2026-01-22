from .nodes.extract import ExtractSkeletonFromPose
from .nodes.transform import ComputeRetargetTransform, ApplyRetargetTransform
from .nodes.visualization import SkeletonToOpenPoseImage

NODE_CLASS_MAPPINGS = {
    "ExtractSkeletonFromPose": ExtractSkeletonFromPose,
    "ComputeRetargetTransform": ComputeRetargetTransform,
    "ApplyRetargetTransform": ApplyRetargetTransform,
    "SkeletonToOpenPoseImage": SkeletonToOpenPoseImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractSkeletonFromPose": "Extract Skeleton From Pose",
    "ComputeRetargetTransform": "Compute Retarget Transform",
    "ApplyRetargetTransform": "Apply Retarget Transform",
    "SkeletonToOpenPoseImage": "Skeleton To OpenPose Image"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
