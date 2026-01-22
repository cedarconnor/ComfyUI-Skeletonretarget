import torch
import numpy as np
import logging

try:
    from ..utils.geometry import track_person_across_frames, select_largest_person
except ImportError:
    from utils.geometry import track_person_across_frames, select_largest_person

logger = logging.getLogger(__name__)

class ExtractSkeletonFromPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSE_KEYPOINT",),
                "format": (["COCO-18", "COCO-133", "BODY-25"], {"default": "COCO-18"}),
                "person_selection": (["largest", "index", "track"], {"default": "largest"}),
            },
            "optional": {
                "person_index": ("INT", {"default": 0, "min": 0, "max": 10}),
                "tracking_threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05}),
                "min_confidence": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "image_for_dimensions": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("SKELETON", "SKELETON_SEQ", "MASK")
    RETURN_NAMES = ("skeleton", "skeleton_sequence", "valid_mask")
    FUNCTION = "extract"
    CATEGORY = "SkeletonRetarget/Extract"
    
    def extract(self, pose_data, format, person_selection, person_index=0, tracking_threshold=0.3, min_confidence=0.1, image_for_dimensions=None):
        """
        Convert DWPose/OpenPose detector output to skeleton tensor format.
        """
        # pose_data structure validation and inspection
        # Typically pose_data is a list of frames, where each frame is a list of people or a dict
        # ComfyUI DWPose often returns: List[Dict{'people': List[Dict]}]
        
        frames = pose_data
        
        # Determine Keypoint Count based on format
        k_count = 18
        if format == "BODY-25":
            k_count = 25
        elif format == "COCO-133":
            k_count = 133
            
        all_skeletons = []
        valid_frames_mask = []
        
        # Pre-process: Convert all people in all frames to tensors [K, 3]
        processed_frames = [] # List[List[Tensor]]
        
        for frame_idx, frame_content in enumerate(frames):
            frame_people_tensors = []
            
            # Helper to get people list from frame
            people_list = []
            if isinstance(frame_content, dict) and 'people' in frame_content:
                people_list = frame_content['people']
            elif isinstance(frame_content, list):
                people_list = frame_content # Some nodes return list of people directly
            
            for person in people_list:
                keypoints_list = []
                if 'pose_keypoints_2d' in person:
                    keypoints_list = person['pose_keypoints_2d']
                elif 'keypoints' in person:
                    keypoints_list = person['keypoints']
                
                # Reshape to [K, 3]
                # Flat list: x, y, c, x, y, c...
                if len(keypoints_list) > 0:
                    try:
                        kp_np = np.array(keypoints_list).reshape(-1, 3)
                        
                        # Fix Keypoint Count mismatch if needed (common with 18 vs 17 vs 133)
                        # If we have more keypoints than expected, truncate. If less, pad.
                        if kp_np.shape[0] != k_count:
                             logger.warning(f"Expected {k_count} keypoints, got {kp_np.shape[0]}. Adjusting.")
                             if kp_np.shape[0] > k_count:
                                 kp_np = kp_np[:k_count, :]
                             else:
                                 # Pad with zeros
                                 padding = np.zeros((k_count - kp_np.shape[0], 3))
                                 kp_np = np.vstack([kp_np, padding])
                        
                        # Normalize coordinates if they are pixel values
                        # We need image dimensions. If not provided, try to guess or assume normalized?
                        # OpenPose JSON is typically pixel coordinates.
                        # Some Comfy nodes return normalized [0,1].
                        # Inspect first point value.
                        if kp_np[:, :2].max() > 1.1: # Likely pixel coordinates
                            # We need image dimensions to normalize.
                            if image_for_dimensions is not None:
                                _, H, W, _ = image_for_dimensions.shape
                                kp_np[:, 0] /= W
                                kp_np[:, 1] /= H
                            else:
                                logger.warning("Detected pixel coordinates but no image_for_dimensions provided. Coordinates may be incorrect.")
                        
                        frame_people_tensors.append(torch.from_numpy(kp_np).float())
                        
                    except Exception as e:
                        print(f"Error parsing person in frame {frame_idx}: {e}")
                        continue
            
            processed_frames.append(frame_people_tensors)

        # Apply Selection Logic
        selected_indices = []
        
        if person_selection == "track":
            selected_indices = track_person_across_frames(processed_frames, tracking_threshold, format, min_confidence)
        else:
            # Per-frame independent selection
            for frame_poses in processed_frames:
                if len(frame_poses) == 0:
                    selected_indices.append(-1)
                elif person_selection == "largest":
                    selected_indices.append(select_largest_person(frame_poses, format, min_confidence))
                else: # index
                    if person_index < len(frame_poses):
                        selected_indices.append(person_index)
                    else:
                        selected_indices.append(-1)
        
        # Build Results
        for frame_idx, person_idx in enumerate(selected_indices):
            if person_idx != -1:
                skeleton = processed_frames[frame_idx][person_idx]
                all_skeletons.append(skeleton)
                valid_frames_mask.append(True)
            else:
                # Add dummy skeleton (zeros)
                all_skeletons.append(torch.zeros((k_count, 3)))
                valid_frames_mask.append(False)
                
        # Stack
        if len(all_skeletons) > 0:
            skeleton_seq = torch.stack(all_skeletons) # [N, K, 3]
        else:
            # Handle empty input: return single dummy frame with matching mask
            skeleton_seq = torch.zeros((1, k_count, 3))
            valid_frames_mask = [False]
            
        valid_mask = torch.tensor(valid_frames_mask, dtype=torch.bool)
        
        # First valid skeleton for output
        first_valid_idx = -1
        for i, valid in enumerate(valid_frames_mask):
            if valid:
                first_valid_idx = i
                break
                
        first_skeleton = skeleton_seq[first_valid_idx] if first_valid_idx != -1 else torch.zeros((k_count, 3))
        
        return (first_skeleton, skeleton_seq, valid_mask)
