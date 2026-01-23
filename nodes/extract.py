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
                # Initialize fully zeroed array for this person
                final_kp_np = np.zeros((k_count, 3))
                
                # 1. Body / Pose
                pose_kp = []
                if 'pose_keypoints_2d' in person:
                    pose_kp = person['pose_keypoints_2d']
                elif 'keypoints' in person:
                    pose_kp = person['keypoints']
                
                if len(pose_kp) > 0:
                    pose_np = np.array(pose_kp).reshape(-1, 3)
                    num_body = pose_np.shape[0]
                    
                    if format == "COCO-133":
                        # Special handling for COCO-133 construction
                        if num_body == 18:
                            # Remap OpenPose-18 to COCO-133 Body slots
                            # OP18: 0=Nose, 1=Neck, 2=RSho, 3=RElb, 4=RWri, 5=LSho, 6=LElb, 7=LWri, 8=RHip...
                            # C133: 0=Nose, 1=LEye, 2=REye, 3=LEar, 4=REar, 5=LSho, 6=RSho, 7=LElb, 8=RElb...
                            
                            # Map: Source Index -> Target Index
                            op18_to_c133 = {
                                0: 0,   # Nose -> Nose
                                # 1 (Neck) -> Skip
                                2: 6,   # RSho -> RSho
                                3: 8,   # RElb -> RElb
                                4: 10,  # RWri -> RWri
                                5: 5,   # LSho -> LSho
                                6: 7,   # LElb -> LElb
                                7: 9,   # LWri -> LWri
                                8: 12,  # RHip -> RHip
                                9: 14,  # RKnee -> RKnee
                                10: 16, # RAnk -> RAnk
                                11: 11, # LHip -> LHip
                                12: 13, # LKnee -> LKnee
                                13: 15, # LAnk -> LAnk
                                14: 2,  # REye -> REye
                                15: 1,  # LEye -> LEye
                                16: 4,  # REar -> REar
                                17: 3   # LEar -> LEar
                            }
                            
                            for src_i, tgt_i in op18_to_c133.items():
                                if src_i < num_body:
                                    final_kp_np[tgt_i] = pose_np[src_i]
                                    
                        elif num_body == 133:
                            # Already 133, assume correct mapping
                            final_kp_np = pose_np
                        else:
                            # Fallback: exact copy up to length (e.g. COCO-17)
                            limit = min(num_body, 133)
                            final_kp_np[:limit] = pose_np[:limit]
                            
                        # 2. Face (Only for COCO-133)
                        # keys: face_keypoints_2d
                        if 'face_keypoints_2d' in person:
                            face_kp = person['face_keypoints_2d']
                            if face_kp is not None and len(face_kp) > 0:
                                face_np = np.array(face_kp).reshape(-1, 3)
                                # COCO-133 Face starts at 23, length usually 68 (sometimes 70 in OP)
                                # DWPose 68 points usually map to 23-90
                                n_face = face_np.shape[0]
                                limit_face = min(n_face, 68)
                                final_kp_np[23:23+limit_face] = face_np[:limit_face]
                        
                        # 3. Hands
                        # keys: hand_left_keypoints_2d, hand_right_keypoints_2d
                        # Left Hand starts at 91 (21 points)
                        if 'hand_left_keypoints_2d' in person:
                            lhand_kp = person['hand_left_keypoints_2d']
                            if lhand_kp is not None and len(lhand_kp) > 0:
                                lh_np = np.array(lhand_kp).reshape(-1, 3)
                                final_kp_np[91:91+21] = lh_np[:21]

                        # Right Hand starts at 112 (21 points)
                        if 'hand_right_keypoints_2d' in person:
                            rhand_kp = person['hand_right_keypoints_2d']
                            if rhand_kp is not None and len(rhand_kp) > 0:
                                rh_np = np.array(rhand_kp).reshape(-1, 3)
                                final_kp_np[112:112+21] = rh_np[:21]
                                
                    else:
                        # Non-133 formats (legacy behavior)
                        # Fix Keypoint Count mismatch
                        if pose_np.shape[0] > k_count:
                             pose_np = pose_np[:k_count, :]
                        elif pose_np.shape[0] < k_count:
                             padding = np.zeros((k_count - pose_np.shape[0], 3))
                             pose_np = np.vstack([pose_np, padding])
                        final_kp_np = pose_np

                try:
                    # Normalize coordinates if they are pixel values
                    # Check first few points for value > 1.0 (assuming normalized is 0-1)
                    # Use final_kp_np
                    
                    # We need to detect if it was pixel coords. 
                    # If we built it from parts, we need to check the parts?
                    # final_kp_np might have mix if sources differed? Unlikely.
                    
                    # Heuristic: check max value in x,y columns
                    max_val = final_kp_np[:, :2].max()
                    
                    if max_val > 1.1: # Likely pixel coordinates
                        if image_for_dimensions is not None:
                            _, H, W, _ = image_for_dimensions.shape
                            final_kp_np[:, 0] /= W
                            final_kp_np[:, 1] /= H
                        else:
                            # Warn once?
                            pass
                    
                    frame_people_tensors.append(torch.from_numpy(final_kp_np).float())
                    
                except Exception as e:
                    print(f"Error processing person in frame {frame_idx}: {e}")
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
