import math
import torch
from typing import List, Tuple, Optional
from .definitions import KEYPOINT_MAPPING, get_anchor_indices

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_torso_anchor(skeleton: torch.Tensor, format: str = "COCO-18") -> Tuple[float, float]:
    """
    Compute default anchor (approx center of torso) for a single skeleton.
    Useful for tracking.
    """
    indices = get_anchor_indices(format, "torso")
    # skeleton is [K, 3]
    
    sum_x, sum_y, count = 0.0, 0.0, 0
    for idx in indices:
        if idx is not None and skeleton[idx, 2] > 0.05: # Minimal confidence
            sum_x += skeleton[idx, 0]
            sum_y += skeleton[idx, 1]
            count += 1
            
    if count == 0:
        return (0.5, 0.5) # Fallback center
        
    return (sum_x / count, sum_y / count)

def select_largest_person(pose_list: List[torch.Tensor], format: str) -> int:
    """
    Select index of person with largest bounding box area.
    pose_list: List of [K, 3] tensors
    """
    best_idx = -1
    max_area = -1.0
    
    for i, pose in enumerate(pose_list):
        # Compute bbox
        valid = pose[:, 2] > 0.05
        if not valid.any():
            continue
            
        points = pose[valid, :2]
        min_x = points[:, 0].min()
        max_x = points[:, 0].max()
        min_y = points[:, 1].min()
        max_y = points[:, 1].max()
        
        area = (max_x - min_x) * (max_y - min_y)
        if area > max_area:
            max_area = area
            best_idx = i
            
    return best_idx if best_idx != -1 else 0

def track_person_across_frames(
    pose_sequences: List[List[torch.Tensor]], 
    tracking_threshold: float,
    format: str = "COCO-18"
) -> List[int]:
    """
    Track a single person across video frames using anchor-based matching.
    
    pose_sequences: List of lists, where each inner list contains skeletons found in that frame.
    Returns: List of indices (one per frame), -1 if lost.
    """
    selected = []
    prev_anchor: Optional[Tuple[float, float]] = None
    
    for frame_idx, frame_poses in enumerate(pose_sequences):
        if len(frame_poses) == 0:
            selected.append(-1)
            continue
            
        if prev_anchor is None:
            # Frame 0: pick largest
            best_idx = select_largest_person(frame_poses, format)
            if best_idx != -1:
                prev_anchor = compute_torso_anchor(frame_poses[best_idx], format)
                selected.append(best_idx)
            else:
                selected.append(-1)
        else:
            # Find closest match to previous anchor
            best_idx = -1
            best_dist = float('inf')
            
            for person_idx, person in enumerate(frame_poses):
                anchor = compute_torso_anchor(person, format)
                dist = euclidean_distance(anchor, prev_anchor)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = person_idx
            
            if best_dist > tracking_threshold:
                # Lost tracking
                selected.append(-1)
                # Don't update anchor if lost? Or keep searching?
                # We'll stick to last known good anchor if lost.
            else:
                selected.append(best_idx)
                # Smooth anchor update (momentum)
                new_anchor = compute_torso_anchor(frame_poses[best_idx], format)
                prev_anchor = (
                    0.7 * new_anchor[0] + 0.3 * prev_anchor[0],
                    0.7 * new_anchor[1] + 0.3 * prev_anchor[1]
                )
    
    return selected

def compute_rotation(skeleton: torch.Tensor, format: str, min_confidence: float) -> float:
    """
    Compute body orientation angle with fallback strategies.
    Returns angle in radians.
    """
    mapping = KEYPOINT_MAPPING[format]
    
    # Try shoulders first
    l_sh = mapping.get("left_shoulder")
    r_sh = mapping.get("right_shoulder")
    
    if l_sh is not None and r_sh is not None:
        # Check confidence
        if skeleton[l_sh, 2] >= min_confidence and skeleton[r_sh, 2] >= min_confidence:
            dx = skeleton[r_sh, 0] - skeleton[l_sh, 0]
            dy = skeleton[r_sh, 1] - skeleton[l_sh, 1]
            return math.atan2(dy, dx)
    
    # Fallback to hips
    l_hip = mapping.get("left_hip")
    r_hip = mapping.get("right_hip")
    
    if l_hip is not None and r_hip is not None:
        if skeleton[l_hip, 2] >= min_confidence and skeleton[r_hip, 2] >= min_confidence:
            dx = skeleton[r_hip, 0] - skeleton[l_hip, 0]
            dy = skeleton[r_hip, 1] - skeleton[l_hip, 1]
            return math.atan2(dy, dx)
            
    return 0.0

def auto_select_anchor_mode(skeleton: torch.Tensor, format: str, min_confidence: float) -> str:
    """
    Automatically select best anchor mode based on keypoint confidence.
    """
    def avg_confidence(indices):
        confs = [skeleton[i, 2] for i in indices if i is not None and skeleton[i, 2] >= min_confidence]
        return sum(confs) / len(confs) if confs else 0.0
    
    # Check each mode's viability
    modes = []
    
    # Torso: need at least 3 valid points
    torso_indices = get_anchor_indices(format, "torso")
    valid_torso = sum(1 for i in torso_indices if i is not None and skeleton[i, 2] >= min_confidence)
    if valid_torso >= 3:
        modes.append(("torso", avg_confidence(torso_indices) * 1.2))  # Prioritize torso
    
    # Hips
    hip_indices = get_anchor_indices(format, "hips")
    if all(i is not None and skeleton[i, 2] >= min_confidence for i in hip_indices):
        modes.append(("hips", avg_confidence(hip_indices)))
    
    # Shoulders
    shoulder_indices = get_anchor_indices(format, "shoulders")
    if all(i is not None and skeleton[i, 2] >= min_confidence for i in shoulder_indices):
        modes.append(("shoulders", avg_confidence(shoulder_indices)))
    
    # If nothing specific found, list is empty?
    if not modes:
         return "neck" # Default fallback
    
    # Return highest scoring mode
    return max(modes, key=lambda x: x[1])[0]
