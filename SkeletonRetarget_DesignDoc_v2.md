# ComfyUI-SkeletonRetarget Design Document v2

## Overview

**Project Name:** ComfyUI-SkeletonRetarget  
**Version:** 2.0 (revised)  
**Purpose:** Align and retarget skeletal pose data from a driving video sequence to match a reference image's skeleton, enabling consistent motion transfer for AI video generation.

**Problem Statement:**  
When using motion control features (like Kling AI), the driving video's skeleton often doesn't align with the reference image's pose. This causes jarring jumps at the start of generated videos or forces users to carefully match poses manually. By computing the offset between skeletons and applying it consistently, we can make any driving video work naturally with any reference pose.

---

## Architecture

### High-Level Flow

```
┌─────────────────┐     ┌─────────────────┐
│ Reference Image │     │  Driving Video  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Pose Detector  │     │  Pose Detector  │
│    (DWPose)     │     │    (DWPose)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Single Skeleton │     │ Skeleton Sequence│
│   [18/133, 3]   │     │  [N, 18/133, 3] │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  SkeletonRetarget     │
         │  - Compute offset     │
         │  - Apply transform    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ Retargeted Sequence   │
         │    [N, 18/133, 3]     │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Pose Renderer       │
         │ (OpenPose/DWPose vis) │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  ControlNet / Export  │
         └───────────────────────┘
```

### Node Categories

The pack provides nodes in three categories:

1. **Extraction** - Convert pose detector output to skeleton tensors
2. **Transform** - Compute and apply retargeting transforms
3. **Visualization** - Render skeletons back to images/video

---

## Data Structures

### Skeleton Tensor Format

All skeleton data uses a standardized tensor format:

```python
# Single skeleton
skeleton: torch.Tensor  # Shape: [K, 3] where K = keypoint count
                        # Channel 0: x coordinate (0.0 - 1.0 normalized)
                        # Channel 1: y coordinate (0.0 - 1.0 normalized)  
                        # Channel 2: confidence score (0.0 - 1.0)

# Skeleton sequence (video)
skeleton_sequence: torch.Tensor  # Shape: [N, K, 3] where N = frame count
```

### Keypoint Index Mapping (Cross-Format)

To support multiple skeleton formats, we define explicit mappings for anchor keypoints:

```python
# Canonical keypoint indices per format
KEYPOINT_MAPPING = {
    "COCO-18": {
        "nose": 0,
        "neck": 1,
        "right_shoulder": 2,
        "right_elbow": 3,
        "right_wrist": 4,
        "left_shoulder": 5,
        "left_elbow": 6,
        "left_wrist": 7,
        "right_hip": 8,
        "right_knee": 9,
        "right_ankle": 10,
        "left_hip": 11,
        "left_knee": 12,
        "left_ankle": 13,
        "right_eye": 14,
        "left_eye": 15,
        "right_ear": 16,
        "left_ear": 17,
    },
    "BODY-25": {
        "nose": 0,
        "neck": 1,
        "right_shoulder": 2,
        "right_elbow": 3,
        "right_wrist": 4,
        "left_shoulder": 5,
        "left_elbow": 6,
        "left_wrist": 7,
        "mid_hip": 8,  # BODY-25 has mid-hip instead of separate L/R at root
        "right_hip": 9,
        "right_knee": 10,
        "right_ankle": 11,
        "left_hip": 12,
        "left_knee": 13,
        "left_ankle": 14,
        "right_eye": 15,
        "left_eye": 16,
        "right_ear": 17,
        "left_ear": 18,
        "left_big_toe": 19,
        "left_small_toe": 20,
        "left_heel": 21,
        "right_big_toe": 22,
        "right_small_toe": 23,
        "right_heel": 24,
    },
    "COCO-133": {
        # Body keypoints (indices 0-16, similar to COCO-17)
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
        # Foot keypoints (17-22)
        "left_big_toe": 17,
        "left_small_toe": 18,
        "left_heel": 19,
        "right_big_toe": 20,
        "right_small_toe": 21,
        "right_heel": 22,
        # Face keypoints (23-90) - 68 points
        "face_start": 23,
        "face_end": 90,
        # Left hand (91-111) - 21 points
        "left_hand_start": 91,
        "left_hand_end": 111,
        # Right hand (112-132) - 21 points
        "right_hand_start": 112,
        "right_hand_end": 132,
        # Note: COCO-133 has no explicit "neck" - compute as midpoint of shoulders
        "neck": None,  # Computed: (left_shoulder + right_shoulder) / 2
    },
}

# Anchor groups with format-specific indices
def get_anchor_indices(format: str, anchor_mode: str) -> List[int]:
    """Get keypoint indices for anchor computation, handling format differences."""
    mapping = KEYPOINT_MAPPING[format]
    
    if anchor_mode == "hips":
        if format == "BODY-25":
            # BODY-25 has mid_hip we can use directly, or average L/R
            return [mapping["right_hip"], mapping["left_hip"]]
        elif format == "COCO-133":
            return [mapping["right_hip"], mapping["left_hip"]]
        else:  # COCO-18
            return [mapping["right_hip"], mapping["left_hip"]]
    
    elif anchor_mode == "shoulders":
        if format == "COCO-133":
            return [mapping["right_shoulder"], mapping["left_shoulder"]]
        else:
            return [mapping["right_shoulder"], mapping["left_shoulder"]]
    
    elif anchor_mode == "neck":
        if format == "COCO-133":
            # No neck in COCO-133, return shoulders to compute midpoint
            return [mapping["left_shoulder"], mapping["right_shoulder"]]
        else:
            return [mapping["neck"]]
    
    elif anchor_mode == "torso":
        if format == "COCO-133":
            return [
                mapping["left_shoulder"], mapping["right_shoulder"],
                mapping["left_hip"], mapping["right_hip"]
            ]
        elif format == "BODY-25":
            return [
                mapping["neck"],
                mapping["left_shoulder"], mapping["right_shoulder"],
                mapping["left_hip"], mapping["right_hip"]
            ]
        else:  # COCO-18
            return [
                mapping["neck"],
                mapping["left_shoulder"], mapping["right_shoulder"],
                mapping["left_hip"], mapping["right_hip"]
            ]
    
    raise ValueError(f"Unknown anchor_mode: {anchor_mode}")
```

### Limb Connections (Per Format)

```python
LIMB_CONNECTIONS = {
    "COCO-18": [
        # Body
        (0, 1),   # nose -> neck
        (1, 2),   # neck -> right_shoulder
        (2, 3),   # right_shoulder -> right_elbow
        (3, 4),   # right_elbow -> right_wrist
        (1, 5),   # neck -> left_shoulder
        (5, 6),   # left_shoulder -> left_elbow
        (6, 7),   # left_elbow -> left_wrist
        (1, 8),   # neck -> right_hip
        (8, 9),   # right_hip -> right_knee
        (9, 10),  # right_knee -> right_ankle
        (1, 11),  # neck -> left_hip
        (11, 12), # left_hip -> left_knee
        (12, 13), # left_knee -> left_ankle
        (0, 14),  # nose -> right_eye
        (14, 16), # right_eye -> right_ear
        (0, 15),  # nose -> left_eye
        (15, 17), # left_eye -> left_ear
    ],
    "BODY-25": [
        # Body
        (0, 1),   # nose -> neck
        (1, 2), (2, 3), (3, 4),     # right arm
        (1, 5), (5, 6), (6, 7),     # left arm
        (1, 8),   # neck -> mid_hip
        (8, 9), (9, 10), (10, 11),  # right leg
        (8, 12), (12, 13), (13, 14), # left leg
        # Feet
        (11, 22), (11, 24), (22, 23), # right foot
        (14, 19), (14, 21), (19, 20), # left foot
        # Face
        (0, 15), (0, 16), (15, 17), (16, 18),
    ],
    "COCO-133": {
        "body": [
            (0, 1), (0, 2),  # nose -> eyes
            (1, 3), (2, 4),  # eyes -> ears
            (5, 7), (7, 9),  # left arm
            (6, 8), (8, 10), # right arm
            (5, 6),          # shoulder connection
            (5, 11), (6, 12), # shoulders -> hips
            (11, 12),        # hip connection
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16), # right leg
            # Feet
            (15, 17), (15, 18), (15, 19),
            (16, 20), (16, 21), (16, 22),
        ],
        "face": [
            # Jaw line (0-16 relative to face_start)
            *[(23 + i, 23 + i + 1) for i in range(16)],
            # Eyebrows, nose, eyes, mouth - standard 68-point connections
            # (omitted for brevity, but follows standard face mesh topology)
        ],
        "left_hand": [
            # Thumb
            (91, 92), (92, 93), (93, 94), (94, 95),
            # Index
            (91, 96), (96, 97), (97, 98), (98, 99),
            # Middle
            (91, 100), (100, 101), (101, 102), (102, 103),
            # Ring
            (91, 104), (104, 105), (105, 106), (106, 107),
            # Pinky
            (91, 108), (108, 109), (109, 110), (110, 111),
        ],
        "right_hand": [
            # Same topology as left hand, offset by 21
            (112, 113), (113, 114), (114, 115), (115, 116),
            (112, 117), (117, 118), (118, 119), (119, 120),
            (112, 121), (121, 122), (122, 123), (123, 124),
            (112, 125), (125, 126), (126, 127), (127, 128),
            (112, 129), (129, 130), (130, 131), (131, 132),
        ],
    },
}
```

### Transform Data Structure (Simplified)

**Change from v1:** Removed redundant `translation` field. The transform is now defined purely by anchor points and scale/rotation. Translation is implicit: transform maps `driving_anchor` → `reference_anchor`.

```python
@dataclass
class SkeletonTransform:
    """
    Transformation that maps driving skeleton space to reference skeleton space.
    
    Apply logic:
    1. Translate point so driving_anchor is at origin
    2. Apply scale (uniform)
    3. Apply rotation (around origin)
    4. Translate so origin maps to reference_anchor
    """
    scale: float                       # Uniform scale factor (1.0 = no change)
    rotation: float                    # Rotation angle in radians (0.0 = no change)
    driving_anchor: Tuple[float, float]   # Center point in driving skeleton
    reference_anchor: Tuple[float, float] # Target center point in reference skeleton
    
    # Metadata for debugging/validation
    anchor_mode: str                   # Which anchor mode was used
    format: str                        # Skeleton format
    driving_scale_reference: float     # The measurement used for scale (e.g., shoulder width)
    reference_scale_reference: float   # Same measurement on reference

    def is_identity(self, tolerance: float = 1e-6) -> bool:
        """Check if this transform is approximately identity."""
        return (
            abs(self.scale - 1.0) < tolerance and
            abs(self.rotation) < tolerance and
            abs(self.driving_anchor[0] - self.reference_anchor[0]) < tolerance and
            abs(self.driving_anchor[1] - self.reference_anchor[1]) < tolerance
        )
```

---

## Node Specifications

### Node 1: ExtractSkeletonFromPose

**Purpose:** Convert DWPose/OpenPose detector output to skeleton tensor format.

**Inputs:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `pose_data` | `POSE_KEYPOINT` | required | Output from DWPose/OpenPose detector node |
| `format` | `COMBO` | "COCO-18" | Keypoint format: "COCO-18", "COCO-133", "BODY-25" |
| `person_selection` | `COMBO` | "largest" | "largest", "index", "track" |
| `person_index` | `INT` | 0 | Which person to extract (when selection="index") |
| `tracking_threshold` | `FLOAT` | 0.3 | Max anchor distance to consider same person (normalized coords) |
| `min_confidence` | `FLOAT` | 0.1 | Minimum confidence to consider keypoint valid |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `skeleton` | `SKELETON` | First frame skeleton [K, 3] |
| `skeleton_sequence` | `SKELETON_SEQ` | Full sequence [N, K, 3] |
| `valid_mask` | `MASK` | Boolean mask of frames with valid detection |

**Person Selection Modes:**

- `largest`: Select person with largest bounding box area (frame-by-frame, can switch people)
- `index`: Select person at fixed index (fails if fewer people detected)
- `track`: **Temporal tracking mode** - maintains identity across frames

**Video Person Tracking Strategy (track mode):**

```python
def track_person_across_frames(pose_sequence, tracking_threshold, min_confidence):
    """
    Track a single person across video frames using anchor-based matching.
    
    Algorithm:
    1. Frame 0: Select largest detected person as target
    2. Compute target's torso anchor (average of shoulders + hips)
    3. For each subsequent frame:
       a. For each detected person, compute torso anchor
       b. Find person whose anchor is closest to previous frame's anchor
       c. If distance > tracking_threshold, mark frame as "lost" 
       d. Update target anchor for next frame (with momentum smoothing)
    4. For "lost" frames, optionally interpolate or mark invalid
    
    Returns:
        selected_indices: List[int] - person index per frame (-1 if lost)
        confidence: List[float] - tracking confidence per frame
    """
    selected = []
    prev_anchor = None
    
    for frame_idx, frame_poses in enumerate(pose_sequence):
        if len(frame_poses) == 0:
            selected.append(-1)
            continue
            
        if prev_anchor is None:
            # Frame 0: pick largest
            best_idx = select_largest_person(frame_poses)
            prev_anchor = compute_torso_anchor(frame_poses[best_idx])
            selected.append(best_idx)
        else:
            # Find closest match to previous anchor
            best_idx = -1
            best_dist = float('inf')
            
            for person_idx, person in enumerate(frame_poses):
                anchor = compute_torso_anchor(person)
                dist = euclidean_distance(anchor, prev_anchor)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = person_idx
            
            if best_dist > tracking_threshold:
                # Lost tracking
                selected.append(-1)
            else:
                selected.append(best_idx)
                # Smooth anchor update (momentum)
                new_anchor = compute_torso_anchor(frame_poses[best_idx])
                prev_anchor = (
                    0.7 * new_anchor[0] + 0.3 * prev_anchor[0],
                    0.7 * new_anchor[1] + 0.3 * prev_anchor[1]
                )
    
    return selected
```

**Implementation:**
```python
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
```

---

### Node 2: ComputeRetargetTransform

**Purpose:** Calculate the transformation needed to align driving skeleton to reference skeleton.

**Inputs:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `reference_skeleton` | `SKELETON` | required | Target pose to match |
| `driving_skeleton` | `SKELETON` | required | First frame of driving sequence |
| `format` | `COMBO` | "COCO-18" | Skeleton format (must match extraction) |
| `anchor_mode` | `COMBO` | "hips" | Which body part(s) to use as anchor |
| `enable_scale` | `BOOLEAN` | True | Whether to scale skeleton to match proportions |
| `enable_rotation` | `BOOLEAN` | False | Whether to rotate to match orientation |
| `min_anchor_confidence` | `FLOAT` | 0.3 | Minimum confidence for anchor keypoints |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `transform` | `SKELETON_TRANSFORM` | Computed transformation parameters |
| `debug_info` | `STRING` | JSON with transform details |

**Anchor Modes and Scale Metrics:**

| Mode | Anchor Computation | Scale Metric |
|------|-------------------|--------------|
| `hips` | Midpoint of left_hip, right_hip | Hip-to-hip distance |
| `shoulders` | Midpoint of left_shoulder, right_shoulder | Shoulder-to-shoulder distance |
| `neck` | Neck keypoint (or shoulder midpoint for COCO-133) | Shoulder width |
| `torso` | Centroid of neck, shoulders, hips (weighted by confidence) | Average of shoulder and hip widths |
| `auto` | Select based on keypoint confidence | Use metric from selected mode |

**Auto Mode Selection Logic:**

```python
def auto_select_anchor_mode(skeleton, format, min_confidence):
    """
    Automatically select best anchor mode based on keypoint confidence.
    
    Priority: torso > hips > shoulders > neck
    (More points = more robust, but requires higher overall confidence)
    """
    mapping = KEYPOINT_MAPPING[format]
    
    def avg_confidence(indices):
        confs = [skeleton[i, 2] for i in indices if skeleton[i, 2] >= min_confidence]
        return sum(confs) / len(confs) if confs else 0.0
    
    # Check each mode's viability
    modes = []
    
    # Torso: need at least 3 of 5 points
    torso_indices = get_anchor_indices(format, "torso")
    valid_torso = sum(1 for i in torso_indices if skeleton[i, 2] >= min_confidence)
    if valid_torso >= 3:
        modes.append(("torso", avg_confidence(torso_indices) * 1.2))  # Bonus for robustness
    
    # Hips
    hip_indices = get_anchor_indices(format, "hips")
    if all(skeleton[i, 2] >= min_confidence for i in hip_indices):
        modes.append(("hips", avg_confidence(hip_indices)))
    
    # Shoulders
    shoulder_indices = get_anchor_indices(format, "shoulders")
    if all(skeleton[i, 2] >= min_confidence for i in shoulder_indices):
        modes.append(("shoulders", avg_confidence(shoulder_indices)))
    
    # Neck (always available as fallback via shoulder midpoint)
    modes.append(("neck", 0.5))  # Lower priority
    
    # Return highest scoring mode
    return max(modes, key=lambda x: x[1])[0]
```

**Rotation Computation with Fallback:**

```python
def compute_rotation(skeleton, format, min_confidence):
    """
    Compute body orientation angle with fallback strategies.
    
    Primary: Shoulder angle (atan2 of shoulder-to-shoulder vector)
    Fallback 1: Hip angle
    Fallback 2: Neck-to-hip-midpoint angle (vertical alignment)
    Fallback 3: Return 0.0 (no rotation)
    """
    mapping = KEYPOINT_MAPPING[format]
    
    # Try shoulders first
    l_sh = mapping.get("left_shoulder")
    r_sh = mapping.get("right_shoulder")
    if l_sh and r_sh:
        if skeleton[l_sh, 2] >= min_confidence and skeleton[r_sh, 2] >= min_confidence:
            dx = skeleton[r_sh, 0] - skeleton[l_sh, 0]
            dy = skeleton[r_sh, 1] - skeleton[l_sh, 1]
            return math.atan2(dy, dx)
    
    # Fallback to hips
    l_hip = mapping.get("left_hip")
    r_hip = mapping.get("right_hip")
    if l_hip and r_hip:
        if skeleton[l_hip, 2] >= min_confidence and skeleton[r_hip, 2] >= min_confidence:
            dx = skeleton[r_hip, 0] - skeleton[l_hip, 0]
            dy = skeleton[r_hip, 1] - skeleton[l_hip, 1]
            return math.atan2(dy, dx)
    
    # Fallback to vertical axis (neck to hip midpoint)
    # ... (similar logic)
    
    # Ultimate fallback: no rotation data available
    return 0.0
```

**Implementation:**
```python
class ComputeRetargetTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_skeleton": ("SKELETON",),
                "driving_skeleton": ("SKELETON",),
                "format": (["COCO-18", "COCO-133", "BODY-25"], {"default": "COCO-18"}),
                "anchor_mode": (["hips", "shoulders", "neck", "torso", "auto"], {"default": "hips"}),
                "enable_scale": ("BOOLEAN", {"default": True}),
                "enable_rotation": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "min_anchor_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("SKELETON_TRANSFORM", "STRING")
    RETURN_NAMES = ("transform", "debug_info")
    FUNCTION = "compute"
    CATEGORY = "SkeletonRetarget/Transform"
```

---

### Node 3: ApplyRetargetTransform

**Purpose:** Apply computed transformation to entire skeleton sequence.

**Inputs:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `skeleton_sequence` | `SKELETON_SEQ` | required | Driving pose sequence [N, K, 3] |
| `transform` | `SKELETON_TRANSFORM` | required | Transform from ComputeRetargetTransform |
| `bounds_mode` | `COMBO` | "none" | How to handle out-of-bounds points |
| `min_confidence` | `FLOAT` | 0.0 | Points below this confidence are not transformed |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `retargeted_sequence` | `SKELETON_SEQ` | Transformed skeleton sequence |
| `bounds_mask` | `MASK` | Mask indicating which frames have OOB points |

**Bounds Handling Modes:**

| Mode | Behavior |
|------|----------|
| `none` | Allow coordinates outside [0,1] - defer to render time |
| `clamp` | Clamp individual keypoints to [0,1] (can distort limbs) |
| `scale_to_fit` | Uniformly scale entire pose to fit within bounds |
| `flag_only` | Don't modify, but set bounds_mask for affected frames |

**Why "none" is Default:**

Clamping at the skeleton level distorts geometry (e.g., a wrist "sticks" to frame edge while elbow is inside, creating impossible limb). Better to:
1. Keep true coordinates in skeleton space
2. Let the renderer handle clipping/cropping
3. Or use `scale_to_fit` if the whole pose must be visible

**Vectorized Implementation (Performance Fix):**

```python
class ApplyRetargetTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton_sequence": ("SKELETON_SEQ",),
                "transform": ("SKELETON_TRANSFORM",),
                "bounds_mode": (["none", "clamp", "scale_to_fit", "flag_only"], {"default": "none"}),
            },
            "optional": {
                "min_confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("SKELETON_SEQ", "MASK")
    RETURN_NAMES = ("retargeted_sequence", "bounds_mask")
    FUNCTION = "apply"
    CATEGORY = "SkeletonRetarget/Transform"
    
    def apply(self, skeleton_sequence, transform, bounds_mode, min_confidence=0.0):
        """
        Apply transform using vectorized torch operations for performance.
        
        For a 1000-frame COCO-133 sequence:
        - Naive Python loops: ~2 seconds
        - Vectorized torch: ~5ms
        """
        N, K, _ = skeleton_sequence.shape
        result = skeleton_sequence.clone()
        
        # Extract coordinates and confidence
        xy = result[:, :, :2]  # [N, K, 2]
        conf = result[:, :, 2]  # [N, K]
        
        # Create confidence mask
        conf_mask = conf >= min_confidence  # [N, K]
        
        # Step 1: Translate to driving anchor origin
        driving_anchor = torch.tensor(transform.driving_anchor, dtype=xy.dtype, device=xy.device)
        xy = xy - driving_anchor  # Broadcasting: [N, K, 2] - [2]
        
        # Step 2: Apply scale
        if transform.scale != 1.0:
            xy = xy * transform.scale
        
        # Step 3: Apply rotation (if non-zero)
        if abs(transform.rotation) > 1e-6:
            cos_r = math.cos(transform.rotation)
            sin_r = math.sin(transform.rotation)
            
            # Rotation matrix multiplication
            x_new = xy[:, :, 0] * cos_r - xy[:, :, 1] * sin_r
            y_new = xy[:, :, 0] * sin_r + xy[:, :, 1] * cos_r
            xy = torch.stack([x_new, y_new], dim=2)
        
        # Step 4: Translate to reference anchor
        reference_anchor = torch.tensor(transform.reference_anchor, dtype=xy.dtype, device=xy.device)
        xy = xy + reference_anchor
        
        # Apply confidence mask (don't transform low-confidence points)
        # Keep original coordinates for low-confidence points
        xy = torch.where(conf_mask.unsqueeze(-1), xy, skeleton_sequence[:, :, :2])
        
        # Handle bounds
        bounds_mask = torch.zeros(N, dtype=torch.bool, device=xy.device)
        
        if bounds_mode == "clamp":
            xy = torch.clamp(xy, 0.0, 1.0)
        elif bounds_mode == "scale_to_fit":
            xy, bounds_mask = self._scale_to_fit(xy, conf_mask)
        elif bounds_mode == "flag_only" or bounds_mode == "none":
            # Check which frames have OOB points
            oob = (xy < 0.0) | (xy > 1.0)  # [N, K, 2]
            oob_per_point = oob.any(dim=2)  # [N, K]
            oob_and_valid = oob_per_point & conf_mask  # Only count valid points
            bounds_mask = oob_and_valid.any(dim=1)  # [N]
        
        # Reassemble result
        result[:, :, :2] = xy
        
        return (result, bounds_mask)
    
    def _scale_to_fit(self, xy, conf_mask):
        """Scale each frame's pose to fit within [0,1] bounds."""
        N, K, _ = xy.shape
        bounds_mask = torch.zeros(N, dtype=torch.bool, device=xy.device)
        
        for i in range(N):
            valid_points = xy[i][conf_mask[i]]  # [V, 2] where V = valid count
            if len(valid_points) == 0:
                continue
            
            min_xy = valid_points.min(dim=0).values
            max_xy = valid_points.max(dim=0).values
            
            # Check if scaling needed
            if min_xy.min() < 0.0 or max_xy.max() > 1.0:
                bounds_mask[i] = True
                
                # Compute scale factor to fit
                center = (min_xy + max_xy) / 2
                extent = (max_xy - min_xy) / 2
                
                # Scale so extent fits in [0,1] with some margin
                margin = 0.05
                max_extent = max(extent[0], extent[1])
                if max_extent > 0:
                    scale_factor = (0.5 - margin) / max_extent
                    scale_factor = min(scale_factor, 1.0)  # Don't scale up
                    
                    # Apply scale around center, then translate center to 0.5
                    xy[i] = (xy[i] - center) * scale_factor + 0.5
        
        return xy, bounds_mask
```

---

### Node 4: SkeletonToOpenPoseImage

**Purpose:** Render skeleton tensor(s) to OpenPose-style visualization images.

**Inputs:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `skeleton_sequence` | `SKELETON_SEQ` | required | Skeleton data to render |
| `format` | `COMBO` | "COCO-18" | Skeleton format (for correct topology) |
| `width` | `INT` | 512 | Output image width |
| `height` | `INT` | 512 | Output image height |
| `line_width` | `INT` | 4 | Skeleton line thickness |
| `point_radius` | `INT` | 4 | Keypoint circle radius |
| `background` | `COMBO` | "black" | "black", "white", "transparent" |
| `render_hands` | `BOOLEAN` | True | Render hand keypoints (COCO-133 only) |
| `render_face` | `BOOLEAN` | False | Render face keypoints (COCO-133 only) |
| `min_confidence` | `FLOAT` | 0.1 | Don't render points below this confidence |
| `clip_to_bounds` | `BOOLEAN` | True | Clip rendering to image bounds |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `images` | `IMAGE` | Rendered pose images [N, H, W, 3] |

**Rendering Logic:**

```python
def render_skeleton(skeleton, format, width, height, options):
    """
    Render a single skeleton to an image.
    
    Handles out-of-bounds gracefully:
    - If clip_to_bounds=True: clip line segments at image boundary
    - If clip_to_bounds=False: allow drawing outside (will be cropped by image)
    """
    canvas = create_canvas(width, height, options.background)
    
    connections = LIMB_CONNECTIONS[format]
    colors = LIMB_COLORS[format]
    
    # Handle COCO-133 sub-topologies
    if format == "COCO-133":
        if isinstance(connections, dict):
            # Render body
            render_limbs(canvas, skeleton, connections["body"], colors["body"], options)
            
            # Optionally render hands
            if options.render_hands:
                render_limbs(canvas, skeleton, connections["left_hand"], colors["hand"], options)
                render_limbs(canvas, skeleton, connections["right_hand"], colors["hand"], options)
            
            # Optionally render face
            if options.render_face:
                render_limbs(canvas, skeleton, connections["face"], colors["face"], options)
    else:
        render_limbs(canvas, skeleton, connections, colors, options)
    
    return canvas

def render_limbs(canvas, skeleton, connections, colors, options):
    """Render limb connections with proper out-of-bounds handling."""
    for idx, (start, end) in enumerate(connections):
        # Check confidence
        if skeleton[start, 2] < options.min_confidence:
            continue
        if skeleton[end, 2] < options.min_confidence:
            continue
        
        # Get coordinates (may be outside [0,1])
        x1 = skeleton[start, 0] * options.width
        y1 = skeleton[start, 1] * options.height
        x2 = skeleton[end, 0] * options.width
        y2 = skeleton[end, 1] * options.height
        
        if options.clip_to_bounds:
            # Clip line segment to image bounds using Cohen-Sutherland or similar
            clipped = clip_line_to_rect(x1, y1, x2, y2, 0, 0, options.width, options.height)
            if clipped is None:
                continue  # Entirely outside
            x1, y1, x2, y2 = clipped
        
        color = colors[idx % len(colors)]
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, options.line_width)
    
    # Render keypoints
    for idx in range(len(skeleton)):
        if skeleton[idx, 2] < options.min_confidence:
            continue
        
        x = skeleton[idx, 0] * options.width
        y = skeleton[idx, 1] * options.height
        
        if options.clip_to_bounds:
            if x < 0 or x >= options.width or y < 0 or y >= options.height:
                continue
        
        cv2.circle(canvas, (int(x), int(y)), options.point_radius, (255, 255, 255), -1)
```

---

### Node 5: SkeletonRetargetSimple (All-in-One)

**Purpose:** Simplified single-node workflow combining extraction, transform computation, and application.

**Inputs:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `reference_pose` | `POSE_KEYPOINT` | required | Pose data from reference image |
| `driving_poses` | `POSE_KEYPOINT` | required | Pose sequence from driving video |
| `format` | `COMBO` | "COCO-18" | Keypoint format |
| `anchor_mode` | `COMBO` | "auto" | Alignment anchor selection |
| `enable_scale` | `BOOLEAN` | True | Scale to match proportions |
| `person_selection` | `COMBO` | "track" | How to select person in driving video |
| `width` | `INT` | 512 | Output render width |
| `height` | `INT` | 512 | Output render height |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `retargeted_images` | `IMAGE` | Rendered retargeted pose sequence |
| `original_images` | `IMAGE` | Rendered original (for comparison) |
| `transform` | `SKELETON_TRANSFORM` | Computed transform (for debugging/chaining) |

---

### Node 6: SkeletonBlend

**Purpose:** Blend between two skeleton sequences with controllable weight.

**Inputs:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `skeleton_a` | `SKELETON_SEQ` | required | First skeleton sequence |
| `skeleton_b` | `SKELETON_SEQ` | required | Second skeleton sequence |
| `blend_weight` | `FLOAT` | 0.5 | 0.0 = all A, 1.0 = all B |
| `blend_mode` | `COMBO` | "linear" | Blending method |
| `keypoint_mask` | `STRING` | "" | Optional: comma-separated keypoint indices to blend |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `blended` | `SKELETON_SEQ` | Interpolated skeleton sequence |

**Blend Modes:**

- `linear`: Simple linear interpolation
- `ease_in_out`: Smooth transition curve
- `per_keypoint`: Use keypoint_mask to blend only specific points (e.g., blend upper body from A, lower body from B)

---

### Node 7: SkeletonTemporalFilter

**Purpose:** Apply temporal smoothing to reduce jitter.

**Inputs:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `skeleton_sequence` | `SKELETON_SEQ` | required | Input sequence |
| `filter_type` | `COMBO` | "one_euro" | Filter algorithm |
| `strength` | `FLOAT` | 0.5 | Filter strength (interpretation varies by filter) |
| `preserve_confidence` | `BOOLEAN` | True | Keep original confidence values |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `filtered_sequence` | `SKELETON_SEQ` | Smoothed skeleton sequence |

**Filter Types:**

| Filter | Strength Interpretation | Best For |
|--------|------------------------|----------|
| `gaussian` | Sigma of gaussian kernel (in frames) | Strong smoothing, offline |
| `moving_average` | Window size (in frames) | Simple smoothing |
| `one_euro` | Min cutoff frequency | Adaptive, preserves quick movements |
| `exponential` | Alpha (0-1), higher = less smooth | Simple, real-time |

**Note on Transform vs Keypoint Smoothing:**

This node smooths **keypoint positions**. For some use cases, smoothing the **transform parameters** instead yields better results (maintains limb structure better). Consider using this node on the driving sequence *before* computing the transform, or on the final output, but not both.

---

## Custom Types

```python
# Type definitions for ComfyUI's type system
# Note: ComfyUI uses string-based typing at runtime; these classes are 
# primarily for documentation and IDE support.

SKELETON_TYPE = "SKELETON"           # Single skeleton [K, 3]
SKELETON_SEQ_TYPE = "SKELETON_SEQ"   # Sequence [N, K, 3]  
SKELETON_TRANSFORM_TYPE = "SKELETON_TRANSFORM"  # Transform parameters

# In __init__.py
NODE_CLASS_MAPPINGS = {
    "ExtractSkeletonFromPose": ExtractSkeletonFromPose,
    "ComputeRetargetTransform": ComputeRetargetTransform,
    "ApplyRetargetTransform": ApplyRetargetTransform,
    "SkeletonToOpenPoseImage": SkeletonToOpenPoseImage,
    "SkeletonRetargetSimple": SkeletonRetargetSimple,
    "SkeletonBlend": SkeletonBlend,
    "SkeletonTemporalFilter": SkeletonTemporalFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractSkeletonFromPose": "Extract Skeleton from Pose",
    "ComputeRetargetTransform": "Compute Retarget Transform",
    "ApplyRetargetTransform": "Apply Retarget Transform",
    "SkeletonToOpenPoseImage": "Skeleton to OpenPose Image",
    "SkeletonRetargetSimple": "Skeleton Retarget (Simple)",
    "SkeletonBlend": "Skeleton Blend",
    "SkeletonTemporalFilter": "Skeleton Temporal Filter",
}
```

---

## Testing Plan

### Invariants to Verify

```python
def test_identity_transform():
    """If reference == driving, transform should be identity."""
    skeleton = create_test_skeleton()
    transform = compute_retarget_transform(skeleton, skeleton, "hips", True, False)
    
    assert transform.is_identity()
    assert abs(transform.scale - 1.0) < 1e-6
    assert abs(transform.rotation) < 1e-6

def test_anchor_maps_correctly():
    """Applying transform to driving anchor should yield reference anchor."""
    ref = create_test_skeleton(hip_center=(0.6, 0.5))
    drv = create_test_skeleton(hip_center=(0.3, 0.4))
    
    transform = compute_retarget_transform(ref, drv, "hips", False, False)
    
    # Apply transform to driving anchor point
    result = apply_point(drv_anchor, transform)
    
    assert abs(result[0] - ref_anchor[0]) < 1e-6
    assert abs(result[1] - ref_anchor[1]) < 1e-6

def test_low_confidence_unchanged():
    """Points with confidence=0 should not be transformed."""
    skeleton = create_test_skeleton()
    skeleton[5, 2] = 0.0  # Set left_shoulder confidence to 0
    original_pos = skeleton[5, :2].clone()
    
    transform = create_test_transform(translation=(0.1, 0.1))
    result = apply_retarget_transform(skeleton.unsqueeze(0), transform, "none", 0.1)
    
    assert torch.allclose(result[0, 5, :2], original_pos)

def test_round_trip_consistency():
    """Transform A->B then B->A should recover original."""
    ref = create_test_skeleton()
    drv = create_test_skeleton_different()
    
    transform_ab = compute_retarget_transform(ref, drv, "torso", True, True)
    transform_ba = compute_retarget_transform(drv, ref, "torso", True, True)
    
    # Apply A->B then B->A
    intermediate = apply_retarget_transform(drv.unsqueeze(0), transform_ab)
    recovered = apply_retarget_transform(intermediate, transform_ba)
    
    assert torch.allclose(recovered[0], drv, atol=1e-5)

def test_vectorized_matches_loop():
    """Vectorized implementation should match naive loop implementation."""
    sequence = create_test_sequence(100)  # 100 frames
    transform = create_test_transform()
    
    result_vectorized = apply_retarget_transform_vectorized(sequence, transform)
    result_loop = apply_retarget_transform_loop(sequence, transform)  # Reference impl
    
    assert torch.allclose(result_vectorized, result_loop, atol=1e-6)
```

### Edge Cases to Test

| Case | Expected Behavior |
|------|-------------------|
| Empty skeleton sequence | Return empty sequence, no crash |
| All keypoints confidence=0 | Return unchanged, valid_mask all False |
| Single frame sequence | Works correctly (degenerate case) |
| Person lost mid-sequence (tracking mode) | valid_mask marks lost frames |
| Extreme scale difference (10x) | Transform computed, may need bounds handling |
| 180° rotation | Correctly rotates, doesn't "flip" unexpectedly |
| COCO-133 with only body keypoints | Falls back gracefully for missing hands/face |

---

## Example Workflows

### Workflow 1: Basic Retargeting

```
Reference Image ──► Load Image ──► DWPose Detector ──► ExtractSkeleton ──┐
                                                        (format=COCO-18)  │
                                                                          ▼
                                                          ComputeRetargetTransform
                                                            (anchor=hips)
                                                                          │
Driving Video ──► Load Video ──► DWPose Detector ──► ExtractSkeleton ──┬──┘
                                                      (person=track)   │
                                                                       ▼
                                                          ApplyRetargetTransform
                                                            (bounds=none)
                                                                       │
                                                                       ▼
                                                          SkeletonToOpenPoseImage
                                                                       │
                                                                       ▼
                                                              ControlNet ──► AnimateDiff
```

### Workflow 2: Quick One-Node Workflow

```
Reference Image ──► DWPose ──┐
                             │
                             ▼
                 SkeletonRetargetSimple ──► ControlNet
                             ▲
                             │
Driving Video ──► DWPose ────┘
```

---

## File Structure

```
ComfyUI-SkeletonRetarget/
├── __init__.py              # Node registration
├── requirements.txt         # Dependencies
├── README.md               # User documentation
│
├── nodes/
│   ├── __init__.py
│   ├── extract.py          # ExtractSkeletonFromPose
│   ├── transform.py        # ComputeRetargetTransform, ApplyRetargetTransform
│   ├── render.py           # SkeletonToOpenPoseImage
│   ├── simple.py           # SkeletonRetargetSimple
│   ├── blend.py            # SkeletonBlend
│   └── filter.py           # SkeletonTemporalFilter
│
├── core/
│   ├── __init__.py
│   ├── skeleton.py         # Skeleton data structures and utilities
│   ├── transform.py        # Transform math (vectorized)
│   ├── constants.py        # Keypoint mappings, colors, connections
│   ├── tracking.py         # Person tracking across frames
│   └── render.py           # OpenPose-style rendering utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_transform.py   # Transform invariants
│   ├── test_extract.py     # Extraction tests
│   ├── test_tracking.py    # Person tracking tests
│   └── test_render.py      # Rendering tests
│
├── workflows/
│   ├── basic_retarget.json
│   ├── simple_retarget.json
│   └── blended_motion.json
│
└── examples/
    ├── reference_images/
    └── sample_outputs/
```

---

## Dependencies

```
# requirements.txt
torch>=2.0.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
```

**ComfyUI Node Dependencies:**
- `comfyui_controlnet_aux` - For DWPose/OpenPose detection

---

## Changelog

### v2.0.0 (Design Revision)
- **Fixed:** Removed redundant `translation` field from transform struct
- **Added:** Keypoint index mapping for COCO-18, BODY-25, COCO-133
- **Added:** Person tracking strategy for video sequences
- **Added:** Configurable confidence thresholds (promoted from hardcoded)
- **Added:** Scale metric tied to anchor mode
- **Added:** Rotation fallback behavior specification
- **Changed:** Bounds handling now defaults to "none" to prevent limb distortion
- **Changed:** Apply transform uses vectorized torch operations
- **Added:** Limb connections for all formats including COCO-133 hands/face
- **Added:** Testing plan with invariants

### v1.0.0 (Initial Draft)
- Core skeleton extraction and conversion
- Basic retargeting with translation/scale/rotation
- OpenPose-style rendering
