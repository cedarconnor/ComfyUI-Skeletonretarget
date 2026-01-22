from typing import Dict, List, Tuple

# Canonical keypoint indices per format
KEYPOINT_MAPPING: Dict[str, Dict[str, int]] = {
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
            # BODY-25 has mid_hip we can use directly, but for consistency with others 
            # and stability, we'll use R/L hips if available, otherwise just mid_hip?
            # Actually BODY-25 has explicit hips at 9 and 12.
            return [mapping["right_hip"], mapping["left_hip"]]
        elif format == "COCO-133":
            return [mapping["right_hip"], mapping["left_hip"]]
        else:  # COCO-18
            return [mapping["right_hip"], mapping["left_hip"]]
    
    elif anchor_mode == "shoulders":
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

# Hand Topology Helpers
# Connect wrist (0) to finger bases (1, 5, 9, 13, 17) relative to hand start
# Then finger segments
def _generate_hand_connections(start_idx):
    wrist = start_idx
    # Thumb: 0->1->2->3->4
    thumb = [(wrist, start_idx+1), (start_idx+1, start_idx+2), (start_idx+2, start_idx+3), (start_idx+3, start_idx+4)]
    # Index: 0->5->6->7->8
    index = [(wrist, start_idx+5), (start_idx+5, start_idx+6), (start_idx+6, start_idx+7), (start_idx+7, start_idx+8)]
    # Middle: 0->9->10->11->12
    middle = [(wrist, start_idx+9), (start_idx+9, start_idx+10), (start_idx+10, start_idx+11), (start_idx+11, start_idx+12)]
    # Ring: 0->13->14->15->16
    ring = [(wrist, start_idx+13), (start_idx+13, start_idx+14), (start_idx+14, start_idx+15), (start_idx+15, start_idx+16)]
    # Pinky: 0->17->18->19->20
    pinky = [(wrist, start_idx+17), (start_idx+17, start_idx+18), (start_idx+18, start_idx+19), (start_idx+19, start_idx+20)]
    return thumb + index + middle + ring + pinky

# Face Topology Helpers (Simple outline + eyes/nose/mouth)
# This is simplified; COCO-133 face is 68 points (23-90)
def _generate_face_connections(start_idx):
    # Just doing jawline and basic features for now to keep it sane
    # Jaw: 0-16 (0->1->...->16)
    jaw = [(start_idx + i, start_idx + i + 1) for i in range(16)]
    # Eyebrows
    # Left: 17-21
    l_brow = [(start_idx + i, start_idx + i + 1) for i in range(17, 21)]
    # Right: 22-26
    r_brow = [(start_idx + i, start_idx + i + 1) for i in range(22, 26)]
    # Nose
    nose_bridge = [(start_idx+27, start_idx+28), (start_idx+28, start_idx+29), (start_idx+29, start_idx+30)]
    nose_base = [(start_idx+31, start_idx+32), (start_idx+32, start_idx+33), (start_idx+33, start_idx+34), (start_idx+34, start_idx+35)]
    # Eyes
    l_eye = [(start_idx+36, start_idx+37), (start_idx+37, start_idx+38), (start_idx+38, start_idx+39), (start_idx+39, start_idx+40), (start_idx+40, start_idx+41), (start_idx+41, start_idx+36)]
    r_eye = [(start_idx+42, start_idx+43), (start_idx+43, start_idx+44), (start_idx+44, start_idx+45), (start_idx+45, start_idx+46), (start_idx+46, start_idx+47), (start_idx+47, start_idx+42)]
    # Mouth
    mouth_outer = [(start_idx+48, start_idx+49), (start_idx+49, start_idx+50), (start_idx+50, start_idx+51), (start_idx+51, start_idx+52), (start_idx+52, start_idx+53), (start_idx+53, start_idx+54),
                   (start_idx+54, start_idx+55), (start_idx+55, start_idx+56), (start_idx+56, start_idx+57), (start_idx+57, start_idx+58), (start_idx+58, start_idx+59), (start_idx+59, start_idx+48)]
    
    return jaw + l_brow + r_brow + nose_bridge + nose_base + l_eye + r_eye + mouth_outer

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
            # Connect body wrists to hand wrists
            (9, 91),   # left_wrist -> left_hand_start
            (10, 112), # right_wrist -> right_hand_start
        ],
        "face": _generate_face_connections(23),
        "left_hand": _generate_hand_connections(91),
        "right_hand": _generate_hand_connections(112),
    },
}

# Define colors for visualization (B, G, R) format for OpenCV
LIMB_COLORS = {
    # Body
    "nose_neck": (0, 0, 255),
    "right_arm": (0, 128, 255),  # Orange-ish
    "left_arm": (255, 128, 0),   # Blue-ish
    "right_leg": (0, 255, 0),    # Green
    "left_leg": (255, 0, 0),     # Blue
    "feet": (0, 255, 255),       # Yellow
    
    # Face
    "face": (255, 255, 255),     # White
    
    # Hands
    "left_hand": (200, 50, 200), # Purple
    "right_hand": (50, 200, 200),# Teal
}
