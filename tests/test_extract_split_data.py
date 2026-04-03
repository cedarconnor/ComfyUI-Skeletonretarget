
import sys
import os
import unittest
import torch
import numpy as np

# Add custom nodes directory to path so we can import nodes
current_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.dirname(current_dir)
sys.path.insert(0, custom_nodes_dir)

from nodes.extract import ExtractSkeletonFromPose

class TestExtractSplitData(unittest.TestCase):
    def test_merge_split_data(self):
        extractor = ExtractSkeletonFromPose()
        
        # Mock DWPose output (Body-18 + Face + Hands split)
        # 18 Body points
        body_kps = [0.1 * i for i in range(18 * 3)] 
        # 68 Face points
        face_kps = [0.2 for _ in range(68 * 3)]
        # 21 Left Hand points
        lhand_kps = [0.3 for _ in range(21 * 3)]
        # 21 Right Hand points
        rhand_kps = [0.4 for _ in range(21 * 3)]
        
        mock_person = {
            "pose_keypoints_2d": body_kps,
            "face_keypoints_2d": face_kps,
            "hand_left_keypoints_2d": lhand_kps,
            "hand_right_keypoints_2d": rhand_kps
        }

        # Test resilience: person with None keys
        mock_person_none = {
             "pose_keypoints_2d": body_kps,
             "face_keypoints_2d": None,
             "hand_left_keypoints_2d": None,
             "hand_right_keypoints_2d": None
        }
        
        # Mock pose object (list of frames, each frame is list of people or dicts)
        # The node expects `pose` to be a list of frames.
        # Inside the node: 
        # if isinstance(frame_content, list): people_list = frame_content
        # else: people_list = [frame_content] (if it's a dict per frame?)
        # Let's check the code: "if isinstance(frame_content, dict) and 'people' in frame_content: people_list = frame_content['people']"
        # The standard OpenPose JSON structure is a dict with "people" key.
        
        mock_pose_data = [
            {"people": [mock_person]},
            {"people": [mock_person_none]}  # Second frame with None data
        ]
        
        # Call extract
        # Format "COCO-133", person_selection="largest"
        result = extractor.extract(mock_pose_data, "COCO-133", "largest", min_confidence=0.0)
        
        # Result is (first_skeleton, skeleton_sequence, mask)
        # We want the sequence
        skeleton_seq = result[1]
        
        # Expected shape: [Frames, Keypoints, 3] -> [2, 133, 3] (2 frames now)
        # (Node selects 1 person per frame)
        print(f"Output shape: {skeleton_seq.shape}")
        self.assertEqual(skeleton_seq.shape, (2, 133, 3))
        
        # Verify Body Mapping
        # Nose is index 0 in both OP18 and C133
        # data[0] = body_kps[0] = 0.0
        self.assertAlmostEqual(skeleton_seq[0, 0, 0].item(), 0.0)
        
        # Verify Face Mapping
        # Face starts at index 23 in COCO-133
        # Should be 0.2
        print(f"Face start value: {skeleton_seq[0, 23, 0].item()}")
        self.assertAlmostEqual(skeleton_seq[0, 23, 0].item(), 0.2)
        
        # Verify Hand Mapping
        # Left Hand starts at 91
        # Should be 0.3
        self.assertAlmostEqual(skeleton_seq[0, 91, 0].item(), 0.3)
        
        # Right Hand starts at 112
        # Should be 0.4
        self.assertAlmostEqual(skeleton_seq[0, 112, 0].item(), 0.4)

if __name__ == '__main__':
    try:
        t = TestExtractSplitData()
        t.test_merge_split_data()
        print("TEST PASSED")
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        print("TEST FAILED")
        sys.exit(1)
