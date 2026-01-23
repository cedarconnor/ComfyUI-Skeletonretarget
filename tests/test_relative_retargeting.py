
import unittest
import torch
import sys
import os
import math

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.transform import ComputeRetargetTransform, ApplyRetargetTransform
from utils.definitions import LIMB_CONNECTIONS

class TestRetargeting(unittest.TestCase):
    def setUp(self):
        # Mock class for ComfyUI nodes (since they use @classmethod for INPUT_TYPES but instance methods for execution)
        self.compute_node = ComputeRetargetTransform()
        self.apply_node = ApplyRetargetTransform()

    def test_face_connections_exist(self):
        """Verify inner mouth connections are present in COCO-133 definitions."""
        face_conns = LIMB_CONNECTIONS["COCO-133"]["face"]
        
        # Inner mouth indices are 60-67 (relative to face start 23 -> 83-90 in global? No, definitions use relative generation then offset?)
        # Let's check definitions.py logic.
        # _generate_face_connections(start_idx)
        # face start is 23.
        # Inner mouth 60-67 relative to start.
        # So exact indices should be 23+60=83 to 23+67=90.
        
        # Check if connections involving these indices exist
        found_inner_mouth = False
        for p1, p2 in face_conns:
            if (p1 >= 83 and p1 <= 90) or (p2 >= 83 and p2 <= 90):
                found_inner_mouth = True
                break
        
        self.assertTrue(found_inner_mouth, "Inner mouth connections (indices 83-90) not found in text definitions.")

    def test_relative_retargeting_logic(self):
        """Verify relative retargeting math."""
        # Setup Tensors
        # Single point for simplicity: index 0
        # Format COCO-18 (18 points)
        
        # Driving Skeletons
        # Initial: (0, 0)
        # Current: (10, 10) -> Motion is (+10, +10)
        driving_initial = torch.zeros((18, 3))
        driving_initial[:, 2] = 1.0 # Conf
        
        driving_current = torch.zeros((1, 18, 3)) # Seq of 1 frame
        driving_current[0, :, 0] = 0.5 # x moves +0.5
        driving_current[0, :, 1] = 0.5 # y moves +0.5
        driving_current[0, :, 2] = 1.0
        
        # Reference Skeleton
        # Initial: (100, 100) (Profile view far away)
        reference = torch.zeros((18, 3))
        reference[:, 0] = 100.0
        reference[:, 1] = 100.0
        reference[:, 2] = 1.0
        
        # 1. Compute Transform
        # hips anchor (indices 8, 11 for COCO-18)
        # We need valid hips for anchor computation
        driving_initial[8, :2] = torch.tensor([0.0, 0.0])
        driving_initial[11, :2] = torch.tensor([0.0, 0.0])
        
        reference[8, :2] = torch.tensor([100.0, 100.0])
        reference[11, :2] = torch.tensor([100.0, 100.0])
        
        transform, _ = self.compute_node.compute(
            reference_skeleton=reference,
            driving_skeleton=driving_initial,
            format="COCO-18",
            anchor_mode="hips",
            enable_scale=False, # Disable scale to test pure relative motion
            enable_rotation=False
        )
        
        # Verify initial poses are stored
        self.assertIn("driving_initial_pose", transform)
        self.assertIn("reference_initial_pose", transform)
        
        # 2. Apply Relative
        result_relative, _ = self.apply_node.apply(
            skeleton_sequence=driving_current,
            transform=transform,
            bounds_mode="none",
            retargeting_mode="relative",
            min_confidence=0.0
        )
        
        # Expected: Reference Initial (100, 100) + Delta (0.5, 0.5) = (100.5, 100.5)
        # Note: Delta = driving_current (0.5) - driving_initial (0.0)
        
        res_x = result_relative[0, 0, 0].item()
        res_y = result_relative[0, 0, 1].item()
        
        print(f"Relative Result: ({res_x}, {res_y})")
        
        self.assertAlmostEqual(res_x, 100.5, places=4)
        self.assertAlmostEqual(res_y, 100.5, places=4)
        
    def test_absolute_retargeting_logic(self):
        """Verify absolute retargeting (legacy) works as expected."""
         # Driving: Initial (0,0), Current (0.5, 0.5)
        driving_initial = torch.zeros((18, 3))
        driving_initial[:, 2] = 1.0
        
        driving_current = torch.zeros((1, 18, 3))
        driving_current[0, :, 0] = 0.5
        driving_current[0, :, 1] = 0.5
        driving_current[0, :, 2] = 1.0
        
        # Reference: (100, 100)
        reference = torch.zeros((18, 3))
        reference[:, 0] = 100.0
        reference[:, 1] = 100.0
        reference[:, 2] = 1.0
        
        # Anchors (hips 8, 11)
        driving_initial[8:12, :2] = 0.0 # Anchor at 0,0
        reference[8:12, :2] = 100.0 # Anchor at 100,100
        
        transform, _ = self.compute_node.compute(
            reference, driving_initial, "COCO-18", "hips", False, False
        )
        
        # Apply Absolute
        result_absolute, _ = self.apply_node.apply(
            driving_current, transform, "none", "absolute", 0.0
        )
        
        # Absolute Logic: 
        # 1. Translate driving current to driving anchor origin: (0.5, 0.5) - (0,0) = (0.5, 0.5)
        # 2. Add reference anchor: (0.5, 0.5) + (100, 100) = (100.5, 100.5)
        
        # Wait, if driving anchor is computed from DRIVING SKELETON (initial), then:
        # driving_anchor = (0,0)
        # sequence point = (0.5, 0.5)
        # centered = (0.5, 0.5)
        # ref anchor = (100, 100)
        # result = (100.5, 100.5)
        
        # It seems simple translation yields same result for simple translation case.
        # Let's differentiate them.
        
        # Case where they differ: Scale or Rotation?
        # Or if Reference Initial Pose != Reference Anchor?
        # If Reference is standing at X=100, but we want to map it to Driving X=0.
        # Absolute maps Driving 0.5 -> Ref 100.5 (centering match).
        
        # Relative maps Motion (+0.5) -> Ref (+0.5).
        
        # They produce the same result for pure translation if scales match.
        
        res_x = result_absolute[0, 0, 0].item()
        res_y = result_absolute[0, 0, 1].item()
        
        self.assertAlmostEqual(res_x, 100.5, places=4)

if __name__ == '__main__':
    unittest.main()
