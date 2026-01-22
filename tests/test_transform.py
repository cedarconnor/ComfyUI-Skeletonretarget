import torch
import unittest
import sys
import os

# Add parent dir to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.transform import ComputeRetargetTransform, ApplyRetargetTransform
from utils.definitions import KEYPOINT_MAPPING

class TestRetargeting(unittest.TestCase):
    def setUp(self):
        self.compute_node = ComputeRetargetTransform()
        self.apply_node = ApplyRetargetTransform()
        
    def test_translation_only(self):
        """Test simple translation retargeting."""
        # Driving: a single point at (0.2, 0.2)
        # Reference: a single point at (0.8, 0.8)
        # Mode: hips (indices 8, 11 for COCO-18)
        
        # Create dummy skeletons
        K = 18
        # Format COCO-18
        driving = torch.zeros((K, 3))
        # Hips at (0.2, 0.2)
        driving[8] = torch.tensor([0.2, 0.2, 1.0])
        driving[11] = torch.tensor([0.2, 0.2, 1.0]) # Same pos for simplicity
        
        ref = torch.zeros((K, 3))
        # Hips at (0.8, 0.8)
        ref[8] = torch.tensor([0.8, 0.8, 1.0])
        ref[11] = torch.tensor([0.8, 0.8, 1.0])
        
        # Compute
        transform, _ = self.compute_node.compute(
            ref, driving, "COCO-18", "hips", 
            enable_scale=False, enable_rotation=False
        )
        
        # Verify anchor calculation
        # Driving anchor should be (0.2, 0.2)
        # Ref anchor should be (0.8, 0.8)
        self.assertAlmostEqual(transform['driving_anchor'][0], 0.2)
        self.assertAlmostEqual(transform['reference_anchor'][0], 0.8)
        
        # Apply to a sequence
        # Sequence has 1 frame, with a point at (0.3, 0.2) -> relative to anchor (0.1, 0)
        # Expectation: Output should be at Ref Anchor (0.8, 0.8) + (0.1, 0) = (0.9, 0.8)
        
        seq = torch.zeros((1, K, 3))
        seq[0] = driving.clone()
        seq[0, 0] = torch.tensor([0.3, 0.2, 1.0]) # Nose
        
        retargeted, _ = self.apply_node.apply(seq, transform, "none")
        
        # Check Nose
        nose = retargeted[0, 0]
        self.assertAlmostEqual(nose[0].item(), 0.9, places=5)
        self.assertAlmostEqual(nose[1].item(), 0.8, places=5)
        
    def test_scale(self):
        """Test scale retargeting."""
        K = 18
        driving = torch.zeros((K, 3))
        # Hips width = 0.2 (0.4 to 0.6)
        driving[8] = torch.tensor([0.4, 0.5, 1.0])
        driving[11] = torch.tensor([0.6, 0.5, 1.0])
        
        ref = torch.zeros((K, 3))
        # Hips width = 0.4 (0.3 to 0.7) -> Scale should be 2.0
        ref[8] = torch.tensor([0.3, 0.5, 1.0])
        ref[11] = torch.tensor([0.7, 0.5, 1.0])
        
        transform, _ = self.compute_node.compute(
            ref, driving, "COCO-18", "hips",
            enable_scale=True, enable_rotation=False
        )
        
        self.assertAlmostEqual(transform['scale'], 2.0, places=5)
        
        # Apply
        # Point at 0.1 offset from center in driving.
        # Format check: 
        # Driving Center: (0.5, 0.5)
        # Driving Point: (0.6, 0.5) [Right Hip]
        # Dist = 0.1
        # Ref Center: (0.5, 0.5)
        # Scale = 2.0 -> New Dist = 0.2
        # Result should be 0.5 + 0.2 = 0.7
        
        seq = driving.unsqueeze(0)
        retargeted, _ = self.apply_node.apply(seq, transform, "none")
        
        r_hip_x = retargeted[0, 11, 0].item() # 0.7
        self.assertAlmostEqual(r_hip_x, 0.7, places=5)

if __name__ == '__main__':
    unittest.main()
