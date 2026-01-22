import torch
import numpy as np
import cv2
try:
    from ..utils.definitions import KEYPOINT_MAPPING, LIMB_CONNECTIONS, LIMB_COLORS
except ImportError:
    from utils.definitions import KEYPOINT_MAPPING, LIMB_CONNECTIONS, LIMB_COLORS

class SkeletonToOpenPoseImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton_sequence": ("SKELETON_SEQ",),
                "format": (["COCO-18", "COCO-133", "BODY-25"], {"default": "COCO-18"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "line_width": ("INT", {"default": 4, "min": 1, "max": 20}),
                "point_radius": ("INT", {"default": 4, "min": 1, "max": 20}),
                "background": (["black", "white", "transparent"], {"default": "black"}),
                "render_hands": ("BOOLEAN", {"default": True}),
                "render_face": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "min_confidence": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "clip_to_bounds": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "render"
    CATEGORY = "SkeletonRetarget/Visualization"
    
    def render(self, skeleton_sequence, format, width, height, line_width, point_radius, background, render_hands, render_face, min_confidence=0.1, clip_to_bounds=True):
        
        N, K, _ = skeleton_sequence.shape
        
        # Prepare batch output
        # ComfyUI IMAGE is [N, H, W, 3] usually float32 0-1
        
        output_images = []
        
        # Determine connections to draw
        connections = []
        
        # Color mapping helper
        # We need to map connection index or simple heuristic to color
        # For now, simplistic consistent coloring.
        
        format_connections = LIMB_CONNECTIONS.get(format)
        if not format_connections:
            # Fallback
            connections_to_draw = []
        elif isinstance(format_connections, list):
             connections_to_draw = format_connections # COCO-18, BODY-25
        elif isinstance(format_connections, dict):
             # COCO-133
             connections_to_draw = format_connections["body"]
             if render_hands:
                 connections_to_draw.extend(format_connections["left_hand"])
                 connections_to_draw.extend(format_connections["right_hand"])
             if render_face:
                 connections_to_draw.extend(format_connections["face"])
        
        # Color strategy:
        # Just cycle colors or use specific simple mapping?
        # OpenPose standard colors are multi-colored. 
        # Using a fixed palette list for limbs.
        
        # BGR (OpenCV)
        palette = [
            (0, 0, 255), (0, 85, 255), (0, 170, 255), (0, 255, 255),
            (0, 255, 170), (0, 255, 85), (0, 255, 0), (85, 255, 0),
            (170, 255, 0), (255, 255, 0), (255, 170, 0), (255, 85, 0),
            (255, 0, 0), (255, 0, 85), (255, 0, 170), (255, 0, 255)
        ]
        
        skeleton_cpu = skeleton_sequence.cpu().numpy()
        
        for i in range(N):
            # Create Canvas
            if background == "black":
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
            elif background == "white":
                canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            else: # transparent - handled as RGBA? Comfy IMAGE is usually RGB.
                # If transparent request, usually means black background for masks/latent?
                # Or we can return RGBA? 
                # Let's stick to black for now to avoid tensor shape mismatch if downstream expects RGB.
                # Actually Comfy supports RGBA.
                # But standard is RGB. Let's do black for simplicity unless requested.
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                
            skel = skeleton_cpu[i] # [K, 3]
            
            # Draw Limbs
            for idx, (p1_idx, p2_idx) in enumerate(connections_to_draw):
                # Check bounds
                if p1_idx >= K or p2_idx >= K:
                    continue
                    
                x1, y1, c1 = skel[p1_idx]
                x2, y2, c2 = skel[p2_idx]
                
                if c1 < min_confidence or c2 < min_confidence:
                    continue
                
                # Convert to pixel
                px1, py1 = int(x1 * width), int(y1 * height)
                px2, py2 = int(x2 * width), int(y2 * height)
                
                # Check OOB
                if clip_to_bounds:
                    if (px1 < 0 or px1 >= width or py1 < 0 or py1 >= height) and \
                       (px2 < 0 or px2 >= width or py2 < 0 or py2 >= height):
                       # Both OOB - check if line crosses? 
                       # cv2.line clips automatically but performance-wise?
                       # If drastically OOB, cv2 can lag or overflow.
                       pass # Allow cv2 to handle minor OOB clipping
                       
                color = palette[idx % len(palette)]
                
                cv2.line(canvas, (px1, py1), (px2, py2), color, line_width)
                
            # Draw Points (optional?) usually standard OpenPose has ellipses/circles
            # Only draw points for valid landmarks
            for idx in range(K):
                x, y, c = skel[idx]
                if c >= min_confidence:
                    px, py = int(x * width), int(y * height)
                    if clip_to_bounds:
                        if px < 0 or px >= width or py < 0 or py >= height:
                            continue
                            
                    # Color logic for points?
                    cv2.circle(canvas, (px, py), point_radius, (255, 255, 255), -1) # White joints?
            
            # Convert BGR to RGB
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            output_images.append(canvas_rgb)
        
        # Stack
        output = np.array(output_images).astype(np.float32) / 255.0
        return (torch.from_numpy(output),)
