import torch
import math
import json
try:
    from ..utils.geometry import compute_torso_anchor, compute_rotation, get_anchor_indices, euclidean_distance, auto_select_anchor_mode
except ImportError:
    from utils.geometry import compute_torso_anchor, compute_rotation, get_anchor_indices, euclidean_distance, auto_select_anchor_mode

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
    
    def compute(self, reference_skeleton, driving_skeleton, format, anchor_mode, enable_scale, enable_rotation, min_anchor_confidence=0.3):
        # Auto-select anchor mode if requested
        selected_mode = anchor_mode
        if anchor_mode == "auto":
            selected_mode = auto_select_anchor_mode(driving_skeleton, format, min_anchor_confidence)
            # Check if reference also supports it?
            # Assuming if driving has it, ref should ideally have it too.
            
        # Compute Anchors
        # Using the helper which handles averaging points
        indices = get_anchor_indices(format, selected_mode)
        
        def get_anchor_point(skeleton, idxs):
            # Compute centroid of valid points
            valid_points = []
            for i in idxs:
                if i is not None and skeleton[i, 2] >= min_anchor_confidence:
                    valid_points.append(skeleton[i, :2])
            
            if not valid_points:
                # Fallback to pure average even if low confidence? Or fail?
                # Let's try to get *any* point from indices
                candidates = [skeleton[i, :2] for i in idxs if i is not None]
                if candidates:
                    return torch.stack(candidates).mean(dim=0)
                return torch.tensor([0.5, 0.5], device=skeleton.device)
                
            return torch.stack(valid_points).mean(dim=0)

        driving_anchor = get_anchor_point(driving_skeleton, indices)
        reference_anchor = get_anchor_point(reference_skeleton, indices)
        
        # Compute Scale
        scale = 1.0
        d_scale_ref = 1.0
        r_scale_ref = 1.0
        
        if enable_scale:
            # Determine scale metric based on mode
            if selected_mode == "hips" or selected_mode == "shoulders":
                # Width based
                p1_idx, p2_idx = indices[0], indices[1]
                if p1_idx is not None and p2_idx is not None:
                     # Check validity
                     d_valid = driving_skeleton[p1_idx, 2] > min_anchor_confidence and driving_skeleton[p2_idx, 2] > min_anchor_confidence
                     r_valid = reference_skeleton[p1_idx, 2] > min_anchor_confidence and reference_skeleton[p2_idx, 2] > min_anchor_confidence
                     
                     if d_valid and r_valid:
                         d_width = torch.norm(driving_skeleton[p1_idx, :2] - driving_skeleton[p2_idx, :2])
                         r_width = torch.norm(reference_skeleton[p1_idx, :2] - reference_skeleton[p2_idx, :2])
                         
                         if d_width > 0.01:
                             scale = (r_width / d_width).item()
                             d_scale_ref = d_width.item()
                             r_scale_ref = r_width.item()
            
            elif selected_mode == "torso" or selected_mode == "neck":
               # Use shoulder width as proxy for scale
               sh_indices = get_anchor_indices(format, "shoulders")
               p1_idx, p2_idx = sh_indices[0], sh_indices[1]
               if p1_idx is not None and p2_idx is not None:
                   d_valid = driving_skeleton[p1_idx, 2] > min_anchor_confidence and driving_skeleton[p2_idx, 2] > min_anchor_confidence
                   r_valid = reference_skeleton[p1_idx, 2] > min_anchor_confidence and reference_skeleton[p2_idx, 2] > min_anchor_confidence
                   
                   if d_valid and r_valid:
                       d_width = torch.norm(driving_skeleton[p1_idx, :2] - driving_skeleton[p2_idx, :2])
                       r_width = torch.norm(reference_skeleton[p1_idx, :2] - reference_skeleton[p2_idx, :2])
                       
                       if d_width > 0.01:
                           scale = (r_width / d_width).item()
                           d_scale_ref = d_width.item()
                           r_scale_ref = r_width.item()

        # Compute Rotation
        rotation = 0.0
        if enable_rotation:
            driving_rot = compute_rotation(driving_skeleton, format, min_anchor_confidence)
            reference_rot = compute_rotation(reference_skeleton, format, min_anchor_confidence)
            rotation = reference_rot - driving_rot
            
        transform = {
            "scale": scale,
            "rotation": rotation,
            "driving_anchor": driving_anchor.tolist(),
            "reference_anchor": reference_anchor.tolist(),
            "anchor_mode": selected_mode,
            "format": format,
            "driving_scale_metric": d_scale_ref,
            "reference_scale_metric": r_scale_ref,
            "driving_initial_pose": driving_skeleton.tolist(),
            "reference_initial_pose": reference_skeleton.tolist()
        }
        
        debug_info = json.dumps(transform, indent=2)
        
        return (transform, debug_info)

class ApplyRetargetTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton_sequence": ("SKELETON_SEQ",),
                "transform": ("SKELETON_TRANSFORM",),
                "retargeting_mode": (["absolute", "relative"], {"default": "absolute"}),
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
    
    def apply(self, skeleton_sequence, transform, bounds_mode, retargeting_mode="absolute", min_confidence=0.0):
        N, K, _ = skeleton_sequence.shape
        device = skeleton_sequence.device
        result = skeleton_sequence.clone()
        
        # Extract coordinates and confidence
        xy = result[:, :, :2]  # [N, K, 2]
        conf = result[:, :, 2]  # [N, K]
        
        # Create confidence mask
        # Points with low confidence should NOT be moved strictly? 
        # Actually usually we transform everything, but maybe mask out low conf ones?
        # The design doc says: "Keep original coordinates for low-confidence points"
        conf_mask = conf >= min_confidence  # [N, K]
        
        if retargeting_mode == "relative":
            # Relative Retargeting: Apply motion DELTA from driving initial to driving current
            # to reference initial.
            # Delta = Driving_Current - Driving_Initial
            # Result = Reference_Initial + Delta * Scale
            
            # Get cached initial poses (ensure they exist in transform)
            if "driving_initial_pose" not in transform or "reference_initial_pose" not in transform:
                # Fallback to absolute if missing data (e.g. old workflow)
                print("Warning: Missing initial pose data for relative retargeting. Falling back to absolute.")
                return self.apply(skeleton_sequence, transform, bounds_mode, "absolute", min_confidence)
                
            driving_initial = torch.tensor(transform["driving_initial_pose"], device=device)[:, :2] # [K, 2]
            reference_initial = torch.tensor(transform["reference_initial_pose"], device=device)[:, :2] # [K, 2]
            
            # Prepare scale
            scale = transform["scale"]
            
            # Current Driving Position (xy) is driving_sequence frame i
            # Delta calculation
            # We need to compute delta for EACH frame relative to driving_initial
            
            # 1. Translate current sequence by subtracting driving initial? 
            # OR better: 
            # Delta = xy - driving_initial (broadcasting [N, K, 2] - [K, 2])
            
            delta = xy - driving_initial
            
            # 2. Apply Scale to Delta
            if scale != 1.0:
                delta = delta * scale
                
            # 3. Apply Rotation to Delta
            rotation = transform["rotation"]
            if abs(rotation) > 1e-6:
                cos_r = math.cos(rotation)
                sin_r = math.sin(rotation)
                
                # Rotation matrix multiplication
                d_x = delta[:, :, 0] * cos_r - delta[:, :, 1] * sin_r
                d_y = delta[:, :, 0] * sin_r + delta[:, :, 1] * cos_r
                delta = torch.stack([d_x, d_y], dim=2)
                
            # 4. Add to Reference Initial
            # result = reference_initial + delta
            xy = reference_initial + delta
            
        else:
            # Absolute Retargeting (Original Logic)
            
            # Step 1: Translate to driving anchor origin
            driving_anchor = torch.tensor(transform["driving_anchor"], dtype=xy.dtype, device=device)
            xy = xy - driving_anchor  # Broadcasting: [N, K, 2] - [2]
            
            # Step 2: Apply scale
            scale = transform["scale"]
            if scale != 1.0:
                xy = xy * scale
            
            # Step 3: Apply rotation (if non-zero)
            rotation = transform["rotation"]
            if abs(rotation) > 1e-6:
                cos_r = math.cos(rotation)
                sin_r = math.sin(rotation)
                
                # Rotation matrix multiplication
                x_new = xy[:, :, 0] * cos_r - xy[:, :, 1] * sin_r
                y_new = xy[:, :, 0] * sin_r + xy[:, :, 1] * cos_r
                xy = torch.stack([x_new, y_new], dim=2)
            
            # Step 4: Translate to reference anchor
            reference_anchor = torch.tensor(transform["reference_anchor"], dtype=xy.dtype, device=device)
            xy = xy + reference_anchor
        
        # Apply confidence mask (revert low-confidence points to original)
        # Using unsqueeze(2) as suggested
        mask_expanded = conf_mask.unsqueeze(2) # [N, K, 1]
        xy = torch.where(mask_expanded, xy, skeleton_sequence[:, :, :2])
        
        # Handle bounds
        bounds_mask = torch.zeros(N, dtype=torch.bool, device=device)
        
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
        
        # Helper for batch bounding box?
        # Doing loop for readability/robustness per frame as scaling varies
        # This part might be slow in python loop if N is large.
        # Vectorizing frame-wise scaling is tricky without padding
        
        for i in range(N):
            mask_i = conf_mask[i]
            if not mask_i.any():
                continue
                
            valid_points = xy[i][mask_i]  # [V, 2]
            
            min_xy = valid_points.min(dim=0).values
            max_xy = valid_points.max(dim=0).values
            
            if min_xy.min() < 0.0 or max_xy.max() > 1.0:
                bounds_mask[i] = True
                
                center = (min_xy + max_xy) / 2
                extent = (max_xy - min_xy) / 2
                
                margin = 0.05
                max_extent = getattr(torch.max(extent), 'item', lambda: max(extent))() # Scalar
                if isinstance(max_extent, torch.Tensor):
                    max_extent = max_extent.item()
                    
                if max_extent > 0:
                    scale_factor = (0.5 - margin) / max_extent
                    scale_factor = min(scale_factor, 1.0) 
                    
                    xy[i] = (xy[i] - center) * scale_factor + 0.5

        return xy, bounds_mask
