import cv2
import numpy as np
import torch
from utils.face_mesh_topology import get_face_triangles
from utils.facing_ratio import compute_facing_ratio


class RenderTexturedFaceMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformed_landmarks": ("FACE_LANDMARKS",),
                "vertex_normals": ("FACE_NORMALS",),
                "landmark_uvs": ("FACE_UVS",),
                "video_frames": ("IMAGE",),
            },
            "optional": {
                "output_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "output_height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "facing_threshold": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "facing_feather": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 90.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("rendered_face", "face_mask", "facing_ratio_mask")
    FUNCTION = "render"
    CATEGORY = "FaceMeshRetarget"

    def render(self, transformed_landmarks, vertex_normals, landmark_uvs,
               video_frames, output_width=512, output_height=512,
               facing_threshold=60.0, facing_feather=20.0):
        B, H_src, W_src, _ = video_frames.shape
        triangles = get_face_triangles().numpy()

        vert_confidence = compute_facing_ratio(
            vertex_normals, facing_threshold, facing_feather
        )

        all_rendered = []
        all_face_masks = []
        all_facing_masks = []

        for b in range(B):
            frame = (video_frames[b].cpu().numpy() * 255).astype(np.uint8)
            verts = transformed_landmarks[b].cpu().numpy()
            uvs = landmark_uvs[b].cpu().numpy()
            conf = vert_confidence[b].cpu().numpy()

            rendered, face_mask, facing_mask = self._render_frame(
                frame, verts, uvs, conf, triangles,
                output_height, output_width, H_src, W_src,
            )

            all_rendered.append(torch.from_numpy(rendered).float() / 255.0)
            all_face_masks.append(torch.from_numpy(face_mask).float() / 255.0)
            all_facing_masks.append(torch.from_numpy(facing_mask).float())

        rendered_batch = torch.stack(all_rendered)
        face_mask_batch = torch.stack(all_face_masks)
        facing_mask_batch = torch.stack(all_facing_masks)

        return (rendered_batch, face_mask_batch, facing_mask_batch)

    @staticmethod
    def _render_frame(frame, verts, uvs, vert_confidence, triangles,
                      out_h, out_w, src_h, src_w):
        """Render one frame via per-triangle OpenCV affine warp."""
        rendered = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        face_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        facing_mask = np.zeros((out_h, out_w), dtype=np.float32)

        dst_px = np.zeros((478, 2), dtype=np.float32)
        dst_px[:, 0] = verts[:, 0] * out_w
        dst_px[:, 1] = verts[:, 1] * out_h

        src_px = np.zeros((478, 2), dtype=np.float32)
        src_px[:, 0] = uvs[:, 0] * src_w
        src_px[:, 1] = uvs[:, 1] * src_h

        for tri in triangles:
            i0, i1, i2 = tri

            src_tri = src_px[tri].astype(np.float32)
            dst_tri = dst_px[tri].astype(np.float32)

            x, y, w, h = cv2.boundingRect(dst_tri)
            if w <= 0 or h <= 0:
                continue
            x = max(x, 0)
            y = max(y, 0)
            w = min(w, out_w - x)
            h = min(h, out_h - y)
            if w <= 0 or h <= 0:
                continue

            dst_local = dst_tri - np.array([x, y], dtype=np.float32)

            M = cv2.getAffineTransform(src_tri, dst_local)
            warped = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)

            tri_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(tri_mask, dst_local.astype(np.int32), 255)

            roi = rendered[y:y + h, x:x + w]
            mask_bool = tri_mask > 0
            roi[mask_bool] = warped[mask_bool]

            face_roi = face_mask[y:y + h, x:x + w]
            face_roi[mask_bool] = 255

            tri_conf = float(vert_confidence[tri].mean())
            facing_roi = facing_mask[y:y + h, x:x + w]
            conf_update = np.full((h, w), tri_conf, dtype=np.float32)
            facing_roi[mask_bool] = np.maximum(facing_roi[mask_bool],
                                                conf_update[mask_bool])

        facing_mask = cv2.GaussianBlur(facing_mask, (5, 5), 1.0)
        facing_mask = np.clip(facing_mask, 0.0, 1.0)

        return rendered, face_mask, facing_mask
