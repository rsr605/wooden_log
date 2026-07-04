"""
Synthetic Wooden Log Sample Data Generator — v3.

Generates highly varied, realistic synthetic images containing wooden logs
with rich textures (bark, grain, knots, cracks), multiple shapes and sizes,
varied backgrounds (forest floor, sawmill, dirt, gravel), and strong
augmentation (lighting, fog, blur, shadows, perspective).

Produces matching YOLO-format annotations for training YOLOv8.
"""

import os
import sys
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from src.utils import ensure_dir, xyxy_to_xywh


class SampleGenerator:
    """
    Generate highly varied synthetic wooden log images for YOLOv8 training.

    Key improvements over v2:
    - Realistic bark texture with procedural noise + wood grain
    - Multiple log shapes: horizontal cylinder, vertical, diagonal, stacked
    - Knots, cracks, and end-grain ring patterns
    - 8+ background types: forest floor, sawmill dirt, gravel, grass, snow, mud
    - Strong augmentation: brightness, contrast, blur, fog, color jitter
    - Multiple image resolutions (640x480, 800x600, 1024x768)
    - Dense overlapping piles and scattered individual logs
    """

    # Wood color palette in BGR — CORRECT brown/tan colors
    # (Blue channel is lowest, Red channel is highest = brown in RGB)
    LOG_COLORS = [
        (33, 67, 101),     # brown  — BGR: B=33 G=67 R=101
        (28, 58, 86),      # dark brown
        (40, 80, 120),     # light brown
        (20, 45, 65),      # dark walnut
        (50, 95, 140),     # tan
        (30, 60, 90),      # medium brown
        (15, 35, 50),      # very dark
        (60, 110, 160),    # pine / light tan
        (25, 50, 75),      # chestnut
        (42, 75, 110),     # oak
        (12, 30, 45),      # ebony-ish
        (48, 90, 130),     # birch bark
        (35, 65, 95),      # cedar
        (70, 130, 180),    # bleached / sun-weathered
        (55, 100, 145),    # amber
        (38, 72, 105),     # rich mahogany
        (45, 85, 125),     # redwood
        (62, 115, 165),    # maple
        (25, 55, 80),      # dark cedar
        (48, 88, 128),     # walnut
    ]

    # Background types and their base colors (BGR)
    # Greens/greys/dark — deliberately distinct from brown wood tones
    BG_TYPES = {
        "forest_floor": [(55, 70, 60), (45, 60, 50), (40, 55, 45), (35, 50, 40)],
        "sawmill_dirt": [(60, 75, 90), (50, 65, 80), (55, 70, 85), (65, 80, 95)],
        "gravel":       [(110, 115, 120), (100, 105, 110), (90, 95, 100), (85, 90, 95)],
        "grass":        [(60, 90, 70), (50, 80, 60), (55, 85, 65), (45, 75, 55)],
        "mud":          [(50, 65, 80), (40, 55, 70), (45, 60, 75), (55, 70, 85)],
        "snow":         [(225, 225, 230), (200, 205, 215), (210, 215, 220), (215, 220, 225)],
        "sand":         [(110, 140, 160), (100, 130, 150), (120, 145, 165), (90, 120, 140)],
        "dark_forest":  [(30, 45, 35), (25, 40, 30), (35, 50, 40), (20, 35, 25)],
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Background generation
    # ------------------------------------------------------------------

    def _generate_background(self, width: int, height: int) -> np.ndarray:
        """Create a textured background from a randomly chosen type."""
        bg_type = self.rng.choice(list(self.BG_TYPES.keys()))
        base_color = self.rng.choice(self.BG_TYPES[bg_type])

        bg = np.full((height, width, 3), base_color, dtype=np.uint8)

        # Add fine texture noise
        noise_level = self.rng.randint(10, 30)
        noise = self._np_rng.randint(
            -noise_level, noise_level, (height, width, 3), dtype=np.int16
        )
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add random texture patches based on background type
        n_patches = self.rng.randint(4, 10)
        for _ in range(n_patches):
            px = self.rng.randint(0, width)
            py = self.rng.randint(0, height)
            radius = self.rng.randint(15, 80)
            patch_color = self._random_patch_color(bg_type)
            self._blend_circle(bg, px, py, radius, patch_color, alpha=0.25)

        # Add random small specks (leaves, pebbles, debris) — earth tones only
        n_specks = self.rng.randint(20, 60)
        for _ in range(n_specks):
            sx = self.rng.randint(0, width)
            sy = self.rng.randint(0, height)
            sr = self.rng.randint(1, 4)
            # Earth-tone specks: browns, greys, dark greens — NOT bright colors
            sc = [int(c) for c in self._np_rng.randint(15, 90, 3)]
            cv2.circle(bg, (sx, sy), sr, tuple(sc), -1)

        return bg

    def _random_patch_color(self, bg_type: str) -> Tuple[int, int, int]:
        """Get a darker or lighter shade for background patches."""
        colors = self.BG_TYPES[bg_type]
        base = self.rng.choice(colors)
        variation = self.rng.randint(-25, 25)
        return tuple(int(np.clip(c + variation, 0, 255)) for c in base)

    def _blend_circle(
        self, img: np.ndarray, cx: int, cy: int, r: int,
        color: Tuple[int, int, int], alpha: float
    ):
        """Blend a semi-transparent circle onto the image."""
        h, w = img.shape[:2]
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r)
        y2 = min(h, cy + r)
        if x2 <= x1 or y2 <= y1:
            return
        overlay = img[y1:y2, x1:x2].copy()
        cv2.circle(overlay, (cx - x1, cy - y1), r, color, -1)
        cv2.addWeighted(overlay, alpha, img[y1:y2, x1:x2], 1 - alpha, 0,
                        img[y1:y2, x1:x2])

    # ------------------------------------------------------------------
    # Log rendering — realistic bark texture
    # ------------------------------------------------------------------

    def _draw_end_view_log(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        angle: Optional[float] = None,
    ) -> Tuple[int, int, int, int]:
        """
        Draw a realistic end-view log cross-section (the circular cut face).

        This matches what you see when looking at a log pile from the end —
        circular cross-sections with tree rings, bark edges, and knots.

        Features:
        - Circular/elliptical outer bark ring
        - Concentric growth rings with off-center pith
        - Random ring spacing for realism
        - Radial cracks (from drying)
        - Knots and color variation
        - Bark texture around the edge

        Returns:
            Bounding box (x1, y1, x2, y2) in pixel coordinates.
        """
        base_color = self.rng.choice(self.LOG_COLORS)

        cx = x + w // 2
        cy = y + h // 2
        rx = max(4, w // 2)
        ry = max(4, h // 2)

        # Optional slight rotation for variety
        if angle is None:
            angle = self.rng.choice([0, 0, 0, 0, 5, -5, 10, -10, 15, -15, 30, -30])

        pad = max(10, max(w, h) // 5)
        canvas_w = w + pad * 2
        canvas_h = h + pad * 2
        log_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        cx_local = canvas_w // 2
        cy_local = canvas_h // 2

        # 1. Bark ring (outer dark ring)
        bark_color = tuple(max(0, c - 30) for c in base_color)
        bark_thickness = max(2, min(rx, ry) // 6)
        cv2.ellipse(log_canvas, (cx_local, cy_local), (rx, ry), angle, 0, 360,
                     bark_color, bark_thickness)

        # 2. Fill interior with base wood color
        cv2.ellipse(log_canvas, (cx_local, cy_local),
                     (max(1, rx - bark_thickness), max(1, ry - bark_thickness)),
                     angle, 0, 360, base_color, -1)

        # 3. Concentric growth rings with off-center pith
        knot_offset_x = self.rng.randint(-rx // 4, rx // 4)
        knot_offset_y = self.rng.randint(-ry // 4, ry // 4)
        ring_center = (cx_local + knot_offset_x, cy_local + knot_offset_y)

        n_rings = self.rng.randint(6, 14)
        ring_color_dark = tuple(max(0, c - 30) for c in base_color)
        ring_color_light = tuple(min(255, c + 20) for c in base_color)

        for i in range(n_rings, 0, -1):
            r_factor = i / n_rings
            # Variable ring spacing
            spacing_jitter = self.rng.uniform(0.7, 1.3)
            r_x = max(1, int((rx - bark_thickness - 1) * r_factor * spacing_jitter))
            r_y = max(1, int((ry - bark_thickness - 1) * r_factor * spacing_jitter))

            # Alternate dark and light rings
            if i % 2 == 0:
                rc = ring_color_dark
                thickness = self.rng.choice([1, 1, 2])
            else:
                rc = ring_color_light
                thickness = 1

            cv2.ellipse(log_canvas, ring_center, (r_x, r_y), angle, 0, 360,
                         rc, thickness)

        # 4. Central pith (small dark dot)
        pith_color = tuple(max(0, c - 45) for c in base_color)
        pith_r = max(1, min(rx, ry) // 10)
        cv2.circle(log_canvas, ring_center, pith_r, pith_color, -1)

        # 5. Radial cracks (from wood drying)
        n_cracks = self.rng.randint(0, 4)
        for _ in range(n_cracks):
            crack_angle = self.rng.uniform(0, 2 * math.pi)
            crack_len = self.rng.uniform(0.4, 0.8) * min(rx, ry)
            crack_color = tuple(max(0, c - 50) for c in base_color)
            ex = int(ring_center[0] + crack_len * math.cos(crack_angle))
            ey = int(ring_center[1] + crack_len * math.sin(crack_angle))
            cv2.line(log_canvas, ring_center, (ex, ey), crack_color, 1, cv2.LINE_AA)

        # 6. Random knots within the cross-section — dark brown/black tones
        n_knots = self.rng.randint(0, 3)
        for _ in range(n_knots):
            knot_r = self.rng.uniform(0.2, 0.7)
            knot_angle = self.rng.uniform(0, 2 * math.pi)
            kx = int(ring_center[0] + knot_r * rx * math.cos(knot_angle))
            ky = int(ring_center[1] + knot_r * ry * math.sin(knot_angle))
            ks = max(2, min(rx, ry) // 8 + self.rng.randint(-2, 3))
            # Dark brown knot — no bright colors
            knot_color = (
                self.rng.randint(5, 35),
                self.rng.randint(10, 40),
                self.rng.randint(15, 50),
            )
            cv2.circle(log_canvas, (kx, ky), ks, knot_color, -1)
            # Ring around knot — slightly lighter brown
            ring_around = tuple(min(255, c + 15) for c in knot_color)
            cv2.circle(log_canvas, (kx, ky), ks + 2, ring_around, 1)

        # 7. Subtle texture noise
        mask = cv2.cvtColor(log_canvas, cv2.COLOR_BGR2GRAY)
        mask_bool = mask > 0
        if mask_bool.any():
            noise = self._np_rng.randint(-15, 15, log_canvas.shape, dtype=np.int16)
            log_canvas = np.clip(
                log_canvas.astype(np.int16) + noise * mask_bool[:, :, np.newaxis],
                0, 255
            ).astype(np.uint8)

        # 8. Apply rotation if needed (already applied via angle parameter)
        if angle != 0:
            rot_matrix = cv2.getRotationMatrix2D(
                (cx_local, cy_local), angle, 1.0
            )
            log_canvas = cv2.warpAffine(
                log_canvas, rot_matrix, (canvas_w, canvas_h)
            )

        # Overlay onto main image (mask-based, only non-black pixels)
        paste_x = max(0, x - pad)
        paste_y = max(0, y - pad)
        paste_w = min(canvas_w, image.shape[1] - paste_x)
        paste_h = min(canvas_h, image.shape[0] - paste_y)

        if paste_w <= 0 or paste_h <= 0:
            return (x, y, x + w, y + h)

        src_region = log_canvas[:paste_h, :paste_w]
        dst_region = image[paste_y: paste_y + paste_h, paste_x: paste_x + paste_w]

        mask = cv2.cvtColor(src_region, cv2.COLOR_BGR2GRAY)
        mask = mask > 0
        dst_region[mask] = src_region[mask]

        return (x, y, x + w, y + h)

    def _draw_log(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        angle: Optional[float] = None,
    ) -> Tuple[int, int, int, int]:
        """
        Draw a realistic wooden log (side-view horizontal cylinder) on the image.

        Features:
        - Cylinder body with gradient shading (lighter top, darker bottom)
        - Rounded ends (ellipse caps)
        - Procedural bark texture (vertical streaks + noise)
        - End-grain ring patterns (concentric circles with off-center knot)
        - Random cracks along the body
        - Knots (dark oval spots)

        Returns:
            Bounding box (x1, y1, x2, y2) in pixel coordinates.
        """
        if angle is None:
            angle = self.rng.choice([
                0, 0, 0, 0, 0,  # mostly horizontal
                10, -10, 20, -20,  # slight tilt
                30, -30, 45, -45,  # diagonal
                90, -90,  # vertical (end-facing)
            ])

        base_color = self.rng.choice(self.LOG_COLORS)

        # Canvas for the log (with padding for rotation)
        pad = max(30, max(w, h) // 3)
        canvas_w = w + pad * 2
        canvas_h = h + pad * 2
        log_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        cx_local = canvas_w // 2
        cy_local = canvas_h // 2

        log_x = cx_local - w // 2
        log_y = cy_local - h // 2

        # 1. Main body with gradient shading
        self._draw_cylinder_body(
            log_canvas, log_x, log_y, w, h, base_color
        )

        # 2. Bark texture — vertical streaks
        self._add_bark_texture(log_canvas, log_x, log_y, w, h, base_color)

        # 3. End caps with ring patterns
        self._draw_end_caps(log_canvas, log_x, log_y, w, h, base_color)

        # 4. Random knots
        self._add_knots(log_canvas, log_x, log_y, w, h)

        # 5. Random cracks
        self._add_cracks(log_canvas, log_x, log_y, w, h)

        # 6. Highlight on top edge
        self._add_highlight(log_canvas, log_x, log_y, w, h)

        # 7. Shadow at bottom
        self._add_shadow(log_canvas, log_x, log_y, w, h)

        # 8. Rotate if needed
        if angle != 0:
            rot_matrix = cv2.getRotationMatrix2D(
                (cx_local, cy_local), angle, 1.0
            )
            log_canvas = cv2.warpAffine(
                log_canvas, rot_matrix, (canvas_w, canvas_h)
            )

        # Overlay onto main image (mask-based, only non-black pixels)
        paste_x = max(0, x - pad)
        paste_y = max(0, y - pad)
        paste_w = min(canvas_w, image.shape[1] - paste_x)
        paste_h = min(canvas_h, image.shape[0] - paste_y)

        if paste_w <= 0 or paste_h <= 0:
            return (x, y, x + w, y + h)

        src_region = log_canvas[:paste_h, :paste_w]
        dst_region = image[paste_y: paste_y + paste_h, paste_x: paste_x + paste_w]

        mask = cv2.cvtColor(src_region, cv2.COLOR_BGR2GRAY)
        mask = mask > 0
        dst_region[mask] = src_region[mask]

        return (paste_x, paste_y, paste_x + paste_w, paste_y + paste_h)

    def _draw_cylinder_body(
        self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
        base_color: Tuple[int, int, int]
    ):
        """Draw the cylinder body with vertical gradient (lighter top → darker bottom)."""
        # Draw rounded rectangle for the body
        cv2.rectangle(canvas, (x, y), (x + w, y + h), base_color, -1)

        # Gradient overlay: lighter at top, darker at bottom
        for i in range(h):
            alpha = i / max(1, h)
            shade = int(-30 + 50 * (1 - alpha))  # -30 at bottom, +20 at top
            row_color = tuple(
                int(np.clip(c + shade, 0, 255)) for c in base_color
            )
            cv2.line(canvas, (x, y + i), (x + w, y + i), row_color, 1)

    def _add_bark_texture(
        self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
        base_color: Tuple[int, int, int]
    ):
        """Add vertical bark streaks and noise for realistic texture."""
        # Vertical streaks (darker and lighter)
        n_streaks = max(3, w // 8)
        for _ in range(n_streaks):
            sx = x + self.rng.randint(2, max(3, w - 2))
            darkness = self.rng.randint(-25, 10)
            streak_color = tuple(
                int(np.clip(c + darkness, 0, 255)) for c in base_color
            )
            thickness = self.rng.choice([1, 1, 1, 2])
            cv2.line(
                canvas, (sx, y + 3), (sx, y + h - 3),
                streak_color, thickness, cv2.LINE_AA
            )

        # Fine noise texture within the body
        body_region = canvas[y + 2: y + h - 2, x + 2: x + w - 2]
        if body_region.size > 0:
            noise = self._np_rng.randint(
                -12, 12, body_region.shape, dtype=np.int16
            )
            body_region[:] = np.clip(
                body_region.astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)

    def _draw_end_caps(
        self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
        base_color: Tuple[int, int, int]
    ):
        """Draw rounded end caps with concentric ring patterns (end grain)."""
        cy = y + h // 2
        radius = max(3, h // 2)

        # Left cap
        cv2.ellipse(canvas, (x, cy), (radius, radius), 0, 90, 270, base_color, -1)
        # Right cap
        cv2.ellipse(canvas, (x + w, cy), (radius, radius), 0, -90, 90, base_color, -1)

        # End-grain rings on both caps
        ring_color_dark = tuple(max(0, c - 35) for c in base_color)
        ring_color_light = tuple(min(255, c + 15) for c in base_color)

        for cap_cx in [x, x + w]:
            # Off-center for realism
            knot_offset_x = self.rng.randint(-radius // 4, radius // 4)
            knot_offset_y = self.rng.randint(-radius // 4, radius // 4)
            ring_center = (cap_cx + knot_offset_x, cy + knot_offset_y)

            for r_factor in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]:
                r = max(1, int(radius * r_factor))
                rc = self.rng.choice([ring_color_dark, ring_color_light])
                thickness = 1 if r_factor < 0.5 else 1
                cv2.ellipse(
                    canvas, ring_center, (r, r), 0, 90, 270, rc, thickness
                )
                cv2.ellipse(
                    canvas, ring_center, (r, r), 0, -90, 90, rc, thickness
                )

            # Central pith (small dark dot)
            cv2.circle(canvas, ring_center, max(1, radius // 8),
                       ring_color_dark, -1)

    def _add_knots(
        self, canvas: np.ndarray, x: int, y: int, w: int, h: int
    ):
        """Add random dark oval knots on the log body — natural brown tones."""
        n_knots = self.rng.randint(0, 3)
        for _ in range(n_knots):
            kx = x + self.rng.randint(5, max(6, w - 5))
            ky = y + self.rng.randint(5, max(6, h - 5))
            kw = self.rng.randint(4, max(5, h // 4))
            kh = self.rng.randint(3, max(4, h // 5))
            # Dark brown knot — natural wood tones
            knot_color = (
                self.rng.randint(5, 35),
                self.rng.randint(10, 40),
                self.rng.randint(15, 50),
            )
            angle = self.rng.randint(0, 180)
            cv2.ellipse(canvas, (kx, ky), (kw, kh), angle, 0, 360,
                        knot_color, -1)
            # Ring around knot — slightly lighter brown
            ring_color = tuple(min(255, c + 20) for c in knot_color)
            cv2.ellipse(canvas, (kx, ky), (kw + 2, kh + 2), angle, 0, 360,
                        ring_color, 1)

    def _add_cracks(
        self, canvas: np.ndarray, x: int, y: int, w: int, h: int
    ):
        """Add random vertical cracks along the log body."""
        n_cracks = self.rng.randint(0, 2)
        for _ in range(n_cracks):
            cx = x + self.rng.randint(3, max(4, w - 3))
            cy_start = y + self.rng.randint(0, h // 3)
            cy_end = y + self.rng.randint(h // 2, h)
            crack_color = (self.rng.randint(10, 35),
                           self.rng.randint(10, 35),
                           self.rng.randint(10, 35))
            # Jagged line
            pts = [(cx, cy_start)]
            cur_y = cy_start
            while cur_y < cy_end:
                cur_y += self.rng.randint(3, 8)
                cur_x = cx + self.rng.randint(-3, 3)
                pts.append((cur_x, min(cur_y, cy_end)))
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], crack_color, 1, cv2.LINE_AA)

    def _add_highlight(
        self, canvas: np.ndarray, x: int, y: int, w: int, h: int
    ):
        """Add a subtle highlight along the top edge of the log."""
        highlight_color = (200, 200, 200)
        overlay = canvas.copy()
        cv2.line(overlay, (x + 3, y + 1), (x + w - 3, y + 1),
                 highlight_color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

    def _add_shadow(
        self, canvas: np.ndarray, x: int, y: int, w: int, h: int
    ):
        """Add a subtle shadow along the bottom edge of the log."""
        shadow_color = (10, 10, 10)
        overlay = canvas.copy()
        cv2.line(overlay, (x + 3, y + h - 1), (x + w - 3, y + h - 1),
                 shadow_color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

    # ------------------------------------------------------------------
    # Scene generation
    # ------------------------------------------------------------------

    def generate_single(
        self,
        width: int = 640,
        height: int = 480,
        n_logs: Optional[int] = None,
        view_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Generate a single synthetic image with wooden logs.

        Produces rich, varied scenes with:
        - 2–40 logs per image (mixed sizes)
        - Three view modes: 'side' (horizontal cylinders),
          'end' (circular cross-sections), 'mixed' (both)
        - Dense hexagonal-packed log piles for end view
        - Five size classes: tiny, small, medium, large, xlarge
        - All angles including vertical and diagonal
        - Edge-partial logs
        - Overlapping logs / log piles
        - Strong augmentation: brightness, contrast, blur, fog, color shift
        """
        image = self._generate_background(width, height)

        if view_mode is None:
            view_mode = self.rng.choices(
                population=["side", "end", "mixed"],
                weights=[30, 50, 20],
            )[0]

        # End-view scenes get dense packing (15-40 logs)
        if view_mode in ("end", "mixed") and n_logs is None:
            n_logs = self.rng.choices(
                population=[15, 18, 20, 22, 25, 28, 30, 35, 40],
                weights=[8, 10, 12, 12, 15, 12, 10, 8, 5],
            )[0]
        elif n_logs is None:
            n_logs = self.rng.choices(
                population=[2, 3, 4, 5, 6, 7, 8, 9, 10],
                weights=[5, 10, 12, 15, 15, 15, 12, 8, 8],
            )[0]

        if view_mode == "end":
            bboxes = self._generate_end_view_pile(image, width, height, n_logs)
        elif view_mode == "mixed":
            n_side = max(1, n_logs // 4)
            bboxes = self._generate_side_view_scene(image, width, height, n_side)
            n_end = n_logs - n_side
            bboxes += self._generate_end_view_pile(image, width, height, n_end)
        else:
            bboxes = self._generate_side_view_scene(image, width, height, n_logs)

        # Ensure at least one log
        if not bboxes:
            log_w = self.rng.randint(60, 120)
            log_h = self.rng.randint(40, 60)
            x = self.rng.randint(50, max(51, width - log_w - 50))
            y = self.rng.randint(50, max(51, height - log_h - 50))
            x1, y1, x2, y2 = self._draw_log(image, x, y, log_w, log_h)
            cx, cy, bw, bh = xyxy_to_xywh(x1, y1, x2, y2)
            bboxes.append([0.0, cx / width, cy / height, bw / width, bh / height])

        # Apply augmentation
        image = self._augment(image)

        return image, bboxes

    def _generate_side_view_scene(
        self, image: np.ndarray, width: int, height: int, n_logs: int
    ) -> List[List[float]]:
        """Generate a scene with side-view horizontal cylinder logs."""
        bboxes: List[List[float]] = []

        for _ in range(n_logs):
            size_class = self.rng.choices(
                population=["tiny", "small", "medium", "large", "xlarge"],
                weights=[12, 25, 30, 20, 13],
            )[0]

            if size_class == "tiny":
                log_w = self.rng.randint(25, 45)
                log_h = self.rng.randint(18, 30)
            elif size_class == "small":
                log_w = self.rng.randint(45, 75)
                log_h = self.rng.randint(28, 50)
            elif size_class == "medium":
                log_w = self.rng.randint(75, 130)
                log_h = self.rng.randint(45, 75)
            elif size_class == "large":
                log_w = self.rng.randint(130, 200)
                log_h = self.rng.randint(65, 105)
            else:
                log_w = self.rng.randint(200, 320)
                log_h = self.rng.randint(90, 140)

            allow_edge = self.rng.random() < 0.30
            if allow_edge:
                x = self.rng.randint(-30, max(1, width - log_w + 30))
                y = self.rng.randint(-30, max(1, height - log_h + 30))
            else:
                max_x = max(11, width - log_w - 10)
                max_y = max(11, height - log_h - 10)
                x = self.rng.randint(5, max_x)
                y = self.rng.randint(5, max_y)

            x1, y1, x2, y2 = self._draw_log(image, x, y, log_w, log_h)

            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(width, x2), min(height, y2)
            if (x2_c - x1_c) < 12 or (y2_c - y1_c) < 8:
                continue

            cx, cy, bw, bh = xyxy_to_xywh(x1_c, y1_c, x2_c, y2_c)
            bboxes.append([0.0, cx / width, cy / height, bw / width, bh / height])

        return bboxes

    def _generate_end_view_pile(
        self, image: np.ndarray, width: int, height: int, n_logs: int
    ) -> List[List[float]]:
        """
        Generate a dense log pile seen from the end (circular cross-sections).

        Scatters circular log cross-sections across the image with varied
        sizes and slight randomness, producing dense pile-like layouts.
        """
        bboxes: List[List[float]] = []

        # Pick a base radius range for this pile
        base_radius = self.rng.randint(25, 55)

        placed = 0
        max_attempts = n_logs * 3
        attempts = 0

        while placed < n_logs and attempts < max_attempts:
            attempts += 1

            # Random position within the image
            gx = self.rng.randint(20, max(21, width - 20))
            gy = self.rng.randint(20, max(21, height - 20))

            # Size variation within the pile
            size_var = self.rng.uniform(0.5, 1.4)
            log_w = int(base_radius * 2 * size_var)
            log_h = int(base_radius * 2 * size_var)

            # Make some logs slightly elliptical
            if self.rng.random() < 0.3:
                log_h = int(log_h * self.rng.uniform(0.75, 0.95))

            x = gx - log_w // 2
            y = gy - log_h // 2

            x1, y1, x2, y2 = self._draw_end_view_log(image, x, y, log_w, log_h)

            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(width, x2), min(height, y2)
            if (x2_c - x1_c) < 12 or (y2_c - y1_c) < 12:
                continue

            cx, cy, bw, bh = xyxy_to_xywh(x1_c, y1_c, x2_c, y2_c)
            bboxes.append([0.0, cx / width, cy / height, bw / width, bh / height])
            placed += 1

        return bboxes

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations that make synthetic images look more
        like real photographs, closing the domain gap.

        Key additions over v2:
        - Stronger HSV colour jitter (sensors vary a lot)
        - CLAHE (adaptive histogram equalization) for realistic contrast
        - Camera sensor noise (Gaussian + Poisson-like)
        - JPEG compression artifacts (simulates phone camera storage)
        - Defocus / motion blur at varying levels
        - Directional lighting gradient (sun angle simulation)
        - Mild perspective / scale perturbation
        - Sharpening kernel (some photos are sharpened)
        """
        # 1. Brightness / contrast jitter — wider range for outdoor photos
        alpha = self.rng.uniform(0.6, 1.4)
        beta = self.rng.randint(-35, 35)
        image = np.clip(
            image.astype(np.float64) * alpha + beta, 0, 255
        ).astype(np.uint8)

        # 2. Strong HSV colour jitter (hue, saturation, value) — simulates
        #    different cameras, white balance, and lighting conditions.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
        h_shift = self.rng.uniform(-12, 12)
        s_scale = self.rng.uniform(0.5, 1.6)
        v_scale = self.rng.uniform(0.6, 1.4)
        hsv[:, :, 0] = np.clip((hsv[:, :, 0] + h_shift) % 180, 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 3. CLAHE — adaptive histogram equalization for realistic contrast
        if self.rng.random() < 0.5:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(
                clipLimit=self.rng.uniform(1.5, 3.5),
                tileGridSize=(8, 8),
            )
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 4. Camera sensor noise — Gaussian + impulse
        if self.rng.random() < 0.6:
            noise_level = self.rng.randint(3, 15)
            noise = self._np_rng.normal(0, noise_level, image.shape)
            image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)

        # 5. Blur — defocus (Gaussian) and motion (directional kernel)
        blur_type = self.rng.random()
        if blur_type < 0.25:
            # Defocus blur
            ksize = self.rng.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif blur_type < 0.35:
            # Motion blur (horizontal or vertical streak)
            ksize = self.rng.choice([5, 7, 9])
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            if self.rng.random() < 0.5:
                kernel[ksize // 2, :] = 1.0  # horizontal motion
            else:
                kernel[:, ksize // 2] = 1.0  # vertical motion
            kernel /= ksize
            image = cv2.filter2D(image, -1, kernel)

        # 6. Sharpening kernel — some cameras sharpen edges
        if self.rng.random() < 0.3:
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ], dtype=np.float32)
            sharpened = cv2.filter2D(image, -1, kernel)
            blend = self.rng.uniform(0.3, 0.6)
            image = cv2.addWeighted(sharpened, blend, image, 1 - blend, 0)

        # 7. Directional lighting gradient — simulates sun angle
        if self.rng.random() < 0.4:
            h, w = image.shape[:2]
            gradient = np.linspace(
                self.rng.uniform(-20, 0),
                self.rng.uniform(0, 20),
                w,
            )[np.newaxis, :, np.newaxis]
            gradient = np.broadcast_to(gradient, (h, w, 1)).astype(np.float64)
            image = np.clip(image.astype(np.float64) + gradient, 0, 255).astype(np.uint8)

        # 8. JPEG compression artifacts — encode/decode at low quality
        if self.rng.random() < 0.35:
            quality = self.rng.choice([30, 40, 50, 60, 70])
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, buf = cv2.imencode('.jpg', image, encode_params)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # 9. Color temperature shift (warm/cool white balance)
        if self.rng.random() < 0.4:
            shift = self.rng.randint(-20, 20)
            overlay = image.copy()
            overlay[:, :, 0] = np.clip(overlay[:, :, 0].astype(int) + shift, 0, 255)
            overlay[:, :, 2] = np.clip(overlay[:, :, 2].astype(int) - shift, 0, 255)
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # 10. Fog / haze overlay
        if self.rng.random() < 0.15:
            fog_intensity = self.rng.uniform(0.1, 0.3)
            fog_color = [self.rng.randint(180, 220) for _ in range(3)]
            overlay = np.full_like(image, fog_color)
            cv2.addWeighted(overlay, fog_intensity, image, 1 - fog_intensity, 0, image)

        # 11. Vignette (darkened corners) — simulates lens characteristics
        if self.rng.random() < 0.2:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 1, -1)
            mask = cv2.GaussianBlur(mask, (min(w, 401) | 1, min(h, 401) | 1), 0)
            mask = mask / mask.max()
            vignette = np.clip(image.astype(np.float64) * mask[:, :, np.newaxis],
                               0, 255).astype(np.uint8)
            cv2.addWeighted(vignette, 0.7, image, 0.3, 0, image)

        return image

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        output_dir: str,
        n_images: int = 50,
        width: int = 640,
        height: int = 480,
    ) -> Dict[str, int]:
        """Generate a full synthetic dataset with train/val split."""
        root = Path(output_dir)

        for split in ["train", "val"]:
            ensure_dir(root / "images" / split)
            ensure_dir(root / "labels" / split)

        sample_dir = root / "sample_images"
        ensure_dir(sample_dir)

        n_train = int(n_images * 0.8)
        splits = {"train": n_train, "val": n_images - n_train}

        counts = {"train": 0, "val": 0, "sample": 0}
        idx = 0

        # Vary image resolution for robustness
        resolutions = [(640, 480), (800, 600), (512, 512)]

        for split, count in splits.items():
            for _ in range(count):
                # 70% standard resolution, 30% varied
                if self.rng.random() < 0.7:
                    iw, ih = width, height
                else:
                    iw, ih = self.rng.choice(resolutions)

                image, bboxes = self.generate_single(iw, ih)

                name = f"log_{idx:04d}"
                img_path = root / "images" / split / f"{name}.jpg"
                lbl_path = root / "labels" / split / f"{name}.txt"

                cv2.imwrite(str(img_path), image)
                with open(lbl_path, "w") as f:
                    for b in bboxes:
                        f.write(
                            f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} "
                            f"{b[3]:.6f} {b[4]:.6f}\n"
                        )

                counts[split] += 1
                idx += 1

        # Generate sample images for the web demo
        for i in range(min(5, n_images)):
            image, _ = self.generate_single(width, height)
            cv2.imwrite(str(sample_dir / f"sample_{i}.jpg"), image)
            counts["sample"] += 1

        return counts

    def generate_sample_images(
        self,
        output_dir: str,
        n_images: int = 5,
        width: int = 640,
        height: int = 480,
    ) -> List[str]:
        """Generate sample images for web demo (no labels)."""
        ensure_dir(output_dir)
        paths = []
        for i in range(n_images):
            image, _ = self.generate_single(width, height)
            path = str(Path(output_dir) / f"sample_{i}.jpg")
            cv2.imwrite(path, image)
            paths.append(path)
        return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry: generate synthetic sample data."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic wooden log samples"
    )
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--n", type=int, default=50, help="Number of images")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    gen = SampleGenerator(seed=args.seed)
    result = gen.generate_dataset(
        output_dir=args.output,
        n_images=args.n,
        width=args.width,
        height=args.height,
    )
    print(f"Sample data generated: {result}")

    # Also generate data.yaml
    root = Path(args.output).resolve()
    yaml_content = (
        f"path: {root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: 1\n"
        f"names: ['wooden_log']\n"
    )
    yaml_path = root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"data.yaml created at: {yaml_path}")


if __name__ == "__main__":
    main()
