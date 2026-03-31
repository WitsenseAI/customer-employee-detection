"""
calibrate.py — interactive uniform colour profile builder

Draw rectangles over employee uniform areas on the first video frame.
The script samples all pixels inside your selections, clusters them into
colour bands automatically, and saves a JSON profile that app.py can load.

Usage:
    python calibrate.py --video footage.mp4 --profile profiles/shop1.json
    python calibrate.py --image frame.jpg   --profile profiles/shop1.json

Controls (shown in window title bar):
    Left-click + drag  — draw a sample rectangle over uniform area
    U                  — undo last rectangle
    R                  — reset all rectangles
    P                  — toggle live mask preview (right panel)
    S                  — compute colour profile and save JSON
    Q / Esc            — quit without saving
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Padding added around the computed percentile range so slight colour
# variations (lighting, fabric folds) are still matched.
HUE_PAD  = 8    # ± hue units
SAT_PAD  = 25   # subtracted from S 10th-percentile (lower S threshold)
VAL_PAD  = 30   # subtracted from V 10th-percentile (lower V threshold)

RECT_COLOUR   = (0, 255, 255)   # cyan rectangles while drawing
SAMPLE_COLOUR = (0, 200, 0)     # green rectangles after confirmed
OVERLAY_ALPHA = 0.35

MIN_RECT_PX   = 10    # ignore rectangles smaller than 10×10 px
DEFAULT_COVERAGE = 25  # % of person torso crop that must match → classify as staff


# ─── HSV profile computation ──────────────────────────────────────────────────

def _unwrap_hues(hues: np.ndarray) -> np.ndarray:
    """
    If hues wrap around 0/179 (e.g. reds: 170, 175, 3, 8) the naive
    min/max would span almost the whole circle.  Detect this by checking
    whether shifting by 90 reduces the spread, and unwrap accordingly.
    """
    spread_raw      = int(hues.max()) - int(hues.min())
    shifted         = (hues.astype(int) + 90) % 180
    spread_shifted  = int(shifted.max()) - int(shifted.min())
    return shifted if spread_shifted < spread_raw else hues.astype(int)


def pixels_to_band(hsv_pixels: np.ndarray, name: str = "band") -> dict:
    """
    Convert a flat array of HSV pixels (N×3) into a colour band dict
    using robust percentile statistics.
    """
    h = _unwrap_hues(hsv_pixels[:, 0])
    s = hsv_pixels[:, 1].astype(int)
    v = hsv_pixels[:, 2].astype(int)

    # Percentile range — ignores outlier pixels (glare, shadow edges)
    h_lo = int(np.percentile(h, 5))  - HUE_PAD
    h_hi = int(np.percentile(h, 95)) + HUE_PAD

    # Unwrap back to 0-179 range
    h_lo = h_lo % 180
    h_hi = h_hi % 180
    if h_lo > h_hi:
        h_lo, h_hi = h_hi, h_lo   # ensure lo < hi

    s_min = max(0,   int(np.percentile(s, 10)) - SAT_PAD)
    v_min = max(0,   int(np.percentile(v, 10)) - VAL_PAD)

    return {
        "name":  name,
        "h_min": h_lo,
        "h_max": h_hi,
        "s_min": s_min,
        "v_min": v_min,
    }


def cluster_into_bands(hsv_pixels: np.ndarray, n_bands: int) -> list[dict]:
    """
    Use K-means to split sampled pixels into n_bands colour clusters,
    then compute a band dict for each cluster.
    Automatically names bands by their dominant hue description.
    """
    if len(hsv_pixels) < n_bands * 10:
        # Not enough pixels — treat everything as one band
        return [pixels_to_band(hsv_pixels, "uniform")]

    km = KMeans(n_clusters=n_bands, n_init=10, random_state=0)
    labels = km.fit_predict(hsv_pixels.astype(float))

    bands = []
    for k in range(n_bands):
        cluster_pixels = hsv_pixels[labels == k]
        if len(cluster_pixels) < 5:
            continue
        h_mean = float(np.mean(cluster_pixels[:, 0]))
        name   = _hue_name(h_mean)
        bands.append(pixels_to_band(cluster_pixels, name))
        logger.info(f"  Band '{name}': {len(cluster_pixels)} px  "
                    f"H[{bands[-1]['h_min']}-{bands[-1]['h_max']}]  "
                    f"S≥{bands[-1]['s_min']}  V≥{bands[-1]['v_min']}")
    return bands


def _hue_name(h_mean: float) -> str:
    """Return a human-readable colour name for an OpenCV hue (0-179)."""
    h = h_mean % 180
    if h < 10 or h > 170:  return "red"
    if h < 20:              return "orange"
    if h < 35:              return "yellow"
    if h < 85:              return "green"
    if h < 130:             return "blue"
    if h < 155:             return "purple"
    return "pink"


# ─── Mask preview ─────────────────────────────────────────────────────────────

def bands_to_mask(frame_bgr: np.ndarray, bands: list[dict]) -> np.ndarray:
    """Return a combined binary mask for all colour bands on a BGR frame."""
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for b in bands:
        m = cv2.inRange(
            hsv,
            np.array([b["h_min"], b["s_min"], b["v_min"]]),
            np.array([b["h_max"], 255,         255]),
        )
        mask = cv2.bitwise_or(mask, m)
    return mask


def draw_mask_overlay(frame_bgr: np.ndarray, bands: list[dict]) -> np.ndarray:
    """Highlight matched pixels in white on a darkened copy of the frame."""
    mask    = bands_to_mask(frame_bgr, bands)
    dark    = (frame_bgr * 0.35).astype(np.uint8)
    preview = dark.copy()
    preview[mask > 0] = frame_bgr[mask > 0]   # restore original colour in matched areas
    return preview


# ─── Interactive calibration window ──────────────────────────────────────────

class Calibrator:
    def __init__(self, frame: np.ndarray):
        self.original  = frame.copy()
        self.rects: list[tuple[int, int, int, int]] = []   # confirmed rectangles
        self._drawing  = False
        self._start    = (0, 0)
        self._current  = (0, 0)
        self.show_mask = False
        self._bands: list[dict] = []

    # ── mouse callback ────────────────────────────────────────────────────────

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start   = (x, y)
            self._current = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._current = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            x0, y0 = min(self._start[0], x), min(self._start[1], y)
            x1, y1 = max(self._start[0], x), max(self._start[1], y)
            if (x1 - x0) >= MIN_RECT_PX and (y1 - y0) >= MIN_RECT_PX:
                self.rects.append((x0, y0, x1, y1))
                logger.info(f"Added rectangle ({x0},{y0})→({x1},{y1})  "
                            f"[{len(self.rects)} total]")
            self._recompute_bands()

    # ── band computation ──────────────────────────────────────────────────────

    def _recompute_bands(self):
        pixels = self._collect_pixels()
        if len(pixels) < 20:
            self._bands = []
            return
        # Auto-detect number of bands by testing k=1 and k=2 inertia ratio
        n = self._estimate_bands(pixels)
        logger.info(f"Auto-detected {n} colour band(s) from {len(pixels)} pixels")
        self._bands = cluster_into_bands(pixels, n)

    def _collect_pixels(self) -> np.ndarray:
        """Gather all HSV pixels from inside confirmed rectangles."""
        hsv    = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        pixels = []
        for (x0, y0, x1, y1) in self.rects:
            patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
            pixels.append(patch)
        return np.vstack(pixels) if pixels else np.empty((0, 3), dtype=np.uint8)

    def _estimate_bands(self, pixels: np.ndarray, max_k: int = 3) -> int:
        """
        Choose number of K-means clusters by elbow method.
        Returns 1 if adding a second cluster barely reduces inertia.
        """
        if len(pixels) < 40:
            return 1
        inertias = []
        for k in range(1, min(max_k + 1, len(pixels) // 20 + 1)):
            km = KMeans(n_clusters=k, n_init=5, random_state=0)
            km.fit(pixels.astype(float))
            inertias.append(km.inertia_)
        if len(inertias) < 2:
            return 1
        # If k=2 reduces inertia by more than 35%, use 2 bands
        drop = (inertias[0] - inertias[1]) / (inertias[0] + 1e-9)
        return 2 if drop > 0.35 else 1

    # ── rendering ─────────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        """Build the display frame: annotated image + optional mask preview side by side."""
        canvas = self.original.copy()

        # Draw confirmed rectangles
        for (x0, y0, x1, y1) in self.rects:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), SAMPLE_COLOUR, -1)
            canvas = cv2.addWeighted(canvas, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), SAMPLE_COLOUR, 2)

        # Draw in-progress rectangle
        if self._drawing:
            cv2.rectangle(canvas, self._start, self._current, RECT_COLOUR, 2)

        # HUD
        h = canvas.shape[0]
        lines = [
            "Drag to sample uniform  |  U=undo  R=reset  P=mask  S=save  Q=quit",
            f"Rectangles: {len(self.rects)}   "
            f"Bands detected: {len(self._bands)}   "
            f"{'[MASK ON]' if self.show_mask else '[MASK OFF]  press P'}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (10, h - 40 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),   3)
            cv2.putText(canvas, line, (10, h - 40 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if not self.show_mask or not self._bands:
            return canvas

        # Side-by-side mask preview
        preview = draw_mask_overlay(self.original, self._bands)
        cv2.putText(preview, "Mask preview (white=matched)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),   3)
        cv2.putText(preview, "Mask preview (white=matched)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        # Resize preview to same height and concatenate
        ph = canvas.shape[0]
        pw = int(preview.shape[1] * ph / preview.shape[0])
        preview_resized = cv2.resize(preview, (pw, ph))
        return np.hstack([canvas, preview_resized])

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, profile_path: Path, n_bands_override: int | None = None) -> bool:
        """
        Open the calibration window. Returns True if profile was saved.
        """
        win = "Uniform Calibrator"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1400, self.original.shape[1] * 2),
                              min(800,  self.original.shape[0]))
        cv2.setMouseCallback(win, self.on_mouse)

        logger.info("Window open. Draw rectangles over employee uniform areas.")
        logger.info("Press S to save, Q/Esc to quit.")

        saved = False
        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):     # Q or Esc
                break

            elif key == ord('u'):         # undo
                if self.rects:
                    self.rects.pop()
                    self._recompute_bands()
                    logger.info("Undid last rectangle")

            elif key == ord('r'):         # reset
                self.rects.clear()
                self._bands = []
                logger.info("Reset all samples")

            elif key == ord('p'):         # toggle mask preview
                self.show_mask = not self.show_mask

            elif key == ord('s'):         # save
                if not self._bands:
                    logger.warning("No rectangles drawn yet — nothing to save.")
                    continue
                if n_bands_override:
                    pixels = self._collect_pixels()
                    self._bands = cluster_into_bands(pixels, n_bands_override)
                self._save(profile_path)
                saved = True
                break

        cv2.destroyAllWindows()
        return saved

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "name":         path.stem,
            "created":      str(date.today()),
            "bands":        self._bands,
            "coverage_pct": DEFAULT_COVERAGE,
        }
        path.write_text(json.dumps(profile, indent=2))
        logger.info(f"Profile saved → {path}")
        logger.info(f"  Bands: {[b['name'] for b in self._bands]}")
        logger.info(f"  Load with: python app.py --mode uniform --profile {path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive uniform colour calibrator for customer-employee-detection"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Path to a CCTV video file")
    src.add_argument("--image", type=Path, help="Path to a still frame image")
    parser.add_argument(
        "--profile", type=Path, default=Path("profiles/default.json"),
        help="Where to save the colour profile JSON (default: profiles/default.json)",
    )
    parser.add_argument(
        "--bands", type=int, default=None,
        help="Force number of colour bands (default: auto-detect)",
    )
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Which frame number to extract from video (default: 0 = first frame)",
    )
    args = parser.parse_args()

    # ── Load frame ────────────────────────────────────────────────────────────
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            logger.error(f"Could not open video: {args.video}  (file missing or codec unsupported)")
            return
        if args.frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.error(
                f"Could not read frame {args.frame} from {args.video} "
                f"(video has {total} frames)"
            )
            return
        logger.info(f"Loaded frame {args.frame} from {args.video}")
    else:
        frame = cv2.imread(str(args.image))
        if frame is None:
            logger.error(f"Could not load image: {args.image}")
            return
        logger.info(f"Loaded image: {args.image}")

    # ── Run calibrator ────────────────────────────────────────────────────────
    cal   = Calibrator(frame)
    saved = cal.run(args.profile, n_bands_override=args.bands)

    if saved:
        print(f"\n✅  Profile saved to: {args.profile}")
        print(f"    Run detection with:  python app.py --mode uniform --profile {args.profile}")
    else:
        print("\n⚠️  Exited without saving.")


if __name__ == "__main__":
    main()
