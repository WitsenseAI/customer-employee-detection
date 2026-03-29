"""
Staff vs Customer Detection Pipeline
YOLOv8 person detection + two classification modes:
  --mode reid     ResNet50 cosine-similarity re-ID (identify specific people)
  --mode uniform  HSV colour detection (identify by uniform colour — faster,
                  no reference images needed, works for any staff member)

Usage:
    # First: build a colour profile by clicking on employees in a frame
    python calibrate.py --video footage.mp4 --profile profiles/shop1.json

    # Then: run detection using that profile
    python app.py --device cpu  --mode uniform --profile profiles/shop1.json
    python app.py --device cuda --mode reid
    python app.py --device jetson --mode uniform --profile profiles/shop2.json
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# ─── CLI args ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Staff vs Customer Detector")
parser.add_argument(
    "--device", choices=["cpu", "cuda", "jetson"], default="cpu",
    help="Inference device: cpu | cuda | jetson",
)
parser.add_argument(
    "--mode", choices=["reid", "uniform"], default="uniform",
    help=(
        "reid    — ResNet50 cosine similarity against saved staff reference crops\n"
        "uniform — HSV colour detection (use when staff wear a distinct uniform)"
    ),
)
parser.add_argument(
    "--profile", type=Path, default=None,
    help="Path to a colour profile JSON built by calibrate.py (uniform mode only)",
)
args, _ = parser.parse_known_args()
DEVICE_ARG   = args.device
MODE         = args.mode
PROFILE_PATH = args.profile

# Torch device
TORCH_DEVICE = torch.device(
    "cuda" if (DEVICE_ARG in ("cuda", "jetson") and torch.cuda.is_available()) else "cpu"
)
print(f"[Config] --device={DEVICE_ARG}  --mode={MODE}  "
      f"profile={PROFILE_PATH}  torch={TORCH_DEVICE}")

# ─── YOLOv8 ──────────────────────────────────────────────────────────────────

YOLO_ENGINE = "yolov8n.engine"
YOLO_PT     = "yolov8n.pt"


def load_yolo() -> YOLO:
    if DEVICE_ARG == "jetson" and Path(YOLO_ENGINE).exists():
        print(f"[Jetson] Loading TensorRT engine: {YOLO_ENGINE}")
        return YOLO(YOLO_ENGINE)
    if DEVICE_ARG == "jetson":
        print(f"[Jetson] No .engine found, falling back to {YOLO_PT}")
    return YOLO(YOLO_PT)


yolo_model = load_yolo()

# ─── Mode: reid — ResNet50 embeddings ────────────────────────────────────────

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()   # strip classifier → 2048-d feature vector
resnet.eval().to(TORCH_DEVICE)

embed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

STAFF_REFS_DIR       = Path("staff_refs")
SIMILARITY_THRESHOLD = 0.75
STAFF_REFS_DIR.mkdir(exist_ok=True)

staff_embeddings: list[tuple[str, torch.Tensor]] = []


def get_embedding(crop_bgr: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    t   = embed_transform(pil).unsqueeze(0).to(TORCH_DEVICE)
    with torch.no_grad():
        emb = resnet(t)
    return F.normalize(emb, dim=1)


def load_staff_embeddings() -> None:
    global staff_embeddings
    staff_embeddings = []
    for p in sorted(STAFF_REFS_DIR.glob("*.jpg")):
        crop = cv2.imread(str(p))
        if crop is not None:
            staff_embeddings.append((p.stem, get_embedding(crop)))
    print(f"[reid] Loaded {len(staff_embeddings)} staff embedding(s)")


def is_staff_reid(crop_bgr: np.ndarray) -> bool:
    """True if crop's embedding is close enough to any saved staff reference."""
    if not staff_embeddings or crop_bgr.size == 0:
        return False
    emb = get_embedding(crop_bgr)
    return any(
        float((emb * ref).sum()) >= SIMILARITY_THRESHOLD
        for _, ref in staff_embeddings
    )


# ─── Mode: uniform — HSV colour detection ────────────────────────────────────
#
# Colour bands are either:
#   a) Loaded from a JSON profile built by calibrate.py  (recommended)
#   b) Set manually via the Setup tab sliders            (fallback / fine-tuning)
#
# Each band: { name, h_min, h_max, s_min, v_min }
# All values: OpenCV HSV scale  H 0-179 | S 0-255 | V 0-255

# Built-in fallback bands (green + yellow supermarket uniform)
_DEFAULT_BANDS: list[dict] = [
    {"name": "green",  "h_min": 40, "h_max": 85, "s_min": 80,  "v_min": 60},
    {"name": "yellow", "h_min": 18, "h_max": 38, "s_min": 100, "v_min": 130},
]

uniform_bands:    list[dict] = list(_DEFAULT_BANDS)
uniform_coverage: int        = 15    # % of crop pixels that must match

# Active profile name shown in the UI
_active_profile: str = "built-in defaults"


def load_profile(path: Path) -> str:
    """Load a colour profile JSON saved by calibrate.py. Returns status string."""
    global uniform_bands, uniform_coverage, _active_profile
    if not path.exists():
        return f"Profile not found: {path}"
    data              = json.loads(path.read_text())
    uniform_bands     = data.get("bands", _DEFAULT_BANDS)
    uniform_coverage  = data.get("coverage_pct", 15)
    _active_profile   = data.get("name", path.stem)
    print(f"[uniform] Loaded profile '{_active_profile}' "
          f"({len(uniform_bands)} band(s)) from {path}")
    return (f"Profile loaded: '{_active_profile}'  "
            f"| {len(uniform_bands)} colour band(s)  "
            f"| coverage ≥ {uniform_coverage}%")


def build_uniform_mask(hsv: np.ndarray) -> np.ndarray:
    """OR-combine masks for every active colour band."""
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for b in uniform_bands:
        m = cv2.inRange(
            hsv,
            np.array([b["h_min"], b["s_min"], b["v_min"]]),
            np.array([b["h_max"], 255,         255]),
        )
        combined = cv2.bitwise_or(combined, m)
    return combined


def is_staff_uniform(crop_bgr: np.ndarray) -> bool:
    """True if enough crop pixels match any active colour band."""
    if crop_bgr.size == 0:
        return False
    hsv      = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask     = build_uniform_mask(hsv)
    coverage = 100.0 * cv2.countNonZero(mask) / mask.size
    return coverage >= uniform_coverage


# Load profile from CLI if provided
if PROFILE_PATH and MODE == "uniform":
    load_profile(PROFILE_PATH)


# ─── Unified is_staff dispatcher ─────────────────────────────────────────────

def is_staff(crop_bgr: np.ndarray) -> bool:
    return is_staff_uniform(crop_bgr) if MODE == "uniform" else is_staff_reid(crop_bgr)


# ─── Shared: first-frame extraction ──────────────────────────────────────────

_first_frame: np.ndarray | None = None   # cached for uniform preview
detected_persons: list[tuple[tuple[int, int, int, int], np.ndarray]] = []


def extract_first_frame(video_path: str | None) -> tuple:
    """Extract first frame and run YOLOv8 person detection. Returns (annotated_rgb, msg)."""
    global _first_frame, detected_persons
    detected_persons = []
    _first_frame     = None

    if video_path is None:
        return None, "Please upload a video file."

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, "Could not read the video file."

    _first_frame = frame.copy()
    results      = yolo_model(frame, classes=[0], verbose=False)[0]
    annotated    = frame.copy()

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2].copy()
        detected_persons.append(((x1, y1, x2, y2), crop))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 210, 255), 2)
        cv2.putText(annotated, str(i + 1), (x1 + 4, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 255), 2)

    msg = f"Detected {len(detected_persons)} person(s)."
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), msg


# ─── Setup: reid mode ─────────────────────────────────────────────────────────

def reid_save_staff(staff_input: str) -> str:
    if not detected_persons:
        return "No persons detected yet — extract a frame first."
    for f in STAFF_REFS_DIR.glob("*.jpg"):
        f.unlink()
    saved = 0
    for tok in staff_input.split(","):
        tok = tok.strip()
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(detected_persons):
                _, crop = detected_persons[idx]
                cv2.imwrite(str(STAFF_REFS_DIR / f"staff_{idx + 1:03d}.jpg"), crop)
                saved += 1
    load_staff_embeddings()
    return f"Staff profiles saved: {saved} person(s)"


# ─── Setup: uniform mode ──────────────────────────────────────────────────────

def uniform_preview(  # kept for internal use only — UI uses _uniform_preview_simple
    green_h_min: int, green_h_max: int, green_s_min: int, green_v_min: int,
    yellow_h_min: int, yellow_h_max: int, yellow_s_min: int, yellow_v_min: int,
    coverage_pct: int,
) -> tuple:
    """
    Update the global uniform config from slider values and return:
      - annotated frame (uniform pixels highlighted in white overlay)
      - per-person classification preview with Staff/Customer labels
      - status message
    """
    # Update global config
    uniform_cfg.update({
        "green_h_min":  green_h_min,  "green_h_max":  green_h_max,
        "green_s_min":  green_s_min,  "green_v_min":  green_v_min,
        "yellow_h_min": yellow_h_min, "yellow_h_max": yellow_h_max,
        "yellow_s_min": yellow_s_min, "yellow_v_min": yellow_v_min,
        "coverage_pct": coverage_pct,
    })

    if _first_frame is None:
        return None, None, "Extract a frame first."

    frame = _first_frame.copy()
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask  = build_uniform_mask(hsv)

    # ── Image 1: full-frame mask overlay (white highlight on matched pixels) ──
    overlay         = frame.copy()
    overlay[mask > 0] = [255, 255, 255]
    mask_preview    = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    # ── Image 2: per-person classification with current config ────────────────
    person_preview  = frame.copy()
    staff_count = customer_count = 0
    for (x1, y1, x2, y2), crop in detected_persons:
        if is_staff_uniform(crop):
            label, color = "Staff",    (200,  60,  0)
            staff_count += 1
        else:
            label, color = "Customer", (0,   180, 50)
            customer_count += 1
        cv2.rectangle(person_preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(person_preview, label, (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    status = (
        f"Uniform mask active. "
        f"Staff detected: {staff_count} | Customers: {customer_count}\n"
        f"Adjust sliders if any staff are missed or customers are misclassified."
    )
    return (
        cv2.cvtColor(mask_preview,   cv2.COLOR_BGR2RGB),
        cv2.cvtColor(person_preview, cv2.COLOR_BGR2RGB),
        status,
    )


# ─── Detection (shared for both modes) ───────────────────────────────────────

BLUE_BGR  = (200,  60,  0)   # Staff label colour
GREEN_BGR = (  0, 180, 50)   # Customer label colour


def run_detection(video_path: str | None):
    """Generator: streams annotated frames + counts. Writes CSV on completion."""
    if video_path is None:
        yield None, "Please upload a video file.", None
        return

    if MODE == "reid":
        load_staff_embeddings()

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    csv_rows: list[dict]  = []
    total_staff = total_customers = frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_idx / fps, 3)
        frame_staff = frame_customers = 0
        results   = yolo_model(frame, classes=[0], verbose=False)[0]
        annotated = frame.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if is_staff(crop):
                label, color = "Staff",    BLUE_BGR
                frame_staff += 1
            else:
                label, color = "Customer", GREEN_BGR
                frame_customers += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1 + 4, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        total_staff     += frame_staff
        total_customers += frame_customers
        csv_rows.append({
            "timestamp": timestamp, "frame": frame_idx,
            "customer_count": frame_customers, "staff_count": frame_staff,
        })

        yield (
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            f"Frame {frame_idx + 1} / {total_frames}\n"
            f"Staff (cumulative):    {total_staff}\n"
            f"Customers (cumulative): {total_customers}",
            None,
        )
        frame_idx += 1

    cap.release()

    csv_path = "detection_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "frame", "customer_count", "staff_count"])
        writer.writeheader()
        writer.writerows(csv_rows)

    yield (
        None,
        f"Done — {frame_idx} frames processed.\n"
        f"Total Staff: {total_staff}  |  Total Customers: {total_customers}\n"
        f"CSV saved → {csv_path}",
        csv_path,
    )


# ─── Gradio UI ────────────────────────────────────────────────────────────────

MODE_LABEL = "Uniform Colour" if MODE == "uniform" else "Re-ID (ResNet50)"

with gr.Blocks(title="Staff vs Customer Detector") as demo:

    gr.Markdown(
        f"# Staff vs Customer Detector\n"
        f"YOLOv8 · **Mode: {MODE_LABEL}** · Device: **{DEVICE_ARG.upper()}**"
    )

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with gr.Tab("Setup"):

        if MODE == "uniform":
            gr.Markdown(
                "### Uniform Colour Mode\n"
                "**Recommended:** Run `calibrate.py` once per shop to build a colour profile, "
                "then load it here.\n"
                "```\npython calibrate.py --video footage.mp4 --profile profiles/shop1.json\n```\n"
                "1. Load a saved profile below (or use built-in defaults)\n"
                "2. Upload a video and click **Extract Frame**\n"
                "3. Click **Preview Mask** to verify — white pixels should cover uniforms\n"
                "4. Adjust coverage threshold if needed, then run detection"
            )
        else:
            gr.Markdown(
                "### Re-ID Mode\n"
                "1. Upload a video and click **Extract Frame**\n"
                "2. Note the number on each staff member\n"
                "3. Type those numbers and click **Save Staff Profiles**"
            )

        setup_video  = gr.Video(label="CCTV Video", sources=["upload"])
        extract_btn  = gr.Button("Extract Frame & Detect Persons", variant="primary")

        with gr.Row():
            setup_image = gr.Image(label="Detected Persons", type="numpy", height=380)
            setup_msg   = gr.Textbox(label="Status", lines=3, interactive=False)

        extract_btn.click(extract_first_frame,
                          inputs=setup_video,
                          outputs=[setup_image, setup_msg])

        # ── Uniform-mode controls ─────────────────────────────────────────────
        if MODE == "uniform":

            # ── Profile loader ────────────────────────────────────────────────
            gr.Markdown("#### Load a Colour Profile (built by calibrate.py)")
            with gr.Row():
                profile_input = gr.Textbox(
                    label="Profile JSON path",
                    placeholder="profiles/shop1.json",
                    value=str(PROFILE_PATH) if PROFILE_PATH else "",
                    scale=4,
                )
                load_profile_btn = gr.Button("Load Profile", scale=1)
            profile_status = gr.Textbox(
                label="Active profile",
                value=f"Profile: {_active_profile}  | {len(uniform_bands)} band(s)"
                      f" | coverage ≥ {uniform_coverage}%",
                interactive=False, lines=1,
            )

            def _load_profile_ui(path_str: str) -> str:
                if not path_str.strip():
                    return "No path entered."
                return load_profile(Path(path_str.strip()))

            load_profile_btn.click(_load_profile_ui,
                                   inputs=profile_input,
                                   outputs=profile_status)

            gr.Markdown(
                "---\n"
                "#### Preview & Fine-tune  \n"
                "Extract a frame above, then click **Preview** to see how well "
                "the loaded profile detects the uniform on this footage. "
                "Adjust coverage threshold if needed."
            )

            coverage_slider = gr.Slider(
                1, 60, value=uniform_coverage, step=1,
                label="Min coverage % (lower = more sensitive, higher = fewer false positives)",
            )
            preview_btn = gr.Button("Preview Mask on Frame", variant="secondary")

            with gr.Row():
                mask_img   = gr.Image(label="Colour Mask (white = matched pixels)",
                                      type="numpy", height=340)
                person_img = gr.Image(label="Classification Preview",
                                      type="numpy", height=340)
            preview_msg = gr.Textbox(label="Preview Status", lines=2, interactive=False)

            def _uniform_preview_simple(cov: int) -> tuple:
                """Preview with current loaded bands + updated coverage threshold."""
                global uniform_coverage
                uniform_coverage = cov
                if _first_frame is None:
                    return None, None, "Extract a frame first."
                frame = _first_frame.copy()
                hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask  = build_uniform_mask(hsv)

                # Mask overlay
                overlay          = frame.copy()
                overlay[mask > 0] = [255, 255, 255]
                mask_preview     = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

                # Per-person classification
                person_preview = frame.copy()
                staff_n = customer_n = 0
                for (x1, y1, x2, y2), crop in detected_persons:
                    if is_staff_uniform(crop):
                        label, color = "Staff",    (200, 60, 0)
                        staff_n += 1
                    else:
                        label, color = "Customer", (0, 180, 50)
                        customer_n += 1
                    cv2.rectangle(person_preview, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(person_preview, label, (x1 + 4, y1 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                status = (
                    f"Profile: '{_active_profile}' | "
                    f"Staff: {staff_n} | Customers: {customer_n} | "
                    f"Coverage threshold: {cov}%"
                )
                return (cv2.cvtColor(mask_preview,   cv2.COLOR_BGR2RGB),
                        cv2.cvtColor(person_preview, cv2.COLOR_BGR2RGB),
                        status)

            preview_btn.click(_uniform_preview_simple,
                              inputs=coverage_slider,
                              outputs=[mask_img, person_img, preview_msg])

        # ── Reid-mode controls ────────────────────────────────────────────────
        else:
            staff_nums = gr.Textbox(
                label="Staff Person Numbers (comma-separated)",
                placeholder="e.g.  2, 3, 5",
            )
            save_btn = gr.Button("Save Staff Profiles", variant="secondary")
            save_msg = gr.Textbox(label="Result", lines=1, interactive=False)
            save_btn.click(reid_save_staff, inputs=staff_nums, outputs=save_msg)

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with gr.Tab("Run Detection"):
        gr.Markdown(
            f"Process video frame-by-frame using **{MODE_LABEL}** mode.  \n"
            "**Staff → blue box** | **Customer → green box**"
        )
        det_video  = gr.Video(label="CCTV Video", sources=["upload"])
        run_btn    = gr.Button("Start Detection", variant="primary")

        with gr.Row():
            det_image  = gr.Image(label="Live Feed", type="numpy", height=400)
            det_status = gr.Textbox(label="Running Counts", lines=6, interactive=False)

        csv_file = gr.File(label="Download Results CSV")

        run_btn.click(run_detection,
                      inputs=det_video,
                      outputs=[det_image, det_status, csv_file])

# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if MODE == "reid":
        load_staff_embeddings()
    print(f"[Ready] Mode={MODE}  Device={DEVICE_ARG}")
    demo.launch()
