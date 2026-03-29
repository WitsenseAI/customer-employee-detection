"""
Staff vs Customer Detection Pipeline
CCTV footage analysis: YOLOv8 person detection + ResNet50 cosine-similarity re-ID.

Usage:
    python app.py --device cpu       # default
    python app.py --device cuda      # GPU
    python app.py --device jetson    # TensorRT .engine if present, else .pt
"""

import argparse
import csv
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
    "--device",
    choices=["cpu", "cuda", "jetson"],
    default="cpu",
    help="Inference device: cpu | cuda | jetson (TensorRT .engine if available)",
)
# parse_known_args avoids conflicts when Gradio is invoked via `gradio app.py`
args, _ = parser.parse_known_args()
DEVICE_ARG = args.device

# Torch device: Jetson and CUDA both use GPU; fall back to CPU if unavailable
if DEVICE_ARG in ("cuda", "jetson"):
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    TORCH_DEVICE = torch.device("cpu")

print(f"[Config] --device={DEVICE_ARG}  |  torch.device={TORCH_DEVICE}")

# ─── YOLOv8 loading ──────────────────────────────────────────────────────────

YOLO_ENGINE = "yolov8n.engine"   # TensorRT export for Jetson
YOLO_PT     = "yolov8n.pt"       # Standard PyTorch weights (auto-downloaded)


def load_yolo() -> YOLO:
    """Load YOLOv8 — TensorRT engine on Jetson if available, else PyTorch."""
    if DEVICE_ARG == "jetson" and Path(YOLO_ENGINE).exists():
        print(f"[Jetson] Loading TensorRT engine: {YOLO_ENGINE}")
        return YOLO(YOLO_ENGINE)
    if DEVICE_ARG == "jetson":
        print(f"[Jetson] No .engine found, falling back to {YOLO_PT}")
    return YOLO(YOLO_PT)


yolo_model = load_yolo()

# ─── ResNet50 embedding model ─────────────────────────────────────────────────

# Strip the classification head → 2048-d feature vector per person crop
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet.eval().to(TORCH_DEVICE)

embed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Staff reference store ────────────────────────────────────────────────────

STAFF_REFS_DIR      = Path("staff_refs")
SIMILARITY_THRESHOLD = 0.75          # Cosine similarity cut-off for staff match

STAFF_REFS_DIR.mkdir(exist_ok=True)

# Runtime list of (name, L2-normalised embedding) tuples
staff_embeddings: list[tuple[str, torch.Tensor]] = []


def get_embedding(crop_bgr: np.ndarray) -> torch.Tensor:
    """Return L2-normalised ResNet50 embedding for a BGR image crop."""
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    t   = embed_transform(pil).unsqueeze(0).to(TORCH_DEVICE)
    with torch.no_grad():
        emb = resnet(t)
    return F.normalize(emb, dim=1)   # unit vector → dot product == cosine similarity


def is_staff(crop_bgr: np.ndarray) -> bool:
    """Return True if the crop matches any saved staff reference."""
    if not staff_embeddings or crop_bgr.size == 0:
        return False
    emb = get_embedding(crop_bgr)
    return any(
        float((emb * ref_emb).sum()) >= SIMILARITY_THRESHOLD
        for _, ref_emb in staff_embeddings
    )


def load_staff_embeddings() -> None:
    """Read all staff_refs/*.jpg from disk and cache their embeddings."""
    global staff_embeddings
    staff_embeddings = []
    for p in sorted(STAFF_REFS_DIR.glob("*.jpg")):
        crop = cv2.imread(str(p))
        if crop is not None:
            staff_embeddings.append((p.stem, get_embedding(crop)))
    print(f"[Info] Loaded {len(staff_embeddings)} staff reference embedding(s)")


# ─── Tab 1 — Setup logic ──────────────────────────────────────────────────────

# Holds detected person boxes/crops from the last setup frame
detected_persons: list[tuple[tuple[int, int, int, int], np.ndarray]] = []


def setup_extract_frame(video_path: str | None):
    """
    Extract the first frame, run YOLOv8 person detection, and draw numbered
    bounding boxes so the user can identify which persons are staff.
    """
    global detected_persons
    detected_persons = []

    if video_path is None:
        return None, "Please upload a video file."

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, "Could not read the video file."

    # Detect persons (class 0 in COCO)
    results   = yolo_model(frame, classes=[0], verbose=False)[0]
    annotated = frame.copy()

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Clamp coordinates to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2].copy()
        detected_persons.append(((x1, y1, x2, y2), crop))

        # Yellow box + 1-based index number
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 210, 255), 2)
        cv2.putText(annotated, str(i + 1), (x1 + 4, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 255), 2)

    msg = (f"Detected {len(detected_persons)} person(s). "
           "Enter the person number(s) you want to mark as staff.")
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), msg


def setup_save_staff(staff_input: str) -> str:
    """
    Save person crops selected by the user as staff reference images, then
    rebuild the in-memory embedding cache.
    """
    if not detected_persons:
        return "No persons detected yet — extract a frame first."

    # Remove previous references before saving the new set
    for f in STAFF_REFS_DIR.glob("*.jpg"):
        f.unlink()

    saved = 0
    for tok in staff_input.split(","):
        tok = tok.strip()
        if not tok.isdigit():
            continue
        idx = int(tok) - 1   # convert 1-based input to 0-based index
        if 0 <= idx < len(detected_persons):
            _, crop = detected_persons[idx]
            cv2.imwrite(str(STAFF_REFS_DIR / f"staff_{idx + 1:03d}.jpg"), crop)
            saved += 1

    load_staff_embeddings()
    return f"Staff profiles saved: {saved} person(s)"


# ─── Tab 2 — Detection logic ──────────────────────────────────────────────────

# BGR colours for on-screen labels
BLUE_BGR  = (200,  60,   0)   # Staff
GREEN_BGR = (  0, 180,  50)   # Customer


def run_detection(video_path: str | None):
    """
    Generator that streams annotated frames + running counts to Gradio.
    Writes a CSV on completion and yields its path as a downloadable file.
    """
    if video_path is None:
        yield None, "Please upload a video file.", None
        return

    load_staff_embeddings()   # pick up any staff profiles saved in Setup tab

    cap         = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    csv_rows: list[dict] = []
    total_staff = total_customers = frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp       = round(frame_idx / fps, 3)
        frame_staff     = 0
        frame_customers = 0

        results   = yolo_model(frame, classes=[0], verbose=False)[0]
        annotated = frame.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop   = frame[y1:y2, x1:x2]

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
            "timestamp":      timestamp,
            "frame":          frame_idx,
            "customer_count": frame_customers,
            "staff_count":    frame_staff,
        })

        status = (
            f"Frame {frame_idx + 1} / {total_frames}\n"
            f"Staff detections (cumulative):    {total_staff}\n"
            f"Customer detections (cumulative): {total_customers}"
        )

        yield cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), status, None
        frame_idx += 1

    cap.release()

    # Write summary CSV
    csv_path = "detection_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "frame", "customer_count", "staff_count"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    final_status = (
        f"Done — {frame_idx} frame(s) processed.\n"
        f"Total Staff detections:    {total_staff}\n"
        f"Total Customer detections: {total_customers}\n"
        f"CSV saved → {csv_path}"
    )
    yield None, final_status, csv_path


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Staff vs Customer Detector") as demo:

    gr.Markdown(
        f"# Staff vs Customer Detector\n"
        f"YOLOv8 person detection · ResNet50 re-ID · "
        f"Device: **{DEVICE_ARG.upper()}** ({TORCH_DEVICE})"
    )

    # ── Tab 1: Setup ──────────────────────────────────────────────────────────
    with gr.Tab("Setup – Label Staff"):
        gr.Markdown(
            "1. Upload a video  \n"
            "2. Click **Extract Frame**  \n"
            "3. Type the number(s) of staff persons  \n"
            "4. Click **Save Staff Profiles**"
        )

        setup_video = gr.Video(label="CCTV Video", sources=["upload"])
        extract_btn = gr.Button("Extract Frame & Detect Persons", variant="primary")

        with gr.Row():
            setup_image = gr.Image(label="Detected Persons (numbered)",
                                   type="numpy", height=400)
            setup_msg   = gr.Textbox(label="Status", lines=4, interactive=False)

        staff_nums = gr.Textbox(
            label='Staff Person Numbers (comma-separated)',
            placeholder="e.g.  1, 3",
        )
        save_btn = gr.Button("Save Staff Profiles", variant="secondary")
        save_msg = gr.Textbox(label="Result", lines=1, interactive=False)

        extract_btn.click(
            fn=setup_extract_frame,
            inputs=setup_video,
            outputs=[setup_image, setup_msg],
        )
        save_btn.click(
            fn=setup_save_staff,
            inputs=staff_nums,
            outputs=save_msg,
        )

    # ── Tab 2: Run Detection ──────────────────────────────────────────────────
    with gr.Tab("Run Detection"):
        gr.Markdown(
            "Upload the same (or a different) video and click **Start Detection**. "
            "Persons are labelled **Staff** (blue) or **Customer** (green) in real time."
        )

        det_video = gr.Video(label="CCTV Video", sources=["upload"])
        run_btn   = gr.Button("Start Detection", variant="primary")

        with gr.Row():
            det_image  = gr.Image(label="Live Detection Feed",
                                  type="numpy", height=400)
            det_status = gr.Textbox(label="Running Counts", lines=6, interactive=False)

        csv_file = gr.File(label="Download Results CSV")

        run_btn.click(
            fn=run_detection,
            inputs=det_video,
            outputs=[det_image, det_status, csv_file],
        )

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_staff_embeddings()   # load any previously saved staff profiles on startup
    demo.launch()
