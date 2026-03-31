"""
Staff vs Customer Detection Pipeline
=====================================
Setup tab  : extract one frame, pick staff persons, enter uniform description
Detection  : YOLOv8 track → classify new IDs with SmolVLM → maintain staff/customer sets
             Lost tracks are removed from the sets.
             Counts (= active track IDs in each set) refresh every 10 seconds.

Usage:
    python app.py                  # CPU
    python app.py --device cuda    # GPU
    python app.py --device jetson  # TensorRT engine if present
"""

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import Idefics3ForConditionalGeneration, Idefics3Processor
from ultralytics import YOLO

# ─── Device ───────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda", "jetson"], default="cpu")
args, _ = parser.parse_known_args()
DEVICE_ARG = args.device

TORCH_DEVICE = torch.device(
    "cuda" if DEVICE_ARG in ("cuda", "jetson") and torch.cuda.is_available() else "cpu"
)
print(f"[Config] device={DEVICE_ARG}  torch={TORCH_DEVICE}")

# ─── YOLOv8 ───────────────────────────────────────────────────────────────────

_ENGINE = "yolov8n.engine"
_PT     = "yolov8n.pt"


def _load_yolo() -> YOLO:
    if DEVICE_ARG == "jetson" and Path(_ENGINE).exists():
        print(f"[YOLO] Loading TensorRT engine: {_ENGINE}")
        return YOLO(_ENGINE)
    return YOLO(_PT)


yolo_model = _load_yolo()

# Detection params used in both tabs
DETECT_CONF = 0.4   # higher than default (0.25) — filters weak detections
DETECT_IOU  = 0.4   # lower than default (0.7)  — suppresses overlapping boxes

# ─── SmolVLM ──────────────────────────────────────────────────────────────────

_VLM_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
print(f"[VLM] Loading {_VLM_ID} …")
_vlm_proc  = Idefics3Processor.from_pretrained(_VLM_ID)
_vlm_model = Idefics3ForConditionalGeneration.from_pretrained(
    _VLM_ID,
    torch_dtype=torch.float16 if TORCH_DEVICE.type == "cuda" else torch.float32,
).to(TORCH_DEVICE)
_vlm_model.eval()
print("[VLM] Ready.")

# ─── Shared state (written in Setup, read in Detection) ───────────────────────

STAFF_REFS_DIR = Path("staff_refs")
STAFF_REFS_DIR.mkdir(exist_ok=True)

_uniform_description: str = ""          # set during Setup
_setup_persons: list[tuple[tuple, np.ndarray]] = []   # (bbox, crop) from first frame


# ─── Tab 1 — Setup ────────────────────────────────────────────────────────────

def setup_extract_frame(video_path: str | None):
    """
    Read the first frame of the uploaded video, run person detection,
    draw 1-based numbered boxes, return the annotated image and a status message.
    """
    global _setup_persons
    _setup_persons = []

    if video_path is None:
        return None, "Upload a video first."

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, "Could not read the video."

    results   = yolo_model.predict(
        frame, classes=[0], conf=DETECT_CONF, iou=DETECT_IOU, verbose=False
    )[0]
    annotated = frame.copy()

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2].copy()
        _setup_persons.append(((x1, y1, x2, y2), crop))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 210, 255), 2)
        cv2.putText(
            annotated, str(i + 1), (x1 + 4, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 210, 255), 2,
        )

    return (
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        f"Detected {len(_setup_persons)} person(s). "
        "Enter the number(s) of staff below, then click Save.",
    )


def setup_save_staff(staff_nums: str, description: str) -> str:
    """
    Save the chosen person crops as staff reference images and store
    the uniform description for use by the VLM during detection.
    """
    global _uniform_description

    if not _setup_persons:
        return "Extract a frame first."
    if not description.strip():
        return "Enter a uniform description."

    _uniform_description = description.strip()

    for f in STAFF_REFS_DIR.glob("*.jpg"):
        f.unlink()

    saved = 0
    for tok in staff_nums.split(","):
        tok = tok.strip()
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(_setup_persons):
                _, crop = _setup_persons[idx]
                cv2.imwrite(str(STAFF_REFS_DIR / f"staff_{idx + 1:03d}.jpg"), crop)
                saved += 1

    return f"Saved {saved} staff profile(s).  Uniform: \"{_uniform_description}\""


# ─── VLM staff check ──────────────────────────────────────────────────────────

def _is_staff_vlm(crop_bgr: np.ndarray) -> bool:
    """
    Ask SmolVLM whether the person crop matches the stored uniform description.
    Crops to the torso region (20 %–75 % of height) before querying.
    Returns True  → Staff,  False → Customer.
    """
    h     = crop_bgr.shape[0]
    torso = crop_bgr[int(h * 0.20): int(h * 0.75), :]
    if torso.size == 0:
        torso = crop_bgr

    pil_img  = Image.fromarray(cv2.cvtColor(torso, cv2.COLOR_BGR2RGB))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text",
             "text": (
                 f"Is this person wearing {_uniform_description}? "
                 "Answer only yes or no."
             )},
        ],
    }]
    prompt = _vlm_proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _vlm_proc(
        text=prompt, images=[pil_img], return_tensors="pt"
    ).to(TORCH_DEVICE)
    with torch.no_grad():
        out = _vlm_model.generate(**inputs, max_new_tokens=5, do_sample=False)
    new_toks = out[0][inputs["input_ids"].shape[1]:]
    answer   = _vlm_proc.decode(new_toks, skip_special_tokens=True).strip().lower()
    return answer.startswith("yes")


# ─── Tab 2 — Detection ────────────────────────────────────────────────────────

_BLUE  = (255,   0,   0)   # Staff     (blue  in BGR)
_GREEN = (  0, 180,  50)   # Customer  (green in BGR)
_GREY  = (150, 150, 150)   # Not yet classified

COUNT_INTERVAL_SEC = 10    # recalculate & refresh counts every N seconds


def _counts_html(staff: int, customers: int) -> str:
    return (
        "<div style='padding:16px 22px;background:#111827;border-radius:12px;"
        "color:white;font-size:24px;font-weight:700;line-height:2.2'>"
        f"<span style='color:#93c5fd'>&#128100;&nbsp;Staff &nbsp;&nbsp;&nbsp;: {staff}</span><br>"
        f"<span style='color:#6ee7b7'>&#128722;&nbsp;Customers : {customers}</span>"
        "</div>"
    )


def run_detection(video_path: str | None):
    """
    Generator — yields (annotated_frame, counts_html) for each frame.

    Algorithm
    ---------
    • Use model.track(persist=True) so ByteTrack maintains stable IDs.
    • When a track ID appears for the first time:
        – call _is_staff_vlm() on its crop
        – add the ID to staff_set  OR  customer_set
    • Build current_ids from boxes present in this frame.
    • Remove any IDs that are no longer visible (track lost) from both sets.
    • Every COUNT_INTERVAL_SEC seconds recalculate the displayed counts
      from len(staff_set) and len(customer_set).
    • Label drawn on each box: "Staff #<id>" or "Customer #<id>" or "#<id>" (pending).
    """
    if video_path is None:
        yield None, _counts_html(0, 0)
        return

    if not _uniform_description:
        yield None, "<p style='color:red;font-size:18px'>⚠ Complete Setup first.</p>"
        return

    # Hard-reset ByteTrack so IDs restart from 1 for every new run
    if yolo_model.predictor is not None:
        yolo_model.predictor = None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count_every = max(1, int(fps * COUNT_INTERVAL_SEC))

    staff_set:    set[int] = set()   # track IDs of persons currently in frame as staff
    customer_set: set[int] = set()   # track IDs of persons currently in frame as customer
    classified:   set[int] = set()   # IDs that have already been through _is_staff_vlm

    last_html = _counts_html(0, 0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results   = yolo_model.track(
            frame,
            classes=[0],
            conf=DETECT_CONF,
            iou=DETECT_IOU,
            persist=True,
            verbose=False,
        )[0]
        annotated    = frame.copy()
        current_ids: set[int] = set()

        for box in results.boxes:
            if box.id is None:
                continue                      # ByteTrack hasn't committed an ID yet
            tid = int(box.id)
            current_ids.add(tid)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ── Classify on first sight ───────────────────────────────────────
            if tid not in classified:
                classified.add(tid)
                if _is_staff_vlm(crop):
                    staff_set.add(tid)
                else:
                    customer_set.add(tid)

            # ── Draw box + label ──────────────────────────────────────────────
            if tid in staff_set:
                label, color = f"Staff #{tid}",    _BLUE
            elif tid in customer_set:
                label, color = f"Customer #{tid}", _GREEN
            else:
                label, color = f"#{tid}",          _GREY   # classifying…

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label, (x1 + 4, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
            )

        # ── Remove lost tracks ────────────────────────────────────────────────
        lost = (staff_set | customer_set) - current_ids
        staff_set    -= lost
        customer_set -= lost

        # ── Refresh counts every COUNT_INTERVAL_SEC seconds ───────────────────
        if frame_idx % count_every == 0:
            last_html = _counts_html(len(staff_set), len(customer_set))

        yield cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), last_html
        frame_idx += 1

    cap.release()
    # Final update with whatever is left visible in the last frame
    yield None, _counts_html(len(staff_set), len(customer_set))


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Staff vs Customer Detector", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Staff vs Customer Detector")

    # ── Tab 1: Setup ──────────────────────────────────────────────────────────
    with gr.Tab("Setup"):
        gr.Markdown(
            "**Step 1** Upload the CCTV video  \n"
            "**Step 2** Click *Extract Frame* — persons are numbered automatically  \n"
            "**Step 3** Type the number(s) of staff, describe their uniform, click *Save*"
        )

        setup_video = gr.Video(label="Upload Video", sources=["upload"])
        extract_btn = gr.Button("Extract Frame & Detect Persons", variant="primary")

        with gr.Row():
            setup_image = gr.Image(
                label="Detected Persons (numbered)",
                type="numpy", height=420,
            )
            with gr.Column():
                setup_status = gr.Textbox(
                    label="Status", lines=2, interactive=False,
                )
                staff_nums_box = gr.Textbox(
                    label="Staff person number(s) — comma-separated",
                    placeholder="e.g.  2, 4",
                )
                uniform_box = gr.Textbox(
                    label="Uniform description",
                    placeholder="e.g.  yellow and green t-shirt with dark trousers",
                    lines=3,
                )
                save_btn    = gr.Button("Save Staff Profiles", variant="secondary")
                save_status = gr.Textbox(label="", lines=1, interactive=False)

        extract_btn.click(
            fn=setup_extract_frame,
            inputs=setup_video,
            outputs=[setup_image, setup_status],
        )
        save_btn.click(
            fn=setup_save_staff,
            inputs=[staff_nums_box, uniform_box],
            outputs=save_status,
        )

    # ── Tab 2: Detection ──────────────────────────────────────────────────────
    with gr.Tab("Detection"):
        gr.Markdown(
            "Upload the video and click **Start Detection**.  \n"
            "Each tracked person shows their track ID.  "
            "**Blue = Staff · Green = Customer**  \n"
            "Counts refresh every 10 seconds and reflect persons currently visible."
        )

        det_video = gr.Video(label="Upload Video", sources=["upload"])
        run_btn   = gr.Button("Start Detection", variant="primary", size="lg")

        with gr.Row():
            live_feed  = gr.Image(
                label="Live Detection Feed", type="numpy",
                height=450, show_label=False,
            )
            count_box = gr.HTML(_counts_html(0, 0))

        run_btn.click(
            fn=run_detection,
            inputs=det_video,
            outputs=[live_feed, count_box],
        )

# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[Ready] device={DEVICE_ARG}")
    demo.launch()
