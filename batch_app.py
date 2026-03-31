"""
Staff vs Customer — Batch Video Processor
==========================================
Setup tab   : Upload video → auto-picks a random frame → YOLO person detection →
              user selects staff numbers + enters uniform description → save profiles.

Process tab : Upload video → runs YOLOv8 track on every frame → draws bounding boxes
              and live staff/customer counts on each frame → saves annotated output
              video → user downloads it.

Usage:
    python batch_app.py
    python batch_app.py --device cuda
"""

import argparse
import random
import tempfile
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

yolo_model = YOLO("yolov8n.pt")

DETECT_CONF = 0.4
DETECT_IOU  = 0.4

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

# ─── Shared setup state ───────────────────────────────────────────────────────

STAFF_REFS_DIR = Path("staff_refs")
STAFF_REFS_DIR.mkdir(exist_ok=True)

_uniform_description: str = ""
_setup_persons: list[tuple[tuple, np.ndarray]] = []


# ─── Tab 1 — Setup ────────────────────────────────────────────────────────────

def setup_auto_frame(video_path: str | None):
    """
    Pick a random frame from the video (between 10 % and 90 % of total length
    to avoid blank intros/outros), run YOLO person detection, return numbered
    annotated image and status message.
    """
    global _setup_persons
    _setup_persons = []

    if video_path is None:
        return None, "Upload a video first."

    cap         = cv2.VideoCapture(video_path)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None, "Could not read frame count."

    # Pick a random frame between 10 % and 90 % of the video
    pick = random.randint(max(0, int(total * 0.10)), min(total - 1, int(total * 0.90)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, pick)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, f"Could not read frame {pick}."

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

    pct = round(pick / total * 100)
    return (
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        f"Auto-selected frame {pick}/{total} ({pct} %).  "
        f"Detected {len(_setup_persons)} person(s).  "
        "Enter staff number(s) and uniform description below, then click Save.",
    )


def setup_save(staff_nums: str, description: str) -> str:
    global _uniform_description

    if not _setup_persons:
        return "Run auto-frame detection first."
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
             "text": f"Is this person wearing {_uniform_description}? Answer only yes or no."},
        ],
    }]
    prompt = _vlm_proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _vlm_proc(text=prompt, images=[pil_img], return_tensors="pt").to(TORCH_DEVICE)
    with torch.no_grad():
        out = _vlm_model.generate(**inputs, max_new_tokens=5, do_sample=False)
    new_toks = out[0][inputs["input_ids"].shape[1]:]
    answer   = _vlm_proc.decode(new_toks, skip_special_tokens=True).strip().lower()
    return answer.startswith("yes")


# ─── Tab 2 — Process & Download ───────────────────────────────────────────────

_BLUE   = (255,   0,   0)
_GREEN  = (  0, 180,  50)
_WHITE  = (255, 255, 255)
_BLACK  = (  0,   0,   0)

COUNT_INTERVAL_SEC = 10


def _draw_count_overlay(frame: np.ndarray, staff: int, customers: int) -> None:
    """Burn staff/customer counts into the top-right corner of frame (in-place)."""
    h, w = frame.shape[:2]
    lines = [
        (f"Staff    : {staff}",    _BLUE),
        (f"Customers: {customers}", _GREEN),
    ]
    box_w, box_h, pad = 260, 70, 10
    x0, y0 = w - box_w - pad, pad
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), _BLACK, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (80, 80, 80), 1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame, text, (x0 + 8, y0 + 22 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
        )


def process_video(video_path: str | None):
    """
    Generator — yields (preview_frame_rgb, progress_text) while processing,
    then yields (None, final_message, output_video_path) when done.
    """
    if video_path is None:
        yield None, "Upload a video first.", None
        return

    if not _uniform_description:
        yield None, "Complete Setup first (uniform description is missing).", None
        return

    # Reset ByteTrack
    if yolo_model.predictor is not None:
        yolo_model.predictor = None

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video file
    out_path = tempfile.mktemp(suffix="_annotated.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    staff_set:    set[int] = set()
    customer_set: set[int] = set()
    classified:   set[int] = set()

    count_every    = max(1, int(fps * COUNT_INTERVAL_SEC))
    display_staff  = 0
    display_cust   = 0
    frame_idx      = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results      = yolo_model.track(
            frame, classes=[0], conf=DETECT_CONF, iou=DETECT_IOU,
            persist=True, verbose=False,
        )[0]
        annotated    = frame.copy()
        current_ids: set[int] = set()

        for box in results.boxes:
            if box.id is None:
                continue
            tid = int(box.id)
            current_ids.add(tid)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if tid not in classified:
                classified.add(tid)
                if _is_staff_vlm(crop):
                    staff_set.add(tid)
                else:
                    customer_set.add(tid)

            if tid in staff_set:
                label, color = f"Staff #{tid}",    _BLUE
            else:
                label, color = f"Customer #{tid}", _GREEN

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1 + 4, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Remove lost tracks
        lost = (staff_set | customer_set) - current_ids
        staff_set    -= lost
        customer_set -= lost

        # Refresh counts every COUNT_INTERVAL_SEC seconds
        if frame_idx % count_every == 0:
            display_staff = len(staff_set)
            display_cust  = len(customer_set)

        # Burn count overlay onto the frame
        _draw_count_overlay(annotated, display_staff, display_cust)

        writer.write(annotated)

        # Stream a preview every 5 frames so the UI feels live
        pct = round(frame_idx / max(total, 1) * 100)
        if frame_idx % 5 == 0:
            yield (
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                f"Processing … {frame_idx}/{total} frames ({pct} %)",
                None,
            )

        frame_idx += 1

    cap.release()
    writer.release()

    yield (
        None,
        f"Done — {frame_idx} frames processed.  Download the annotated video below.",
        out_path,
    )


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Staff vs Customer — Batch Processor", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Staff vs Customer — Batch Video Processor")

    # ── Tab 1: Setup ──────────────────────────────────────────────────────────
    with gr.Tab("Setup"):
        gr.Markdown(
            "**Step 1** Upload the video — a random frame is picked automatically  \n"
            "**Step 2** Note the numbered persons, enter the staff number(s)  \n"
            "**Step 3** Describe their uniform and click *Save*"
        )
        setup_video = gr.Video(label="Upload Video", sources=["upload"])
        auto_btn    = gr.Button("Auto-Pick Frame & Detect Persons", variant="primary")

        with gr.Row():
            setup_image  = gr.Image(label="Auto-selected Frame (numbered)", type="numpy", height=420)
            with gr.Column():
                setup_status = gr.Textbox(label="Status", lines=3, interactive=False)
                staff_nums   = gr.Textbox(
                    label="Staff person number(s) — comma-separated",
                    placeholder="e.g.  1, 3",
                )
                uniform_desc = gr.Textbox(
                    label="Uniform description",
                    placeholder="e.g.  yellow and green t-shirt with dark trousers",
                    lines=3,
                )
                save_btn    = gr.Button("Save Staff Profiles", variant="secondary")
                save_status = gr.Textbox(label="", lines=1, interactive=False)

        auto_btn.click(fn=setup_auto_frame, inputs=setup_video,
                       outputs=[setup_image, setup_status])
        save_btn.click(fn=setup_save, inputs=[staff_nums, uniform_desc],
                       outputs=save_status)

    # ── Tab 2: Process & Download ─────────────────────────────────────────────
    with gr.Tab("Process & Download"):
        gr.Markdown(
            "Upload the video and click **Process Video**.  \n"
            "Every frame is annotated with bounding boxes and a live count overlay.  \n"
            "When done, download the fully annotated video."
        )
        proc_video   = gr.Video(label="Upload Video", sources=["upload"])
        process_btn  = gr.Button("Process Video", variant="primary", size="lg")

        with gr.Row():
            preview    = gr.Image(label="Processing Preview", type="numpy",
                                  height=420, show_label=True)
            with gr.Column():
                progress   = gr.Textbox(label="Progress", lines=2, interactive=False)
                output_vid = gr.File(label="Download Annotated Video")

        process_btn.click(
            fn=process_video,
            inputs=proc_video,
            outputs=[preview, progress, output_vid],
        )

# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[Ready] device={DEVICE_ARG}")
    demo.launch()