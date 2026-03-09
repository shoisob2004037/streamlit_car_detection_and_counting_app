"""
vehicle_counter_app.py
Run with:  streamlit run vehicle_counter_app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import io
import time
from datetime import datetime
from collections import deque
from ultralytics import YOLO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Counter",
    page_icon="🚗",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Dark background */
[data-testid="stAppViewContainer"] {
    background: #0e1117;
}
[data-testid="stSidebar"] {
    background: #161b26;
}

/* Metric cards */
.metric-card {
    background: #1c2233;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    border-left: 4px solid;
    margin-bottom: 10px;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #8899aa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.72rem;
    margin-top: 6px;
    color: #8899aa;
}

/* Direction badges */
.badge {
    display: inline-block;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 2px;
}
.badge-down { background:#143d1e; color:#4dff88; }
.badge-up   { background:#0d2244; color:#66aaff; }

/* Section headers */
.section-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #556677;
    margin: 14px 0 6px 0;
    border-bottom: 1px solid #1e2736;
    padding-bottom: 4px;
}

/* Sidebar text colors */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] .stMarkdown {
    color: #ccd6f0 !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00c2ff !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2a3550;
}
/* Slider label and value */
[data-testid="stSidebar"] [data-testid="stSlider"] label,
[data-testid="stSidebar"] [data-testid="stSlider"] p {
    color: #aabbcc !important;
}
/* Caption text */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #6677aa !important;
}
/* How it works bullet list */
[data-testid="stSidebar"] ul li {
    color: #99aacc !important;
    margin-bottom: 4px;
}
/* Sidebar headings */
[data-testid="stSidebar"] h2 {
    color: #00c2ff !important;
    font-size: 1.05rem;
    letter-spacing: 1px;
}
[data-testid="stSidebar"] h3 {
    color: #7ec8e3 !important;
}
[data-testid="stSidebar"] ul {
    padding-left: 18px;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Progress bar color */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00c2ff, #0066ff);
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #0066ff 0%, #00c2ff 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
}
div.stButton > button:hover {
    opacity: 0.88;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
TARGET_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

CLASS_COLORS_BGR = {
    2: (0, 230, 0),
    3: (0, 150, 255),
    5: (0, 220, 255),
    7: (255, 80, 200),
}

CLASS_COLORS_HEX = {
    2: "#00e600",
    3: "#ff9600",
    5: "#00dcff",
    7: "#ff50c8",
}

CLASS_EMOJI = {2: "🚗", 3: "🏍️", 5: "🚌", 7: "🚚"}

CROSSING_BUFFER = 3
TRACK_TTL       = 60


# ─────────────────────────────────────────────
#  PDF Report Generator
# ─────────────────────────────────────────────
def generate_pdf_report(counts_down, counts_up, total_frames,
                         video_name, confidence, line_pct):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm
    )

    # ── Colour palette ──
    DARK       = colors.HexColor("#0e1117")
    NAVY       = colors.HexColor("#1c2233")
    CYAN       = colors.HexColor("#00c2ff")
    GREEN      = colors.HexColor("#00e600")
    ORANGE     = colors.HexColor("#ff9600")
    YELLOW     = colors.HexColor("#00dcff")
    MAGENTA    = colors.HexColor("#ff50c8")
    WHITE      = colors.HexColor("#e0e8f0")
    MUTED      = colors.HexColor("#8899aa")
    DOWN_COLOR = colors.HexColor("#4dff88")
    UP_COLOR   = colors.HexColor("#66aaff")

    styles = getSampleStyleSheet()

    def style(name, **kwargs):
        return ParagraphStyle(name, **kwargs)

    title_style = style("Title",
        fontSize=26, textColor=CYAN, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=4)

    subtitle_style = style("Subtitle",
        fontSize=11, textColor=MUTED, alignment=TA_CENTER,
        fontName="Helvetica", spaceAfter=2)

    section_style = style("Section",
        fontSize=13, textColor=CYAN, fontName="Helvetica-Bold",
        spaceBefore=14, spaceAfter=6)

    body_style = style("Body",
        fontSize=10, textColor=WHITE, fontName="Helvetica",
        spaceAfter=4, leading=15)

    label_style = style("Label",
        fontSize=9, textColor=MUTED, fontName="Helvetica")

    story = []

    # ── Header ──
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("VEHICLE COUNTER", title_style))
    story.append(Paragraph("Automated Traffic Analysis Report", subtitle_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y  —  %H:%M:%S')}",
        subtitle_style))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=CYAN, spaceAfter=16))

    # ── Video Info ──
    story.append(Paragraph("Video Information", section_style))

    info_data = [
        ["Field", "Value"],
        ["File Name",        video_name],
        ["Total Frames",     str(total_frames)],
        ["Confidence Level", f"{int(confidence * 100)}%"],
        ["Counting Line",    f"{line_pct}% from top"],
        ["Processed On",     datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]

    info_table = Table(info_data, colWidths=[5*cm, 11*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  CYAN),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        ("BACKGROUND",   (0, 1), (-1, -1), DARK),
        ("TEXTCOLOR",    (0, 1), (0, -1),  MUTED),
        ("TEXTCOLOR",    (1, 1), (1, -1),  WHITE),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#141824"), colors.HexColor("#1a2030")]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#2a3550")),
        ("TOPPADDING",   (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Overall Summary ──
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#2a3550"), spaceAfter=10))
    story.append(Paragraph("Overall Summary", section_style))

    total_all  = sum(counts_down.values()) + sum(counts_up.values())
    total_down = sum(counts_down.values())
    total_up   = sum(counts_up.values())

    summary_data = [
        ["Metric", "Count", "Percentage"],
        ["Total Vehicles Counted", str(total_all), "100%"],
        ["Moving Down  (↓)",
         str(total_down),
         f"{(total_down/total_all*100):.1f}%" if total_all else "0%"],
        ["Moving Up    (↑)",
         str(total_up),
         f"{(total_up/total_all*100):.1f}%"   if total_all else "0%"],
    ]

    sum_table = Table(summary_data, colWidths=[8*cm, 4*cm, 4*cm])
    sum_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  10),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("BACKGROUND",    (0, 1), (-1, 1),  colors.HexColor("#0d2a1a")),
        ("TEXTCOLOR",     (0, 1), (-1, 1),  WHITE),
        ("FONTNAME",      (0, 1), (-1, 1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 2), (1, 2),   DOWN_COLOR),
        ("TEXTCOLOR",     (1, 3), (1, 3),   UP_COLOR),
        ("BACKGROUND",    (0, 2), (-1, 2),  colors.HexColor("#141824")),
        ("BACKGROUND",    (0, 3), (-1, 3),  colors.HexColor("#1a2030")),
        ("TEXTCOLOR",     (0, 2), (-1, -1), WHITE),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#2a3550")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(sum_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Per-Class Breakdown ──
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#2a3550"), spaceAfter=10))
    story.append(Paragraph("Breakdown by Vehicle Type", section_style))

    cls_colors_map = {
        2: GREEN, 3: ORANGE, 5: YELLOW, 7: MAGENTA
    }
    cls_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

    breakdown_data = [
        ["Vehicle Type", "Down (↓)", "Up (↑)", "Total", "Share %"]
    ]
    for cls in [2, 3, 5, 7]:
        d = counts_down[cls]
        u = counts_up[cls]
        t = d + u
        pct = f"{(t/total_all*100):.1f}%" if total_all else "0%"
        breakdown_data.append([cls_names[cls], str(d), str(u), str(t), pct])

    # Totals row
    breakdown_data.append([
        "TOTAL",
        str(total_down),
        str(total_up),
        str(total_all),
        "100%"
    ])

    col_w = [5*cm, 3*cm, 3*cm, 3*cm, 3*cm]
    brk_table = Table(breakdown_data, colWidths=col_w)

    row_styles = [
        ("BACKGROUND",    (0, 0), (-1, 0),   NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),   CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),   "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),   10),
        ("ALIGN",         (1, 0), (-1, -1),  "CENTER"),
        ("GRID",          (0, 0), (-1, -1),  0.5, colors.HexColor("#2a3550")),
        ("TOPPADDING",    (0, 0), (-1, -1),  8),
        ("BOTTOMPADDING", (0, 0), (-1, -1),  8),
        ("LEFTPADDING",   (0, 0), (-1, -1),  10),
        # Totals row
        ("BACKGROUND",    (0, -1), (-1, -1), NAVY),
        ("TEXTCOLOR",     (0, -1), (-1, -1), CYAN),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
    ]

    bg_rows = [colors.HexColor("#141824"), colors.HexColor("#1a2030")]
    for i, cls in enumerate([2, 3, 5, 7]):
        row = i + 1
        row_styles += [
            ("BACKGROUND", (0, row), (-1, row), bg_rows[i % 2]),
            ("TEXTCOLOR",  (0, row), (0, row),  cls_colors_map[cls]),
            ("FONTNAME",   (0, row), (0, row),  "Helvetica-Bold"),
            ("TEXTCOLOR",  (1, row), (-1, row), WHITE),
        ]

    brk_table.setStyle(TableStyle(row_styles))
    story.append(brk_table)
    story.append(Spacer(1, 0.8*cm))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#2a3550"), spaceAfter=8))
    story.append(Paragraph(
        "Built by <b>Mahadi Hasan Shaisob</b>  |  "
        "Powered by YOLO11 + BotSORT + Streamlit",
        style("Footer", fontSize=8, textColor=MUTED,
              fontName="Helvetica", alignment=TA_CENTER)))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model():
    return YOLO("yolo11s.pt")


def overlay_dashboard(frame, counts_down, counts_up, line_y, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 44), (10, 14, 22), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    def put(img, txt, pos, scale=0.55, color=(255,255,255), bold=False):
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2 if bold else 1, cv2.LINE_AA)

    put(frame, "VEHICLE COUNTER", (12, 28), scale=0.65,
        color=(0, 200, 255), bold=True)
    put(frame, f"FPS {fps:.1f}", (w - 110, 28), scale=0.55,
        color=(160, 160, 160))

    # Counting line (shadow + colour)
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 0), 4)
    cv2.line(frame, (0, line_y), (w, line_y), (0, 80, 255), 2)
    put(frame, "-- COUNTING LINE --", (10, line_y - 8),
        scale=0.48, color=(0, 120, 255))

    # Bottom-left live count chip
    chip_x, chip_y = 10, h - 120
    cv2.rectangle(frame, (chip_x - 4, chip_y - 20),
                  (chip_x + 200, chip_y + 100), (12, 16, 26), -1)
    cv2.rectangle(frame, (chip_x - 4, chip_y - 20),
                  (chip_x + 200, chip_y + 100), (40, 50, 70), 1)

    labels  = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}
    row_y   = chip_y
    for cls in [2, 3, 5, 7]:
        total = counts_down[cls] + counts_up[cls]
        color = CLASS_COLORS_BGR[cls]
        put(frame, f"{labels[cls]:<6} {total:>3}", (chip_x, row_y),
            scale=0.50, color=color)
        row_y += 22

    return frame

def process_video(video_path, confidence, line_fraction, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video file."); return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    line_y       = int(frame_h * line_fraction)

    counts_down  = {2: 0, 3: 0, 5: 0, 7: 0}
    counts_up    = {2: 0, 3: 0, 5: 0, 7: 0}
    crossed_ids  = {}
    side_history = {}
    prev_stable  = {}
    track_ttl    = {}
    id_map       = {}
    next_clean   = [1]

    frame_idx = 0
    t_prev    = time.time()
    fps       = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            now  = time.time()
            fps  = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            results = model.track(frame, persist=True,
                                  tracker="botsort.yaml", verbose=False)

            boxes_list   = []
            ids_list     = []
            classes_list = []
            active_ids   = set()

            for result in results:
                if result.boxes.id is None:
                    continue
                for box, track_id in zip(result.boxes, result.boxes.id):
                    cls = int(box.cls)
                    if cls in TARGET_CLASSES and float(box.conf) > confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        raw_id = int(track_id)
                        active_ids.add(raw_id)

                        if raw_id not in id_map:
                            id_map[raw_id] = next_clean[0]
                            next_clean[0] += 1

                        boxes_list.append((x1, y1, x2, y2))
                        ids_list.append(raw_id)
                        classes_list.append(cls)

            # TTL management
            for rid in list(track_ttl.keys()):
                if rid in active_ids:
                    track_ttl[rid] = TRACK_TTL
                else:
                    track_ttl[rid] -= 1
                    if track_ttl[rid] <= 0:
                        track_ttl.pop(rid, None)
                        side_history.pop(rid, None)
                        prev_stable.pop(rid, None)
            for rid in active_ids:
                if rid not in track_ttl:
                    track_ttl[rid] = TRACK_TTL

            # Buffered crossing detection
            for (x1, y1, x2, y2), raw_id, cls in zip(boxes_list, ids_list, classes_list):
                cy   = (y1 + y2) // 2
                side = 1 if cy >= line_y else -1

                if raw_id not in side_history:
                    side_history[raw_id] = deque(maxlen=CROSSING_BUFFER)
                side_history[raw_id].append(side)

                buf = side_history[raw_id]
                if len(buf) < CROSSING_BUFFER:
                    continue
                if not all(s == buf[0] for s in buf):
                    continue

                current_stable = buf[0]
                key = (raw_id, cls)

                if key not in crossed_ids:
                    old_side = prev_stable.get(raw_id)
                    if old_side is not None and old_side != current_stable:
                        if old_side == -1 and current_stable == 1:
                            crossed_ids[key] = "down"
                            counts_down[cls] += 1
                        elif old_side == 1 and current_stable == -1:
                            crossed_ids[key] = "up"
                            counts_up[cls] += 1

                prev_stable[raw_id] = current_stable

            # Draw bounding boxes
            for (x1, y1, x2, y2), raw_id, cls in zip(boxes_list, ids_list, classes_list):
                clean_id = id_map[raw_id]
                color    = CLASS_COLORS_BGR[cls]
                label    = TARGET_CLASSES[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} #{clean_id}",
                            (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                cx = (x1 + x2) // 2
                cy_dot = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy_dot), 5, (0, 0, 220), -1)

                key = (raw_id, cls)
                if key in crossed_ids:
                    arrow = "v" if crossed_ids[key] == "down" else "^"
                    cv2.putText(frame, arrow,
                                (cx - 8, max(y1 - 26, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

            # Overlay dashboard onto frame
            frame = overlay_dashboard(frame, counts_down, counts_up, line_y, fps)

            progress = frame_idx / max(total_frames, 1)

            yield (
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                dict(counts_down),
                dict(counts_up),
                progress,
                frame_idx,
                total_frames,
            )
    finally:
        cap.release()   


# ─────────────────────────────────────────────
#  Sidebar — settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    confidence = st.slider(
        "Detection Confidence", 0.10, 0.90, 0.35, 0.05,
        help="Minimum YOLO confidence to accept a detection."
    )
    line_pct = st.slider(
        "Counting Line Position", 10, 90, 67, 1,
        format="%d%%",
        help="Vertical position of the counting line as % of frame height."
    )
    line_fraction = line_pct / 100.0

    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown("""
- **Upload** a video file  
- **Click** Start Counting  
- YOLO11 detects vehicles each frame  
- BotSORT assigns persistent IDs  
- A 3-frame buffer confirms line crossings  
- Counts update live in the dashboard
""")
    st.markdown("---")
    st.caption("Built with YOLO11 + BotSORT + Streamlit")


st.markdown("# 🚗 Vehicle Counter")
st.markdown("Upload a traffic video and count vehicles crossing a line — by type and direction.")

st.markdown("---")

# ── Upload ──
uploaded = st.file_uploader(
    "📁 Upload your traffic video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Supports MP4, AVI, MOV, MKV"
)

if uploaded:
    # Save to temp file
    suffix = os.path.splitext(uploaded.name)[1]
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    video_path = tmp.name

    st.success(f"✅ **{uploaded.name}** uploaded successfully!")

    # Preview
    with st.expander("👁️ Preview uploaded video", expanded=False):
        st.video(video_path)

    st.markdown("---")

    # ── Start button ──
    start = st.button("▶  Start Counting")

    if start:
        model = load_model()

        # Layout: video left, stats right
        col_video, col_stats = st.columns([3, 1], gap="large")

        with col_video:
            st.markdown('<div class="section-header">Live Feed</div>',
                        unsafe_allow_html=True)
            video_placeholder = st.empty()
            progress_bar      = st.progress(0)
            status_text       = st.empty()

        with col_stats:
            st.markdown('<div class="section-header">Live Counts</div>',
                        unsafe_allow_html=True)

            # Total card
            total_placeholder = st.empty()

            # Direction row
            dir_placeholder = st.empty()

            st.markdown('<div class="section-header">Per Vehicle Type</div>',
                        unsafe_allow_html=True)
            cards = {cls: st.empty() for cls in [2, 3, 5, 7]}

        # ── Process ──
        for (rgb_frame, counts_down, counts_up,
             progress, frame_idx, total_frames) in process_video(
                video_path, confidence, line_fraction, model):

            # Update video
            video_placeholder.image(rgb_frame, channels="RGB",
                                    width="stretch")

            # Update progress
            progress_bar.progress(min(progress, 1.0))
            status_text.caption(
                f"Frame {frame_idx} / {total_frames}  "
                f"({progress*100:.1f}% complete)"
            )

            # Total card
            total_all = sum(counts_down.values()) + sum(counts_up.values())
            total_placeholder.markdown(f"""
<div class="metric-card" style="border-color:#00c2ff;">
  <div class="label">Total Vehicles</div>
  <div class="value" style="color:#00c2ff;">{total_all}</div>
  <div class="sub">all types combined</div>
</div>""", unsafe_allow_html=True)

            # Direction row
            td = sum(counts_down.values())
            tu = sum(counts_up.values())
            dir_placeholder.markdown(f"""
<div style="display:flex; gap:8px; margin-bottom:10px;">
  <div class="metric-card" style="border-color:#4dff88; flex:1;">
    <div class="label">Down ↓</div>
    <div class="value" style="color:#4dff88;">{td}</div>
  </div>
  <div class="metric-card" style="border-color:#66aaff; flex:1;">
    <div class="label">Up ↑</div>
    <div class="value" style="color:#66aaff;">{tu}</div>
  </div>
</div>""", unsafe_allow_html=True)

            # Per-class cards
            for cls in [2, 3, 5, 7]:
                d     = counts_down[cls]
                u     = counts_up[cls]
                name  = TARGET_CLASSES[cls]
                emoji = CLASS_EMOJI[cls]
                color = CLASS_COLORS_HEX[cls]
                cards[cls].markdown(f"""
<div class="metric-card" style="border-color:{color};">
  <div class="label">{emoji} {name}</div>
  <div class="value" style="color:{color};">{d+u}</div>
  <div style="margin-top:6px;">
    <span class="badge badge-down">↓ {d}</span>
    <span class="badge badge-up">↑ {u}</span>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Done ──
        progress_bar.progress(1.0)
        status_text.success("✅ Processing complete!")

        # Final summary table
        st.markdown("---")
        st.markdown("### 📊 Final Results")

        total_all = sum(counts_down.values()) + sum(counts_up.values())

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Vehicles", total_all)
        col2.metric("Moving Down ↓", sum(counts_down.values()))
        col3.metric("Moving Up ↑",   sum(counts_up.values()))

        st.markdown("#### Breakdown by Vehicle Type")

        rows = []
        for cls in [2, 3, 5, 7]:
            rows.append({
                "Type":    f"{CLASS_EMOJI[cls]} {TARGET_CLASSES[cls]}",
                "Down ↓":  counts_down[cls],
                "Up ↑":    counts_up[cls],
                "Total":   counts_down[cls] + counts_up[cls],
            })
        st.table(rows)

        # ── PDF Report ──
        st.markdown("---")
        st.markdown("### 📄 Detailed Report")

        with st.spinner("Generating PDF report..."):
            pdf_bytes = generate_pdf_report(
                counts_down  = counts_down,
                counts_up    = counts_up,
                total_frames = total_frames,
                video_name   = uploaded.name,
                confidence   = confidence,
                line_pct     = line_pct,
            )

        st.success("✅ Report ready!")

        # Preview
        with st.expander("👁️ Preview Report", expanded=True):
            # Embed PDF inline using base64
            import base64
            b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_display = f"""
            <iframe
                src="data:application/pdf;base64,{b64}"
                width="100%" height="600px"
                style="border:none; border-radius:8px;">
            </iframe>"""
            st.markdown(pdf_display, unsafe_allow_html=True)

        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label     = "⬇️  Download PDF Report",
            data      = pdf_bytes,
            file_name = f"vehicle_report_{timestamp}.pdf",
            mime      = "application/pdf",
        )

        try:
            os.unlink(video_path)
        except PermissionError:
            pass   

else:
    # Empty state
    st.markdown("""
<div style="
    background:#1c2233;
    border: 2px dashed #2a3550;
    border-radius: 16px;
    padding: 60px 40px;
    text-align: center;
    color: #556677;
    margin-top: 20px;
">
  <div style="font-size:3rem; margin-bottom:16px;">🎥</div>
  <div style="font-size:1.1rem; font-weight:600; color:#8899aa; margin-bottom:8px;">
    No video uploaded yet
  </div>
  <div style="font-size:0.85rem;">
    Upload a traffic video above to get started.<br>
    Supports MP4, AVI, MOV, MKV.
  </div>
</div>
""", unsafe_allow_html=True)