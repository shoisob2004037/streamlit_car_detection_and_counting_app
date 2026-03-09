import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import deque
from ultralytics import RTDETR

VIDEO_PATH      = "car_motor.mp4"
CONFIDENCE      = 0.35
PANEL_WIDTH     = 340         


CROSSING_BUFFER = 3

TRACK_TTL       = 60        

TARGET_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

CLASS_COLORS = {         
    2: (0, 230, 0),       
    3: (0, 150, 255),    
    5: (0, 220, 255),     
    7: (255,  80, 200),   
}

CLASS_ICONS = {
    2:  "Car",
    3:  "Moto",
    5:  "Bus",
    7:  "Truck",
}
# ===================================================

# ── Utility ────────────────────────────────────────────────────────────────
def put_text(img, txt, pos, scale=0.6, color=(255, 255, 255),
             bold=False, align="left"):
    thickness = 2 if bold else 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    if align != "left":
        (tw, th), _ = cv2.getTextSize(txt, font, scale, thickness)
        x, y = pos
        if align == "right":
            x -= tw
        elif align == "center":
            x -= tw // 2
        pos = (x, y)
    cv2.putText(img, txt, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    """Filled rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)


def divider(img, y, W, alpha=80):
    cv2.line(img, (12, y), (W - 12, y), (alpha, alpha, alpha), 1)


# ── Dashboard ──────────────────────────────────────────────────────────────
def draw_dashboard(panel, counts_down, counts_up, fps, frame_idx):
    panel[:] = (18, 20, 26)          # near-black blue-tinted background
    W, H = panel.shape[1], panel.shape[0]
    PAD = 14

    # ── Header bar ──
    cv2.rectangle(panel, (0, 0), (W, 52), (30, 34, 44), -1)
    put_text(panel, "VEHICLE COUNTER", (PAD, 34), scale=0.72,
             color=(0, 210, 255), bold=True)
    put_text(panel, f"FPS {fps:4.1f}", (W - PAD, 34), scale=0.55,
             color=(160, 160, 160), align="right")

    y = 70
    divider(panel, y, W); y += 18

    # ── Per-class rows ──
    for cls in [2, 3, 5, 7]:
        d = counts_down[cls]
        u = counts_up[cls]
        total = d + u
        color = CLASS_COLORS[cls]
        name  = CLASS_ICONS[cls]

        # Color accent strip on the left
        cv2.rectangle(panel, (PAD, y - 2), (PAD + 3, y + 26), color, -1)

        # Class name
        put_text(panel, name, (PAD + 10, y + 18), scale=0.65,
                 color=color, bold=True)

        # Directional mini-badges  ↓ count  ↑ count
        bx = PAD + 80
        # Down badge
        draw_rounded_rect(panel, (bx, y + 1), (bx + 52, y + 24),
                          (20, 60, 20), radius=5)
        put_text(panel, f"v {d}", (bx + 6, y + 18), scale=0.52,
                 color=(100, 255, 100))
        # Up badge
        bx2 = bx + 60
        draw_rounded_rect(panel, (bx2, y + 1), (bx2 + 52, y + 24),
                          (20, 30, 70), radius=5)
        put_text(panel, f"^ {u}", (bx2 + 6, y + 18), scale=0.52,
                 color=(120, 180, 255))

        # Total (right-aligned)
        put_text(panel, str(total), (W - PAD, y + 18), scale=0.72,
                 color=(230, 230, 230), bold=True, align="right")

        y += 36

    y += 6
    divider(panel, y, W); y += 20

    # ── Totals block ──
    total_all  = sum(counts_down.values()) + sum(counts_up.values())
    total_down = sum(counts_down.values())
    total_up   = sum(counts_up.values())

    # Large total
    put_text(panel, "Total Vehicles", (PAD, y + 18), scale=0.60,
             color=(180, 180, 180))
    put_text(panel, str(total_all), (W - PAD, y + 18), scale=0.85,
             color=(255, 255, 255), bold=True, align="right")
    y += 36

    # Direction summary
    put_text(panel, "Moving Down  v", (PAD, y + 18), scale=0.58,
             color=(100, 255, 100))
    put_text(panel, str(total_down), (W - PAD, y + 18), scale=0.65,
             color=(100, 255, 100), bold=True, align="right")
    y += 28

    put_text(panel, "Moving Up    ^", (PAD, y + 18), scale=0.58,
             color=(120, 180, 255))
    put_text(panel, str(total_up), (W - PAD, y + 18), scale=0.65,
             color=(120, 180, 255), bold=True, align="right")
    y += 32

    divider(panel, y, W); y += 18

    # ── Frame counter (subtle) ──
    put_text(panel, f"Frame #{frame_idx}", (PAD, y + 16), scale=0.48,
             color=(70, 70, 70))

    return panel


# ── Model & capture ────────────────────────────────────────────────────────
model = RTDETR("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Cannot open video file"); exit()

frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
LINE_Y  = frame_h * 2 // 3

counts_down = {2: 0, 3: 0, 5: 0, 7: 0}
counts_up   = {2: 0, 3: 0, 5: 0, 7: 0}

crossed_ids  = {}

side_history : dict[int, deque] = {}

track_ttl    = {}

id_map = {}
next_clean_id = 1
frame_idx = 0
t_prev = time.time()
fps = 0.0

print(f"Counting line at Y = {LINE_Y}  (2/3 down the frame)")
print("Press 'q' to quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # FPS (rolling)
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
    t_prev = now

    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          verbose=False)

    boxes_list   = []
    ids_list     = []
    classes_list = []

    active_ids = set()

    for result in results:
        if result.boxes.id is None:
            continue
        for box, track_id in zip(result.boxes, result.boxes.id):
            cls = int(box.cls)
            if cls in TARGET_CLASSES and float(box.conf) > CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_id = int(track_id)
                active_ids.add(raw_id)

                if raw_id not in id_map:
                    id_map[raw_id] = next_clean_id
                    next_clean_id += 1

                boxes_list.append((x1, y1, x2, y2))
                ids_list.append(raw_id)
                classes_list.append(cls)

    for rid in list(track_ttl.keys()):
        if rid in active_ids:
            track_ttl[rid] = TRACK_TTL
        else:
            track_ttl[rid] -= 1
            if track_ttl[rid] <= 0:
                track_ttl.pop(rid, None)
                side_history.pop(rid, None)

    for rid in active_ids:
        if rid not in track_ttl:
            track_ttl[rid] = TRACK_TTL

    # ── Buffered crossing detection ──
    for (x1, y1, x2, y2), raw_id, cls in zip(boxes_list, ids_list, classes_list):
        cy   = (y1 + y2) // 2
        side = 1 if cy >= LINE_Y else -1   # +1 = below, -1 = above

        if raw_id not in side_history:
            side_history[raw_id] = deque(maxlen=CROSSING_BUFFER)
        side_history[raw_id].append(side)

        key = (raw_id, cls)
        if key in crossed_ids:
            continue   

        buf = side_history[raw_id]
        if len(buf) < CROSSING_BUFFER:
            continue   

        if all(s == 1 for s in buf):   
            pass  

    for (x1, y1, x2, y2), raw_id, cls in zip(boxes_list, ids_list, classes_list):
        cy   = (y1 + y2) // 2
        side = 1 if cy >= LINE_Y else -1

        buf = side_history.get(raw_id, deque(maxlen=CROSSING_BUFFER))
        side_history[raw_id] = buf
        buf.append(side)

        key = (raw_id, cls)
        if key in crossed_ids or len(buf) < CROSSING_BUFFER:
            continue

        # Only act when the buffer is fully uniform (stable on one side)
        if not (all(s == buf[0] for s in buf)):
            continue

        current_stable = buf[0]

        # Compare to the stable side BEFORE this sequence.
        # We piggyback on the existing prev_stable dict below.

    # Store prev_stable outside the loop
    if not hasattr(draw_dashboard, "_prev_stable"):
        draw_dashboard._prev_stable = {}

    prev_stable = draw_dashboard._prev_stable

    for (x1, y1, x2, y2), raw_id, cls in zip(boxes_list, ids_list, classes_list):
        cy   = (y1 + y2) // 2
        side = 1 if cy >= LINE_Y else -1

        buf = side_history.get(raw_id)
        if buf is None or len(buf) < CROSSING_BUFFER:
            continue

        if not all(s == buf[0] for s in buf):
            continue   # buffer not yet stable

        current_stable = buf[0]
        key = (raw_id, cls)

        if key not in crossed_ids:
            old_side = prev_stable.get(raw_id)
            if old_side is not None and old_side != current_stable:
                # Direction of travel
                if old_side == -1 and current_stable == 1:   # above → below = down
                    crossed_ids[key] = "down"
                    counts_down[cls] += 1
                elif old_side == 1 and current_stable == -1:  # below → above = up
                    crossed_ids[key] = "up"
                    counts_up[cls] += 1

        prev_stable[raw_id] = current_stable

    # ── Drawing ────────────────────────────────────────────────────────────
    for (x1, y1, x2, y2), raw_id, cls in zip(boxes_list, ids_list, classes_list):
        clean_id = id_map[raw_id]
        color    = CLASS_COLORS[cls]
        label    = TARGET_CLASSES[cls]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} #{clean_id}",
                    (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 220), -1)

        key = (raw_id, cls)
        if key in crossed_ids:
            arrow = "v" if crossed_ids[key] == "down" else "^"
            cv2.putText(frame, arrow, (cx - 8, max(y1 - 28, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Counting line with gradient-like double draw for visibility
    cv2.line(frame, (0, LINE_Y), (frame_w, LINE_Y), (0, 0, 0), 4)
    cv2.line(frame, (0, LINE_Y), (frame_w, LINE_Y), (0, 80, 255), 2)
    cv2.putText(frame, "-- COUNTING LINE --",
                (10, LINE_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1, cv2.LINE_AA)

    # ── Dashboard panel ────────────────────────────────────────────────────
    panel = np.zeros((frame_h, PANEL_WIDTH, 3), dtype=np.uint8)
    draw_dashboard(panel, counts_down, counts_up, fps, frame_idx)

    combined = np.hstack([frame, panel])
    cv2.imshow("YOLOv11 Vehicle Counter  |  q = quit", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ── Final summary ──────────────────────────────────────────────────────────
print("\n========== FINAL RESULTS ==========")
print(f"{'Type':<14} {'Down v':>8} {'Up ^':>8} {'Total':>8}")
print("-" * 44)
for cls in [2, 3, 5, 7]:
    name = TARGET_CLASSES[cls]
    d    = counts_down[cls]
    u    = counts_up[cls]
    print(f"{name:<14} {d:>8} {u:>8} {d+u:>8}")
print("-" * 44)
td = sum(counts_down.values())
tu = sum(counts_up.values())
print(f"{'TOTAL':<14} {td:>8} {tu:>8} {td+tu:>8}")