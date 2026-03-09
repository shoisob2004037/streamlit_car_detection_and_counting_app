# car_counting_yolo11.py
import cv2
from ultralytics import YOLO
import numpy as np

# ==================== SETTINGS ====================
VIDEO_PATH = "car.mp4"
LINE_Y = 400
CONFIDENCE = 0.5
CLASS_CAR = 2
# ===================================================

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

car_count = 0
crossed_ids = set()         # raw track IDs that already crossed
prev_centroids = {}         # {track_id: centroid_y} from previous frame
id_map = {}                 # {raw_track_id: clean_sequential_id}
next_clean_id = 1           # counter for clean IDs

print("Press 'q' to quit the window")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 detection + tracking
    results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

    car_boxes = []
    car_ids = []

    for result in results:
        if result.boxes.id is None:
            continue
        for box, track_id in zip(result.boxes, result.boxes.id):
            if int(box.cls) == CLASS_CAR and float(box.conf) > CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_id = int(track_id)

                # Remap to clean sequential ID
                if raw_id not in id_map:
                    id_map[raw_id] = next_clean_id
                    next_clean_id += 1

                car_boxes.append((x1, y1, x2, y2))
                car_ids.append(raw_id)

    # ── Crossing detection (needs previous frame position) ──
    current_centroids = {}

    for (x1, y1, x2, y2), raw_id in zip(car_boxes, car_ids):
        cy = (y1 + y2) // 2
        current_centroids[raw_id] = cy

        # Only count if we saw this car in the previous frame
        if raw_id in prev_centroids:
            prev_cy = prev_centroids[raw_id]

            # Crossed downward: was above (cy < LINE_Y), now below (cy >= LINE_Y)
            crossed_down = prev_cy < LINE_Y and cy >= LINE_Y
            # Crossed upward: was below, now above
            crossed_up   = prev_cy >= LINE_Y and cy < LINE_Y

            if (crossed_down or crossed_up) and raw_id not in crossed_ids:
                crossed_ids.add(raw_id)
                car_count += 1

    prev_centroids = current_centroids  # update for next frame

    # ── Drawing ──
    # ── Drawing ──
    for (x1, y1, x2, y2), raw_id in zip(car_boxes, car_ids):
        clean_id = id_map[raw_id]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Car #{clean_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── Red dot at centroid (tracked by BoT-SORT) ──
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)   # filled red dot

    # Counting line (red)
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 3)

    # Count display
    cv2.putText(frame, f"Cars Passed: {car_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.imshow("YOLOv11 + BoT-SORT Car Counting (Press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal car count: {car_count}")
print(f"Unique cars that crossed the line: {len(crossed_ids)}")