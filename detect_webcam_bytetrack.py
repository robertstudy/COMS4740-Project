import cv2
import numpy as np
import re
from ultralytics import YOLO
from collections import defaultdict
import time


# ----------------------------
# CONFIG
# ----------------------------
MISSING_TIMEOUT = 1.0  # seconds
TRACK_MATCH_DIST2 = 5000  # center distance threshold for same card track
PARTIAL_MARGIN = 12  # pixels to expand remembered box for corner evidence


# ----------------------------
# CARD VALUE
# ----------------------------
def card_value(label):
    m = re.match(r"^(10|[2-9]|[AJQK])", label.upper())
    if not m:
        return 0
    r = m.group(1)
    if r in ["J", "Q", "K"]:
        return 10
    if r == "A":
        return 1
    return int(r)


# ----------------------------
# CORNER PAIRING → CARD BOX
# ----------------------------
def build_cards(dets):
    """
    dets = [(x1,y1,x2,y2,conf,label), ...]

    We assume:
    - each card has 2 corners visible
    - corners belong close spatially
    """

    cards = []
    used = set()

    for i, d1 in enumerate(dets):
        if i in used:
            continue

        x1, y1, x2, y2, c1, l1 = d1

        best_j = None
        best_dist = 1e9

        for j, d2 in enumerate(dets):
            if i == j or j in used:
                continue

            x1b, y1b, x2b, y2b, c2, l2 = d2

            cx1 = (x1 + x2) / 2
            cy1 = (y1 + y2) / 2
            cx2 = (x1b + x2b) / 2
            cy2 = (y1b + y2b) / 2

            dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is not None:
            d2 = dets[best_j]

            x1b, y1b, x2b, y2b, c2, l2 = d2

            # merge into card box
            bx1 = min(x1, x1b)
            by1 = min(y1, y1b)
            bx2 = max(x2, x2b)
            by2 = max(y2, y2b)

            conf = max(c1, c2)

            cards.append([bx1, by1, bx2, by2, conf, l1])

            used.add(i)
            used.add(best_j)

    return cards


# ----------------------------
# MAIN
# ----------------------------
def main():
    model = YOLO("runs/playing_cards_yolo/weights/best.pt")
    cap = cv2.VideoCapture(1)

    # track memory
    track_last_seen = {}
    track_data = {}

    track_id_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, conf=0.25, verbose=False)[0]

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        names = result.names if isinstance(result.names, dict) else {
            i: n for i, n in enumerate(result.names)
        }

        raw = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            conf = float(confs[i])
            label = names[int(classes[i])]
            raw.append((x1, y1, x2, y2, conf, label))

        # ----------------------------
        # STEP 1: build card boxes
        # ----------------------------
        cards = build_cards(raw)

        now = time.time()
        active_ids = set()

        # ----------------------------
        # STEP 2: assign simple tracking
        # (NO DeepSORT / ByteTrack needed anymore)
        # ----------------------------
        unmatched_track_ids = set(track_data.keys())
        for (x1, y1, x2, y2, conf, label) in cards:

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            matched_id = None

            # match to nearest unmatched track with same label
            best_dist = 1e9
            for tid in list(unmatched_track_ids):
                tinfo = track_data[tid]
                tx, ty = tinfo["center"]
                if tinfo["label"] != label:
                    continue
                dist = (cx - tx) ** 2 + (cy - ty) ** 2
                if dist < TRACK_MATCH_DIST2 and dist < best_dist:
                    best_dist = dist
                    matched_id = tid

            if matched_id is None:
                matched_id = track_id_counter
                track_id_counter += 1

            track_data[matched_id] = {
                "center": (cx, cy),
                "bbox": (x1, y1, x2, y2),
                "label": label
            }
            track_last_seen[matched_id] = now
            active_ids.add(matched_id)
            unmatched_track_ids.discard(matched_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {matched_id} {label}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

        # ----------------------------
        # STEP 2.5: keep remembered boxes while partially visible
        # ----------------------------
        for tid in list(unmatched_track_ids):
            tinfo = track_data.get(tid)
            if tinfo is None:
                continue

            x1, y1, x2, y2 = tinfo["bbox"]
            label = tinfo["label"]
            ex1 = x1 - PARTIAL_MARGIN
            ey1 = y1 - PARTIAL_MARGIN
            ex2 = x2 + PARTIAL_MARGIN
            ey2 = y2 + PARTIAL_MARGIN

            partial_visible = False
            for rx1, ry1, rx2, ry2, _rconf, rlabel in raw:
                if rlabel != label:
                    continue
                rcx = (rx1 + rx2) // 2
                rcy = (ry1 + ry2) // 2
                if ex1 <= rcx <= ex2 and ey1 <= rcy <= ey2:
                    partial_visible = True
                    break

            if partial_visible:
                # Keep this track alive if at least one matching corner is still visible.
                track_last_seen[tid] = now
                active_ids.add(tid)

            # Draw remembered box even when corners cannot be paired this frame.
            box_color = (0, 255, 255) if partial_visible else (0, 165, 255)
            status = "partial" if partial_visible else "memory"
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"ID {tid} {label} ({status})",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2)

        # ----------------------------
        # STEP 3: remove stale cards
        # ----------------------------
        for tid in list(track_last_seen.keys()):
            if now - track_last_seen[tid] > MISSING_TIMEOUT:
                del track_last_seen[tid]
                if tid in track_data:
                    del track_data[tid]

        cv2.imshow("Card Tracker (Corner-Based)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()