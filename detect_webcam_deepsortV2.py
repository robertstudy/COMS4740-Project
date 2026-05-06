import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError as exc:
    raise ImportError(
        "DeepSORT dependency missing. Install with: pip install deep-sort-realtime"
    ) from exc

# python detect_webcam_deepsortV2.py --camera 1


def resolve_weights_path(project_root: Path, weights: str) -> Path:
    p = Path(weights)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Weights not found: {p}")
    return p


@dataclass
class CornerDetection:
    class_id: int
    label: str
    bbox: Tuple[int, int, int, int]
    conf: float


@dataclass
class MemoryTrack:
    track_id: int
    label: str
    bbox: Tuple[int, int, int, int]
    missing_frames: int = 0
    occluded: bool = False
    last_seen_frame: int = 0
    first_seen_frame: int = 0


def box_intersects(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1 -= margin
    ay1 -= margin
    ax2 += margin
    ay2 += margin
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def has_visible_corner_for_track(
    label: str,
    track_bbox: Tuple[int, int, int, int],
    corners: List[CornerDetection],
    margin: int = 14,
) -> bool:
    for det in corners:
        if det.label != label:
            continue
        if box_intersects(track_bbox, det.bbox, margin=margin):
            return True
    return False


def extract_corner_detections(result, class_names: Dict[int, str]) -> List[CornerDetection]:
    detections: List[CornerDetection] = []
    boxes = result.boxes
    if boxes is None:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        detections.append(
            CornerDetection(
                class_id=int(cls_id),
                label=class_names.get(int(cls_id), str(cls_id)),
                bbox=(x1, y1, x2, y2),
                conf=float(conf),
            )
        )
    return detections


def big_box_from_pair(
    d1: CornerDetection, d2: CornerDetection, frame_shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    x1 = min(d1.bbox[0], d2.bbox[0])
    y1 = min(d1.bbox[1], d2.bbox[1])
    x2 = max(d1.bbox[2], d2.bbox[2])
    y2 = max(d1.bbox[3], d2.bbox[3])

    frame_h, frame_w = frame_shape[:2]
    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w - 1, x2))
    y2 = max(0, min(frame_h - 1, y2))
    return x1, y1, x2, y2


def bbox_iou(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    return inter / max(1.0, area_a + area_b - inter)


def bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def center_distance(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def best_pair_for_track(
    available: List[CornerDetection],
    track_bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
) -> Optional[Tuple[int, int]]:
    best_score = -1.0
    best_pair: Optional[Tuple[int, int]] = None
    frame_diag = (frame_shape[0] * frame_shape[0] + frame_shape[1] * frame_shape[1]) ** 0.5
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            candidate_bbox = big_box_from_pair(available[i], available[j], frame_shape)
            iou = bbox_iou(candidate_bbox, track_bbox)
            dist_norm = center_distance(candidate_bbox, track_bbox) / max(1.0, frame_diag)
            # Prefer high overlap; distance is a tie-breaker.
            score = iou - (0.35 * dist_norm)
            if score > best_score:
                best_score = score
                best_pair = (i, j)
    if best_pair is None:
        return None
    # Keep assignment conservative so we don't cross-pair two identical cards far apart.
    i, j = best_pair
    candidate_bbox = big_box_from_pair(available[i], available[j], frame_shape)
    if bbox_iou(candidate_bbox, track_bbox) <= 0.0 and center_distance(candidate_bbox, track_bbox) > (
        0.25 * frame_diag
    ):
        return None
    return best_pair

def create_player_box(
        track_ids: set[int], memory_tracks: List[MemoryTrack], frame_shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = frame_shape[1], frame_shape[0], 0, 0
    for track_id in track_ids:
        card = memory_tracks[track_id]
        x1 = min(x1, card.bbox[0])
        y1 = min(y1, card.bbox[1])
        x2 = max(x2, card.bbox[2])
        y2 = max(y2, card.bbox[3])
    return x1, y1, x2, y2

def pair_same_class_boxes(
    corners: List[CornerDetection],
    memory_tracks: Dict[int, MemoryTrack],
    frame_shape: Tuple[int, int, int],
    frame_idx: int,
    max_track_stale_frames: int = 10,
) -> List[Tuple[CornerDetection, CornerDetection]]:
    by_class: Dict[int, List[CornerDetection]] = {}
    for det in corners:
        by_class.setdefault(det.class_id, []).append(det)

    pairs: List[Tuple[CornerDetection, CornerDetection]] = []
    for class_corners in by_class.values():
        if len(class_corners) < 2:
            continue

        available = list(class_corners)
        label = class_corners[0].label
        recent_tracks = [
            track
            for track in memory_tracks.values()
            if track.label == label and (frame_idx - track.last_seen_frame) <= max_track_stale_frames
        ]
        # Older tracks first so recently-created IDs don't steal pairing from stable cards.
        recent_tracks.sort(key=lambda t: (t.first_seen_frame, t.track_id))

        for track in recent_tracks:
            if len(available) < 2:
                break
            pair_idx = best_pair_for_track(available, track.bbox, frame_shape)
            if pair_idx is None:
                continue
            i, j = pair_idx
            d1 = available[i]
            d2 = available[j]
            pairs.append((d1, d2))
            for idx in sorted((i, j), reverse=True):
                del available[idx]

        # Pair any remaining detections by nearest-neighbor center distance.
        while len(available) >= 2:
            best_i = 0
            best_j = 1
            best_dist = float("inf")
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    d1 = available[i]
                    d2 = available[j]
                    d = center_distance(d1.bbox, d2.bbox)
                    if d < best_dist:
                        best_dist = d
                        best_i = i
                        best_j = j
            d1 = available[best_i]
            d2 = available[best_j]
            pairs.append((d1, d2))
            for idx in sorted((best_i, best_j), reverse=True):
                del available[idx]

    return pairs


# HiLo Values
HILO_VALUES = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1
}

def get_rank(label: str) -> str:
    clean = label.strip().upper()
    match = re.match(r"^(10|[2-9]|[AJQK])", clean)
    return match.group(1) if match else ""

def calculate_hand_value(ranks: List[str]) -> Tuple[int, bool]:
    value = 0
    aces = 0
    for r in ranks:
        if r in {"J", "Q", "K", "10"}:
            value += 10
        elif r == "A":
            aces += 1
            value += 11
        else:
            value += int(r)
    
    while value > 21 and aces > 0:
        value -= 10
        aces -= 1
    
    return value, (aces > 0)

def get_optimal_action(player_total: int, is_soft: bool, dealer_rank: str, running_count: int, hand_cards: List[str] = None) -> str:
    if not dealer_rank:
        return "WAIT"
    
    # Convert dealer_rank to value (A=11)
    if dealer_rank in {"J", "Q", "K", "10"}:
        dv = 10
    elif dealer_rank == "A":
        dv = 11
    else:
        dv = int(dealer_rank)

    # Insurance
    if running_count >= 3 and dealer_rank == 'A':
        return "INSURANCE"

    # Handling pairs for SPLIT
    if hand_cards and len(hand_cards) == 2 and hand_cards[0] == hand_cards[1]:
        rank = hand_cards[0]
        # Illustrious 18 / Fab 4 / Common SPLIT strategies
        if rank == 'A' or rank == '8':
            return "SPLIT"
        if rank == '10':
            # TTv5: Split at +5 or higher, stand otherwise
            if dv == 5 and running_count >= 5: return "SPLIT"
            # TTv6: Split at +4 or higher, stand otherwise
            if dv == 6 and running_count >= 4: return "SPLIT"
            return "STAND"
        if rank == '9':
            if dv in [2, 3, 4, 5, 6, 8, 9]: return "SPLIT"
            return "STAND"
        if rank == '7' or rank == '2' or rank == '3':
            if dv <= 7: return "SPLIT"
        if rank == '6':
            if dv <= 6: return "SPLIT"
        if rank == '4':
            if dv in [5, 6]: return "SPLIT"
        return "HIT"

    # Standalone Illustrious 18 & Fab 4 deviations for total/soft
    if player_total == 16:
        if dv == 10:
            if running_count >= 0: return "STAND"
            return "SURRENDER"
        if dv == 9 and running_count >= 5: return "STAND"
    
    if player_total == 15:
        if dv == 10:
            if running_count >= 4: return "STAND"
            return "SURRENDER"
        if dv == 9 and running_count >= 2: return "SURRENDER"
        if dv == 11 and running_count >= 1: return "SURRENDER"

    if player_total == 14 and dv == 10 and running_count >= 3:
        return "SURRENDER"

    if player_total == 13:
        if dv == 2 and running_count >= -1: return "STAND"
        if dv == 3 and running_count >= -2: return "STAND"

    if player_total == 12:
        if dv == 2 and running_count >= 3: return "STAND"
        if dv == 3 and running_count >= 2: return "STAND"
        if dv == 4 and running_count >= 0: return "STAND"
        if dv == 5 and running_count >= -2: return "STAND"
        if dv == 6 and running_count >= -1: return "STAND"

    if player_total == 11 and dv == 11 and running_count >= 1:
        return "DOUBLE"
    
    if player_total == 10 and dv == 10 and running_count >= 4:
        return "DOUBLE"

    if player_total == 9:
        if dv == 2 and running_count >= 1: return "DOUBLE"
        if dv == 7 and running_count >= 3: return "DOUBLE"

    # Standard Strategy
    if not is_soft:
        if player_total <= 11:
            return "DOUBLE" if player_total >= 9 else "HIT"
        if 12 <= player_total <= 16:
            return "STAND" if 2 <= dv <= 6 else "HIT"
        return "STAND"
    else: # Soft totals
        if player_total <= 17:
            return "DOUBLE" if 3 <= dv <= 6 else "HIT"
        if player_total == 18:
            if dv <= 6: return "DOUBLE"
            if dv <= 8: return "STAND"
            return "HIT"
        return "STAND"

def draw_count_panel(frame, running_count: int, num_players: int) -> None:
    lines = [
        f"HiLo Running Count: {running_count}",
        f"Players Detected: {num_players}",
        "Press 'r' to reset count",
        "Press 'q' to quit"
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    line_h = 28
    pad = 12

    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_width = max(max_width, w)

    panel_w = max_width + pad * 2
    panel_h = len(lines) * line_h + pad * 2
    x0, y0 = 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        y = y0 + pad + (i + 1) * line_h - 8
        cv2.putText(
            frame,
            line,
            (x0 + pad, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track connected big boxes using DeepSORT."
    )
    parser.add_argument(
        "--weights",
        default="runs/playing_cards_yolo/weights/best.pt",
        help="Path to .pt weights (best.pt or last.pt from your run).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (0 = default camera, try 1 if 0 fails).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence for corner detections.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (higher helps small objects; multiple of 32 is typical).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested webcam frame width (actual may differ).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested webcam frame height (actual may differ).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: e.g. 0, cpu. Default: Ultralytics auto-select.",
    )
    parser.add_argument(
        "--track-max-age",
        type=int,
        default=30,
        help="DeepSORT max_age (frames to keep lost tracks).",
    )
    parser.add_argument(
        "--track-init",
        type=int,
        default=2,
        help="DeepSORT n_init (hits before a track is confirmed).",
    )
    parser.add_argument(
        "--memory-frames",
        type=int,
        default=30,
        help="How long to keep last known card box/value after occlusion.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a preview window (still runs inference; useful headless).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    weights_path = resolve_weights_path(project_root, args.weights)

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    tracker = DeepSort(
        max_age=args.track_max_age,
        n_init=args.track_init,
        embedder="mobilenet",
        half=True,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(
        f"Webcam index: {args.camera}  requested={args.width}x{args.height}  actual={actual_w}x{actual_h}"
    )
    print(
        f"Inference: conf={args.conf} imgsz={args.imgsz} device={args.device} "
        f"max_age={args.track_max_age} memory_frames={args.memory_frames}"
    )
    print("Press 'q' in the preview window to quit.\n")
    memory_tracks: Dict[int, MemoryTrack] = {}
    players: List[set[int]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam.")

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            raw_names = results[0].names
            if isinstance(raw_names, dict):
                class_names = {int(k): v for k, v in raw_names.items()}
            else:
                class_names = {i: name for i, name in enumerate(raw_names)}

            corners = extract_corner_detections(results[0], class_names)
            pairs = pair_same_class_boxes(
                corners=corners,
                memory_tracks=memory_tracks,
                frame_shape=frame.shape,
                frame_idx=frame_idx,
            )

            detections_for_tracker = []
            for d1, d2 in pairs:
                x1, y1, x2, y2 = big_box_from_pair(d1, d2, frame.shape)
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                conf = min(d1.conf, d2.conf)
                detections_for_tracker.append(([x1, y1, w, h], conf, d1.label))

            tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
            seen_track_ids = set()
            for track in tracks:
                if not track.is_confirmed():
                    continue
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = [int(v) for v in ltrb]
                label = str(track.get_det_class() or "card")
                track_id = int(track.track_id)
                seen_track_ids.add(track_id)
                current_bbox = (x1, y1, x2, y2)
                existing = memory_tracks.get(track_id)
                if existing is None:
                    memory_tracks[track_id] = MemoryTrack(
                        track_id=track_id,
                        label=label,
                        bbox=current_bbox,
                        missing_frames=0,
                        occluded=False,
                        last_seen_frame=frame_idx,
                        first_seen_frame=frame_idx,
                    )
                    continue

                # Keep the original remembered box position while at least one
                # matching corner remains visible inside that anchored region.
                corner_visible_in_anchor = has_visible_corner_for_track(
                    existing.label, existing.bbox, corners
                )
                if corner_visible_in_anchor:
                    anchored_bbox = existing.bbox
                    occluded = False
                else:
                    anchored_bbox = current_bbox
                    occluded = not has_visible_corner_for_track(
                        label, current_bbox, corners
                    )

                memory_tracks[track_id] = MemoryTrack(
                    track_id=track_id,
                    label=label,
                    bbox=anchored_bbox,
                    missing_frames=0,
                    occluded=occluded,
                    last_seen_frame=frame_idx,
                    first_seen_frame=existing.first_seen_frame,
                )

            for track_id in list(memory_tracks.keys()):
                if track_id in seen_track_ids:
                    continue
                mem = memory_tracks[track_id]
                corner_visible = has_visible_corner_for_track(mem.label, mem.bbox, corners)
                if corner_visible:
                    mem.missing_frames = 0
                    mem.occluded = False
                    continue
                mem.missing_frames += 1
                mem.occluded = True
                if mem.missing_frames > args.memory_frames:
                    del memory_tracks[track_id]

            # Group overlapping cards as different players
            players.clear()
            track_ids = list(memory_tracks.keys())
            adj = {tid: [] for tid in track_ids}

            for i in range(len(track_ids)):
                for j in range(len(track_ids)):
                    id1, id2 = track_ids[i], track_ids[j]
                    if box_intersects(memory_tracks[id1].bbox, memory_tracks[id2].bbox):
                        adj[id1].append(id2)
                        adj[id2].append(id1)
            
            visited = set()

            for track_id in track_ids:
                if track_id not in visited:
                    hand = set()
                    stack = [track_id]

                    while stack:
                        card = stack.pop()
                        if card not in visited:
                            visited.add(card)
                            hand.add(card)
                            stack.extend(adj[card])
                    players.append(hand)

            # Update HiLo count
            for tid in track_ids:
                mem = memory_tracks[tid]
                if tid not in counted_ids:
                    rank = get_rank(mem.label)
                    if rank:
                        running_hilo_count += HILO_VALUES.get(rank, 0)
                        counted_ids.add(tid)

            # Identify Dealer (top-most hand)
            players_with_y = []
            for p_set in players:
                y_min = min(memory_tracks[tid].bbox[1] for tid in p_set)
                players_with_y.append((y_min, p_set))
            
            players_with_y.sort()
            sorted_hands = [p[1] for p in players_with_y]
            
            dealer_hand = sorted_hands[0] if sorted_hands else None
            dealer_upcard_rank = ""
            if dealer_hand:
                # Use the first card in the dealer's hand as the upcard
                dealer_upcard_rank = get_rank(memory_tracks[list(dealer_hand)[0]].label)

            if not args.no_show:
                annotated = frame.copy()
                for det in corners:
                    x1, y1, x2, y2 = det.bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 200), 1)

                for mem in memory_tracks.values():
                    x1, y1, x2, y2 = mem.bbox
                    is_live = mem.missing_frames == 0
                    color = (0, 220, 0) if is_live else (0, 165, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    if is_live:
                        suffix = " (occluded)" if mem.occluded else ""
                    else:
                        suffix = f" (mem {mem.missing_frames})"
                    cv2.putText(
                        annotated,
                        f"ID {mem.track_id} {mem.label}{suffix}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                for pi, player_set in enumerate(sorted_hands):
                    is_dealer = (player_set == dealer_hand)
                    x1, y1, x2, y2 = create_player_box(player_set, memory_tracks, frame.shape)
                    
                    hand_ranks = [get_rank(memory_tracks[tid].label) for tid in player_set]
                    hand_val, is_soft = calculate_hand_value(hand_ranks)
                    
                    label_prefix = "Dealer" if is_dealer else f"Player {pi}"
                    display_text = f"{label_prefix}: {hand_val}"
                    if is_soft and hand_val < 21:
                        display_text += " (Soft)"
                    
                    margin = 30
                    color = (255, 0, 0) if is_dealer else (0, 0, 255)
                    cv2.rectangle(annotated, (x1 - margin, y1 - margin), (x2 + margin, y2 + margin), color, 2)
                    cv2.putText(
                        annotated,
                        display_text,
                        (x1, max(20, y1 - (margin + 8))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    # Strategy Advice
                    if not is_dealer and dealer_upcard_rank:
                        advice = get_optimal_action(hand_val, is_soft, dealer_upcard_rank, running_hilo_count, hand_ranks)
                        cv2.putText(
                            annotated,
                            f"Advice: {advice}",
                            (x1, y2 + margin + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                draw_count_panel(annotated, running_hilo_count, len(sorted_hands))

                cv2.imshow("Blackjack HiLo & Strategy Bot", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "\nTips: install deep-sort-realtime, try --camera 1, reduce --imgsz, or use --device cpu.",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
