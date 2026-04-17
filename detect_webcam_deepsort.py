import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError as exc:
    raise ImportError(
        "DeepSORT dependency missing. Install with: pip install deep-sort-realtime"
    ) from exc

# python detect_webcam_bigbox_deepsort.py --camera 1


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
    sticky: bool = False


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


def pair_same_class_boxes(corners: List[CornerDetection]) -> List[Tuple[CornerDetection, CornerDetection]]:
    by_class: Dict[int, List[CornerDetection]] = {}
    for det in corners:
        by_class.setdefault(det.class_id, []).append(det)

    pairs: List[Tuple[CornerDetection, CornerDetection]] = []
    for class_corners in by_class.values():
        ordered = sorted(class_corners, key=lambda d: d.conf, reverse=True)
        for i in range(0, len(ordered) - 1, 2):
            pairs.append((ordered[i], ordered[i + 1]))

    return pairs


def card_value_from_label(label: str) -> int:
    clean = label.strip().upper()
    match = re.match(r"^(10|[2-9]|[AJQK])", clean)
    if not match:
        return 0
    rank = match.group(1)
    if rank in {"J", "Q", "K"}:
        return 10
    if rank == "A":
        return 1
    return int(rank)


def draw_count_panel(frame, values: List[int]) -> None:
    tuple_text = "(" + ",".join(str(v) for v in values) + ")" if values else "()"
    total_text = f"Running Total: {sum(values)}"
    values_text = f"Cards: {tuple_text}"
    lines = [values_text, total_text]

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
    max_seen_track_id = 0

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
            pairs = pair_same_class_boxes(corners)

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
                max_seen_track_id = max(max_seen_track_id, track_id)
                memory_tracks[track_id] = MemoryTrack(
                    track_id=track_id,
                    label=label,
                    bbox=(x1, y1, x2, y2),
                    missing_frames=0,
                    sticky=(track_id in memory_tracks and memory_tracks[track_id].sticky),
                )

            for track_id in list(memory_tracks.keys()):
                if track_id in seen_track_ids:
                    continue
                memory_tracks[track_id].missing_frames += 1
                if track_id < max_seen_track_id:
                    # Keep older IDs pinned once newer IDs exist.
                    memory_tracks[track_id].sticky = True
                if memory_tracks[track_id].sticky:
                    continue
                if memory_tracks[track_id].missing_frames > args.memory_frames:
                    del memory_tracks[track_id]

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
                        suffix = ""
                    elif mem.sticky:
                        suffix = f" (sticky mem {mem.missing_frames})"
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

                active_values = [
                    card_value_from_label(mem.label)
                    for _, mem in sorted(memory_tracks.items(), key=lambda item: item[0])
                ]
                active_values = [v for v in active_values if v > 0]
                draw_count_panel(annotated, active_values)

                cv2.imshow("Playing Card Big Box + DeepSORT", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
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
