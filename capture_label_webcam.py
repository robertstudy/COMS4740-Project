import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import yaml


@dataclass
class BoxAnnotation:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


class AnnotationSession:
    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self.current_class_idx = 0
        self.annotations: list[BoxAnnotation] = []
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_x = 0
        self.current_y = 0

    def reset(self) -> None:
        self.annotations.clear()
        self.drawing = False

    def change_class(self, step: int) -> None:
        count = len(self.class_names)
        self.current_class_idx = (self.current_class_idx + step) % count

    def set_class(self, idx: int) -> None:
        if 0 <= idx < len(self.class_names):
            self.current_class_idx = idx


def load_class_names(data_yaml_path: Path) -> list[str]:
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Could not find class yaml: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Invalid names list in yaml: {data_yaml_path}")
    return [str(name) for name in names]


def yolo_line(box: BoxAnnotation, width: int, height: int) -> str:
    x1 = max(0, min(box.x1, width - 1))
    y1 = max(0, min(box.y1, height - 1))
    x2 = max(0, min(box.x2, width - 1))
    y2 = max(0, min(box.y2, height - 1))
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    xc = ((left + right) / 2.0) / width
    yc = ((top + bottom) / 2.0) / height
    w = (right - left) / width
    h = (bottom - top) / height

    return f"{box.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def draw_overlay(frame, session: AnnotationSession, mode_text: str):
    overlay = frame.copy()

    for ann in session.annotations:
        color = (0, 255, 0)
        cv2.rectangle(overlay, (ann.x1, ann.y1), (ann.x2, ann.y2), color, 2)
        label = session.class_names[ann.class_id]
        cv2.putText(
            overlay,
            label,
            (min(ann.x1, ann.x2), max(18, min(ann.y1, ann.y2) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if session.drawing:
        cv2.rectangle(
            overlay,
            (session.start_x, session.start_y),
            (session.current_x, session.current_y),
            (0, 200, 255),
            2,
        )

    info_lines = [
        f"Mode: {mode_text}",
        f"Current class: [{session.current_class_idx}] {session.class_names[session.current_class_idx]}",
        "Keys: c=capture  s=save  r=discard  u=undo  a/d=class  0-9=quick class  q=quit",
        "Draw boxes: click + drag left mouse",
    ]
    y = 24
    for line in info_lines:
        cv2.putText(
            overlay,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24

    return overlay


def save_sample(
    frozen_frame,
    annotations: list[BoxAnnotation],
    images_dir: Path,
    labels_dir: Path,
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = images_dir / f"card_{timestamp}.jpg"
    label_path = labels_dir / f"card_{timestamp}.txt"

    ok = cv2.imwrite(str(image_path), frozen_frame)
    if not ok:
        raise RuntimeError(f"Could not write image: {image_path}")

    h, w = frozen_frame.shape[:2]
    lines = [yolo_line(ann, w, h) for ann in annotations]
    label_path.write_text("\n".join(lines), encoding="utf-8")
    return image_path, label_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture webcam images and label YOLO boxes.")
    parser.add_argument("--dataset-dir", default="training_data", help="Dataset root directory.")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"], help="Split to save into.")
    parser.add_argument("--data-yaml", default="training_data/data_local.yaml", help="Path to dataset yaml with class names.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional separate output root for captures. If set, saves into <output-dir>/<split>/{images,labels}.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--external-webcam",
        action="store_true",
        help="Use external webcam preset (camera index 1).",
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow saving image even when no boxes are drawn.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset_dir = (root / args.dataset_dir).resolve()
    data_yaml_path = (root / args.data_yaml).resolve()
    output_root = (root / args.output_dir).resolve() if args.output_dir else dataset_dir

    camera_index = args.camera
    if args.external_webcam:
        camera_index = 1

    class_names = load_class_names(data_yaml_path)
    session = AnnotationSession(class_names)

    images_dir = output_root / args.split / "images"
    labels_dir = output_root / args.split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    window_name = "Webcam YOLO Label Capture"
    cv2.namedWindow(window_name)

    state = {"frozen": None}

    def on_mouse(event, x, y, flags, param):
        _ = (flags, param)
        if state["frozen"] is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            session.drawing = True
            session.start_x, session.start_y = x, y
            session.current_x, session.current_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and session.drawing:
            session.current_x, session.current_y = x, y
        elif event == cv2.EVENT_LBUTTONUP and session.drawing:
            session.drawing = False
            session.current_x, session.current_y = x, y

            min_size = 4
            if abs(session.current_x - session.start_x) >= min_size and abs(session.current_y - session.start_y) >= min_size:
                session.annotations.append(
                    BoxAnnotation(
                        class_id=session.current_class_idx,
                        x1=session.start_x,
                        y1=session.start_y,
                        x2=session.current_x,
                        y2=session.current_y,
                    )
                )

    cv2.setMouseCallback(window_name, on_mouse)

    print("Starting webcam label tool.")
    print("Class list:")
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name}")
    print(f"Saving captures to: {output_root}")
    print(f"Using camera index: {camera_index}")
    print("Press 'c' to capture a frame. Annotate with mouse. Press 's' to save image+labels.")

    try:
        while True:
            if state["frozen"] is None:
                ok, frame = cap.read()
                if not ok:
                    print("Warning: camera frame read failed.")
                    continue
                shown = draw_overlay(frame, session, "LIVE (press c to capture)")
                cv2.imshow(window_name, shown)
            else:
                shown = draw_overlay(state["frozen"], session, "ANNOTATE (draw boxes, then s to save)")
                cv2.imshow(window_name, shown)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if key == ord("q"):
                break
            if key == ord("a"):
                session.change_class(-1)
                continue
            if key == ord("d"):
                session.change_class(1)
                continue
            if ord("0") <= key <= ord("9"):
                session.set_class(key - ord("0"))
                continue
            if key == ord("u"):
                if session.annotations:
                    session.annotations.pop()
                continue

            if key == ord("c") and state["frozen"] is None:
                ok, frame = cap.read()
                if ok:
                    state["frozen"] = frame.copy()
                    session.reset()
                continue

            if key == ord("r") and state["frozen"] is not None:
                state["frozen"] = None
                session.reset()
                continue

            if key == ord("s") and state["frozen"] is not None:
                if not session.annotations and not args.allow_empty:
                    print("No boxes drawn. Draw at least one box or use --allow-empty.")
                    continue
                image_path, label_path = save_sample(state["frozen"], session.annotations, images_dir, labels_dir)
                print(f"Saved image: {image_path}")
                print(f"Saved label: {label_path}")
                state["frozen"] = None
                session.reset()
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
