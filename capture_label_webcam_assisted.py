import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

# Example:
# python capture_label_webcam_assisted.py --external-webcam --split train --output-dir supplemental_capture


@dataclass
class BoxAnnotation:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    source: str = "manual"  # "manual" or "auto"
    track_id: int | None = None

    def contains(self, x: int, y: int) -> bool:
        left = min(self.x1, self.x2)
        right = max(self.x1, self.x2)
        top = min(self.y1, self.y2)
        bottom = max(self.y1, self.y2)
        return left <= x <= right and top <= y <= bottom


class AnnotationSession:
    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self.current_class_idx = 0
        self.annotations: list[BoxAnnotation] = []
        self.selected_idx: int | None = None
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_x = 0
        self.current_y = 0

    def reset(self) -> None:
        self.annotations.clear()
        self.selected_idx = None
        self.drawing = False

    def change_class(self, step: int) -> None:
        self.current_class_idx = (self.current_class_idx + step) % len(self.class_names)

    def set_class(self, idx: int) -> None:
        if 0 <= idx < len(self.class_names):
            self.current_class_idx = idx

    def select_box_at(self, x: int, y: int) -> bool:
        for i in range(len(self.annotations) - 1, -1, -1):
            if self.annotations[i].contains(x, y):
                self.selected_idx = i
                return True
        self.selected_idx = None
        return False

    def relabel_selected(self, class_id: int) -> bool:
        if self.selected_idx is None:
            return False
        if not (0 <= self.selected_idx < len(self.annotations)):
            self.selected_idx = None
            return False
        self.annotations[self.selected_idx].class_id = class_id
        return True

    def delete_selected(self) -> bool:
        if self.selected_idx is None:
            return False
        if not (0 <= self.selected_idx < len(self.annotations)):
            self.selected_idx = None
            return False
        self.annotations.pop(self.selected_idx)
        self.selected_idx = None
        return True

    def replace_auto_boxes(self, new_auto_boxes: list[BoxAnnotation]) -> None:
        manual = [ann for ann in self.annotations if ann.source != "auto"]
        self.annotations = manual + new_auto_boxes
        self.selected_idx = None


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


def predict_boxes(model: YOLO, frame, conf: float, max_class_id: int, use_tracking: bool) -> list[BoxAnnotation]:
    if use_tracking:
        result = model.track(source=frame, conf=conf, persist=True, verbose=False)[0]
    else:
        result = model.predict(source=frame, conf=conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    track_ids = None
    if use_tracking and result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy()

    output: list[BoxAnnotation] = []
    for i, (box_xyxy, class_float) in enumerate(zip(xyxy, cls)):
        class_id = int(class_float)
        if not (0 <= class_id <= max_class_id):
            continue
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        track_id = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None
        output.append(
            BoxAnnotation(
                class_id=class_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                source="auto",
                track_id=track_id,
            )
        )
    return output


def draw_overlay(frame, session: AnnotationSession, mode_text: str):
    overlay = frame.copy()

    for i, ann in enumerate(session.annotations):
        selected = i == session.selected_idx
        color = (255, 200, 0) if ann.source == "auto" else (0, 255, 0)
        if selected:
            color = (0, 0, 255)

        cv2.rectangle(overlay, (ann.x1, ann.y1), (ann.x2, ann.y2), color, 2)
        label = session.class_names[ann.class_id]
        cv2.putText(
            overlay,
            label,
            (min(ann.x1, ann.x2), max(18, min(ann.y1, ann.y2) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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

    selected_text = "None"
    if session.selected_idx is not None and 0 <= session.selected_idx < len(session.annotations):
        selected_text = f"{session.selected_idx}:{session.class_names[session.annotations[session.selected_idx].class_id]}"

    if "LIVE" in mode_text:
        info_lines = [
            f"Mode: {mode_text}",
            "Predicted ranks are shown next to cards.",
            "Keys: c=capture  Esc=quit",
        ]
    else:
        info_lines = [
            f"Mode: {mode_text}",
            f"Current class: [{session.current_class_idx}] {session.class_names[session.current_class_idx]}",
            f"Selected box: {selected_text}",
            "Keys: s=save  r=discard  p=refresh auto  x=delete selected  u=undo  Esc=quit",
            "Relabel selected box with: A, 2-9, T(10), J, Q, K",
            "Mouse: click box to select OR drag to draw new box",
        ]

    y = 24
    for line in info_lines:
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22

    return overlay


def save_sample(frame, annotations: list[BoxAnnotation], images_dir: Path, labels_dir: Path) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = images_dir / f"card_{timestamp}.jpg"
    label_path = labels_dir / f"card_{timestamp}.txt"

    ok = cv2.imwrite(str(image_path), frame)
    if not ok:
        raise RuntimeError(f"Could not write image: {image_path}")

    h, w = frame.shape[:2]
    lines = [yolo_line(ann, w, h) for ann in annotations]
    label_path.write_text("\n".join(lines), encoding="utf-8")
    return image_path, label_path


def build_rank_key_map(class_names: list[str]) -> dict[int, int]:
    key_map: dict[int, int] = {}
    for idx, name in enumerate(class_names):
        token = str(name).strip().upper()
        if token == "10":
            key_map[ord("t")] = idx
            key_map[ord("T")] = idx
        elif len(token) == 1 and token in "A23456789JQK":
            key_map[ord(token.lower())] = idx
            key_map[ord(token)] = idx
    return key_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam capture with YOLO-assisted labeling.")
    parser.add_argument("--dataset-dir", default="training_data", help="Dataset root directory.")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"], help="Split to save into.")
    parser.add_argument("--data-yaml", default="training_data/data_local.yaml", help="Path to dataset yaml with class names.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output root. If set, saves to <output-dir>/<split>/{images,labels}.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--external-webcam", action="store_true", help="Use camera index 1.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--allow-empty", action="store_true", help="Allow saving image with no boxes.")
    parser.add_argument("--model", default="runs/playing_cards_yolo/weights/best.pt", help="YOLO weights for auto suggestions.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for auto predictions.")
    parser.add_argument("--no-auto", action="store_true", help="Disable auto-predict on capture.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset_dir = (root / args.dataset_dir).resolve()
    data_yaml_path = (root / args.data_yaml).resolve()
    output_root = (root / args.output_dir).resolve() if args.output_dir else dataset_dir

    camera_index = 1 if args.external_webcam else args.camera
    class_names = load_class_names(data_yaml_path)
    session = AnnotationSession(class_names)
    rank_key_map = build_rank_key_map(class_names)

    images_dir = output_root / args.split / "images"
    labels_dir = output_root / args.split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if not args.no_auto:
        model_path = (root / args.model).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        model = YOLO(str(model_path))

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    window_name = "Webcam YOLO Assisted Labeling"
    cv2.namedWindow(window_name)
    state = {"frozen": None, "live_frame": None, "live_boxes": []}

    def on_mouse(event, x, y, flags, param):
        _ = (flags, param)
        if state["frozen"] is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if session.select_box_at(x, y):
                return
            session.selected_idx = None
            session.drawing = True
            session.start_x, session.start_y = x, y
            session.current_x, session.current_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and session.drawing:
            session.current_x, session.current_y = x, y
        elif event == cv2.EVENT_LBUTTONUP and session.drawing:
            session.drawing = False
            session.current_x, session.current_y = x, y
            if abs(session.current_x - session.start_x) >= 4 and abs(session.current_y - session.start_y) >= 4:
                session.annotations.append(
                    BoxAnnotation(
                        class_id=session.current_class_idx,
                        x1=session.start_x,
                        y1=session.start_y,
                        x2=session.current_x,
                        y2=session.current_y,
                        source="manual",
                    )
                )
                session.selected_idx = len(session.annotations) - 1

    cv2.setMouseCallback(window_name, on_mouse)

    print("Starting webcam assisted label tool.")
    print(f"Saving captures to: {output_root}")
    print(f"Using camera index: {camera_index}")
    print(f"Auto-predict on capture: {'off' if model is None else 'on'}")
    print("Press 'c' to capture; correct labels/boxes; press 's' to save.")

    try:
        while True:
            if state["frozen"] is None:
                ok, frame = cap.read()
                if not ok:
                    continue
                state["live_frame"] = frame.copy()
                if model is not None:
                    state["live_boxes"] = predict_boxes(
                        model=model,
                        frame=frame,
                        conf=args.conf,
                        max_class_id=len(class_names) - 1,
                        use_tracking=True,
                    )
                    session.replace_auto_boxes(state["live_boxes"])
                else:
                    session.replace_auto_boxes([])
                shown = draw_overlay(frame, session, "LIVE (press c to capture)")
                cv2.imshow(window_name, shown)
            else:
                shown = draw_overlay(state["frozen"], session, "ANNOTATE")
                cv2.imshow(window_name, shown)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if key == 27:
                break
                continue
            if key == ord("x"):
                session.delete_selected()
                continue
            if key == ord("u"):
                if session.annotations:
                    session.annotations.pop()
                    session.selected_idx = None
                continue

            if key in rank_key_map:
                class_idx = rank_key_map[key]
                if not session.relabel_selected(class_idx):
                    session.set_class(class_idx)
                continue

            if key == ord("c") and state["frozen"] is None:
                if state["live_frame"] is not None:
                    state["frozen"] = state["live_frame"].copy()
                    session.reset()
                    if model is not None:
                        session.replace_auto_boxes(list(state["live_boxes"]))
                        print(f"Loaded {len(state['live_boxes'])} tracked boxes from live view.")
                continue

            if key == ord("p") and state["frozen"] is not None and model is not None:
                auto_boxes = predict_boxes(
                    model=model,
                    frame=state["frozen"],
                    conf=args.conf,
                    max_class_id=len(class_names) - 1,
                    use_tracking=False,
                )
                session.replace_auto_boxes(auto_boxes)
                print(f"Refreshed auto predictions: {len(auto_boxes)} boxes.")
                continue

            if key == ord("r") and state["frozen"] is not None:
                state["frozen"] = None
                session.reset()
                continue

            if key == ord("s") and state["frozen"] is not None:
                if not session.annotations and not args.allow_empty:
                    print("No boxes drawn. Draw/add at least one, or pass --allow-empty.")
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
