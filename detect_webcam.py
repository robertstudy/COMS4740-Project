import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# python detect_webcam.py --camera 1 (for external webcam)

def resolve_weights_path(project_root: Path, weights: str) -> Path:
    p = Path(weights)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Weights not found: {p}")
    return p

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect playing cards from webcam with a trained YOLO model."
    )
    parser.add_argument(
        "--weights",
        #default="runs/playing_cards_yolo/weights/best_supplemental.pt",
        default="runs/playing_cards_yolo/weights/best_kaggle.pt",
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
        help="Minimum confidence for a box to be drawn.",
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
        "--no-show",
        action="store_true",
        help="Do not open a preview window (still runs inference; useful headless).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    weights_path = resolve_weights_path(project_root, args.weights)

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

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
    print(f"Inference: conf={args.conf} imgsz={args.imgsz} device={args.device}")
    print("Press 'q' in the preview window to quit.\n")

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

            if not args.no_show:
                annotated = results[0].plot()
                cv2.imshow("Playing Card Detection", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "\nTips: Try --camera 1, reduce --imgsz, or use --device cpu if GPU errors occur.",
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
