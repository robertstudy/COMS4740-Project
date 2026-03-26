
# python undistort_webcam.py

from __future__ import annotations

import argparse
import sys
from typing import Any

import cv2
import numpy as np

# Trackbar maxima (integer-only in OpenCV); values are mapped in callbacks.
K_SLIDER_MAX = 2000  # center 1000 -> coefficient 0
P_SLIDER_MAX = 2000
FOCAL_MIN = 200
FOCAL_MAX = 2500

# OpenCV distCoeffs order is (k1, k2, p1, p2, k3). Labels spell out what that means.
# k1–k3: radial distortion (how much straight lines bow away from the center).
# p1, p2: tangential / decentering (lens not perfectly centered on the sensor).
TB_RADIAL_R2 = "radial r2 (main barrel / pincushion)"
TB_RADIAL_R4 = "radial r4 (higher order)"
TB_RADIAL_R6 = "radial r6 (higher order)"
TB_TANGENTIAL_P1 = "tangential p1 (decentering)"
TB_TANGENTIAL_P2 = "tangential p2 (decentering)"
TB_FOCAL_PX = "focal length (pixels)"


def _k_from_slider(v: int) -> float:
    """Map 0..K_SLIDER_MAX to roughly -1 .. +1 (center at 1000 -> 0)."""
    return (float(v) - K_SLIDER_MAX / 2.0) / (K_SLIDER_MAX / 2.0)


def _p_from_slider(v: int) -> float:
    """Tangential coeffs are usually smaller."""
    return (float(v) - P_SLIDER_MAX / 2.0) / (P_SLIDER_MAX / 2.0) * 0.1


def _slider_from_k(k: float) -> int:
    return int(round(k * (K_SLIDER_MAX / 2.0) + K_SLIDER_MAX / 2.0))


def _slider_from_p(p: float) -> int:
    return int(round((p / 0.1) * (P_SLIDER_MAX / 2.0) + P_SLIDER_MAX / 2.0))


def build_camera_matrix(w: int, h: int, focal_px: float) -> np.ndarray:
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    return np.array(
        [[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment with OpenCV undistortion on a live webcam (no ML)."
    )
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        default=0,
        metavar="N",
        help="Webcam device index (0 = default). Example: --camera 1 or -c 1.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="cv2.getOptimalNewCameraMatrix alpha in [0,1]: 1 keeps all pixels (may add border), 0 crops.",
    )
    parser.add_argument(
        "--focal",
        type=float,
        default=None,
        help="Initial focal length in pixels (fx=fy). Default: ~0.9 * max(w,h).",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open webcam index {args.camera}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))

    ok, probe = cap.read()
    if not ok:
        print("Failed to read first frame.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    h, w = probe.shape[:2]
    focal0 = args.focal if args.focal is not None else 0.9 * float(max(w, h))
    focal0 = float(np.clip(focal0, FOCAL_MIN, FOCAL_MAX))

    window = "Undistortion (raw | corrected)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    state: dict[str, Any] = {"dirty": True}

    def mark_dirty(_: int | None = None) -> None:
        state["dirty"] = True

    cv2.createTrackbar(
        TB_RADIAL_R2, window, _slider_from_k(0.0), K_SLIDER_MAX, mark_dirty
    )
    cv2.createTrackbar(
        TB_RADIAL_R4, window, _slider_from_k(0.0), K_SLIDER_MAX, mark_dirty
    )
    cv2.createTrackbar(
        TB_RADIAL_R6, window, _slider_from_k(0.0), K_SLIDER_MAX, mark_dirty
    )
    cv2.createTrackbar(
        TB_TANGENTIAL_P1, window, _slider_from_p(0.0), P_SLIDER_MAX, mark_dirty
    )
    cv2.createTrackbar(
        TB_TANGENTIAL_P2, window, _slider_from_p(0.0), P_SLIDER_MAX, mark_dirty
    )
    cv2.createTrackbar(
        TB_FOCAL_PX,
        window,
        int(focal0),
        FOCAL_MAX,
        mark_dirty,
    )

    map1: np.ndarray | None = None
    map2: np.ndarray | None = None
    new_mtx: np.ndarray | None = None

    print(
        f"Webcam {args.camera}  frame {w}x{h}  alpha={args.alpha}\n"
        "Sliders: k1–k3 = radial distortion (r^2, r^4, r^6 terms); "
        "p1,p2 = tangential; focal = rough fx=fy in pixels.\n"
        "Adjust trackbars; maps update when you change a value.\n"
        "Keys: q/ESC quit, r reset.\n"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed.", file=sys.stderr)
                break

            k1 = _k_from_slider(cv2.getTrackbarPos(TB_RADIAL_R2, window))
            k2 = _k_from_slider(cv2.getTrackbarPos(TB_RADIAL_R4, window))
            k3 = _k_from_slider(cv2.getTrackbarPos(TB_RADIAL_R6, window))
            p1 = _p_from_slider(cv2.getTrackbarPos(TB_TANGENTIAL_P1, window))
            p2 = _p_from_slider(cv2.getTrackbarPos(TB_TANGENTIAL_P2, window))
            focal = float(
                np.clip(
                    cv2.getTrackbarPos(TB_FOCAL_PX, window), FOCAL_MIN, FOCAL_MAX
                )
            )

            dist = np.array([[k1, k2, p1, p2, k3]], dtype=np.float64)
            mtx = build_camera_matrix(w, h, focal)

            if state["dirty"] or map1 is None:
                new_mtx, _ = cv2.getOptimalNewCameraMatrix(
                    mtx, dist, (w, h), args.alpha
                )
                map1, map2 = cv2.initUndistortRectifyMap(
                    mtx,
                    dist,
                    None,
                    new_mtx,
                    (w, h),
                    cv2.CV_16SC2,
                )
                state["dirty"] = False

            assert map1 is not None and map2 is not None
            undist = cv2.remap(
                frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            label = (
                f"radial k1={k1:.4f} k2={k2:.4f} k3={k3:.4f} "
                f"tang p1={p1:.5f} p2={p2:.5f} focal={focal:.0f}px"
            )
            vis = np.hstack([frame, undist])
            y0 = min(28, h - 4)
            cv2.putText(
                vis,
                label,
                (8, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                cv2.setTrackbarPos(TB_RADIAL_R2, window, _slider_from_k(0.0))
                cv2.setTrackbarPos(TB_RADIAL_R4, window, _slider_from_k(0.0))
                cv2.setTrackbarPos(TB_RADIAL_R6, window, _slider_from_k(0.0))
                cv2.setTrackbarPos(TB_TANGENTIAL_P1, window, _slider_from_p(0.0))
                cv2.setTrackbarPos(TB_TANGENTIAL_P2, window, _slider_from_p(0.0))
                cv2.setTrackbarPos(
                    TB_FOCAL_PX,
                    window,
                    int(np.clip(focal0, FOCAL_MIN, FOCAL_MAX)),
                )
                state["dirty"] = True
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
