"""
Drone Visual Stabilization via Optical Flow
============================================
Estimates camera motion frame-by-frame using Lucas-Kanade optical flow
and applies a correction to stabilize the video.

Directly inspired by visual odometry principles used in embedded drone systems
(e.g. Parrot Anafi UKR GPS-denied navigation).

Author: Benjamin Madar
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from scipy.signal import butter, filtfilt


# ─────────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────────
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=300,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=3,
)

CUTOFF_HZ    = 1.5  # low-pass cutoff in Hz — shake is above, intentional motion below
FILTER_ORDER = 4    # Butterworth order — higher = steeper cutoff


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def smooth_trajectory(trajectory: np.ndarray, fps: float,
                      cutoff_hz: float = CUTOFF_HZ,
                      order: int = FILTER_ORDER) -> np.ndarray:
    """
    Smooth camera trajectory using a Butterworth low-pass filter.
    Preserves intentional motion (below cutoff_hz) and removes shake (above).
    Uses filtfilt for zero-phase distortion — same approach as IMU filtering on
    embedded drone systems.
    """
    nyquist = fps / 2.0
    normal_cutoff = np.clip(cutoff_hz / nyquist, 1e-4, 0.999)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    smoothed = np.copy(trajectory)
    for col in range(trajectory.shape[1]):
        smoothed[:, col] = filtfilt(b, a, trajectory[:, col])
    return smoothed


def estimate_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple:
    """
    Estimate rigid motion (dx, dy, dangle) between two consecutive frames
    using Lucas-Kanade sparse optical flow.

    Returns (dx, dy, da) — translation in pixels and rotation in radians.
    """
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)

    if prev_pts is None or len(prev_pts) < 10:
        return 0.0, 0.0, 0.0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **LK_PARAMS
    )

    # Keep only successfully tracked points
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]

    if len(good_prev) < 6:
        return 0.0, 0.0, 0.0

    # Estimate affine transform (rotation + translation)
    transform, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)

    if transform is None:
        return 0.0, 0.0, 0.0

    dx = transform[0, 2]
    dy = transform[1, 2]
    da = np.arctan2(transform[1, 0], transform[0, 0])

    return dx, dy, da


def apply_transform(frame: np.ndarray, dx: float, dy: float, da: float) -> np.ndarray:
    """Apply a rigid correction transform to a frame."""
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2

    # Build correction matrix
    cos_a, sin_a = np.cos(da), np.sin(da)
    M = np.array([
        [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + dx],
        [sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx + dy],
    ], dtype=np.float32)

    stabilized = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
    return stabilized


def draw_overlay(frame: np.ndarray, prev_pts, curr_pts, dx: float, dy: float,
                 frame_idx: int) -> np.ndarray:
    """Draw optical flow vectors and motion info on frame."""
    out = frame.copy()

    # Draw flow vectors
    if prev_pts is not None and curr_pts is not None:
        for p, c in zip(prev_pts.reshape(-1, 2), curr_pts.reshape(-1, 2)):
            pt1 = tuple(p.astype(int))
            pt2 = tuple(c.astype(int))
            cv2.arrowedLine(out, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)
            cv2.circle(out, pt1, 2, (0, 200, 255), -1)

    # Motion magnitude bar
    magnitude = np.sqrt(dx**2 + dy**2)
    bar_max = 50
    bar_len = min(int(magnitude * 3), bar_max * 3)
    cv2.rectangle(out, (10, 10), (10 + bar_max * 3, 25), (50, 50, 50), -1)
    cv2.rectangle(out, (10, 10), (10 + bar_len, 25), (0, 200, 255), -1)

    # Text info
    cv2.putText(out, f"Frame: {frame_idx}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(out, f"dx={dx:+.1f}px  dy={dy:+.1f}px", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(out, f"Motion magnitude: {magnitude:.1f}px", (10, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    return out


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────
def stabilize(input_path: str, output_path: str, show: bool = False,
              cutoff: float = CUTOFF_HZ) -> None:

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Input : {input_path}")
    print(f"[INFO] Frames: {n_frames} | FPS: {fps:.1f} | Size: {w}x{h}")

    # ── Pass 1: accumulate trajectory ──────────────────────────────────────
    print("[PASS 1] Estimating camera trajectory...")
    transforms = []   # list of (dx, dy, da) per frame transition

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for i in range(n_frames - 1):
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        dx, dy, da = estimate_motion(prev_gray, curr_gray)
        transforms.append((dx, dy, da))
        prev_gray = curr_gray

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_frames - 1} frames processed")

    transforms = np.array(transforms, dtype=np.float64)   # (N, 3)

    # Cumulative trajectory
    trajectory = np.cumsum(transforms, axis=0)

    # Smooth trajectory
    smoothed = smooth_trajectory(trajectory, fps, cutoff_hz=cutoff)

    # Correction = difference between smooth and raw trajectory
    corrections = smoothed - trajectory   # (N, 3)

    print(f"[INFO] Max translation correction: "
          f"dx={np.abs(corrections[:,0]).max():.1f}px, "
          f"dy={np.abs(corrections[:,1]).max():.1f}px")

    # ── Pass 2: apply corrections ───────────────────────────────────────────
    print("[PASS 2] Applying stabilization...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for i, (dx_c, dy_c, da_c) in enumerate(corrections):
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Raw flow for overlay
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
        curr_pts = None
        if prev_pts is not None:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **LK_PARAMS
            )
            prev_pts = prev_pts[status == 1]
            curr_pts = curr_pts[status == 1]

        # Stabilized frame
        stabilized = apply_transform(curr, dx_c, dy_c, -da_c)

        # Overlay on original
        original_overlay = draw_overlay(
            curr, prev_pts, curr_pts,
            transforms[i, 0], transforms[i, 1], i
        )

        # Side-by-side: original (left) | stabilized (right)
        side_by_side = np.hstack([original_overlay, stabilized])

        # Labels
        cv2.putText(side_by_side, "ORIGINAL", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(side_by_side, "STABILIZED", (w + 10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        out_writer.write(side_by_side)

        if show:
            cv2.imshow("Drone Visual Stabilization", side_by_side)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        prev_gray = curr_gray

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(corrections)} frames stabilized")

    cap.release()
    out_writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"[DONE] Output saved: {output_path}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drone video stabilization via Lucas-Kanade optical flow"
    )
    parser.add_argument("input", help="Path to input video (e.g. drone_footage.mp4)")
    parser.add_argument("-o", "--output", default=None,
                        help="Path to output video (default: input_stabilized.mp4)")
    parser.add_argument("--cutoff", type=float, default=CUTOFF_HZ,
                        help=f"Low-pass cutoff frequency in Hz (default: {CUTOFF_HZ}). Lower = smoother.")
    parser.add_argument("--show", action="store_true",
                        help="Display video while processing")

    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / (p.stem + "_stabilized" + p.suffix))

    stabilize(args.input, args.output, show=args.show, cutoff=args.cutoff)
