"""
Drone Visual Stabilization — Kalman Filter
============================================
Same optical flow pipeline as stabilizer.py but replaces the Butterworth
low-pass filter with a Kalman filter for trajectory smoothing.

The Kalman filter models the camera trajectory as a dynamic system:
  - State: [position, velocity] for each axis (tx, ty, rotation)
  - Prediction: camera continues moving at current velocity
  - Update: correct prediction with observed optical flow measurement

This is the standard approach in embedded drone navigation systems —
the same filter used for IMU/GPS fusion in flight controllers (PX4, ArduPilot).

Author: Benjamin Madar
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

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


# ─────────────────────────────────────────────
#  Kalman smoother
# ─────────────────────────────────────────────

class KalmanSmoother1D:
    """
    1D Kalman filter for smoothing a scalar signal (position).

    State vector: [position, velocity]
    Observation:  [position]

    Process noise Q controls how much we trust the motion model.
    Measurement noise R controls how much we trust the observations.
    High Q/R ratio → trusts observations more → less smoothing.
    Low Q/R ratio  → trusts model more        → more smoothing.
    """

    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 5e-2):
        self.Q = process_noise
        self.R = measurement_noise

        # State: [pos, vel]
        self.x = np.zeros(2)
        # Covariance
        self.P = np.eye(2) * 1.0

        # State transition: pos += vel
        self.F = np.array([[1, 1],
                           [0, 1]])
        # Observation: we observe position only
        self.H = np.array([[1, 0]])

        # Noise matrices
        self.Q_mat = np.array([[self.Q, 0],
                               [0,      self.Q]])
        self.R_mat = np.array([[self.R]])

    def update(self, measurement: float) -> float:
        """Run one Kalman predict+update cycle. Returns smoothed position."""
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q_mat

        # Kalman gain
        S = self.H @ P_pred @ self.H.T + self.R_mat
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update
        z = np.array([measurement])
        self.x = x_pred + K @ (z - self.H @ x_pred)
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return self.x[0]


def smooth_trajectory_kalman(trajectory: np.ndarray,
                              process_noise: float = 1e-4,
                              measurement_noise: float = 5e-2) -> np.ndarray:
    """
    Apply independent 1D Kalman filters to each trajectory axis (tx, ty, rot).
    Runs forward then backward (RTS-style) for zero-lag smoothing.
    """
    smoothed = np.zeros_like(trajectory)

    for col in range(trajectory.shape[1]):
        signal = trajectory[:, col]

        # Forward pass
        kf = KalmanSmoother1D(process_noise, measurement_noise)
        fwd = np.zeros_like(signal)
        for i, z in enumerate(signal):
            fwd[i] = kf.update(z)

        # Backward pass
        kf = KalmanSmoother1D(process_noise, measurement_noise)
        bwd = np.zeros_like(signal)
        for i, z in enumerate(reversed(signal)):
            bwd[len(signal) - 1 - i] = kf.update(z)

        # Average forward + backward → zero phase lag
        smoothed[:, col] = (fwd + bwd) / 2.0

    return smoothed


# ─────────────────────────────────────────────
#  Optical flow (same as stabilizer.py)
# ─────────────────────────────────────────────

def estimate_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple:
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
    if prev_pts is None or len(prev_pts) < 10:
        return 0.0, 0.0, 0.0
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **LK_PARAMS)
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]
    if len(good_prev) < 6:
        return 0.0, 0.0, 0.0
    transform, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
    if transform is None:
        return 0.0, 0.0, 0.0
    dx = transform[0, 2]
    dy = transform[1, 2]
    da = np.arctan2(transform[1, 0], transform[0, 0])
    return dx, dy, da


def apply_transform(frame, dx, dy, da):
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    cos_a, sin_a = np.cos(da), np.sin(da)
    M = np.array([
        [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + dx],
        [sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx + dy],
    ], dtype=np.float32)
    return cv2.warpAffine(frame, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def draw_overlay(frame, prev_pts, curr_pts, dx, dy, frame_idx):
    out = frame.copy()
    if prev_pts is not None and curr_pts is not None:
        for p, c in zip(prev_pts.reshape(-1, 2), curr_pts.reshape(-1, 2)):
            cv2.arrowedLine(out, tuple(p.astype(int)), tuple(c.astype(int)),
                            (0, 255, 0), 1, tipLength=0.3)
            cv2.circle(out, tuple(p.astype(int)), 2, (0, 200, 255), -1)
    magnitude = np.sqrt(dx**2 + dy**2)
    bar_len = min(int(magnitude * 3), 150)
    cv2.rectangle(out, (10, 10), (160, 25), (50, 50, 50), -1)
    cv2.rectangle(out, (10, 10), (10 + bar_len, 25), (0, 200, 255), -1)
    cv2.putText(out, f"Frame: {frame_idx}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(out, f"dx={dx:+.1f}px  dy={dy:+.1f}px", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(out, f"Motion: {magnitude:.1f}px", (10, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    cv2.putText(out, "Kalman filter", (10, 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)
    return out


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────

def stabilize(input_path: str, output_path: str, show: bool = False,
              process_noise: float = 1e-4,
              measurement_noise: float = 5e-2) -> None:

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Input : {input_path}")
    print(f"[INFO] Frames: {n_frames} | FPS: {fps:.1f} | Size: {w}x{h}")
    print(f"[INFO] Kalman Q={process_noise} R={measurement_noise}")

    # ── Pass 1: trajectory ─────────────────────────────────────
    print("[PASS 1] Estimating camera trajectory...")
    transforms = []
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
            print(f"  {i+1}/{n_frames-1}")

    transforms = np.array(transforms, dtype=np.float64)
    trajectory = np.cumsum(transforms, axis=0)

    # ── Kalman smoothing ───────────────────────────────────────
    print("[INFO] Applying Kalman filter...")
    smoothed    = smooth_trajectory_kalman(trajectory, process_noise, measurement_noise)
    corrections = smoothed - trajectory

    print(f"[INFO] Max correction: "
          f"dx={np.abs(corrections[:,0]).max():.1f}px  "
          f"dy={np.abs(corrections[:,1]).max():.1f}px")

    # ── Pass 2: apply corrections ──────────────────────────────
    print("[PASS 2] Applying stabilization...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for i, (dx_c, dy_c, da_c) in enumerate(corrections):
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
        curr_pts = None
        if prev_pts is not None:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **LK_PARAMS)
            prev_pts = prev_pts[status == 1]
            curr_pts = curr_pts[status == 1]

        stabilized       = apply_transform(curr, dx_c, dy_c, -da_c)
        original_overlay = draw_overlay(curr, prev_pts, curr_pts,
                                        transforms[i, 0], transforms[i, 1], i)
        side_by_side     = np.hstack([original_overlay, stabilized])

        cv2.putText(side_by_side, "ORIGINAL", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(side_by_side, "STABILIZED (Kalman)", (w + 10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        out_writer.write(side_by_side)
        if show:
            cv2.imshow("Drone Stabilization — Kalman", side_by_side)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        prev_gray = curr_gray
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(corrections)}")

    cap.release()
    out_writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"[DONE] Output saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drone video stabilization using Kalman filter"
    )
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--process-noise", type=float, default=1e-4,
                        help="Kalman process noise Q (default: 1e-4). "
                             "Higher = trusts observations more = less smoothing.")
    parser.add_argument("--measurement-noise", type=float, default=5e-2,
                        help="Kalman measurement noise R (default: 5e-2). "
                             "Higher = trusts model more = more smoothing.")
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / (p.stem + "_kalman" + p.suffix))

    stabilize(args.input, args.output,
              show=args.show,
              process_noise=args.process_noise,
              measurement_noise=args.measurement_noise)
