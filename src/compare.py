"""
Butterworth vs Kalman — Side-by-side Comparison
=================================================
Generates a 3-column video:
  [ORIGINAL] | [BUTTERWORTH] | [KALMAN]

With real-time stability metrics displayed at the bottom:
  - Per-column motion variance (lower = more stable)
  - Stability improvement % vs original

Usage:
    python src/compare.py drone_shaky.mp4
    python src/compare.py drone_shaky.mp4 -o assets/compare.mp4 --show
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from scipy.signal import butter, filtfilt

# ── Import shared utilities from sibling modules ─────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from stabilizer_butterworth import (
    estimate_motion, apply_transform,
    FEATURE_PARAMS, LK_PARAMS, CUTOFF_HZ, FILTER_ORDER
)
from stabilizer_kalman import (
    smooth_trajectory_kalman, KalmanSmoother1D
)


# ─────────────────────────────────────────────
#  Butterworth smoother (local, for compare)
# ─────────────────────────────────────────────

def smooth_butterworth(trajectory: np.ndarray, fps: float,
                       cutoff_hz: float = CUTOFF_HZ) -> np.ndarray:
    nyquist = fps / 2.0
    normal_cutoff = np.clip(cutoff_hz / nyquist, 1e-4, 0.999)
    b, a = butter(FILTER_ORDER, normal_cutoff, btype="low", analog=False)
    smoothed = np.copy(trajectory)
    for col in range(trajectory.shape[1]):
        smoothed[:, col] = filtfilt(b, a, trajectory[:, col])
    return smoothed


# ─────────────────────────────────────────────
#  Overlay helpers
# ─────────────────────────────────────────────

def draw_flow_overlay(frame: np.ndarray, prev_gray, curr_gray,
                      dx: float, dy: float, frame_idx: int,
                      label: str, label_color: tuple) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Optical flow vectors
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
    if prev_pts is not None:
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **LK_PARAMS)
        for p, c, s in zip(prev_pts.reshape(-1, 2),
                           curr_pts.reshape(-1, 2),
                           status.flatten()):
            if s:
                cv2.arrowedLine(out, tuple(p.astype(int)),
                                tuple(c.astype(int)),
                                (0, 255, 0), 1, tipLength=0.4)
                cv2.circle(out, tuple(p.astype(int)), 2, (0, 200, 255), -1)

    # Motion magnitude bar
    mag = np.sqrt(dx**2 + dy**2)
    bar_w = int(min(mag * 4, w - 20))
    cv2.rectangle(out, (10, 10), (w - 10, 28), (40, 40, 40), -1)
    bar_color = (0, 200, 255) if mag < 8 else (0, 100, 255) if mag < 15 else (0, 50, 200)
    cv2.rectangle(out, (10, 10), (10 + bar_w, 28), bar_color, -1)

    # Text overlay
    cv2.putText(out, f"Frame {frame_idx}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(out, f"dx={dx:+.1f} dy={dy:+.1f}px", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(out, f"mag={mag:.1f}px", (10, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

    # Column label at bottom
    cv2.rectangle(out, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.putText(out, label, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2)
    return out


def draw_metrics_bar(frame_orig, frame_bw, frame_kl,
                     var_orig, var_bw, var_kl, bar_h=56) -> np.ndarray:
    """Bottom metrics bar showing stability comparison."""
    total_w = frame_orig.shape[1] * 3
    bar = np.zeros((bar_h, total_w, 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)

    # Compute improvement percentages
    imp_bw = max(0, (1 - var_bw / (var_orig + 1e-8)) * 100)
    imp_kl = max(0, (1 - var_kl / (var_orig + 1e-8)) * 100)

    col_w = frame_orig.shape[1]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # Original
    cv2.putText(bar, f"Variance: {var_orig:.2f}px²",
                (10, 22), font, 0.5, (180, 180, 180), 1)
    cv2.putText(bar, "Reference", (10, 44), font, 0.5, (150, 150, 150), 1)

    # Butterworth
    bw_color = (100, 220, 100) if imp_bw > 20 else (200, 200, 100)
    cv2.putText(bar, f"Variance: {var_bw:.2f}px²",
                (col_w + 10, 22), font, 0.5, (180, 180, 180), 1)
    cv2.putText(bar, f"Shake -{imp_bw:.0f}%",
                (col_w + 10, 44), font, 0.5, bw_color, 1)

    # Kalman
    kl_color = (100, 255, 100) if imp_kl > 30 else (150, 220, 100)
    cv2.putText(bar, f"Variance: {var_kl:.2f}px²",
                (col_w * 2 + 10, 22), font, 0.5, (180, 180, 180), 1)
    cv2.putText(bar, f"Shake -{imp_kl:.0f}%",
                (col_w * 2 + 10, 44), font, 0.5, kl_color, 1)

    # Dividers
    cv2.line(bar, (col_w, 0), (col_w, bar_h), (60, 60, 60), 1)
    cv2.line(bar, (col_w * 2, 0), (col_w * 2, bar_h), (60, 60, 60), 1)

    return bar


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────

def compare(input_path: str, output_path: str, show: bool = False,
            cutoff: float = CUTOFF_HZ) -> None:

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] {n_frames} frames | {fps:.1f} FPS | {w}x{h}")

    # ── Pass 1: estimate trajectory ────────────────────────────
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

    # ── Smooth with both methods ───────────────────────────────
    print("[INFO] Computing Butterworth + Kalman smoothing...")
    smoothed_bw = smooth_butterworth(trajectory, fps, cutoff)
    smoothed_kl = smooth_trajectory_kalman(trajectory)

    corr_bw = smoothed_bw - trajectory
    corr_kl = smoothed_kl - trajectory

    # ── Pass 2: render 3-column video ─────────────────────────
    print("[PASS 2] Rendering comparison video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    bar_h  = 56
    out_h  = h + bar_h
    out_w  = w * 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Rolling variance window for real-time metrics (30 frames)
    WIN = 30
    mag_orig = []
    mag_bw   = []
    mag_kl   = []

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for i, (dx_bw, dy_bw, da_bw) in enumerate(corr_bw):
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        dx_kl, dy_kl, da_kl = corr_kl[i]
        dx_raw, dy_raw       = transforms[i, 0], transforms[i, 1]

        # Stabilized frames
        frame_bw = apply_transform(curr, dx_bw, dy_bw, -da_bw)
        frame_kl = apply_transform(curr, dx_kl, dy_kl, -da_kl)

        # Overlays
        col_orig = draw_flow_overlay(curr, prev_gray, curr_gray,
                                     dx_raw, dy_raw, i,
                                     "ORIGINAL", (255, 255, 255))
        col_bw   = draw_flow_overlay(frame_bw, prev_gray, curr_gray,
                                     dx_bw, dy_bw, i,
                                     "BUTTERWORTH", (100, 220, 255))
        col_kl   = draw_flow_overlay(frame_kl, prev_gray, curr_gray,
                                     dx_kl, dy_kl, i,
                                     "KALMAN", (100, 255, 100))

        # Rolling variance
        mag_orig.append(np.sqrt(dx_raw**2 + dy_raw**2))
        mag_bw.append(np.sqrt(dx_bw**2 + dy_bw**2))
        mag_kl.append(np.sqrt(dx_kl**2 + dy_kl**2))
        if len(mag_orig) > WIN:
            mag_orig.pop(0); mag_bw.pop(0); mag_kl.pop(0)

        var_orig = float(np.var(mag_orig))
        var_bw   = float(np.var(mag_bw))
        var_kl   = float(np.var(mag_kl))

        metrics = draw_metrics_bar(col_orig, col_bw, col_kl,
                                   var_orig, var_bw, var_kl, bar_h)

        frame_3col = np.hstack([col_orig, col_bw, col_kl])
        final      = np.vstack([frame_3col, metrics])

        writer.write(final)

        if show:
            # Scale down for display if too wide
            display = cv2.resize(final, (min(out_w, 1800),
                                         int(out_h * min(1800/out_w, 1))))
            cv2.imshow("Butterworth vs Kalman", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        prev_gray = curr_gray
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(corr_bw)}")

    writer.release()
    cap.release()
    if show:
        cv2.destroyAllWindows()
    print(f"[DONE] Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison: Original | Butterworth | Kalman"
    )
    parser.add_argument("input", help="Shaky input video")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: assets/compare.mp4)")
    parser.add_argument("--cutoff", type=float, default=CUTOFF_HZ,
                        help=f"Butterworth cutoff Hz (default: {CUTOFF_HZ})")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input).resolve()
        assets = p.parent
        for candidate in [p.parent, p.parent.parent]:
            if (candidate / "assets").exists() or (candidate / "README.md").exists():
                assets = candidate / "assets"
                assets.mkdir(exist_ok=True)
                break
        args.output = str(assets / "compare.mp4")

    compare(args.input, args.output, show=args.show, cutoff=args.cutoff)
