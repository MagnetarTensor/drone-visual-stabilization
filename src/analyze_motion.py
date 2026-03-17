"""
Camera Motion Analysis
======================
Plots the raw vs smoothed camera trajectory from a drone video.
Useful for visualizing the stabilization effect before rendering the full output.

Usage:
    python analyze_motion.py input.mp4
    python analyze_motion.py input.mp4 --smoothing 50
"""

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from stabilizer import estimate_motion, smooth_trajectory, FEATURE_PARAMS, LK_PARAMS


def analyze(input_path: str, smoothing: int = 30) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Analyzing {n_frames} frames at {fps:.1f} FPS...")

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

    transforms = np.array(transforms)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed = smooth_trajectory(trajectory, smoothing)

    time = np.arange(len(trajectory)) / fps

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Camera Motion Analysis — Drone Visual Odometry", fontsize=14, fontweight="bold")

    labels = ["Translation X (px)", "Translation Y (px)", "Rotation (rad)"]
    colors_raw = ["#E74C3C", "#3498DB", "#2ECC71"]
    colors_smooth = ["#C0392B", "#2980B9", "#27AE60"]

    for idx, (ax, label, cr, cs) in enumerate(zip(axes, labels, colors_raw, colors_smooth)):
        ax.plot(time, trajectory[:, idx], color=cr, alpha=0.4, linewidth=0.8, label="Raw trajectory")
        ax.plot(time, smoothed[:, idx], color=cs, linewidth=1.8, label=f"Smoothed (r={smoothing})")
        ax.fill_between(time, trajectory[:, idx], smoothed[:, idx],
                        alpha=0.1, color=cr)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()

    # Save to assets/ directory (project root)
    p = Path(input_path).resolve()
    assets = p.parent
    for candidate in [p.parent, p.parent.parent]:
        if (candidate / "assets").exists() or (candidate / "README.md").exists():
            assets = candidate / "assets"
            assets.mkdir(exist_ok=True)
            break
    out_path = str(assets / (Path(input_path).stem + "_motion_analysis.png"))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[DONE] Plot saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot camera motion from drone video")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("--smoothing", type=int, default=30, help="Smoothing radius")
    args = parser.parse_args()
    analyze(args.input, args.smoothing)
