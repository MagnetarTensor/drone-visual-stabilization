"""
Realistic Camera Shake Generator
==================================
Simulates periodic drone camera shake — stable phases alternating with
shake bursts, like a real drone hitting wind gusts or turbulence.

Usage:
    python add_shake.py input.mp4
    python add_shake.py input.mp4 -o shaky.mp4 --intensity 1.0
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def generate_shake(n: int, fps: float, intensity: float = 1.0, seed: int = 42) -> tuple:
    """
    Generate periodic shake bursts separated by calm phases.

    Structure:
      - Envelope: smooth sine-based mask that creates calm/shake alternation
      - Base shake: gentle mid-frequency jitter (2-4 Hz)
      - Occasional impulse: single spike every ~3-5 seconds
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps

    # ── Shake envelope — creates calm/shake rhythm ───────────────
    # Oscillates between 0 (calm) and 1 (full shake) every ~2.5s
    envelope = (np.sin(2 * np.pi * 0.2 * t) ** 2) * 0.7
    # Add a second slower oscillation for variety
    envelope += (np.sin(2 * np.pi * 0.08 * t + 1.2) ** 2) * 0.3
    envelope = np.clip(envelope, 0, 1)

    def layer(freq, amp, phase_seed):
        phase = rng.uniform(0, 2 * np.pi)
        return amp * np.sin(2 * np.pi * freq * t + phase)

    # ── Base shake (gentle, 2-4 Hz) ──────────────────────────────
    x = layer(2.5, 3.0, 1) + layer(3.8, 1.5, 2)
    y = layer(2.1, 3.0, 3) + layer(4.2, 1.5, 4)
    r = layer(1.8, 0.25, 5)

    # Apply envelope — shake only during burst phases
    x *= envelope
    y *= envelope
    r *= envelope

    # ── Rare impulse spikes (~every 4s) ──────────────────────────
    spike_interval = int(fps * 4)
    for sf in range(spike_interval, n - 20, spike_interval + rng.integers(-30, 30)):
        amp = rng.uniform(3, 6)
        decay = np.exp(-np.arange(min(15, n - sf)) * 0.5)
        x[sf:sf + len(decay)] += rng.choice([-1, 1]) * amp * decay
        y[sf:sf + len(decay)] += rng.choice([-1, 1]) * amp * decay

    # ── Scale by intensity ────────────────────────────────────────
    x *= intensity
    y *= intensity
    r *= intensity

    return x, y, r


def add_shake(input_path: str, output_path: str, intensity: float = 1.0) -> None:

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Adding periodic shake — {n_frames} frames, intensity={intensity}")
    shake_x, shake_y, shake_r = generate_shake(n_frames, fps, intensity)
    print(f"[INFO] X: {shake_x.std():.1f}px rms | Y: {shake_y.std():.1f}px rms")

    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    cx, cy     = w / 2.0, h / 2.0

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        da = np.radians(shake_r[i])
        cos_a, sin_a = np.cos(da), np.sin(da)
        M = np.array([
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + shake_x[i]],
            [sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx + shake_y[i]],
        ], dtype=np.float32)
        shaky = cv2.warpAffine(frame, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        out_writer.write(shaky)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_frames}")

    cap.release()
    out_writer.release()
    print(f"[DONE] Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--intensity", type=float, default=1.0,
                        help="Shake intensity multiplier (default: 1.0)")
    args = parser.parse_args()
    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / (p.stem + "_shaky" + p.suffix))
    add_shake(args.input, args.output, intensity=args.intensity)
