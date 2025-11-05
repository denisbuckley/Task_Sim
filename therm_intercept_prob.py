#!/usr/bin/env python3
"""
Speed-sweep thermal intercept probability (Poisson field, straight line, no detours)

What this does
--------------
- Each trial creates a NEW homogeneous 2D Poisson field of thermal centers (areal density λ / km^2).
- A straight glide segment is flown. Its length depends on AIRSPEED via the sink polar:
      L(v) = v * (H0 - Hmin) / S(v)
  Optionally capped by --cap-km.
- Success if the line segment intersects the disk of ANY **updraft** thermal (circle–segment test).
- Trials are Bernoulli; we aggregate probability and Wilson 95% CI across trials per speed.

Key differences from step-like models
-------------------------------------
- No reseeding inside loops, fresh field per trial → genuinely random outcomes.
- No "at least one thermal" clamp; Poisson can be 0.
- Geometry-based success via robust circle–segment intersection.

Outputs
-------
- Plot: Probability vs speed (+ shaded 95% CI)
- CSV: columns [speed_kmh, trials, success, prob, ci_low, ci_high]

Usage
-----
  python speed_prob_sweep.py \
      --trials 1000 --lambda-a 0.01 --p-up 0.5 \
      --radius-mode fixed --radius-m 250 \
      --h0-m 2500 --hmin-m 500 --cap-km 100 \
      --seed 123

  # Sniffing (cubic) radius style:
  python speed_prob_sweep.py \
      --radius-mode sniff --wt 6.0 --mc-sniff 2.0 --sniff-k 8.0

Notes
-----
- Units: distances in meters internally; λ in thermals/km^2.
- The sampling window is a rectangle tightly padded around the segment so we don’t waste
  time generating far-away thermals that cannot intersect.
"""

from __future__ import annotations
import math
import csv
from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
VERSION = "sim_prob_sweep_v1a"
print(f"[RUN] {VERSION} from {Path(__file__).resolve()}")

# --------------------------- Defaults (edit as you like) ---------------------------
SPEED_MIN_KMH = 100
SPEED_MAX_KMH = 250
TRIALS_DEFAULT = 1000

LAMBDA_A_DEFAULT = 0.01       # thermals / km^2 (areal density)
P_UP_DEFAULT = 0.5            # fraction of thermals that are updrafts

# Polar: simple convex quadratic around a min-sink point (approx LS10-ish)
POLAR_V0 = 27.0               # m/s at min sink
POLAR_S0 = 0.55               # m/s minimum sink
POLAR_K  = 0.0020             # curvature; S(v) = S0 + K*(v - V0)^2

H0_M_DEFAULT = 2500.0         # start altitude (m)
HMIN_M_DEFAULT = 500.0        # minimum (landing) altitude (m)
CAP_KM_DEFAULT = 100.0        # optional cap on segment length (km). Set <=0 to disable.

# Fixed thermal radius default (when --radius-mode fixed)
RADIUS_M_DEFAULT = 250.0      # meters

# Sniffing (cubic) radius defaults (when --radius-mode sniff)
WT_DEFAULT = 6.0              # e.g., Wt (m/s)
MC_SNIFF_DEFAULT = 2.0        # MC threshold for sniff (m/s)
SNIFF_K_DEFAULT = 8.0         # scale factor for cubic radius, r = k * max(Wt - MC, 0)^3  (meters)

OUT_CSV_DEFAULT = "speed_vs_probability.csv"
PLOT_TITLE = "Probability vs Airspeed (Poisson field; straight line; polar-coupled range)"
# ------------------------------------------------------------------------------------


# ------------------------------- Helpers -------------------------------------------
def sink_ms(v_ms: float) -> float:
    """Convex quadratic still-air sink polar (m/s)."""
    return POLAR_S0 + POLAR_K * (v_ms - POLAR_V0) ** 2


def wilson_ci(success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = success / n
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z ** 2) / (4 * n * n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def circle_line_segment_intersections(
    cx: float, cy: float, r: float,
    x1: float, y1: float, x2: float, y2: float,
    eps: float = 1e-12
) -> bool:
    """
    Robust circle–line-segment intersection.
    Returns True if the line segment from (x1,y1) to (x2,y2) intersects the circle
    centered at (cx,cy) with radius r.
    """
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy

    a = dx * dx + dy * dy
    if a < eps:
        # Degenerate segment (point): check distance to center
        return (fx * fx + fy * fy) <= r * r

    b = 2.0 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - r * r

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Intersection if either root lies within the finite segment
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def sniff_radius_m(mode: str, radius_m: float, wt: float, mc_sniff: float, k: float) -> float:
    """
    Return thermal capture/sniffing radius in meters.
    mode = "fixed" -> return radius_m
    mode = "sniff" -> cubic law: r = k * max(Wt - MC, 0)^3
    """
    mode = mode.lower()
    if mode == "fixed":
        return float(radius_m)
    if mode == "sniff":
        return float(k * max(wt - mc_sniff, 0.0) ** 3)
    raise ValueError("radius-mode must be 'fixed' or 'sniff'")


@dataclass
class Result:
    speed_kmh: int
    trials: int
    success: int
    prob: float
    ci_low: float
    ci_high: float


def trial_once(
    rng: np.random.Generator,
    seg_len_m: float,
    r_m: float,
    lambda_a: float,   # thermals per km^2
    p_up: float
) -> bool:
    """
    Run a single trial:
      - Build a sampling rectangle that tightly pads the segment by r_m (plus small margin).
      - Draw N ~ Poisson(lambda_a * area_km2) thermal centers uniformly in that rectangle.
      - Flip a coin for each: updraft with prob p_up.
      - Return True on first updraft whose disk intersects the segment.
    We fix the segment along the x-axis: (0,0) → (seg_len_m, 0).
    This is equivalent to a random heading in an isotropic Poisson field.
    """
    margin = max(5.0, 0.05 * r_m)   # meters; tiny safety margin
    pad = r_m + margin

    # Sampling rectangle (meters): x in [-pad, seg_len_m + pad], y in [-pad, +pad]
    width_m = seg_len_m + 2.0 * pad
    height_m = 2.0 * pad
    area_km2 = (width_m * height_m) / 1e6  # m^2 → km^2

    mu = lambda_a * area_km2
    n = rng.poisson(mu)  # allow n=0 (no clamp)

    if n == 0:
        return False

    # Sample centers uniformly in the rectangle
    x = rng.uniform(-pad, seg_len_m + pad, size=n)
    y = rng.uniform(-pad, pad, size=n)

    # Label updrafts
    is_up = rng.random(n) < p_up
    if not np.any(is_up):
        return False

    # Segment endpoints
    x1, y1 = 0.0, 0.0
    x2, y2 = seg_len_m, 0.0

    # Check intersections (early exit if any updraft hits)
    for xi, yi, up in zip(x, y, is_up):
        if not up:
            continue
        if circle_line_segment_intersections(xi, yi, r_m, x1, y1, x2, y2):
            return True
    return False


def run_for_speed(
    speed_kmh: int,
    trials: int,
    rng: np.random.Generator,
    lambda_a: float,
    p_up: float,
    radius_mode: str,
    radius_m_fixed: float,
    wt: float,
    mc_sniff: float,
    sniff_k: float,
    h0_m: float,
    hmin_m: float,
    cap_km: float
) -> Result:
    """Aggregate trials for a single speed."""
    v_ms = speed_kmh / 3.6
    sink = sink_ms(v_ms)          # m/s
    if sink <= 0:
        # Defensive; should not happen with our convex polar
        return Result(speed_kmh, trials, 0, 0.0, 0.0, 0.0)

    time_until_hmin = max(h0_m - hmin_m, 0.0) / sink  # seconds
    seg_len_m = v_ms * time_until_hmin               # meters

    if cap_km and cap_km > 0:
        seg_len_m = min(seg_len_m, cap_km * 1000.0)

    r_m = sniff_radius_m(radius_mode, radius_m_fixed, wt, mc_sniff, sniff_k)

    success = 0
    for _ in range(trials):
        if trial_once(rng, seg_len_m, r_m, lambda_a, p_up):
            success += 1

    p = success / trials if trials > 0 else 0.0
    lo, hi = wilson_ci(success, trials)
    return Result(speed_kmh, trials, success, p, lo, hi)


# ----------------------------------- Main -----------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Speed sweep: thermal intercept probability (Poisson field + polar range)")
    ap.add_argument("--trials", type=int, default=TRIALS_DEFAULT, help="Trials per speed (default: %(default)s)")
    ap.add_argument("--speed-min", type=int, default=SPEED_MIN_KMH, help="Min speed km/h (default: %(default)s)")
    ap.add_argument("--speed-max", type=int, default=SPEED_MAX_KMH, help="Max speed km/h (default: %(default)s)")
    ap.add_argument("--lambda-a", type=float, default=LAMBDA_A_DEFAULT, help="Thermal areal density /km^2 (default: %(default)s)")
    ap.add_argument("--p-up", type=float, default=P_UP_DEFAULT, help="Probability a thermal is an updraft (default: %(default)s)")
    ap.add_argument("--radius-mode", choices=["fixed", "sniff"], default="fixed", help="Capture radius mode (default: %(default)s)")
    ap.add_argument("--radius-m", type=float, default=RADIUS_M_DEFAULT, help="Fixed capture radius in meters (if radius-mode=fixed)")
    ap.add_argument("--wt", type=float, default=WT_DEFAULT, help="Wt (m/s) for sniff radius (radius-mode=sniff)")
    ap.add_argument("--mc-sniff", type=float, default=MC_SNIFF_DEFAULT, help="MC threshold for sniff (radius-mode=sniff)")
    ap.add_argument("--sniff-k", type=float, default=SNIFF_K_DEFAULT, help="Scale k in r = k * max(Wt - MC, 0)^3 (radius-mode=sniff)")
    ap.add_argument("--h0-m", type=float, default=H0_M_DEFAULT, help="Start altitude m (default: %(default)s)")
    ap.add_argument("--hmin-m", type=float, default=HMIN_M_DEFAULT, help="Landing floor m (default: %(default)s)")
    ap.add_argument("--cap-km", type=float, default=CAP_KM_DEFAULT, help="Optional cap on segment length (km). Set <=0 to disable. (default: %(default)s)")
    ap.add_argument("--csv", type=Path, default=Path(OUT_CSV_DEFAULT), help="Output CSV path")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed (omit for true MC)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    speeds = list(range(args.speed_min, args.speed_max + 1))
    results = [
        run_for_speed(
            speed_kmh=s,
            trials=args.trials,
            rng=rng,
            lambda_a=args.lambda_a,
            p_up=args.p_up,
            radius_mode=args.radius_mode,
            radius_m_fixed=args.radius_m,
            wt=args.wt,
            mc_sniff=args.mc_sniff,
            sniff_k=args.sniff_k,
            h0_m=args.h0_m,
            hmin_m=args.hmin_m,
            cap_km=args.cap_km,
        )
        for s in speeds
    ]

    # Write CSV
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speed_kmh", "trials", "success", "prob", "ci_low", "ci_high"])
        for r in results:
            w.writerow([r.speed_kmh, r.trials, r.success, f"{r.prob:.6f}", f"{r.ci_low:.6f}", f"{r.ci_high:.6f}"])
    print(f"[OK] Wrote CSV → {args.csv}")

    # Plot
    x = [r.speed_kmh for r in results]
    y = [r.prob for r in results]
    lo = [r.ci_low for r in results]
    hi = [r.ci_high for r in results]

    plt.figure()
    plt.plot(x, y, label="Probability")
    plt.fill_between(x, lo, hi, alpha=0.15, label="95% CI")
    plt.ylim(-0.02, 1.02)
    plt.xlim(min(x), max(x))
    plt.xlabel("Airspeed (km/h)")
    plt.ylabel("Success probability (intercept an updraft before Hmin)")
    plt.title(PLOT_TITLE)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Console best
    best = max(results, key=lambda r: r.prob)
    print(f"Peak P: {best.prob:.3f} at {best.speed_kmh} km/h (95% CI {best.ci_low:.3f}-{best.ci_high:.3f})")


if __name__ == "__main__":
    main()