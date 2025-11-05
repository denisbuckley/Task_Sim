#!/usr/bin/env python3
"""
Thermal intercept probability vs airspeed
-----------------------------------------
Monte Carlo: straight-line flight through a 2-D Poisson thermal field
(no detours/search). Probability = chance of intersecting at least one
**updraft** thermal disk before reaching Hmin.

Adds optional ANALYTIC OVERLAY:
    P(v) = 1 - exp(-λ * p_up * (2r) * L(v))
with L(v) derived from the sink polar; capped by --cap-km if provided.

Outputs:
- Plot: Monte Carlo curve (+ 95% CI band) and optional Analytic overlay.
- CSV: speed_kmh, trials, success, prob, ci_low, ci_high

Usage (examples):
    # Baseline (uncapped) with overlay
    python therm_intercept_prob.py --cap-km 0 --lambda-a 0.02 --p-up 0.6 \
        --radius-mode fixed --radius-m 300 --trials 1000 --analytic-overlay

    # Sniff (cubic) radius style
    python therm_intercept_prob.py --cap-km 0 --radius-mode sniff --wt 6 --mc-sniff 2 --sniff-k 6 \
        --lambda-a 0.02 --p-up 0.6 --trials 1000 --analytic-overlay
"""

from __future__ import annotations
import math
import csv
from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


# --------------------------- Run banner (sanity) ---------------------------
VERSION = "therm_intercept_prob_v1b"
print(f"[RUN] {VERSION} @ {Path(__file__).resolve()}")


# --------------------------- Defaults (edit freely) ------------------------
SPEED_MIN_KMH = 100
SPEED_MAX_KMH = 250
TRIALS_DEFAULT = 1000

LAMBDA_A_DEFAULT = 0.01       # thermals / km^2 (areal density)
P_UP_DEFAULT = 0.5            # fraction of thermals that are updrafts

# Polar: simple convex quadratic around min-sink point (approx LS10-ish)
POLAR_V0 = 27.0               # m/s at min sink
POLAR_S0 = 0.55               # m/s minimum sink
POLAR_K  = 0.0020             # curvature; S(v) = S0 + K*(v - V0)^2

H0_M_DEFAULT = 2500.0         # start altitude (m)
HMIN_M_DEFAULT = 500.0        # minimum (landing) altitude (m)
CAP_KM_DEFAULT = 100.0        # cap segment length (km). Set 0 to disable.

# Fixed capture radius default (when --radius-mode fixed)
RADIUS_M_DEFAULT = 250.0      # meters

# Sniff (cubic) capture radius defaults (when --radius-mode sniff)
WT_DEFAULT = 6.0              # e.g., Wt (m/s)
MC_SNIFF_DEFAULT = 2.0        # MC threshold for sniff (m/s)
SNIFF_K_DEFAULT = 8.0         # r = k * max(Wt - MC, 0)^3  (meters)

OUT_CSV_DEFAULT = "speed_vs_probability.csv"
PLOT_TITLE = "Probability vs Airspeed (Poisson field; straight line; polar-coupled range)"
# ---------------------------------------------------------------------------


# ------------------------------- Helpers -----------------------------------
def sink_ms(v_ms: float) -> float:
    """Still-air sink polar (m/s)."""
    return POLAR_S0 + POLAR_K * (v_ms - POLAR_V0) ** 2


def wilson_ci(success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = success / n
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z ** 2) / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def circle_line_segment_intersections(
    cx: float, cy: float, r: float,
    x1: float, y1: float, x2: float, y2: float,
    eps: float = 1e-12
) -> bool:
    """
    Robust circle–line-segment intersection.
    Returns True if segment (x1,y1)->(x2,y2) intersects circle center (cx,cy), radius r.
    """
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy

    a = dx * dx + dy * dy
    if a < eps:  # degenerate segment as a point
        return (fx * fx + fy * fy) <= r * r

    b = 2.0 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - r * r

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def sniff_radius_m(mode: str, radius_m: float, wt: float, mc_sniff: float, k: float) -> float:
    """
    Capture/sniff radius (m).
    mode = "fixed" -> radius_m
    mode = "sniff" -> r = k * max(Wt - MC, 0)^3
    """
    mode = mode.lower()
    if mode == "fixed":
        return float(radius_m)
    if mode == "sniff":
        return float(k * max(wt - mc_sniff, 0.0) ** 3)
    raise ValueError("radius-mode must be 'fixed' or 'sniff'")


def glide_length_m(v_kmh: float, h0_m: float, hmin_m: float, cap_km: float | None) -> float:
    """Glide distance until Hmin using polar (m); optionally capped."""
    v_ms = v_kmh / 3.6
    s = sink_ms(v_ms)
    t = max(h0_m - hmin_m, 0.0) / s  # seconds
    L = v_ms * t                     # meters
    if cap_km and cap_km > 0:
        L = min(L, cap_km * 1000.0)
    return L


def analytic_prob(v_kmh: float, lambda_a: float, p_up: float, r_m: float,
                  h0_m: float, hmin_m: float, cap_km: float | None) -> float:
    """Closed-form probability for Poisson tube model."""
    Lm = glide_length_m(v_kmh, h0_m, hmin_m, cap_km)
    Lkm = Lm / 1000.0
    rkm = r_m / 1000.0
    mu = (lambda_a * p_up) * (2.0 * rkm * Lkm)
    return 1.0 - math.exp(-mu)


@dataclass
class Result:
    speed_kmh: int
    trials: int
    success: int
    prob: float
    ci_low: float
    ci_high: float


# ------------------------------- Simulation --------------------------------
def trial_once(
    rng: np.random.Generator,
    seg_len_m: float,
    r_m: float,
    lambda_a: float,   # thermals per km^2
    p_up: float
) -> bool:
    """
    Single trial:
      - Segment along x-axis: (0,0) -> (seg_len_m, 0)
      - Sampling rectangle pads the segment by r_m + small margin.
      - N ~ Poisson(λ * area_km^2) thermals uniformly in the rectangle.
      - Each thermal is updraft with probability p_up.
      - Return True if any updraft circle intersects the segment.
    """
    margin = max(5.0, 0.05 * r_m)   # meters; tiny safety margin
    pad = r_m + margin

    width_m = seg_len_m + 2.0 * pad
    height_m = 2.0 * pad
    area_km2 = (width_m * height_m) / 1e6  # m^2 -> km^2

    n = rng.poisson(lambda_a * area_km2)  # allow n=0
    if n == 0:
        return False

    # Sample centers
    x = rng.uniform(-pad, seg_len_m + pad, size=n)
    y = rng.uniform(-pad, pad, size=n)

    # Updraft labels
    is_up = rng.random(n) < p_up
    if not np.any(is_up):
        return False

    # Segment endpoints
    x1, y1 = 0.0, 0.0
    x2, y2 = seg_len_m, 0.0

    # Intersections (early exit)
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
    """Aggregate Bernoulli trials for one speed."""
    v_ms = speed_kmh / 3.6
    if sink_ms(v_ms) <= 0:
        return Result(speed_kmh, trials, 0, 0.0, 0.0, 0.0)

    seg_len_m = glide_length_m(speed_kmh, h0_m, hmin_m, cap_km)
    r_m = sniff_radius_m(radius_mode, radius_m_fixed, wt, mc_sniff, sniff_k)

    success = 0
    for _ in range(trials):
        if trial_once(rng, seg_len_m, r_m, lambda_a, p_up):
            success += 1

    p = success / trials if trials > 0 else 0.0
    lo, hi = wilson_ci(success, trials)
    return Result(speed_kmh, trials, success, p, lo, hi)


# ------------------------------------ Main ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="Thermal intercept probability vs airspeed (Poisson field, straight line)")
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
    ap.add_argument("--cap-km", type=float, default=CAP_KM_DEFAULT, help="Optional cap on segment length (km). Set 0 to disable. (default: %(default)s)")
    ap.add_argument("--csv", type=Path, default=Path(OUT_CSV_DEFAULT), help="Output CSV path")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed (omit for true MC)")
    ap.add_argument("--no-analytic-overlay", action="store_true",
                    help="Disable the analytic overlay (enabled by default)")
    args = ap.parse_args()
    # Show analytic overlay by default unless explicitly disabled
    show_overlay = not args.no_analytic_overlay

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

    # Prepare plot data
    x = [r.speed_kmh for r in results]
    y = [r.prob for r in results]
    lo = [r.ci_low for r in results]
    hi = [r.ci_high for r in results]

    # Plot MC curve + CI band
    plt.figure()
    plt.plot(x, y, label="Monte Carlo")
    plt.fill_between(x, lo, hi, alpha=0.15, label="95% CI")
    plt.ylim(-0.02, 1.02)
    plt.xlim(min(x), max(x))
    plt.xlabel("Airspeed (km/h)")
    plt.ylabel("Success probability (intercept an updraft before Hmin)")
    plt.title(PLOT_TITLE)
    plt.grid(True, alpha=0.3)

    # Analytic overlay (same capture radius as MC at this parameter set)
    if show_overlay:
        r_overlay_m = sniff_radius_m(args.radius_mode, args.radius_m, args.wt, args.mc_sniff, args.sniff_k)
        y_ana = [analytic_prob(s, args.lambda_a, args.p_up, r_overlay_m,
                               args.h0_m, args.hmin_m, args.cap_km) for s in speeds]
        # draw on top of the CI band and make it clearly visible
        plt.plot(x, y_ana, linestyle="--", linewidth=1.4, color="black", zorder=10, label="Analytic")

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Console best
    best = max(results, key=lambda r: r.prob)
    print(f"Peak P: {best.prob:.3f} at {best.speed_kmh} km/h (95% CI {best.ci_low:.3f}-{best.ci_high:.3f})")


if __name__ == "__main__":
    main()