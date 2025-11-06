#!/usr/bin/env python3
"""
Thermal intercept probability vs airspeed — Block Flying, multi-glider compare (v1j)
-----------------------------------------------------------------------------------
Monte Carlo through a 2-D Poisson thermal field using digitized **table polars**
with linear interpolation (no synthetic quadratic). Optionally plots **both** gliders
on a single figure and writes a **combined CSV**.

Gliders:
- ls10_18m_600 : LS10 18 m @ 600 kg (table from manufacturer plot)
- cirrus_std_35: Standard Cirrus, WITH WATER BALLAST (W/S ≈ 35 kg/m²), dashed right curve

Block Flying:
- ΔH = (H0 − Hmin). B1 gets fraction f; sweep B1 speed (x-axis).
- B2 & B3 split remainder equally at best-glide v* (overrideable).
- L = L_B1(v_B1; f·ΔH) + L_B2(v2; (1−f)/2·ΔH) + L_B3(v3; (1−f)/2·ΔH)

Analytic overlay:
P = 1 − exp(−λ p_up · 2 r · L) using the SAME L as the sim.

Wizard appears if no CLI args; otherwise use flags (see argparse below).
"""

from __future__ import annotations
import math
import csv
from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Run banner ------------------------------------
VERSION = "therm_intercept_prob_v1j_multi"
print(f"[RUN] {VERSION} @ {Path(__file__).resolve()}")

# --------------------------- Defaults --------------------------------------
SPEED_MIN_KMH = 100
SPEED_MAX_KMH = 250
TRIALS_DEFAULT = 1000

LAMBDA_A_DEFAULT = 0.0145
P_UP_DEFAULT     = 0.60

H0_M_DEFAULT    = 2500.0
HMIN_M_DEFAULT  = 500.0
CAP_KM_DEFAULT  = 0.0

# Capture radius / sniff defaults (sniff is default mode)
RADIUS_M_DEFAULT = 300.0
WT_DEFAULT       = 6.0
MC_SNIFF_DEFAULT = 2.0
SNIFF_K_DEFAULT  = 6.0

# Block share default: B1 gets one third of ΔH
B1_FRAC_DEFAULT  = 1.0 / 3.0

OUT_CSV_DEFAULT  = "speed_vs_probability_block.csv"

# --------------------------- Table polars ----------------------------------
# LS10 18 m @ 600 kg (dark blue curve)
POLAR_LS10_18M_600: List[Tuple[float, float]] = [
    (80.0, 0.63), (90.0, 0.56), (100.0, 0.58), (110.0, 0.63), (120.0, 0.70),
    (130.0, 0.80), (140.0, 0.93), (150.0, 1.07), (160.0, 1.23), (170.0, 1.40),
    (180.0, 1.58), (190.0, 1.77), (200.0, 1.97), (210.0, 2.20), (220.0, 2.45),
    (230.0, 2.72), (240.0, 3.00),
]

# Standard Cirrus — WITH WATER BALLAST (dashed right curve, W/S ≈ 35 kg/m²)
POLAR_CIRRUS_STD_35: List[Tuple[float, float]] = [
    (80.0, 0.65), (90.0, 0.60), (100.0, 0.63), (110.0, 0.72), (120.0, 0.84),
    (130.0, 1.00), (140.0, 1.20), (150.0, 1.45), (160.0, 1.75), (170.0, 2.05),
    (180.0, 2.38), (190.0, 2.70), (200.0, 3.00), (210.0, 3.30), (220.0, 3.60),
    (230.0, 3.95), (240.0, 4.30),
]

POLARS: Dict[str, List[Tuple[float, float]]] = {
    "ls10_18m_600": POLAR_LS10_18M_600,
    "cirrus_std_35": POLAR_CIRRUS_STD_35,
}

# globals filled per selected polar
_POLAR_KMH = None  # np.ndarray
_POLAR_SNK = None  # np.ndarray

def select_polar(name: str) -> None:
    """Choose which table polar to use (fills global interp arrays)."""
    global _POLAR_KMH, _POLAR_SNK
    pts = sorted(POLARS[name], key=lambda t: t[0])
    _POLAR_KMH = np.array([p[0] for p in pts], dtype=float)
    _POLAR_SNK = np.array([p[1] for p in pts], dtype=float)
    print(f"[INFO] Using polar: {name}  (range {float(_POLAR_KMH[0])}-{float(_POLAR_KMH[-1])} km/h)")

# --------------------------- Interp + range helpers ------------------------
def sink_ms(v_ms: float) -> float:
    """Interpolate sink from the currently selected table (linear; clamps at ends)."""
    v_kmh = v_ms * 3.6
    return float(np.interp(v_kmh, _POLAR_KMH, _POLAR_SNK))

def best_glide_speed_kmh(vmin_kmh: float = 70.0, vmax_kmh: float = 280.0, step: float = 0.2) -> float:
    """Maximize v / s(v) given the selected table polar."""
    v = vmin_kmh
    best_v, best_ratio = vmin_kmh, -1.0
    while v <= vmax_kmh:
        v_ms = v / 3.6
        s = sink_ms(v_ms)
        if s > 0:
            ratio = v_ms / s
            if ratio > best_ratio:
                best_ratio, best_v = ratio, v
        v += step
    return best_v

def glide_length_for_band_m(v_kmh: float, band_height_m: float) -> float:
    """Distance (m) flown in one band of height at speed v, from selected polar."""
    v_ms = v_kmh / 3.6
    s = sink_ms(v_ms)
    if s <= 0:
        return 0.0
    t = band_height_m / s
    return v_ms * t

# ----------------------------- Other helpers -------------------------------
def wilson_ci(success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p_hat = success / n
    den = 1.0 + (z*z)/n
    ctr = (p_hat + (z*z)/(2*n)) / den
    half = z * math.sqrt((p_hat*(1-p_hat))/n + (z*z)/(4*n*n)) / den
    return (max(0.0, ctr - half), min(1.0, ctr + half))

def circle_line_segment_intersections(
    cx: float, cy: float, r: float,
    x1: float, y1: float, x2: float, y2: float,
    eps: float = 1e-12
) -> bool:
    # FIXED: compute both components
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy
    a = dx*dx + dy*dy
    if a < eps:
        return (fx*fx + fy*fy) <= r*r
    b = 2.0 * (fx*dx + fy*dy)
    c = (fx*fx + fy*fy) - r*r
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    sd = math.sqrt(disc)
    t1 = (-b - sd) / (2*a)
    t2 = (-b + sd) / (2*a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)

def sniff_radius_m(mode: str, radius_m: float, wt: float, mc_sniff: float, k: float) -> float:
    mode = mode.lower()
    if mode == "fixed": return float(radius_m)
    if mode == "sniff": return float(k * max(wt - mc_sniff, 0.0) ** 3)
    raise ValueError("radius-mode must be 'fixed' or 'sniff'")

def total_block_length_m(
    v_b1_kmh: float,
    h0_m: float,
    hmin_m: float,
    b1_frac: float,
    v_b2_kmh: Optional[float],
    v_b3_kmh: Optional[float],
    cap_km: float
) -> float:
    """
    L_total = L_B1(v_B1; f·ΔH) + L_B2(v2; (1−f)/2·ΔH) + L_B3(v3; (1−f)/2·ΔH)
    If v2/v3 are None, use best-glide automatically. Cap applied to final sum.
    """
    dh = max(h0_m - hmin_m, 0.0)
    if dh <= 0:
        return 0.0

    f1 = max(0.0, min(1.0, b1_frac))
    band1_h = f1 * dh
    band_rest = dh - band1_h
    band2_h = band3_h = band_rest / 2.0

    v_star = best_glide_speed_kmh()
    v2 = v_b2_kmh if v_b2_kmh is not None else v_star
    v3 = v_b3_kmh if v_b3_kmh is not None else v_star

    L1 = glide_length_for_band_m(v_b1_kmh, band1_h)
    L2 = glide_length_for_band_m(v2,        band2_h)
    L3 = glide_length_for_band_m(v3,        band3_h)
    L  = L1 + L2 + L3

    if cap_km and cap_km > 0:
        L = min(L, cap_km * 1000.0)
    return L

def analytic_prob_block(
    v_b1_kmh: float,
    lambda_a: float, p_up: float, r_m: float,
    h0_m: float, hmin_m: float,
    b1_frac: float,
    v_b2_kmh: Optional[float],
    v_b3_kmh: Optional[float],
    cap_km: float
) -> float:
    Lm = total_block_length_m(v_b1_kmh, h0_m, hmin_m, b1_frac, v_b2_kmh, v_b3_kmh, cap_km)
    mu = (lambda_a * p_up) * (2.0 * (r_m/1000.0) * (Lm/1000.0))
    return 1.0 - math.exp(-mu)

@dataclass
class Result:
    glider: str
    speed_kmh: int
    trials: int
    success: int
    prob: float
    ci_low: float
    ci_high: float

# -------------------------- Interactive helpers ---------------------------
def _ask(prompt: str, cast, default):
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "":
        return default
    try:
        return cast(s)
    except Exception:
        print(f"Invalid input. Using default: {default}")
        return default

def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    chs = "/".join(choices)
    s = input(f"{prompt} ({chs}) [{default}]: ").strip().lower()
    if s == "":
        return default
    if s in choices:
        return s
    print(f"Invalid choice. Using default: {default}")
    return default

def _ask_bool(prompt: str, default_yes: bool=False) -> bool:
    d = "Y/n" if default_yes else "y/N"
    s = input(f"{prompt} ({d}) ").strip().lower()
    if s == "":
        return default_yes
    return s in ("y","yes")

# ------------------------------- Simulation --------------------------------
def trial_once(
    rng: np.random.Generator,
    seg_len_m: float,
    r_m: float,
    lambda_a: float,
    p_up: float
) -> bool:
    margin = max(5.0, 0.05 * r_m)
    pad = r_m + margin
    width_m  = seg_len_m + 2.0 * pad
    height_m = 2.0 * pad
    area_km2 = (width_m * height_m) / 1e6

    n = rng.poisson(lambda_a * area_km2)
    if n == 0:
        return False

    x = rng.uniform(-pad, seg_len_m + pad, size=n)
    y = rng.uniform(-pad, pad, size=n)
    is_up = rng.random(n) < p_up
    if not np.any(is_up):
        return False

    x1, y1 = 0.0, 0.0
    x2, y2 = seg_len_m, 0.0
    for xi, yi, up in zip(x, y, is_up):
        if not up:
            continue
        if circle_line_segment_intersections(xi, yi, r_m, x1, y1, x2, y2):
            return True
    return False

def run_for_speed_block(
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
    cap_km: float,
    b1_frac: float,
    b2_speed_kmh: Optional[float],
    b3_speed_kmh: Optional[float]
) -> Result:
    r_m = sniff_radius_m(radius_mode, radius_m_fixed, wt, mc_sniff, sniff_k)
    seg_len_m = total_block_length_m(speed_kmh, h0_m, hmin_m, b1_frac, b2_speed_kmh, b3_speed_kmh, cap_km)

    success = 0
    for _ in range(trials):
        if trial_once(rng, seg_len_m, r_m, lambda_a, p_up):
            success += 1

    p = success / trials if trials > 0 else 0.0
    lo, hi = wilson_ci(success, trials)
    # glider filled by caller
    return Result("", speed_kmh, trials, success, p, lo, hi)

# --------------- Per-glider wrapper (select polar, clamp, run) -------------
def run_glider(
    glider: str,
    rng: np.random.Generator,
    speed_min: int,
    speed_max: int,
    trials: int,
    lambda_a: float,
    p_up: float,
    radius_mode: str,
    radius_m: float,
    wt: float,
    mc_sniff: float,
    sniff_k: float,
    h0_m: float,
    hmin_m: float,
    cap_km: float,
    b1_frac: float,
    b2_speed: Optional[float],
    b3_speed: Optional[float]
) -> Tuple[str, List[Result], List[int]]:
    select_polar(glider)
    v_star = best_glide_speed_kmh()
    print(f"[INFO] ({glider}) Best-glide v* ≈ {v_star:.1f} km/h")

    # Clamp sweep to polar range
    min_tab, max_tab = int(_POLAR_KMH[0]), int(_POLAR_KMH[-1])
    if speed_max > max_tab or speed_min < min_tab:
        print(f"[WARN] ({glider}) Clamping sweep to {min_tab}-{max_tab} km/h.")
    speeds = [s for s in range(speed_min, speed_max + 1) if min_tab <= s <= max_tab]
    if not speeds:
        raise SystemExit(f"[ERR] ({glider}) No speeds within polar range {min_tab}-{max_tab} km/h")

    results = []
    for s in speeds:
        r = run_for_speed_block(
            speed_kmh=s, trials=trials, rng=rng,
            lambda_a=lambda_a, p_up=p_up,
            radius_mode=radius_mode, radius_m_fixed=radius_m,
            wt=wt, mc_sniff=mc_sniff, sniff_k=sniff_k,
            h0_m=h0_m, hmin_m=hmin_m, cap_km=cap_km,
            b1_frac=b1_frac, b2_speed_kmh=b2_speed, b3_speed_kmh=b3_speed
        )
        r.glider = glider
        results.append(r)

    return glider, results, speeds

# ------------------------------------ Main ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="Thermal intercept probability vs airspeed (Block; multi-glider compare)")
    # Sim sweep
    ap.add_argument("--trials", type=int, default=TRIALS_DEFAULT)
    ap.add_argument("--speed-min", type=int, default=SPEED_MIN_KMH)
    ap.add_argument("--speed-max", type=int, default=SPEED_MAX_KMH)
    ap.add_argument("--lambda-a", type=float, default=LAMBDA_A_DEFAULT)
    ap.add_argument("--p-up", type=float, default=P_UP_DEFAULT)
    # Capture radius (default sniff)
    ap.add_argument("--radius-mode", choices=["fixed", "sniff"], default="sniff")
    ap.add_argument("--radius-m", type=float, default=RADIUS_M_DEFAULT)
    ap.add_argument("--wt", type=float, default=WT_DEFAULT)
    ap.add_argument("--mc-sniff", type=float, default=MC_SNIFF_DEFAULT)
    ap.add_argument("--sniff-k", type=float, default=SNIFF_K_DEFAULT)
    # Heights & cap
    ap.add_argument("--h0-m", type=float, default=H0_M_DEFAULT)
    ap.add_argument("--hmin-m", type=float, default=HMIN_M_DEFAULT)
    ap.add_argument("--cap-km", type=float, default=CAP_KM_DEFAULT)
    # Block params
    ap.add_argument("--b1-frac", type=float, default=B1_FRAC_DEFAULT)
    ap.add_argument("--b2-speed", type=float, default=None)
    ap.add_argument("--b3-speed", type=float, default=None)
    # Glider selector & compare toggle
    ap.add_argument("--glider", choices=list(POLARS.keys()), default="ls10_18m_600")
    ap.add_argument("--compare-both", action="store_true", help="Run and plot both gliders together")
    # I/O
    ap.add_argument("--csv", type=Path, default=Path(OUT_CSV_DEFAULT))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-analytic-overlay", action="store_true")

    # -------- Wizard when no CLI args --------
    if len(sys.argv) == 1:
        print("\n[Wizard] Enter values or press Enter for defaults.\n")
        compare_both = _ask_bool("Compare both gliders?", False)
        glider     = "ls10_18m_600" if compare_both else _ask_choice("Glider", list(POLARS.keys()), "ls10_18m_600")
        speed_min  = _ask("B1 sweep start speed (km/h)", int, SPEED_MIN_KMH)
        speed_max  = _ask("B1 sweep end speed (km/h)",   int, SPEED_MAX_KMH)
        trials     = _ask("Trials per speed",            int, TRIALS_DEFAULT)
        lambda_a   = _ask("Thermal density λ (/km^2)",   float, LAMBDA_A_DEFAULT)
        p_up       = _ask("Updraft fraction p_up",       float, P_UP_DEFAULT)

        radius_mode = _ask_choice("Radius mode", ["fixed","sniff"], "sniff")
        if radius_mode == "fixed":
            radius_m = _ask("Fixed capture radius r (m)", float, RADIUS_M_DEFAULT)
            wt, mc_sniff, sniff_k = WT_DEFAULT, MC_SNIFF_DEFAULT, SNIFF_K_DEFAULT
        else:
            wt       = _ask("Wt (m/s)", float, WT_DEFAULT)
            mc_sniff = _ask("MC sniff threshold (m/s)", float, MC_SNIFF_DEFAULT)
            sniff_k  = _ask("Sniff k (meters per (Wt-MC)^3)", float, SNIFF_K_DEFAULT)
            radius_m = RADIUS_M_DEFAULT  # unused in sniff

        h0_m    = _ask("Start altitude H0 (m)", float, H0_M_DEFAULT)
        hmin_m  = _ask("Minimum altitude Hmin (m)", float, HMIN_M_DEFAULT)
        cap_km  = _ask("Cap total length (km; 0=off)", float, CAP_KM_DEFAULT)

        b1_frac = _ask("B1 fraction of ΔH (0–1)", float, B1_FRAC_DEFAULT)
        b2_s = input("Optional B2 speed (km/h) [blank = best-glide]: ").strip()
        b3_s = input("Optional B3 speed (km/h) [blank = best-glide]: ").strip()
        b2_speed = float(b2_s) if b2_s else None
        b3_speed = float(b3_s) if b3_s else None

        csv_out = input(f"CSV output path [{OUT_CSV_DEFAULT}]: ").strip() or OUT_CSV_DEFAULT
        seed    = None
        show_overlay = True

        # Inject as CLI
        sys.argv.extend([
            "--speed-min", str(speed_min),
            "--speed-max", str(speed_max),
            "--trials", str(trials),
            "--lambda-a", str(lambda_a),
            "--p-up", str(p_up),
            "--radius-mode", radius_mode,
            "--h0-m", str(h0_m),
            "--hmin-m", str(hmin_m),
            "--cap-km", str(cap_km),
            "--b1-frac", str(b1_frac),
            "--csv", str(csv_out),
        ])
        if not compare_both:
            sys.argv.extend(["--glider", glider])
        else:
            sys.argv.extend(["--compare-both"])
        if b2_speed is not None: sys.argv.extend(["--b2-speed", str(b2_speed)])
        if b3_speed is not None: sys.argv.extend(["--b3-speed", str(b3_speed)])
        if radius_mode == "fixed":
            sys.argv.extend(["--radius-m", str(radius_m)])
        else:
            sys.argv.extend(["--wt", str(wt), "--mc-sniff", str(mc_sniff), "--sniff-k", str(sniff_k)])
        if seed is not None:
            sys.argv.extend(["--seed", str(seed)])
        if not show_overlay:
            sys.argv.extend(["--no-analytic-overlay"])
        print("\n[Wizard] Using parameters:", " ".join(sys.argv[1:]), "\n")

    args = ap.parse_args()
    show_overlay = not args.no_analytic_overlay
    rng = np.random.default_rng(args.seed)

    # Choose which gliders to run
    gliders = list(POLARS.keys()) if args.compare_both else [args.glider]

    # Combined CSV
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fcsv = args.csv.open("w", newline="")
    w = csv.writer(fcsv)
    w.writerow(["glider","speed_kmh","trials","success","prob","ci_low","ci_high"])

    # Plot setup
    plt.figure()
    color_for = {
        "ls10_18m_600": "tab:blue",
        "cirrus_std_35": "tab:red",
    }

    for g in gliders:
        g_name, results, speeds = run_glider(
            glider=g, rng=rng,
            speed_min=args.speed_min, speed_max=args.speed_max,
            trials=args.trials, lambda_a=args.lambda_a, p_up=args.p_up,
            radius_mode=args.radius_mode, radius_m=args.radius_m,
            wt=args.wt, mc_sniff=args.mc_sniff, sniff_k=args.sniff_k,
            h0_m=args.h0_m, hmin_m=args.hmin_m, cap_km=args.cap_km,
            b1_frac=args.b1_frac, b2_speed=args.b2_speed, b3_speed=args.b3_speed
        )

        # write CSV rows
        for r in results:
            w.writerow([g_name, r.speed_kmh, r.trials, r.success,
                        f"{r.prob:.6f}", f"{r.ci_low:.6f}", f"{r.ci_high:.6f}"])

        # plot
        x = [r.speed_kmh for r in results]
        y = [r.prob for r in results]
        lo = [r.ci_low for r in results]
        hi = [r.ci_high for r in results]
        c = color_for.get(g_name, None)

        plt.plot(x, y, label=f"{g_name} — Monte Carlo", color=c)
        plt.fill_between(x, lo, hi, alpha=0.12, label=f"{g_name} — 95% CI", color=c)

        if show_overlay:
            r_overlay_m = sniff_radius_m(args.radius_mode, args.radius_m, args.wt, args.mc_sniff, args.sniff_k)
            y_ana = [
                analytic_prob_block(
                    v_b1_kmh=s,
                    lambda_a=args.lambda_a, p_up=args.p_up, r_m=r_overlay_m,
                    h0_m=args.h0_m, hmin_m=args.hmin_m,
                    b1_frac=args.b1_frac,
                    v_b2_kmh=args.b2_speed, v_b3_kmh=args.b3_speed,
                    cap_km=args.cap_km
                )
                for s in speeds
            ]
            plt.plot(x, y_ana, linestyle="--", linewidth=1.4,
                     color=c, alpha=0.9, label=f"{g_name} — Analytic")

        # console peak
        best = max(results, key=lambda r: r.prob)
        print(f"[{g_name}] Peak P: {best.prob:.3f} at B1={best.speed_kmh} km/h "
              f"(95% CI {best.ci_low:.3f}-{best.ci_high:.3f})")

    fcsv.close()
    print(f"[OK] Wrote combined CSV → {args.csv}")

    plt.ylim(-0.02, 1.02)
    plt.xlim(args.speed_min, args.speed_max)
    plt.xlabel("B1 Airspeed (km/h)")
    plt.ylabel("Success probability (intercept before Hmin)")
    ttl = "Probability vs Airspeed (Block; multi-glider)"
    plt.title(ttl)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()y
