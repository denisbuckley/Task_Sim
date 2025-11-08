#!/usr/bin/env python3
"""
v3b — Explicit sector search to nearest thermal (no tube shortcuts)
-------------------------------------------------------------------
Per glide cycle:
  1) Compute max reachable flown distance from CBL→Hmin using B1/B2/B3 speeds.
  2) Sample a Poisson set of updraft centres uniformly in the sector wedge:
       angle ∈ [-θ,+θ], radius ∈ [0,R] with pdf ∝ r (i.e., r = R*sqrt(U)).
  3) Choose the nearest centre (smallest radius d_hit). If none, landout.
  4) If d_hit > R, landout (insufficient height). Otherwise:
       - Fly straight to the centre on heading ψ.
       - Time = piecewise through B1→B2→B3: sum(d_i / v_i)
       - Course progress = d_hit * cos(ψ)
       - Climb to CBL: time += ΔH / climb_ms
  5) Repeat until accumulated course progress ≥ task_km.

Notes:
  • B1 fraction of ΔH selectable; B2, B3 default to v*+20 and v*, but can be set.
  • Updraft fraction p_up applied when generating centres.
  • Fixed capture radius; no sniff mode.
"""

from __future__ import annotations
import math, sys, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

VERSION = "therm_intercept_prob_v3b_sector"
print(f"[RUN] {VERSION} @ {Path(__file__).resolve()}")

# ---------------- Polars ----------------
POLAR_LS10_18M_600 = [
    (90,0.60),(100,0.62),(110,0.66),(120,0.70),(130,0.77),(140,0.86),
    (150,0.97),(160,1.10),(170,1.24),(180,1.39),(190,1.55),(200,1.70),
    (210,1.85),(220,2.45),(230,2.72),(240,3.00)
]
POLAR_JS3_18M_60 = [
    (90,0.60),(100,0.62),(110,0.65),(120,0.70),(130,0.77),(140,0.86),
    (150,0.97),(160,1.08),(170,1.29),(180,1.55),(190,1.75),(200,1.90),
    (210,2.06),(220,2.28),(230,2.48),(240,2.70),(250,2.93)
]
POLAR_ASG29_18M_600KG = [
    (80,0.70),(90,0.65),(100,0.66),(110,0.70),(120,0.78),(130,0.87),
    (140,0.98),(150,1.10),(160,1.23),(170,1.37),(180,1.52),(190,1.67),
    (200,1.83)
]
POLAR_VENTUS_2CX_18M_565 = [
    (90,0.62),(100,0.68),(110,0.75),(120,0.84),(130,0.96),(140,1.08),
    (150,1.22),(160,1.38),(170,1.56),(180,1.76),(190,1.98),(200,2.22)
]
POLAR_CIRRUS_STD_35 = [
    (80,0.78),(90,0.82),(100,0.87),(110,0.94),(120,1.04),(130,1.16),
    (140,1.30),(150,1.46),(160,1.64),(170,1.84),(180,2.06),(190,2.30)
]

GLIDERS: Dict[str, List[Tuple[float,float]]] = {
    "ls10_18m_600": POLAR_LS10_18M_600,
    "js3_18m_60": POLAR_JS3_18M_60,
    "asg29_18m_600kg": POLAR_ASG29_18M_600KG,
    "ventus_2cx_18m_565": POLAR_VENTUS_2CX_18M_565,
    "cirrus_std_35": POLAR_CIRRUS_STD_35,
}

# ---------------- Defaults ----------------
TASK_KM     = 300.0
HCBL        = 2500.0
HMIN        = 500.0
CLIMB_MS    = 2.0
LAMBDA_A    = 0.015
P_UP        = 0.60
RADIUS_M    = 380.0
SECTOR_DEG  = 20.0
B1_FRAC     = 1.0/3.0
TRIALS      = 1000
SPEED_MIN   = 100
SPEED_MAX   = 240

# ---------------- Helpers ----------------
def interp_sink(glider: str, v_kmh: float) -> float:
    pts = GLIDERS[glider]
    xs, ys = zip(*pts)
    if v_kmh <= xs[0]: return ys[0]
    if v_kmh >= xs[-1]: return ys[-1]
    i = np.searchsorted(xs, v_kmh)
    x1, x2 = xs[i-1], xs[i]; y1, y2 = ys[i-1], ys[i]
    t = (v_kmh - x1)/(x2 - x1)
    return y1 + t*(y2 - y1)

def best_glide(glider: str) -> float:
    v = np.linspace(80, 260, 901)
    gr = [(vk/3.6)/interp_sink(glider, vk) for vk in v]
    return float(v[int(np.argmax(gr))])

def band_capacity_course_m(glider: str, v_kmh: float, band_h_m: float) -> float:
    s = max(interp_sink(glider, v_kmh), 1e-9)
    return (v_kmh/3.6) * (band_h_m / s)

def reachable_distance_m(glider: str, v1: float, v2: float, v3: float,
                         b1_frac: float, h_cbl: float, h_min: float) -> Tuple[float,float,float,float]:
    dh = max(0.0, h_cbl - h_min)
    f = max(0.0, min(1.0, b1_frac))
    h1 = f*dh; h2 = h3 = (dh - h1)/2.0
    L1 = band_capacity_course_m(glider, v1, h1)
    L2 = band_capacity_course_m(glider, v2, h2)
    L3 = band_capacity_course_m(glider, v3, h3)
    return L1, L2, L3, (L1 + L2 + L3)

def split_time_over_bands(distance_m: float, L1: float, L2: float, L3: float,
                          v1_ms: float, v2_ms: float, v3_ms: float) -> float:
    rem = distance_m; t = 0.0
    for L, v in ((L1, v1_ms), (L2, v2_ms), (L3, v3_ms)):
        if rem <= 0: break
        d = min(rem, L)
        t += d / v
        rem -= d
    return t

def sample_nearest_updraft_in_sector(rng: np.random.Generator,
                                     lam_a: float, p_up: float,
                                     theta_rad: float, R_m: float) -> Optional[Tuple[float, float]]:
    """Sample Poisson updrafts in a sector wedge."""
    area_m2 = theta_rad * (R_m ** 2)
    area_km2 = area_m2 / 1e6
    mu = lam_a * p_up * area_km2
    N = rng.poisson(mu)
    if N <= 0:
        return None
    psi = rng.uniform(-theta_rad, +theta_rad, size=N)
    r = R_m * np.sqrt(rng.random(size=N))
    i = int(np.argmin(r))
    return float(r[i]), float(psi[i])

@dataclass
class Trial:
    finished: bool
    time_s: float

# ---------------- Trial ----------------
def run_one_trial(rng: np.random.Generator,
                  glider: str,
                  v_b1_kmh: float, v_b2_kmh: float, v_b3_kmh: float,
                  b1_frac: float,
                  task_km: float,
                  h_cbl_m: float, h_min_m: float,
                  climb_ms: float,
                  lam_a: float, p_up: float,
                  radius_m: float, sector_deg: float) -> Trial:

    L1, L2, L3, R_course = reachable_distance_m(
        glider, v_b1_kmh, v_b2_kmh, v_b3_kmh, b1_frac, h_cbl_m, h_min_m
    )
    if R_course <= 0.0:
        return Trial(False, 0.0)

    v1, v2, v3 = v_b1_kmh/3.6, v_b2_kmh/3.6, v_b3_kmh/3.6
    theta = math.radians(sector_deg)

    remaining_course = task_km * 1000.0
    t_total = 0.0

    while remaining_course > 0.0:
        nearest = sample_nearest_updraft_in_sector(rng, lam_a, p_up, theta, R_course)
        if nearest is None:
            return Trial(False, t_total)
        d_hit, psi = nearest
        if d_hit > R_course + 1e-9:
            return Trial(False, t_total)

        t_glide = split_time_over_bands(d_hit, L1, L2, L3, v1, v2, v3)
        t_total += t_glide

        d_course = d_hit * math.cos(psi)
        remaining_course -= d_course

        dh = max(0.0, h_cbl_m - h_min_m)
        t_total += dh / climb_ms

    return Trial(True, t_total)

# ---------------- Wizard ----------------
def _ask(prompt, cast, default):
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "": return default
    try: return cast(s)
    except: print("Invalid; using default."); return default

def _choose_gliders() -> List[str]:
    keys = list(GLIDERS.keys())
    print("\nGliders:")
    for i,k in enumerate(keys, 1):
        print(f" {i}. {k}")
    s = input("Select glider(s) by number (comma sep) [1]: ").strip()
    if not s: return [keys[0]]
    try:
        idx = [int(x)-1 for x in s.split(",")]
        return [keys[i] for i in idx if 0 <= i < len(keys)]
    except:
        return [keys[0]]

def main():
    print("\n[Wizard] Sector-to-nearest-thermal Monte Carlo (v3b)")
    gliders = _choose_gliders()
    trials   = _ask("Trials per speed", int, TRIALS)
    smin     = _ask("B1 sweep start (km/h)", int, SPEED_MIN)
    smax     = _ask("B1 sweep end (km/h)", int, SPEED_MAX)
    task_km  = _ask("Task distance (km)", float, TASK_KM)
    h_cbl    = _ask("CBL (m)", float, HCBL)
    h_min    = _ask("Hmin (m)", float, HMIN)
    climb    = _ask("Climb (m/s)", float, CLIMB_MS)
    lam_a    = _ask("Lambda (/km^2)", float, LAMBDA_A)
    p_up     = _ask("p_up", float, P_UP)
   #  r_m      = _ask("Capture radius (m, fixed; no sniff)", float, RADIUS_M)
    r_m = RADIUS_M
    theta    = _ask("Sector half-angle (deg)", float, SECTOR_DEG)
    b1_frac  = _ask("B1 fraction of ΔH", float, B1_FRAC)
    b2_in    = input("B2 speed (km/h) [blank = v*+20]: ").strip()
    b3_in    = input("B3 speed (km/h) [blank = v*   ]: ").strip()
    csv_out  = input("CSV output file [out.csv]: ").strip() or "out.csv"

    rows = [["glider","b1_kmh","achieved_kmh","finish_prob"]]

    plt.figure(figsize=(10,6))
    ax1 = plt.gca(); ax2 = ax1.twinx()

    for g in gliders:
        vstar = best_glide(g)
        b2_eff = float(b2_in) if b2_in else vstar + 20.0
        b3_eff = float(b3_in) if b3_in else vstar
        print(f"[INFO] {g}: v*≈{vstar:.1f}  B2={b2_eff:.1f}  B3={b3_eff:.1f}")

        speeds = list(range(smin, smax+1))
        achieved, probs = [], []

        for s in speeds:
            sub_rng = np.random.default_rng(10_001 + 97*s)
            finished = 0
            times = []
            for _ in range(trials):
                tr = run_one_trial(sub_rng, g, s, b2_eff, b3_eff, b1_frac,
                                   task_km, h_cbl, h_min, climb, lam_a, p_up, r_m, theta)
                if tr.finished:
                    finished += 1
                    times.append(tr.time_s)
            p = finished / trials
            v_task = task_km / (np.mean(times)/3600.0) if times else float("nan")
            probs.append(p); achieved.append(v_task)
            rows.append([g, s, f"{v_task:.3f}", f"{p:.6f}"])

        xs = np.array(speeds, float)
        ys = np.array(achieved, float)
        mask = np.isfinite(ys)

        (line_achieved,) = ax1.plot(xs[mask], ys[mask],
                                    label=f"{g} — achieved")
        color = line_achieved.get_color()

        p_arr = np.array(probs, float)
        ax2.plot(xs, p_arr, "--", linewidth=1.8, color=color,
                 label=f"{g} — prob", zorder=6)
        ax2.fill_between(xs, p_arr, 1.0, alpha=0.12, color=color, zorder=5)

        print(f"[STATS] {g}: prob min/median/max = "
              f"{np.nanmin(p_arr):.3f}/"
              f"{np.nanmedian(p_arr):.3f}/"
              f"{np.nanmax(p_arr):.3f}")

    ax1.set_xlabel("B1 flown speed (km/h)")
    ax1.set_ylabel("Achieved task speed (km/h)")
    ax2.set_ylabel("Finish probability")
    ax2.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right"); ax2.legend(loc="upper right")
    plt.title(f"Task vs B1 — sector (±{theta:.0f}°)\n"
              f"Task {task_km:.0f} km; Hcbl {h_cbl:.0f} m; climb {climb:.1f} m/s; "
              f"λ={lam_a:.3f}; p_up={p_up:.2f}")
    plt.tight_layout(); plt.show()

    with open(csv_out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[OK] wrote → {csv_out}")

if __name__ == "__main__":
    main()