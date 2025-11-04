#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task_Sim / therm_arc_single.py
--------------------------------------------------------------------
Transferred from Poisson repository → Task_Sim (2025-11-04)

Description:
  Single-flight + Monte Carlo simulator for glider–thermal interaction.
  Supports Poisson and clustered (Thomas, Neyman–Scott) thermal fields,
  with a downdraft annulus penalty and a uniform-grid spatial index.

Options:
  1) Single run with plot
  2) Monte Carlo probability for current params (headless)
  3) 3D probability surfaces vs speed(MC1) and MC bands
  4) 2D probability curves: x=MC band 1 (0.1 steps), multiple curves for MC band 2 (1.0 steps)

This revision:
  • Adds Option 4: 2D probability plot with MC1 on x-axis and separate curves for MC2.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable
import math, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ======================== Configuration ================================

# Scene / task
CBL_ALT = 2500.0                  # Convective boundary layer top (m)
MIN_SAFE_ALT = 500.0              # Land-out floor (m)
TASK_DISTANCE = 100_000.0         # 100 km goal along +Y axis (m)
SEARCH_ARC_DEG = 30.0             # Forward search sector width (deg)
STEP_MIN = 2_000.0                # Glide step toward goal (m) when no in-arc thermal
STEP_MAX = 20_000.0
GOAL_EPS = 25.0                   # Considered "arrived" if within this (m)

# Thermal field density / strength
LAMBDA_TOTAL = 0.0145             # thermals per km² (empirical target)
LAMBDA_STRENGTH = 2.0             # Poisson mean for strength, clamped to [1..10] m/s

# Cluster controls
THERMAL_FIELD_MODEL = "thomas"    # "poisson", "thomas", or "neyman"
LAMBDA_PARENT: Optional[float] = None     # parents per km²; None → auto LAMBDA_TOTAL/MEAN_CHILDREN
MEAN_CHILDREN = 5.0
THOMAS_SIGMA_M = 150.0            # Thomas (Gaussian) σ for child offsets (m)
NEYMAN_RADIUS_M = 150.0           # Neyman–Scott uniform-disc radius (m)

# Thermal geometry / penalties
C_UPDRAFT_STRENGTH_DECREMENT = 5.9952e-7  # radius ≈ (strength / C)^(1/3)
OUTER_DIAMETER_M = 1200.0
OUTER_RADIUS_M = OUTER_DIAMETER_M / 2.0   # fixed downdraft ring outer radius
DOWNDRAFT_SINK_MS = 0.042194              # extra sink (m/s) while inside ring

# LS10 polar (knots → m/s)
KNOT_TO_MS = 0.514444
POLAR_DATA = pd.DataFrame({
    'V_Knot':     [45, 50, 55, 60, 65, 70, 75, 80,  85,  90,  95, 100, 105, 110, 115, 120],
    'Sink_Knots': [0.55,0.50,0.51,0.55,0.60,0.66,0.72,0.79,0.86,0.94,1.02,1.11,1.20,1.30,1.40,1.50],
})
POLAR_DATA['V_MS']    = POLAR_DATA['V_Knot'] * KNOT_TO_MS
POLAR_DATA['Sink_MS'] = POLAR_DATA['Sink_Knots'] * KNOT_TO_MS
f_sink_from_v = interp1d(POLAR_DATA['V_MS'], POLAR_DATA['Sink_MS'],
                         kind='linear', fill_value='extrapolate', bounds_error=False)

# Band MC (upper 2/3, mid 1/3..2/3, lower <1/3)
MC_BAND1 = 2.0
MC_BAND2 = 1.0

# Monte Carlo
NUM_TRIALS = 1000   # used by Option 2

# Plot padding when drawing field
PLOT_PADDING_M = 20_000.0

# Spatial index
GRID_CELL_SIZE_M = OUTER_RADIUS_M   # ~600 m cells works well
NEAREST_SEARCH_RADIUS_M = 40_000.0  # cap radius to look for nearest thermal

# ====== Sweep settings (Option 3) ======
SWEEP_MC1_MIN, SWEEP_MC1_MAX, SWEEP_MC1_STEP = 0.0, 3.0, 0.2
SWEEP_MC2_MIN, SWEEP_MC2_MAX, SWEEP_MC2_STEP = 0.0, 3.0, 0.2
SWEEP_TRIALS_PER_CELL = 120

# ====== 2D plot settings (Option 4) ======
PLOT2D_MC1_MIN, PLOT2D_MC1_MAX, PLOT2D_MC1_STEP = 0.0, 3.0, 0.1      # band 1 stepping
PLOT2D_MC2_VALUES = [0.0, 1.0, 2.0, 3.0]                              # band 2 curves
PLOT2D_TRIALS_PER_POINT = 120                                         # MC draws per x point

# ======================== Data Structures ==============================

@dataclass
class Thermal:
    center: Tuple[float, float]
    updraft_radius: float
    strength_ms: float  # m/s

# ======================== Utility Geometry =============================

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0]-a[0], b[1]-a[1])

def brg_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx, dy = (b[0]-a[0], b[1]-a[1])
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0  # 0° = +X

def move(a: Tuple[float, float], bearing_deg: float, d: float) -> Tuple[float, float]:
    ang = math.radians(bearing_deg)
    return (a[0] + d * math.cos(ang), a[1] + d * math.sin(ang))

def smallest_angle(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0

def is_in_forward_arc(observer: Tuple[float, float],
                      target: Tuple[float, float],
                      candidate: Tuple[float, float],
                      arc_deg: float) -> bool:
    to_target = brg_deg(observer, target)
    to_point  = brg_deg(observer, candidate)
    delta     = smallest_angle(to_point, to_target)
    return abs(delta) <= arc_deg / 2.0

# ======================== Glider & Bands ===============================

def get_band_and_mc(z_agl: float, z_cbl: float) -> Tuple[int, float]:
    if z_agl >= (2.0/3.0) * z_cbl:
        return (1, MC_BAND1)
    elif z_agl >= (1.0/3.0) * z_cbl:
        return (2, MC_BAND2)
    else:
        return (3, 0.0)

def get_glider_parameters(mc: float) -> Tuple[float, float, float]:
    v = np.linspace(POLAR_DATA['V_MS'].min(), POLAR_DATA['V_MS'].max(), 600)
    sink = f_sink_from_v(v)            # m/s down
    eff = np.maximum(sink - mc, -3.0)  # cap to avoid silly negative
    ratio = eff / np.maximum(v, 1e-6)
    idx = int(np.argmin(ratio))
    v_opt, sink_air = float(v[idx]), float(sink[idx])
    glide = v_opt / max(sink_air, 1e-6)
    return v_opt, sink_air, glide

# ======================== Thermal Helpers ==============================

def _draw_strength(lambda_strength: float) -> float:
    s = 0
    while s == 0:
        s = np.random.poisson(lambda_strength)
    return float(min(10.0, s))

def radius_from_strength(strength_ms: float) -> float:
    strength_ms = max(1.0, min(10.0, strength_ms))
    return (strength_ms / C_UPDRAFT_STRENGTH_DECREMENT) ** (1.0/3.0)

# ======================== Thermal Generators ===========================

def generate_poisson_thermals(sim_side_m: float,
                              lambda_thermals_km2: float,
                              lambda_strength: float) -> List[Thermal]:
    area_km2 = (sim_side_m / 1000.0) ** 2
    n = np.random.poisson(lambda_thermals_km2 * area_km2)
    half = sim_side_m / 2.0
    out: List[Thermal] = []
    for _ in range(n):
        cx = random.uniform(-half, half)
        cy = random.uniform(-half, half)
        s = _draw_strength(lambda_strength)
        r = radius_from_strength(s)
        out.append(Thermal(center=(cx, cy), updraft_radius=r, strength_ms=s))
    return out

def generate_clustered_thomas(sim_side_m: float,
                              lambda_parent_km2: float,
                              mean_children: float,
                              sigma_m: float,
                              lambda_strength: float) -> List[Thermal]:
    area_km2 = (sim_side_m / 1000.0) ** 2
    n_par = np.random.poisson(lambda_parent_km2 * area_km2)
    half = sim_side_m / 2.0
    out: List[Thermal] = []
    for _ in range(n_par):
        px = random.uniform(-half, half)
        py = random.uniform(-half, half)
        k = np.random.poisson(mean_children)
        if k <= 0: continue
        dx = np.random.normal(0.0, sigma_m, k)
        dy = np.random.normal(0.0, sigma_m, k)
        for i in range(k):
            cx, cy = px + dx[i], py + dy[i]
            if abs(cx) > half or abs(cy) > half: continue
            s = _draw_strength(lambda_strength)
            r = radius_from_strength(s)
            out.append(Thermal(center=(cx, cy), updraft_radius=r, strength_ms=s))
    return out

def generate_clustered_neyman(sim_side_m: float,
                              lambda_parent_km2: float,
                              mean_children: float,
                              radius_m: float,
                              lambda_strength: float) -> List[Thermal]:
    area_km2 = (sim_side_m / 1000.0) ** 2
    n_par = np.random.poisson(lambda_parent_km2 * area_km2)
    half = sim_side_m / 2.0
    out: List[Thermal] = []
    for _ in range(n_par):
        px = random.uniform(-half, half)
        py = random.uniform(-half, half)
        k = np.random.poisson(mean_children)
        if k <= 0: continue
        u = np.random.random(k)
        rr = radius_m * np.sqrt(u)
        theta = 2*np.pi*np.random.random(k)
        dx = rr * np.cos(theta); dy = rr * np.sin(theta)
        for i in range(k):
            cx, cy = px + dx[i], py + dy[i]
            if abs(cx) > half or abs(cy) > half: continue
            s = _draw_strength(lambda_strength)
            r = radius_from_strength(s)
            out.append(Thermal(center=(cx, cy), updraft_radius=r, strength_ms=s))
    return out

# ======================== Spatial Index (Uniform Grid) =================

class GridIndex:
    """Uniform grid spatial index for thermals."""
    __slots__ = ("cell", "cell_size")

    def __init__(self, cell_size: float):
        self.cell_size = float(cell_size)
        self.cell: Dict[Tuple[int,int], List[Thermal]] = {}

    def _key(self, x: float, y: float) -> Tuple[int,int]:
        cs = self.cell_size
        return (int(math.floor(x / cs)), int(math.floor(y / cs)))

    def insert(self, th: Thermal) -> None:
        k = self._key(th.center[0], th.center[1])
        self.cell.setdefault(k, []).append(th)

    @classmethod
    def build(cls, thermals: List[Thermal], cell_size: float) -> "GridIndex":
        gi = cls(cell_size)
        for th in thermals:
            gi.insert(th)
        return gi

    def _range_keys(self, xmin: float, ymin: float, xmax: float, ymax: float) -> Iterable[Tuple[int,int]]:
        cs = self.cell_size
        ix0, iy0 = int(math.floor(xmin / cs)), int(math.floor(ymin / cs))
        ix1, iy1 = int(math.floor(xmax / cs)), int(math.floor(ymax / cs))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                yield (ix, iy)

    def query_bbox(self, xmin: float, ymin: float, xmax: float, ymax: float) -> Iterable[Thermal]:
        seen = set()
        for k in self._range_keys(xmin, ymin, xmax, ymax):
            for th in self.cell.get(k, ()):
                x, y = th.center
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    if id(th) not in seen:
                        seen.add(id(th))
                        yield th

    def query_circle(self, cx: float, cy: float, radius: float) -> Iterable[Thermal]:
        xmin, ymin = cx - radius, cy - radius
        xmax, ymax = cx + radius, cy + radius
        r2 = radius * radius
        seen = set()
        for k in self._range_keys(xmin, ymin, xmax, ymax):
            for th in self.cell.get(k, ()):
                if id(th) in seen:
                    continue
                dx, dy = th.center[0] - cx, th.center[1] - cy
                if dx*dx + dy*dy <= r2:
                    seen.add(id(th))
                    yield th

# ======================== Segment / Annulus ============================

def _roots_for_radius(center: Tuple[float,float], R: float,
                      p1: Tuple[float,float], p2: Tuple[float,float]) -> List[float]:
    (cx, cy) = center
    (x1, y1) = (p1[0] - cx, p1[1] - cy)
    (x2, y2) = (p2[0] - cx, p2[1] - cy)
    dx, dy = (x2 - x1, y2 - y1)
    A = dx*dx + dy*dy
    if A == 0.0: return []
    B = 2*(x1*dx + y1*dy)
    C = x1*x1 + y1*y1 - R*R
    disc = B*B - 4*A*C
    if disc < 0.0: return []
    sd = math.sqrt(disc)
    return [(-B - sd) / (2*A), (-B + sd) / (2*A)]

def length_inside_annulus(center: Tuple[float,float],
                          r_in: float, r_out: float,
                          p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    ts = [0.0, 1.0]
    ts += [t for t in _roots_for_radius(center, r_in, p1, p2) if 0.0 <= t <= 1.0]
    ts += [t for t in _roots_for_radius(center, r_out, p1, p2) if 0.0 <= t <= 1.0]
    ts = sorted(set(ts))
    if len(ts) < 2:
        mid = ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)
        d = dist(center, mid)
        return dist(p1, p2) if (r_in <= d <= r_out) else 0.0
    total = 0.0
    for a, b in zip(ts[:-1], ts[1:]):
        ta, tb = max(0.0, a), min(1.0, b)
        if tb <= ta: continue
        pa = (p1[0] + (p2[0]-p1[0])*ta, p1[1] + (p2[1]-p1[1])*ta)
        pb = (p1[0] + (p2[0]-p1[0])*tb, p1[1] + (p2[1]-p1[1])*tb)
        mid = ((pa[0]+pb[0])/2.0, (pa[1]+pb[1])/2.0)
        d = dist(center, mid)
        if r_in <= d <= r_out:
            total += dist(pa, pb)
    return total

# ======================== Single Flight ================================

def simulate_flight(plot: bool = False) -> Tuple[bool, float]:
    """Return (success, final_altitude_m)."""
    origin = (0.0, 0.0)
    goal   = (0.0, TASK_DISTANCE)  # straight “north”
    z_agl  = CBL_ALT

    # sim square covering origin+goal+padding
    minx = min(origin[0], goal[0]) - PLOT_PADDING_M
    maxx = max(origin[0], goal[0]) + PLOT_PADDING_M
    miny = min(origin[1], goal[1]) - PLOT_PADDING_M
    maxy = max(origin[1], goal[1]) + PLOT_PADDING_M
    sim_side_m = max(maxx - minx, maxy - miny)

    # Thermal field
    if THERMAL_FIELD_MODEL == "poisson":
        thermals = generate_poisson_thermals(sim_side_m, LAMBDA_TOTAL, LAMBDA_STRENGTH)
    else:
        lam_parent = LAMBDA_PARENT if LAMBDA_PARENT is not None else (LAMBDA_TOTAL / max(MEAN_CHILDREN, 1e-6))
        if THERMAL_FIELD_MODEL == "thomas":
            thermals = generate_clustered_thomas(sim_side_m, lam_parent, MEAN_CHILDREN, THOMAS_SIGMA_M, LAMBDA_STRENGTH)
        elif THERMAL_FIELD_MODEL == "neyman":
            thermals = generate_clustered_neyman(sim_side_m, lam_parent, MEAN_CHILDREN, NEYMAN_RADIUS_M, LAMBDA_STRENGTH)
        else:
            raise ValueError(f"Unknown THERMAL_FIELD_MODEL='{THERMAL_FIELD_MODEL}'")

    # Build grid index
    grid = GridIndex.build(thermals, GRID_CELL_SIZE_M)

    # Plot
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 10))
        ax.set_aspect("equal")
        for th in thermals:
            core = plt.Circle(th.center, th.updraft_radius, color="red", alpha=0.25, ec="k", lw=0.3)
            ring = plt.Circle(th.center, OUTER_RADIUS_M, fill=False, ec="green", lw=0.5, ls=":")
            ax.add_patch(core); ax.add_patch(ring)
        ax.plot([origin[0], goal[0]], [origin[1], goal[1]], "b--", lw=1.0, label="Task")
        ax.scatter([origin[0]], [origin[1]], c="black", s=40, label="Start")
        ax.scatter([goal[0]], [goal[1]], c="blue",  s=40, label="Goal")
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        ax.set_title(f"Thermal field: {THERMAL_FIELD_MODEL}")
        ax.legend(loc="upper center")

    pos = origin
    safety_iter = 0

    while True:
        safety_iter += 1
        if safety_iter > 20_000:
            return (False, z_agl)

        d_to_goal = dist(pos, goal)
        if d_to_goal < GOAL_EPS:
            if plot:
                plt.plot([pos[0], goal[0]], [pos[1], goal[1]], "k-", lw=1.3, label="Flight path")
                plt.tight_layout(); plt.show(block=True)
            return (True, z_agl)

        if z_agl <= MIN_SAFE_ALT:
            if plot:
                plt.tight_layout(); plt.show(block=True)
            return (False, z_agl)

        # Band + MC → speed & sink
        _, mc = get_band_and_mc(z_agl, CBL_ALT)
        v_ms, sink_air_ms, _ = get_glider_parameters(mc)

        # Bearing to goal
        brg_to_goal = brg_deg(pos, goal)

        # ----- nearest thermal in forward sector via grid circle query -----
        search_r = min(NEAREST_SEARCH_RADIUS_M, d_to_goal + 1.0)
        nearest = None
        nearest_d = float("inf")
        for th in grid.query_circle(pos[0], pos[1], search_r):
            d = dist(pos, th.center)
            if d < 1.0:
                continue
            if d < nearest_d and is_in_forward_arc(pos, goal, th.center, SEARCH_ARC_DEG):
                nearest, nearest_d = th, d

        if nearest is not None:
            next_pt = nearest.center
            seg_len = nearest_d
        else:
            seg_len = max(STEP_MIN, min(STEP_MAX, d_to_goal))
            next_pt = move(pos, brg_to_goal, seg_len)

        if seg_len < 0.1:
            return (False, z_agl)

        # --------- Downdraft penalty via grid bbox prefilter ----------
        xmin = min(pos[0], next_pt[0]) - OUTER_RADIUS_M
        xmax = max(pos[0], next_pt[0]) + OUTER_RADIUS_M
        ymin = min(pos[1], next_pt[1]) - OUTER_RADIUS_M
        ymax = max(pos[1], next_pt[1]) + OUTER_RADIUS_M

        length_in_dd = 0.0
        for th in grid.query_bbox(xmin, ymin, xmax, ymax):
            r_in = th.updraft_radius; r_out = OUTER_RADIUS_M
            if r_out > r_in:
                length_in_dd += length_inside_annulus(th.center, r_in, r_out, pos, next_pt)

        time_seg = seg_len / max(v_ms, 1e-6)
        time_in_dd = length_in_dd / max(v_ms, 1e-6)
        dz = sink_air_ms * time_seg + DOWNDRAFT_SINK_MS * time_in_dd
        z_agl -= dz

        # advance & draw
        if plot:
            plt.plot([pos[0], next_pt[0]], [pos[1], next_pt[1]], "k-", lw=1.3)
        pos = next_pt

        # If arrived at a thermal, evaluate climb
        if nearest is not None and pos == nearest.center:
            if nearest.strength_ms >= mc:
                gain = CBL_ALT - z_agl
                if gain > 0:
                    z_agl = CBL_ALT

        # Safety
        if not math.isfinite(z_agl) or not math.isfinite(pos[0]) or not math.isfinite(pos[1]):
            return (False, 0.0)

# ======================== Monte Carlo (Option 2) =======================

def run_monte_carlo(num_trials: int = NUM_TRIALS) -> None:
    successes = 0
    for _ in tqdm(range(num_trials), desc=f"MC ({THERMAL_FIELD_MODEL})"):
        ok, _ = simulate_flight(plot=False)
        if ok:
            successes += 1
    p = successes / num_trials
    print(f"Trials: {num_trials} | Successes: {successes} | Probability: {p:.4f}")

# ======================== Sweep & 3D Plot (Option 3) ===================

def sweep_probability_surface() -> None:
    global MC_BAND1, MC_BAND2
    mc1_vals = np.round(np.arange(SWEEP_MC1_MIN, SWEEP_MC1_MAX + 1e-9, SWEEP_MC1_STEP), 10)
    mc2_vals = np.round(np.arange(SWEEP_MC2_MIN, SWEEP_MC2_MAX + 1e-9, SWEEP_MC2_STEP), 10)
    X_speed = np.zeros((len(mc2_vals), len(mc1_vals)), dtype=float)
    X_mc1   = np.zeros_like(X_speed)
    Y_mc2   = np.zeros_like(X_speed)
    Z_prob  = np.zeros_like(X_speed)

    for j, mc2 in enumerate(tqdm(mc2_vals, desc="Sweep MC2 rows")):
        for i, mc1 in enumerate(mc1_vals):
            MC_BAND1 = float(mc1); MC_BAND2 = float(mc2)
            v_opt, _, _ = get_glider_parameters(MC_BAND1)
            succ = 0
            for _ in range(SWEEP_TRIALS_PER_CELL):
                ok, _alt = simulate_flight(plot=False)
                succ += int(ok)
            prob = succ / SWEEP_TRIALS_PER_CELL
            X_speed[j, i] = v_opt; X_mc1[j, i] = mc1; Y_mc2[j, i] = mc2; Z_prob[j, i] = prob

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X_speed, Y_mc2, Z_prob, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("Speed v_opt from MC1 (m/s)"); ax.set_ylabel("MC band 2 (m/s)"); ax.set_zlabel("Probability")
    ax.set_title("Probability vs Speed(MC1) and MC2")
    plt.tight_layout(); plt.show()

    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(X_mc1, Y_mc2, Z_prob, linewidth=0, antialiased=True, alpha=0.9)
    ax2.set_xlabel("MC band 1 (m/s)"); ax2.set_ylabel("MC band 2 (m/s)"); ax2.set_zlabel("Probability")
    ax2.set_title("Probability vs MC1 and MC2")
    plt.tight_layout(); plt.show()

# ======================== 2D Plot (Option 4) ===========================

def sweep_2d_curves() -> None:
    """
    2D plot of success probability vs MC band 1.
    - x-axis: MC band 1 in 0.1 steps (PLOT2D_MC1_STEP)
    - separate curves for MC band 2 values (PLOT2D_MC2_VALUES, 1.0 increments by default)
    """
    global MC_BAND1, MC_BAND2

    mc1_vals = np.round(np.arange(PLOT2D_MC1_MIN, PLOT2D_MC1_MAX + 1e-9, PLOT2D_MC1_STEP), 10)

    plt.figure(figsize=(9, 6))
    for mc2 in PLOT2D_MC2_VALUES:
        probs = []
        MC_BAND2 = float(mc2)
        for mc1 in tqdm(mc1_vals, desc=f"Curves MC2={mc2:.1f}"):
            MC_BAND1 = float(mc1)
            succ = 0
            for _ in range(PLOT2D_TRIALS_PER_POINT):
                ok, _alt = simulate_flight(plot=False)
                succ += int(ok)
            probs.append(succ / PLOT2D_TRIALS_PER_POINT)
        plt.plot(mc1_vals, probs, marker="", linewidth=2, label=f"MC2 = {mc2:.1f}")

    plt.xlabel("MC band 1 (m/s)")
    plt.ylabel("Success probability")
    plt.title("Probability vs MC1 with MC2 curves")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Band 2")
    plt.tight_layout()
    plt.show()

# ======================== CLI =========================================

def main():
    print("1 = single run with plot")
    print("2 = Monte Carlo (current params)")
    print("3 = 3D sweep (prob vs speed & MC bands)")
    print("4 = 2D curves (prob vs MC1; separate curves for MC2)")
    opt = input("Enter 1, 2, 3, or 4: ").strip()
    if opt == "1":
        ok, z = simulate_flight(plot=True)
        print("Result:", "Success" if ok else "Landout", f"(final z={z:.0f} m)")
    elif opt == "2":
        run_monte_carlo(num_trials=NUM_TRIALS)
    elif opt == "3":
        sweep_probability_surface()
    elif opt == "4":
        sweep_2d_curves()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()