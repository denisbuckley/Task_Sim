#!/usr/bin/env python3
"""
task_sim_triangle_clustered_v1.py
-----------------------------------------------------
Derived from Gemini’s Poisson-based Monte Carlo thermal encounter simulator.

This version adds:
  • Clustered (Thomas / Neyman–Scott) thermal field generation.
  • Cluster geometry limited to the task corridor.
  • Adjustable parent intensity, child count, and cluster radius/sigma.
  • Same interfaces and outputs as the Poisson baseline.

Commit context:
  feat(sim): integrate clustered thermal generation (Thomas/Neyman–Scott)
  Based on prior Poisson version from task_sim_triangle.py.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random
from tqdm import tqdm
import csv
import pandas as pd
from scipy.interpolate import interp1d

# ------------------------------------------------------------
# --- Constants and physical parameters
# ------------------------------------------------------------
KNOT_TO_MS = 0.514444
FT_TO_M = 0.3048

C_UPDRAFT_STRENGTH_DECREMENT = 5.9952e-7
FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS = 1200.0
FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS = FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS / 2
G = 9.81
W_WING = 125.0
W_PILOT_BAGS = 100.0

# ------------------------------------------------------------
# --- Triangular task geometry
# ------------------------------------------------------------
TRIANGLE_PERIMETER_METERS = 300000.0  # 300 km
PATH_CORRIDOR_WIDTH_METERS = 300.0    # 0.3 km half-width either side
TRIANGLE_SIDE_LENGTH_METERS = TRIANGLE_PERIMETER_METERS / 3.0
PLOT_PADDING_METERS = 10000.0

# ------------------------------------------------------------
# --- Scenario Parameters
# ------------------------------------------------------------
SCENARIO_Z_CBL = 2500.0
SCENARIO_MC_SNIFF = 2
SCENARIO_LAMBDA_STRENGTH = 3  # mean thermal strength (Poisson for updrafts)
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.0145  # empirical λ (thermals/km²)

# Altitude bands for MC sniffing
MC_SNIFF_ALTITUDE_BANDS = {
    "upper": {"min_alt": 1500, "mc": 2.0},
    "lower": {"min_alt": 500, "mc": 1.0},
}

# ------------------------------------------------------------
# --- Clustered field parameters
# ------------------------------------------------------------
SCENARIO_CLUSTER_MODEL = "neyman"   # "thomas" or "neyman"
SCENARIO_LAMBDA_PARENT_PER_SQ_KM = 0.0036
SCENARIO_MEAN_CHILDREN_PER_PARENT = 4.0
SCENARIO_CLUSTER_SIGMA_M = 100     # for Thomas
SCENARIO_CLUSTER_RADIUS_M = 100
# for Neyman

# ------------------------------------------------------------
# --- Polar Data (LS10 18m)
# ------------------------------------------------------------
POLAR_DATA = pd.DataFrame({
    'V_Knot': [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120],
    'Sink_Knots': [0.55, 0.50, 0.51, 0.55, 0.60, 0.66, 0.72, 0.79, 0.86, 0.94, 1.02, 1.11, 1.20, 1.30, 1.40, 1.50],
})
POLAR_DATA['V_MS'] = POLAR_DATA['V_Knot'] * KNOT_TO_MS
POLAR_DATA['Sink_MS'] = POLAR_DATA['Sink_Knots'] * KNOT_TO_MS

f_sink_from_v = interp1d(POLAR_DATA['V_MS'], POLAR_DATA['Sink_MS'], kind='linear', fill_value='extrapolate')

# ------------------------------------------------------------
# --- Helper Functions
# ------------------------------------------------------------
def get_mc_for_sniffing(altitude_agl_meters):
    if altitude_agl_meters >= MC_SNIFF_ALTITUDE_BANDS["upper"]["min_alt"]:
        return MC_SNIFF_ALTITUDE_BANDS["upper"]["mc"]
    elif altitude_agl_meters >= MC_SNIFF_ALTITUDE_BANDS["lower"]["min_alt"]:
        return MC_SNIFF_ALTITUDE_BANDS["lower"]["mc"]
    else:
        return 0


def calculate_sniffing_radius(Wt_ms_ambient, MC_for_sniffing_ms, thermal_type="NORMAL"):
    C_thermal = 0.033 if thermal_type == "NORMAL" else 0.10
    Wt_knots = Wt_ms_ambient / KNOT_TO_MS
    MC_knots = MC_for_sniffing_ms / KNOT_TO_MS
    y = Wt_knots - MC_knots
    if y / C_thermal > 0:
        R_sniff_ft = 100 * ((y / C_thermal) ** (1 / 3))
    else:
        R_sniff_ft = 0
    return (R_sniff_ft * FT_TO_M)


def check_circle_line_segment_intersection(circle_center, radius, line_start, line_end):
    fx, fy = circle_center
    x1, y1 = line_start
    x2, y2 = line_end
    dx = x2 - x1
    dy = y2 - y1
    A = dx ** 2 + dy ** 2
    B = 2 * (dx * (x1 - fx) + dy * (y1 - fy))
    C = (x1 - fx) ** 2 + (y1 - fy) ** 2 - radius ** 2
    disc = B ** 2 - 4 * A * C
    if disc < 0:
        return False, []
    t1 = (-B + math.sqrt(disc)) / (2 * A)
    t2 = (-B - math.sqrt(disc)) / (2 * A)
    pts = []
    for t in (t1, t2):
        if 0 <= t <= 1:
            pts.append((x1 + t * dx, y1 + t * dy))
    return len(pts) > 0, pts

# ------------------------------------------------------------
# --- Corridor geometry helpers
# ------------------------------------------------------------
def segment_length(a, b): return math.hypot(b[0]-a[0], b[1]-a[1])
def unit_normal(a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    L = math.hypot(dx, dy)
    return (-dy/L, dx/L)
def sample_point_in_corridor_along_segment(a, b, half_width):
    t = random.random()
    x, y = a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1])
    nx, ny = unit_normal(a, b)
    s = (random.random() - 0.5) * 2 * half_width
    return (x + s*nx, y + s*ny)
def point_to_segment_distance(px, py, a, b):
    ax, ay = a; bx, by = b
    dx, dy = bx-ax, by-ay
    if dx == dy == 0:
        return math.hypot(px-ax, py-ay)
    t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy)/(dx*dx+dy*dy)))
    cx, cy = ax + t*dx, ay + t*dy
    return math.hypot(px-cx, py-cy)
def in_corridor(px, py, segments, half_width):
    return any(point_to_segment_distance(px, py, a, b) <= half_width for a, b in segments)

# ------------------------------------------------------------
# --- Clustered thermal field generators
# ------------------------------------------------------------
def _draw_strength_and_radius(lambda_strength):
    s = 0
    while s == 0:
        s = np.random.poisson(lambda_strength)
    s = min(10, s)
    r = (s / C_UPDRAFT_STRENGTH_DECREMENT) ** (1/3)
    return s, r

def _expected_area_corridor(segments, half_width):
    return sum(segment_length(a,b) for a,b in segments) * (2*half_width)

def _sample_parents_in_corridor(segments, half_width, lambda_parent_per_km2):
    A_m2 = _expected_area_corridor(segments, half_width)
    A_km2 = A_m2 / 1e6
    n_parents = np.random.poisson(lambda_parent_per_km2 * A_km2)
    seg_lengths = np.array([segment_length(a,b) for a,b in segments])
    seg_probs = seg_lengths / seg_lengths.sum()
    seg_choices = np.random.choice(len(segments), size=n_parents, p=seg_probs)
    return [sample_point_in_corridor_along_segment(*segments[i], half_width) for i in seg_choices]

def generate_clustered_thermals_thomas(segments, half_width, λ_parent, mean_children, σ_m, λ_strength):
    parents = _sample_parents_in_corridor(segments, half_width, λ_parent)
    thermals = []
    for px, py in parents:
        k = np.random.poisson(mean_children)
        if k <= 0: continue
        dx, dy = np.random.normal(0, σ_m, k), np.random.normal(0, σ_m, k)
        for i in range(k):
            cx, cy = px+dx[i], py+dy[i]
            if not in_corridor(cx, cy, segments, half_width): continue
            s, r = _draw_strength_and_radius(λ_strength)
            thermals.append({"center": (cx, cy), "updraft_radius": r, "updraft_strength": s, "intercepted": False})
    return thermals

def generate_clustered_thermals_neyman(segments, half_width, λ_parent, mean_children, R_m, λ_strength):
    parents = _sample_parents_in_corridor(segments, half_width, λ_parent)
    thermals = []
    for px, py in parents:
        k = np.random.poisson(mean_children)
        if k <= 0: continue
        u, θ = np.random.random(k), 2*np.pi*np.random.random(k)
        r = R_m * np.sqrt(u)
        dx, dy = r*np.cos(θ), r*np.sin(θ)
        for i in range(k):
            cx, cy = px+dx[i], py+dy[i]
            if not in_corridor(cx, cy, segments, half_width): continue
            s, rr = _draw_strength_and_radius(λ_strength)
            thermals.append({"center": (cx, cy), "updraft_radius": rr, "updraft_strength": s, "intercepted": False})
    return thermals

# ------------------------------------------------------------
# --- Visualization
# ------------------------------------------------------------
def draw_clustered_thermals_and_path():
    triangle_height = TRIANGLE_SIDE_LENGTH_METERS * math.sqrt(3) / 2
    v1 = (0, triangle_height / 2)
    v2 = (-TRIANGLE_SIDE_LENGTH_METERS / 2, -triangle_height / 2)
    v3 = (TRIANGLE_SIDE_LENGTH_METERS / 2, -triangle_height / 2)
    segments = [(v1, v2), (v2, v3), (v3, v1)]
    half_width = PATH_CORRIDOR_WIDTH_METERS / 2

    if SCENARIO_CLUSTER_MODEL == "neyman":
        thermals = generate_clustered_thermals_thomas(
            segments, half_width,
            SCENARIO_LAMBDA_PARENT_PER_SQ_KM,
            SCENARIO_MEAN_CHILDREN_PER_PARENT,
            SCENARIO_CLUSTER_SIGMA_M,
            SCENARIO_LAMBDA_STRENGTH)
    else:
        thermals = generate_clustered_thermals_neyman(
            segments, half_width,
            SCENARIO_LAMBDA_PARENT_PER_SQ_KM,
            SCENARIO_MEAN_CHILDREN_PER_PARENT,
            SCENARIO_CLUSTER_RADIUS_M,
            SCENARIO_LAMBDA_STRENGTH)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    for a,b in segments:
        ax.plot([a[0], b[0]], [a[1], b[1]], 'b-')

    # corridor shading
    for a,b in segments:
        nx, ny = unit_normal(a,b)
        ax.fill_between([a[0], b[0]], [a[1]-half_width*ny, b[1]-half_width*ny],
                        [a[1]+half_width*ny, b[1]+half_width*ny], color='blue', alpha=0.05)

    for th in thermals:
        cx, cy = th['center']
        s = th['updraft_strength']
        r = th['updraft_radius']
        ax.add_patch(patches.Circle((cx,cy), r, facecolor='red', alpha=0.5*s/10, edgecolor='k', linewidth=0.3))

    ax.set_title(f"Clustered ({SCENARIO_CLUSTER_MODEL}) Thermals λ={SCENARIO_LAMBDA_THERMALS_PER_SQ_KM:.3f}/km²")
    plt.show()

# ------------------------------------------------------------
# --- Monte Carlo Simulation
# ------------------------------------------------------------
def simulate_intercept_experiment_clustered(num_simulations=5000):
    triangle_height = TRIANGLE_SIDE_LENGTH_METERS * math.sqrt(3) / 2
    v1 = (0, triangle_height / 2)
    v2 = (-TRIANGLE_SIDE_LENGTH_METERS / 2, -triangle_height / 2)
    v3 = (TRIANGLE_SIDE_LENGTH_METERS / 2, -triangle_height / 2)
    segments = [(v1, v2), (v2, v3), (v3, v1)]
    half_width = PATH_CORRIDOR_WIDTH_METERS / 2
    λs = SCENARIO_LAMBDA_STRENGTH

    intercepts = 0
    for _ in tqdm(range(num_simulations), desc="Monte Carlo Clustered"):
        if SCENARIO_CLUSTER_MODEL == "thomas":
            thermals = generate_clustered_thermals_thomas(
                segments, half_width,
                SCENARIO_LAMBDA_PARENT_PER_SQ_KM,
                SCENARIO_MEAN_CHILDREN_PER_PARENT,
                SCENARIO_CLUSTER_SIGMA_M,
                λs)
        else:
            thermals = generate_clustered_thermals_neyman(
                segments, half_width,
                SCENARIO_LAMBDA_PARENT_PER_SQ_KM,
                SCENARIO_MEAN_CHILDREN_PER_PARENT,
                SCENARIO_CLUSTER_RADIUS_M,
                λs)

        sniff = calculate_sniffing_radius(λs, SCENARIO_MC_SNIFF)
        path_intercepted = False
        for th in thermals:
            if th["updraft_strength"] < SCENARIO_MC_SNIFF:
                continue
            for a,b in segments:
                hit, _ = check_circle_line_segment_intersection(th["center"], sniff, a, b)
                if hit:
                    path_intercepted = True
                    break
            if path_intercepted:
                break
        if path_intercepted:
            intercepts += 1

    p = intercepts / num_simulations
    print(f"\n--- Monte Carlo Clustered Results ---")
    print(f"Simulations: {num_simulations}")
    print(f"Intercepts:  {intercepts}")
    print(f"Probability: {p:.4f}")

# ------------------------------------------------------------
# --- Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Visualize clustered thermals")
    print("2. Run Monte Carlo simulation")

    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        draw_clustered_thermals_and_path()
    elif choice == "2":
        simulate_intercept_experiment_clustered()
    else:
        print("Invalid selection.")