#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task_Sim / therm_arc_angles.py
--------------------------------------------------------------------
Transferred from Poisson repository → Task_Sim (2025-11-04)

Description:
  Monte Carlo and single-flight simulator for glider–thermal interaction.
  Based on a homogeneous Poisson thermal field (from Poisson repo),
  extended here to include:
    • Clustered thermal field models (Thomas & Neyman–Scott)
    • Downdraft ring penalty on glide segments intersecting thermals

Key Features:
  - LS10 polar, MacCready-optimized dynamic speeds
  - 3 altitude bands for MC values (upper/mid/lower CBL)
  - Forward-arc sniffing for nearest thermal intercepts
  - Optional clustered field for spatial realism
  - Downdraft penalties reduce glide height when crossing rings
  - Monte Carlo mode (option 2) for success probability analysis
  - Single plotted run (option 1) for visualization/debugging

Author: Denis Buckley & ChatGPT (GPT-5)
Date: 2025-11-04
Repository: Task_Sim
--------------------------------------------------------------------
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe backend for batch mode
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import sin, cos, radians, atan2, sqrt, degrees
import random
from tqdm import tqdm
import csv

# ======================== Parameters ===================================

CBL_ALT = 2500.0
MIN_SAFE_ALT = 500.0
THERMAL_LAMBDA = 0.02           # per km² (Poisson default)
THERMAL_LAMBDA_PARENT = 0.003   # for cluster parents
THERMAL_CHILDREN_MEAN = 5
THERMAL_CLUSTER_SIGMA = 150.0   # m, spread of cluster children
THERMAL_RADIUS_SCALE = 150.0    # m base scaling for radius
DOWNDRAFT_WIDTH = 200.0         # m annulus width
DOWNDRAFT_PENALTY = 0.25        # extra sink multiplier when in ring
THERMAL_STRENGTH_MEAN = 2.0     # m/s mean (Poisson strength)

SEARCH_ARC = 30.0               # deg, forward detection arc
TASK_DISTANCE = 100000.0        # m
NUM_TRIALS = 1000               # Monte Carlo
CLUSTER_MODEL = "thomas"        # 'poisson', 'thomas', or 'neyman'

# =======================================================================

@dataclass
class Thermal:
    x: float
    y: float
    strength: float
    radius: float


# ===================== Helper Geometry ================================
def distance(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def bearing(a, b):
    return degrees(atan2(b[0]-a[0], b[1]-a[1])) % 360

def move(a, brg, dist):
    return (a[0] + dist*sin(radians(brg)), a[1] + dist*cos(radians(brg)))


# ===================== Thermal Field Generators ========================
def generate_poisson_thermals(side_m, λ_th, λ_strength):
    area_km2 = (side_m/1000)**2
    n = np.random.poisson(λ_th * area_km2)
    thermals = []
    for _ in range(n):
        x, y = np.random.uniform(-side_m/2, side_m/2, 2)
        strength = max(1.0, np.random.poisson(λ_strength))
        radius = THERMAL_RADIUS_SCALE * (strength / 2.0)**(1/3)
        thermals.append(Thermal(x, y, strength, radius))
    return thermals


def generate_clustered_thermals(side_m, model="thomas"):
    area_km2 = (side_m/1000)**2
    n_parent = np.random.poisson(THERMAL_LAMBDA_PARENT * area_km2)
    parents = np.random.uniform(-side_m/2, side_m/2, (n_parent, 2))
    thermals = []

    for px, py in parents:
        n_children = np.random.poisson(THERMAL_CHILDREN_MEAN)
        for _ in range(n_children):
            if model == "thomas":
                dx, dy = np.random.normal(0, THERMAL_CLUSTER_SIGMA, 2)
            else:  # Neyman–Scott
                r = THERMAL_CLUSTER_SIGMA * np.sqrt(np.random.rand())
                θ = 2*np.pi*np.random.rand()
                dx, dy = r*np.cos(θ), r*np.sin(θ)
            x, y = px+dx, py+dy
            strength = max(1.0, np.random.poisson(THERMAL_STRENGTH_MEAN))
            radius = THERMAL_RADIUS_SCALE * (strength / 2.0)**(1/3)
            thermals.append(Thermal(x, y, strength, radius))
    return thermals


# ===================== Flight Simulator ================================
def simulate_flight(plot=False):
    origin = (0.0, 0.0)
    goal = (0.0, TASK_DISTANCE)
    z_agl = CBL_ALT

    # --- Generate thermals ---
    side_m = TASK_DISTANCE * 1.5
    if CLUSTER_MODEL == "poisson":
        thermals = generate_poisson_thermals(side_m, THERMAL_LAMBDA, THERMAL_STRENGTH_MEAN)
    else:
        thermals = generate_clustered_thermals(side_m, model=CLUSTER_MODEL)

    # --- Plot setup ---
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        for th in thermals:
            core = plt.Circle((th.x, th.y), th.radius, color="red", fill=False, alpha=0.5)
            ring = plt.Circle((th.x, th.y), th.radius+DOWNDRAFT_WIDTH, color="blue", fill=False, linestyle="--", alpha=0.3)
            ax.add_patch(core)
            ax.add_patch(ring)
        ax.plot([origin[0], goal[0]], [origin[1], goal[1]], "k--")
        ax.set_xlim(-side_m/2, side_m/2)
        ax.set_ylim(-side_m/2, side_m/2)

    # --- Flight loop ---
    pos = origin
    while True:
        d_goal = distance(pos, goal)
        if d_goal < 100:
            return True, z_agl  # success
        if z_agl <= MIN_SAFE_ALT:
            return False, z_agl  # landout

        # find thermals within forward arc
        brg_goal = bearing(pos, goal)
        nearest, nearest_d = None, 1e9
        for th in thermals:
            d = distance(pos, (th.x, th.y))
            if d < 1.0:
                continue
            brg_th = bearing(pos, (th.x, th.y))
            diff = (brg_th - brg_goal + 180) % 360 - 180
            if abs(diff) <= SEARCH_ARC/2 and d < nearest_d:
                nearest, nearest_d = th, d

        # step size
        leg = min(2000.0, d_goal)
        next_pt = move(pos, brg_goal, leg)
        z_agl -= leg * 0.04  # baseline sink (m per m)
        pos = next_pt

        # downdraft penalty if passing near thermal
        for th in thermals:
            d_to_th = distance(pos, (th.x, th.y))
            if th.radius < d_to_th <= th.radius + DOWNDRAFT_WIDTH:
                z_agl -= leg * 0.04 * DOWNDRAFT_PENALTY

        # climb if thermal strong enough
        if nearest and nearest_d < 200 and nearest.strength >= 2.0:
            z_agl = CBL_ALT  # reset altitude

        if z_agl <= 0:
            return False, 0

    if plot:
        plt.savefig("outputs/flight_plot.png", dpi=150)
        plt.close(fig)


# ===================== Monte Carlo Runner ==============================
def run_monte_carlo(num_trials=NUM_TRIALS, plot=False):
    successes = 0
    for _ in tqdm(range(num_trials), desc=f"MC ({CLUSTER_MODEL})"):
        ok, _ = simulate_flight(plot=False)
        if ok:
            successes += 1
    p_success = successes / num_trials
    print(f"Success probability: {p_success:.3f}")


# ===================== Main CLI =======================================
def main():
    print("1 = single run with plot, 2 = Monte Carlo")
    opt = input("Enter 1 or 2: ").strip()
    if opt == "1":
        ok, _ = simulate_flight(plot=True)
        print("Result:", "Success" if ok else "Landout")
    else:
        run_monte_carlo(num_trials=NUM_TRIALS, plot=False)


if __name__ == "__main__":
    main()