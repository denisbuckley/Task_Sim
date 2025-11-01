# task_sim_triangle.py
# Continuous 3-leg triangle over a real thermal field — NO "suicide" to Z_MIN.
# Single-run: one random triangle shared by all 9 MC triples; writes a per-run CSV and one plot per triple.
# Batch-run: many random triangles; prints success stats and writes a compact CSV.


import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
THETA_DEG    = 30.0                 # sector half-angle either side of leg bearing (±)
TRI_TOTAL_KM = 300.0                # triangle perimeter (km), legs ≈ TRI_TOTAL_KM/3 each

# Altitude bands (your spec)
Z_TOP        = 3000.0               # top of band 1
Z_MID        = 2400.0               # boundary 1↔2
Z_LOW        = 1600.0               # boundary 2↔3
Z_MIN        = 800.0                # hard floor

STEP_KM      = 2.0                  # glide integration chunk (km)
NEAR_PAD_KM  = 20.0                 # for plotting nearby thermals

# RNG: set to an integer for reproducibility; None -> new triangle each execution
RNG_SEED     = None

# Inputs
IN_CSV = Path("outputs/waypoints/thermal_waypoints_v1.csv")
IN_GJ  = Path("outputs/waypoints/thermal_waypoints_v1.geojson")

# Outputs (single run)
OUT_DIR      = Path("outputs/task_sim/plot_sim")
OUT_CSV_SUM  = Path("outputs/task_sim/triangle_results.csv")

# Outputs (batch run)
BATCH_CSV    = Path("outputs/task_sim/triangle_batch_summary.csv")

# -----------------------------
# Polar nodes: MC ↦ (V, LD) with linear interp
# -----------------------------
MC_nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
V_nodes  = np.array([36.9, 39.6, 41.8, 43.8, 45.6, 48.9])   # m/s
LD_nodes = np.array([48.6, 38.9, 32.4, 28.0, 24.9, 20.6])   # L/D

def V_of_MC(mc: float) -> float:
    return float(np.interp(mc, MC_nodes, V_nodes))

def LD_of_MC(mc: float) -> float:
    return float(np.interp(mc, MC_nodes, LD_nodes))

# Grid for final-glide feasibility (we pick fastest admissible)
MC_FINAL_GRID = np.linspace(0.0, 3.0, 61)  # 0.05 step

# -----------------------------
# Utilities: load & project field
# -----------------------------
def load_points_from_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame(columns=["lat","lon","strength"])
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    lat_col = next((cols[c] for c in ["lat","latitude","y"] if c in cols), None)
    lon_col = next((cols[c] for c in ["lon","longitude","lng","x"] if c in cols), None)
    s_col   = next((cols[c] for c in [
        "strength_mean_all","climb_rate_ms","avg_climb","circle_av_climb_ms"
    ] if c in cols), None)
    out = pd.DataFrame({
        "lat": df[lat_col] if lat_col else np.nan,
        "lon": df[lon_col] if lon_col else np.nan,
        "strength": df[s_col] if s_col else np.nan
    })
    return out.dropna(subset=["lat","lon"])

def load_points_from_geojson(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame(columns=["lat","lon","strength"])
    with open(p, "r") as f:
        gj = json.load(f)
    pts = []
    def add_geom(geom, props=None):
        if not geom or "type" not in geom: return
        t = geom["type"]
        if t == "Point":
            lon, lat = geom["coordinates"][:2]
            s = (props or {}).get("strength_mean_all")
            pts.append((lat, lon, s))
        elif t == "MultiPoint":
            for c in geom["coordinates"]:
                lon, lat = c[:2]; pts.append((lat, lon, None))
        elif t == "FeatureCollection":
            for feat in gj.get("features", []):
                add_geom(feat.get("geometry"), feat.get("properties", {}))
        elif t == "Feature":
            add_geom(geom.get("geometry"), props or {})
        elif t in ("GeometryCollection",):
            for g in geom.get("geometries", []):
                add_geom(g, props or {})
    if "type" in gj and gj["type"] in ("FeatureCollection","Feature","Point","MultiPoint","GeometryCollection"):
        if gj["type"] == "FeatureCollection":
            for feat in gj.get("features", []):
                add_geom(feat.get("geometry"), feat.get("properties", {}))
        elif gj["type"] == "Feature":
            add_geom(gj.get("geometry"), gj.get("properties", {}))
        else:
            add_geom(gj, {})
    return pd.DataFrame(pts, columns=["lat","lon","strength"])

def enu_km(df_ll: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (E_km, N_km, meta) projected about centroid."""
    R_km = 6371.0
    lat0 = float(df_ll["lat"].mean()); lon0 = float(df_ll["lon"].mean())
    lat0_r = np.radians(lat0)
    E = np.radians(df_ll["lon"] - lon0) * np.cos(lat0_r) * R_km
    N = np.radians(df_ll["lat"] - lat0) * R_km
    return E.to_numpy(), N.to_numpy(), {"lat0": lat0, "lon0": lon0}

# -----------------------------
# Geometry & band helpers
# -----------------------------
def pick_random_triangle(pts_xy: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    One triangle inside field bbox. Equilateral with side ≈ TRI_TOTAL_KM/3.
    """
    xmin, ymin = pts_xy.min(axis=0)
    xmax, ymax = pts_xy.max(axis=0)
    M = 15.0  # margin
    if (xmax - xmin) < 2*M or (ymax - ymin) < 2*M:
        cx = 0.5*(xmin+xmax); cy = 0.5*(ymin+ymax)
    else:
        cx = rng.uniform(xmin+M, xmax-M)
        cy = rng.uniform(ymin+M, ymax-M)
    side_km = TRI_TOTAL_KM / 3.0
    r = side_km / math.sqrt(3.0)  # equilateral circumradius
    base = rng.uniform(0, 2*np.pi)
    A = np.array([cx + r*math.cos(base),                cy + r*math.sin(base)], dtype=float)
    B = np.array([cx + r*math.cos(base + 2*np.pi/3.0),  cy + r*math.sin(base + 2*np.pi/3.0)], dtype=float)
    C = np.array([cx + r*math.cos(base + 4*np.pi/3.0),  cy + r*math.sin(base + 4*np.pi/3.0)], dtype=float)
    return A, B, C

def band_of_alt(h: float) -> int:
    # 3000–2400 => band 1; 2400–1600 => band 2; 1600–800 => band 3
    if h > Z_MID:   # (2400, 3000]
        return 1
    if h > Z_LOW:   # (1600, 2400]
        return 2
    return 3        # ≤ 1600

# -----------------------------
# Glide segment (band-splitting, optional V_override)
# -----------------------------
def glide_segment(p0: np.ndarray, p1: np.ndarray, alt0: float,
                  MC1: float, MC2: float, MC3: float,
                  accum: Dict[int,float]|None = None,
                  V_override: float|None = None) -> Tuple[np.ndarray, float, float, float]:
    """
    Glide from p0 toward p1. Switch MC at 3000/2400/1600/800 as we descend.
    Splits exactly at band boundaries to credit distance per band.
    If V_override is given (m/s), fly at that speed (used for final glide).
    Returns (p_end, alt_end, time_s, dist_km).
    """
    total_t = 0.0
    total_d = 0.0
    cur  = p0.copy()
    alt  = float(alt0)
    vec  = p1 - p0
    goal = float(np.hypot(vec[0], vec[1]))
    if goal < 1e-9:
        return p1.copy(), alt, 0.0, 0.0
    u = vec / goal
    remain = goal

    while remain > 1e-6 and alt > Z_MIN:
        b  = band_of_alt(alt)
        MC = MC1 if b == 1 else (MC2 if b == 2 else MC3)
        V  = V_override if (V_override is not None) else V_of_MC(MC)  # m/s
        LD = LD_of_MC(MC)

        # next boundary altitude for this band
        next_floor = Z_MID if b == 1 else (Z_LOW if b == 2 else Z_MIN)
        usable_drop_m = max(0.0, alt - next_floor)
        dist_to_boundary_km = (usable_drop_m * LD) / 1000.0

        piece_km = min(remain, dist_to_boundary_km, STEP_KM)
        if piece_km <= 1e-9:
            break

        dt = (piece_km * 1000.0) / V
        dh = (piece_km * 1000.0) / LD

        new_alt = alt - dh
        # clip if we would go below Z_MIN inside this piece
        if new_alt < Z_MIN:
            dh_to_min = alt - Z_MIN
            piece_km = (dh_to_min * LD) / 1000.0
            dt = (piece_km * 1000.0) / V
            new_alt = Z_MIN

        alt   = new_alt
        cur   = cur + u * piece_km
        remain-= piece_km
        total_d += piece_km
        total_t += dt

        if accum is not None:
            accum[b] = accum.get(b, 0.0) + piece_km

        if alt <= Z_MIN:
            break

    return cur, alt, total_t, total_d

# -----------------------------
# Core: one continuous triangle
# -----------------------------
def simulate_triangle_continuous(P: np.ndarray, strengths: np.ndarray|None,
                                 tri: Tuple[np.ndarray,np.ndarray,np.ndarray],
                                 MC1: float, MC2: float, MC3: float,
                                 rng=np.random.default_rng()) -> Dict[str,Any]:
    """
    Fly A→B, then B→C, then C→A without altitude reset.
    Sector search: nearest thermal within ±THETA. Accept climb to Z_TOP if W ≥ current-band MC.
    Final glide check allowed on each leg; when accepted, glide to that leg end at fastest admissible speed.
    IMPORTANT: We DO NOT force alt=Z_MIN after final glide; the arrived altitude carries into the next leg.
    """
    A, B, C = tri
    legs = [(A,B), (B,C), (C,A)]

    # Strengths
    W = strengths if strengths is not None else np.full(P.shape[0], 2.0, float)
    W = np.asarray(W, float)
    W = np.clip(W, 0.2, 10.0)

    path = [A.copy()]
    cur  = A.copy()
    alt  = Z_TOP
    t_glide = 0.0
    t_climb = 0.0
    track_dist = 0.0
    dist_by_band = {1:0.0, 2:0.0, 3:0.0}

    for (_, E) in legs:
        while True:
            se_vec  = E - cur
            se_dist = float(np.hypot(se_vec[0], se_vec[1]))
            if se_dist < 1e-6:
                # reached turnpoint; proceed to next leg with SAME altitude
                break

            # Final-glide feasibility toward E: arrival >= Z_MIN
            if alt > Z_MIN:
                req_LD = (se_dist*1000.0) / max(1e-6, (alt - Z_MIN))
                LDs = np.interp(MC_FINAL_GRID, MC_nodes, LD_nodes)
                ok  = LDs >= req_LD
                if np.any(ok):
                    Vs   = np.interp(MC_FINAL_GRID, MC_nodes, V_nodes)
                    Vmax = float(np.max(Vs[ok]))   # m/s (fastest admissible)
                    # Final glide with band-split credit — DO NOT force alt to Z_MIN
                    cur, alt, dt, dd = glide_segment(
                        cur, E, alt, MC1, MC2, MC3,
                        accum=dist_by_band, V_override=Vmax
                    )
                    t_glide    += dt
                    track_dist += dd
                    path.append(cur.copy())
                    # arrive at E with whatever alt the physics gave us
                    break  # leg finished

            # If we cannot final-glide and have no altitude left, fail
            if alt <= Z_MIN:
                T_tot = t_glide + t_climb
                return {
                    "success": False,
                    "time_total_s": T_tot,
                    "time_glide_s": t_glide,
                    "time_climb_s": t_climb,
                    "track_dist_km": track_dist,
                    "V_track_kmh": (track_dist / (T_tot/3600.0)) if T_tot>0 else float("nan"),
                    "V_task_kmh": float("nan"),
                    "dist_by_band": dist_by_band,
                    "path_xy": np.vstack(path) if len(path) else np.empty((0,2)),
                    "triangle": (A,B,C)
                }

            # Sector search to E
            hdg = math.degrees(math.atan2(se_vec[1], se_vec[0]))
            vecs  = P - cur
            dists = np.hypot(vecs[:,0], vecs[:,1])
            angs  = np.degrees(np.arctan2(vecs[:,1], vecs[:,0]))
            deltas= np.abs(((angs - hdg + 180.0) % 360.0) - 180.0)
            mask  = (deltas <= THETA_DEG) & (dists > 1e-6)

            if not np.any(mask):
                # No candidate — glide a straight STEP toward E
                step_to = cur + (se_vec / se_dist) * min(STEP_KM, se_dist)
                cur, alt, dt, dd = glide_segment(
                    cur, step_to, alt, MC1, MC2, MC3, accum=dist_by_band
                )
                t_glide    += dt
                track_dist += dd
                path.append(cur.copy())
                continue

            # Nearest thermal in sector
            cand_idx = int(np.argmin(np.where(mask, dists, np.inf)))
            cand_xy  = P[cand_idx]
            cand_W   = float(W[cand_idx])

            # Glide to candidate
            cur, alt, dt, dd = glide_segment(
                cur, cand_xy, alt, MC1, MC2, MC3, accum=dist_by_band
            )
            t_glide    += dt
            track_dist += dd
            path.append(cur.copy())

            # At thermal, accept climb if W ≥ current-band MC
            if np.allclose(cur, cand_xy, atol=1e-3):
                cur_band = band_of_alt(alt)
                cur_MC   = MC1 if cur_band == 1 else (MC2 if cur_band == 2 else MC3)
                if cand_W >= cur_MC:
                    climb = max(0.0, Z_TOP - alt)
                    if climb > 0.0:
                        t_climb += climb / cand_W
                        alt = Z_TOP
                # loop continues toward E with updated alt

    # All three legs completed; success
    T_tot = t_glide + t_climb
    return {
        "success": True,
        "time_total_s": T_tot,
        "time_glide_s": t_glide,
        "time_climb_s": t_climb,
        "track_dist_km": track_dist,
        "V_track_kmh": (track_dist / (T_tot/3600.0)) if T_tot>0 else float("nan"),
        "V_task_kmh": (TRI_TOTAL_KM / (T_tot/3600.0)) if T_tot>0 else float("nan"),
        "dist_by_band": dist_by_band,
        "path_xy": np.vstack(path),
        "triangle": (A,B,C)
    }

# -----------------------------
# Plotting
# -----------------------------
def plot_triangle(pts_km: np.ndarray, tri: Tuple[np.ndarray,np.ndarray,np.ndarray],
                  path_xy: np.ndarray, res: Dict[str,Any], out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    A,B,C = tri
    xmin = min(A[0],B[0],C[0]) - NEAR_PAD_KM
    xmax = max(A[0],B[0],C[0]) + NEAR_PAD_KM
    ymin = min(A[1],B[1],C[1]) - NEAR_PAD_KM
    ymax = max(A[1],B[1],C[1]) + NEAR_PAD_KM
    mask = (pts_km[:,0] >= xmin) & (pts_km[:,0] <= xmax) & (pts_km[:,1] >= ymin) & (pts_km[:,1] <= ymax)
    pts_near = pts_km[mask]

    plt.figure(figsize=(8.0,7.6))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    if len(pts_near):
        ax.scatter(pts_near[:,0], pts_near[:,1], s=6, alpha=0.35, color="gray", label="thermals (nearby)")

    # triangle sides
    tri_x = [A[0],B[0],C[0],A[0]]
    tri_y = [A[1],B[1],C[1],A[1]]
    ax.plot(tri_x, tri_y, 'k--', lw=1.4, label="Triangle")

    # flown track
    if path_xy.size:
        ax.plot(path_xy[:,0], path_xy[:,1], '-', lw=2.0, color="#cc0077", label="track")
        ax.plot(path_xy[:,0], path_xy[:,1], '^', ms=5, color="#cc0077", alpha=0.9)

    # marks
    ax.plot([A[0]],[A[1]], 'go', ms=7, label="A (Start)")
    ax.plot([B[0]],[B[1]], 'bo', ms=7, label="B")
    ax.plot([C[0]],[C[1]], 'ro', ms=7, label="C")

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title("Triangle — Thermals & Flown Track (no forced Z_MIN on FG)")
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    ax.grid(True, ls="--", alpha=0.35)

    # Info box
    d1 = res["dist_by_band"].get(1,0.0); d2 = res["dist_by_band"].get(2,0.0); d3 = res["dist_by_band"].get(3,0.0)
    vtrk = res["V_track_kmh"]; vtask = res.get("V_task_kmh", float("nan"))

    def _mmss(sec: float) -> str:
        if not np.isfinite(sec): return "nan"
        m, s = divmod(int(round(sec)), 60); return f"{m:02d}:{s:02d}"

    box_text = (
        f"{'Success' if res['success'] else 'Fail'}\n"
        f"Ttot={_mmss(res['time_total_s'])} | Tgl={_mmss(res['time_glide_s'])} | Tcl={_mmss(res['time_climb_s'])}\n"
        f"Track={res['track_dist_km']:.1f} km | Vtrack={vtrk:.1f} | Vtask={vtask:.1f}\n"
        f"Dist by band (km): 1={d1:.1f}, 2={d2:.1f}, 3={d3:.1f}"
    )
    ax.text(0.02, 0.02, box_text, transform=ax.transAxes,
            fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#444"))

    ax.legend(loc="upper right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# -----------------------------
# CSV / console helpers
# -----------------------------
def _fmt_time(sec: float) -> str:
    if sec is None or not np.isfinite(sec): return "nan"
    m, s = divmod(int(round(sec)), 60); return f"{m:02d}:{s:02d}"

def _fmt_float(x) -> str:
    return "nan" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{float(x):.3f}"

def write_summary_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w") as f:
            f.write("mc1,mc2,mc3,success,"
                    "time_total,time_glide,time_climb,"
                    "track_dist_km,Vtrack_kmh,Vtrack_int,Vtask_kmh,"
                    "dist_band1_km,dist_band2_km,dist_band3_km\n")

def write_summary_row(path: Path, mc1: float, mc2: float, mc3: float, res: Dict[str,Any]):
    T_tot  = res["time_total_s"]
    Vtrack = res["V_track_kmh"]
    Vtask  = res.get("V_task_kmh", float("nan"))
    d1 = res["dist_by_band"].get(1,0.0)
    d2 = res["dist_by_band"].get(2,0.0)
    d3 = res["dist_by_band"].get(3,0.0)
    with open(path, "a") as f:
        f.write(f"{int(mc1)},{int(mc2)},{int(mc3)},{int(bool(res['success']))},"
                f"{_fmt_time(res['time_total_s'])},{_fmt_time(res['time_glide_s'])},{_fmt_time(res['time_climb_s'])},"
                f"{_fmt_float(res['track_dist_km'])},{_fmt_float(Vtrack)},{int(round(Vtrack)) if np.isfinite(Vtrack) else 'nan'},{_fmt_float(Vtask)},"
                f"{_fmt_float(d1)},{_fmt_float(d2)},{_fmt_float(d3)}\n")

def print_header():
    print("MC1 MC2 MC3 | Success |  Tot(mm:ss) | V_track | V_task | Dist_by_band(km: 1/2/3)")
    print("-"*88)

def print_rule(i):
    if (i+1) % 9 == 0:
        print("="*88)
    elif (i+1) % 3 == 0:
        print("-"*88)

def print_row(m1,m2,m3,res):
    S = "T" if res["success"] else "F"
    vtrk = res["V_track_kmh"]; vtask = res.get("V_task_kmh", float("nan"))
    d1 = res["dist_by_band"].get(1,0.0); d2 = res["dist_by_band"].get(2,0.0); d3 = res["dist_by_band"].get(3,0.0)
    vstr  = f"{int(round(vtrk)):5d}" if np.isfinite(vtrk) else "  —  "
    vtstr = f"{int(round(vtask)):5d}" if np.isfinite(vtask) else "  —  "
    print(f"{int(m1):2d}  {int(m2):2d}  {int(m3):2d} |   {S}    |  {_fmt_time(res['time_total_s'])}  |"
          f"  {vstr} | {vtstr} | {d1:5.1f}/{d2:5.1f}/{d3:5.1f}")

# -----------------------------
# Single run (9 MC triples on one triangle)
# -----------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)

    # load field
    df_all = pd.concat([load_points_from_csv(IN_CSV), load_points_from_geojson(IN_GJ)], ignore_index=True)
    if df_all.empty:
        raise RuntimeError("No thermal points loaded. Check CSV/GeoJSON paths.")

    # project
    E_km, N_km, _ = enu_km(df_all)
    PTS_KM = np.column_stack([E_km, N_km])
    STRENGTH = pd.to_numeric(df_all.get("strength", pd.Series(np.nan)), errors="coerce").to_numpy()
    if not np.isfinite(STRENGTH).any():
        STRENGTH = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_summary_header(OUT_CSV_SUM)

    # One random triangle shared by all 9 runs (new triangle each script execution because RNG_SEED=None)
    A, B, C = pick_random_triangle(PTS_KM, rng)
    triangle = (A,B,C)

    triples = [(1,1,1),(1,2,1),(1,3,1),
               (2,1,1),(2,2,1),(2,3,1),
               (3,1,1),(3,2,1),(3,3,1)]

    print_header()
    total = len(triples)

    for i, (MC1, MC2, MC3) in enumerate(triples):
        print(f"[{i+1}/{total}] MC=({MC1},{MC2},{MC3}) … starting")
        t0 = time.time()
        try:
            res = simulate_triangle_continuous(
                PTS_KM, STRENGTH, triangle, MC1, MC2, MC3, rng=rng
            )
        except Exception as e:
            res = {
                "success": False,
                "time_total_s": float("nan"),
                "time_glide_s": float("nan"),
                "time_climb_s": float("nan"),
                "track_dist_km": float("nan"),
                "V_track_kmh": float("nan"),
                "V_task_kmh": float("nan"),
                "dist_by_band": {1: 0.0, 2: 0.0, 3: 0.0},
                "path_xy": np.empty((0,2)),
                "triangle": triangle
            }
            print(f"(iteration error: {e})")
        elapsed = time.time() - t0
        print(f"… done in {elapsed:.1f}s")
        print_row(MC1, MC2, MC3, res)
        print_rule(i)

        write_summary_row(OUT_CSV_SUM, MC1, MC2, MC3, res)
        try:
            plot_triangle(PTS_KM, triangle, res["path_xy"], res,
                          OUT_DIR / f"triangle_MC_{MC1}_{MC2}_{MC3}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")

# -----------------------------
# Batch run (many triangles; summary only)
# -----------------------------
def run_batch(n_runs: int = 1000):
    rng = np.random.default_rng(RNG_SEED)

    # load field
    df_all = pd.concat([load_points_from_csv(IN_CSV), load_points_from_geojson(IN_GJ)], ignore_index=True)
    if df_all.empty:
        raise RuntimeError("No thermal points loaded. Check CSV/GeoJSON paths.")
    E_km, N_km, _ = enu_km(df_all)
    PTS_KM = np.column_stack([E_km, N_km])
    STRENGTH = pd.to_numeric(df_all.get("strength", pd.Series(np.nan)), errors="coerce").to_numpy()
    if not np.isfinite(STRENGTH).any():
        STRENGTH = None

    triples = [(1,1,1),(1,2,1),(1,3,1),
               (2,1,1),(2,2,1),(2,3,1),
               (3,1,1),(3,2,1),(3,3,1)]

    results = {tr: {"success": 0, "fail": 0, "speeds": []} for tr in triples}
    rows = []

    print(f"\nRunning batch of {n_runs} iterations…")
    for run in range(n_runs):
        # Pick a NEW random triangle per iteration
        A,B,C = pick_random_triangle(PTS_KM, rng)
        tri = (A,B,C)

        for (MC1,MC2,MC3) in triples:
            res = simulate_triangle_continuous(PTS_KM, STRENGTH, tri, MC1, MC2, MC3, rng=rng)
            ok = bool(res["success"])
            Vtask = res.get("V_task_kmh", float("nan"))

            if ok:
                results[(MC1,MC2,MC3)]["success"] += 1
                if np.isfinite(Vtask):
                    results[(MC1,MC2,MC3)]["speeds"].append(Vtask)
            else:
                results[(MC1,MC2,MC3)]["fail"] += 1

            rows.append([MC1,MC2,MC3, run, int(ok), Vtask])

    # Save CSV
    BATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["MC1","MC2","MC3","run","success","Vtask_kmh"]).to_csv(BATCH_CSV, index=False)

    # Console summary
    print("\nMC1 | MC2 | MC3 | Success/Total | Success % | Avg V_task (km/h)")
    print("-"*75)
    for (MC1,MC2,MC3), counts in results.items():
        total = counts["success"] + counts["fail"]
        rate = 100.0 * counts["success"]/total if total>0 else 0.0
        avg_speed = np.mean(counts["speeds"]) if counts["speeds"] else float("nan")
        avg_str = f"{avg_speed:10.1f}" if np.isfinite(avg_speed) else "        —"
        print(f"{MC1:4d} | {MC2:3d} | {MC3:3d} | {counts['success']:6d}/{total:<6d} | {rate:7.2f}% | {avg_str}")
    print("="*75)
    print(f"Wrote: {BATCH_CSV}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # Choose one:
    #   1) Single-run (plots + per-run CSV):
    main()
    #   2) Batch-run (summary console + batch CSV):
    # run_batch(1000)