
import math, time
from typing import Optional

import pulp
from math import erf, sqrt


# ────────────────────────────────────────────────────────────────────────────
#  Exact Gaussian integral over a rectangle – kernel for Eq.(8)
# ────────────────────────────────────────────────────────────────────────────
def rect_gauss(x0, x1, y0, y1, xc, yc, sigma=0.5):
    """∬ gσ(x,y) dx dy  on  [x0,x1]×[y0,y1], centre (xc,yc)."""
    a = erf((x1 - xc) / (sqrt(2) * sigma)) - erf((x0 - xc) / (sqrt(2) * sigma))
    b = erf((y1 - yc) / (sqrt(2) * sigma)) - erf((y0 - yc) / (sqrt(2) * sigma))
    return 0.25 * a * b


# ────────────────────────────────────────────────────────────────────────────
#  Core MILP builder (variable-layer version)
# ────────────────────────────────────────────────────────────────────────────
def build_and_solve_milp_var_layers(
    *,
    obj_bounds: dict,               # {id:(xmin,xmax,ymin,ymax)}
    laplace_pairs: list,            # [(small,big), …]
    proximity_pairs: list,          # [(a,b), …]
    tile_size: float = 0.66,
    n_layers: int = 20,             # generous upper bound
    sigma: float = 0.5,
    c_dry: float = 512.0,
    c_layer: Optional[float] = None,   # penalty per used layer
    solver: pulp.LpSolver = pulp.PULP_CBC_CMD(msg=False),
):
    """Return (assignment, layer_scores, used_layers, total_score)."""

    # Row-printing weight (paper: 24·ℓ_T)
    c_print = 24.0 * tile_size
    if c_layer is None:
        c_layer = c_print           # default: ~ cost of one printed row

    # ── derive grid size ────────────────────────────────────────────────
    xmax = max(b[1] for b in obj_bounds.values())
    ymax = max(b[3] for b in obj_bounds.values())
    n_cols, n_rows = math.ceil(xmax / tile_size), math.ceil(ymax / tile_size)
    cols, rows, layers = range(n_cols), range(n_rows), range(n_layers)
    objects = list(obj_bounds)
    BIG_M, BIG_V = len(objects), 1e9

    # ── pre-compute row membership & Gaussian weights ───────────────────
    pmi, Δs = {}, {}
    for i, (xmin, xmax, ymin, ymax) in obj_bounds.items():
        row_lo, row_hi = int(ymin // tile_size), int(ymax // tile_size)
        col_lo, col_hi = int(xmin // tile_size), int(xmax // tile_size)
        for m in range(row_lo, row_hi + 1):
            pmi[(m, i)] = 1
            yc = (m + 0.5) * tile_size
            for n in range(col_lo, col_hi + 1):
                xc = (n + 0.5) * tile_size
                Δs[(i, m, n)] = rect_gauss(xmin, xmax, ymin, ymax, xc, yc, sigma)

    # ── build MILP model ────────────────────────────────────────────────
    mdl = pulp.LpProblem("Inkjet_VarLayer_MILP", pulp.LpMinimize)

    # Binary variables
    q = pulp.LpVariable.dicts("q", (objects, layers), 0, 1, "Binary")  # Eq.(2)
    y = pulp.LpVariable.dicts("y", layers, 0, 1, "Binary")             # layer flag
    r = pulp.LpVariable.dicts("r", (rows, layers), 0, 1, "Binary")     # Eq.(5)

    # Continuous / integer
    vPrint = pulp.LpVariable.dicts("vPrint", (rows, layers), lowBound=0, cat="Integer")
    vDry   = pulp.LpVariable.dicts("vDry",   (rows, cols, layers))
    vRow   = pulp.LpVariable.dicts("vRow",   (rows, layers))
    vLayer = pulp.LpVariable.dicts("vLayer", layers)

    # (2) each object exactly once
    for i in objects:
        mdl += pulp.lpSum(q[i][l] for l in layers) == 1, f"Eq2_once_{i}"

    # link q → y
    for i in objects:
        for l in layers:
            mdl += q[i][l] <= y[l], f"link_q_y_{i}_{l}"

    # (3) Laplace precedence
    for s, b in laplace_pairs:
        mdl += (
            pulp.lpSum((l + 1) * q[s][l] for l in layers) + 1
            <= pulp.lpSum((l + 1) * q[b][l] for l in layers)
        ), f"Eq3_{s}_{b}"

    # (4) proximity exclusion
    for a, b in proximity_pairs:
        for l in layers:
            mdl += q[a][l] + q[b][l] <= 1, f"Eq4_{a}_{b}_{l}"

    # (5) row indicator (single Big-M)
    for m in rows:
        for l in layers:
            cover = pulp.lpSum(pmi.get((m, i), 0) * q[i][l] for i in objects)
            mdl += cover <= BIG_M * r[m][l], f"Eq5_{m}_{l}"

    # (6) first row printing score
    for l in layers:
        mdl += vPrint[0][l] == r[0][l], f"Eq6_init_{l}"

    # (7) recursion
    for l in layers:
        for m in range(1, n_rows):
            mdl += vPrint[m][l] == vPrint[m - 1][l] + r[m][l], f"Eq7_{m}_{l}"

    # (8) Gaussian drying load
    for l in layers:
        for m in rows:
            for n in cols:
                mdl += (
                    vDry[m][n][l]
                    == pulp.lpSum(Δs.get((i, m, n), 0) * q[i][l] for i in objects)
                ), f"Eq8_{m}_{n}_{l}"

    # (9) manufacturing score per row
    for l in layers:
        for m in rows:
            for n in cols:
                mdl += (
                    vRow[m][l]
                    >= c_print * vPrint[m][l] + c_dry * vDry[m][n][l]
                ), f"Eq9_{m}_{n}_{l}"

    # (10) layer score ≥ every row score, and gate via y
    for l in layers:
        for m in rows:
            mdl += vLayer[l] >= vRow[m][l], f"Eq10_{m}_{l}"
        mdl += vLayer[l] <= BIG_V * y[l], f"gate_{l}"

    # Objective
    mdl += (
        pulp.lpSum(vLayer[l] for l in layers)
        + c_layer * pulp.lpSum(y[l] for l in layers)
    ), "Obj"

    # Solve
    status = mdl.solve(solver)
    if status != pulp.LpStatusOptimal:
        raise RuntimeError("MILP infeasible or time-limit reached")

    # Extract solution
    assign = {i: next(l for l in layers if pulp.value(q[i][l]) > 0.5) for i in objects}
    layer_scores = [pulp.value(vLayer[l]) or 0.0 for l in layers]
    used_layers  = [l for l in layers if pulp.value(y[l]) > 0.5]
    total_score  = sum(layer_scores) + c_layer * len(used_layers)
    return assign, layer_scores, used_layers, total_score


# ────────────────────────────────────────────────────────────────────────────
#  Tiny helper to run & time a single instance
# ────────────────────────────────────────────────────────────────────────────
def run_case(name, bounds, *, tile, laplace=None, prox=None,
             layers=20, clayer=None, solver=None):
    t0 = time.perf_counter()
    assign, scores, used, total = build_and_solve_milp_var_layers(
        obj_bounds=bounds,
        laplace_pairs=laplace or [],
        proximity_pairs=prox or [],
        tile_size=tile,
        n_layers=layers,
        c_layer=clayer,
        solver=solver or pulp.PULP_CBC_CMD(msg=False),
    )
    dt = time.perf_counter() - t0
    print(f"{name:<12}  | layers={used}  | cycle={sum(scores):7.1f}"
          f" | score={total:7.1f} | {dt:5.2f}s")


# ────────────────────────────────────────────────────────────────────────────
#  Demo sweep
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Geometry definitions
    toy_bounds = {0: (0, 2, 0, 2), 1: (0, 4, 2, 4), 2: (2, 4, 4, 6)}
    heater_bounds = {
        0: (0, 30, 0, 5),
        1: (0, 30, 5.5, 10.5),
        2: (0, 30, 11, 16),
        3: (0, 30, 16.5, 21.5),
    }
    dmf_bounds = {
        0: (0, 74.9, 0, 3.3),
        1: (0, 74.9, 3.9, 7.2),
        2: (0, 74.9, 7.8, 11.1),
    }

    # Penalty sweep to illustrate effect
    penalties = [0, 1, 5, 10, 25]   # multiples of c_print for toy (tile=1 → c_print=24)

    print("── Toy geometry (tile=1 mm) ──────────────────────────────────────")
    for k in penalties:
        run_case(f"c={k:>2}", toy_bounds, tile=1.0, layers=10,
                 clayer=k*24.0)

    print("\n── Micro-heater (≈10 k tiles) ──────────────────────────────────")
    heater_tile = math.sqrt(30 * 22 / 1e4)   # ≈0.26 mm
    cprint_heater = 24.0 * heater_tile
    for k in penalties:
        run_case(f"c={k:>2}", heater_bounds, tile=heater_tile, layers=20,
                 clayer=k*cprint_heater)

    print("\n── µ-fluidics (tile=0.66 mm) ──────────────────────────────────")
    cprint_dmf = 24.0 * 0.66
    for k in penalties:
        run_case(f"c={k:>2}", dmf_bounds, tile=0.66, layers=20,
                 clayer=k*cprint_dmf)
