
import math
import pulp
from math import erf, sqrt

# -------- exact Gaussian integral on a rectangle (Eq. 8 kernel) ----------
def rect_gauss(x0, x1, y0, y1, xc, yc, sigma):
    a = erf((x1 - xc) / (sqrt(2)*sigma)) - erf((x0 - xc) / (sqrt(2)*sigma))
    b = erf((y1 - yc) / (sqrt(2)*sigma)) - erf((y0 - yc) / (sqrt(2)*sigma))
    return 0.25 * a * b          # ¼ = ½·½  (normalisation)

# -------- MILP builder – reduced constraint set -------------------------
def build_and_solve_milp_short(
        *,                               # keyword-only API
        obj_bounds: dict,                # {id:(xmin,xmax,ymin,ymax)}
        laplace_pairs: list,             # precedence edges (small,big)
        proximity_pairs: list,           # conflict edges   (a,b)
        tile_size: float = 0.66,
        n_layers:   int   = 8,
        sigma:      float = 0.5,
        c_dry:      float = 512.0):

    c_print = 24.0 * tile_size           # Eq.(9) constant

    # ---- grid dimensions ------------------------------------------------
    xmax = max(b[1] for b in obj_bounds.values())
    ymax = max(b[3] for b in obj_bounds.values())
    n_cols = math.ceil(xmax / tile_size)
    n_rows = math.ceil(ymax / tile_size)
    cols, rows, layers = range(n_cols), range(n_rows), range(n_layers)
    objects = list(obj_bounds)
    BIG_M   = len(objects)

    # ---- pre-compute helpers  pmi  &  Δs_{i,m,n} ------------------------
    pmi, Δs = {}, {}
    for i, (xmin,xmax,ymin,ymax) in obj_bounds.items():
        row_lo, row_hi = int(ymin//tile_size), int(ymax//tile_size)
        col_lo, col_hi = int(xmin//tile_size), int(xmax//tile_size)
        for m in range(row_lo, row_hi+1):
            pmi[(m,i)] = 1
            yc = (m+0.5)*tile_size
            for n in range(col_lo, col_hi+1):
                xc = (n+0.5)*tile_size
                Δs[(i,m,n)] = rect_gauss(xmin,xmax,ymin,ymax, xc,yc, sigma)

    # ---- model & decision variables ------------------------------------
    mdl = pulp.LpProblem("Inkjet_Short_MILP", pulp.LpMinimize)

    q       = pulp.LpVariable.dicts("q",      (objects,layers), 0, 1, 'Binary')   # Eq.(2)
    r_row   = pulp.LpVariable.dicts("rRow",   (rows,layers),    0, 1, 'Binary')   # Eq.(5)
    v_print = pulp.LpVariable.dicts("vPrint", (rows,layers),    lowBound=0, cat='Integer')
    v_dry   = pulp.LpVariable.dicts("vDry",   (rows,cols,layers))
    v_row   = pulp.LpVariable.dicts("vRow",   (rows,layers))
    v_layer = pulp.LpVariable.dicts("vLayer", layers)

    # (2) each object printed exactly once
    for i in objects:
        mdl += pulp.lpSum(q[i][j] for j in layers) == 1

    # (3) Laplace precedence
    for small,big in laplace_pairs:
        mdl += (pulp.lpSum((j+1)*q[small][j] for j in layers) + 1
                <= pulp.lpSum((j+1)*q[big][j]   for j in layers))

    # (4) proximity exclusion
    for a,b in proximity_pairs:
        for j in layers:
            mdl += q[a][j] + q[b][j] <= 1

    # (5) row-printing indicator (single Big-M)
    for m in rows:
        for j in layers:
            cover = pulp.lpSum(pmi.get((m,i),0) * q[i][j] for i in objects)
            mdl += cover <= BIG_M * r_row[m][j]

    # (6) first row printing score
    for j in layers:
        mdl += v_print[0][j] == r_row[0][j]

    # (7) recursion of printed rows
    for j in layers:
        for m in range(1, n_rows):
            mdl += v_print[m][j] == v_print[m-1][j] + r_row[m][j]

    # (8) Gaussian drying load
    for j in layers:
        for m in rows:
            for n in cols:
                mdl += v_dry[m][n][j] == pulp.lpSum(Δs.get((i,m,n),0) * q[i][j]
                                                    for i in objects)

    # (9) manufacturing score per row
    for j in layers:
        for m in rows:
            for n in cols:
                mdl += v_row[m][j] >= c_print * v_print[m][j] + c_dry * v_dry[m][n][j]

    # (10) layer score = max row score
    for j in layers:
        for m in rows:
            mdl += v_layer[j] >= v_row[m][j]

    # Objective (1) – minimise Σ v_layer_j
    mdl += pulp.lpSum(v_layer[j] for j in layers)

    # ---- solve ----------------------------------------------------------
    if mdl.solve(pulp.PULP_CBC_CMD(msg=False)) != 1:
        raise RuntimeError("MILP infeasible")

    # ---- compact result dictionaries -----------------------------------
    assign = {i: next(j for j in layers if pulp.value(q[i][j]) > 0.5)
              for i in objects}
    layer_scores = [pulp.value(v_layer[j]) or 0.0 for j in layers]
    used_layers  = [j for j,s in enumerate(layer_scores) if s > 1e-6]
    return assign, layer_scores, used_layers, sum(layer_scores)

# ---------------- demonstration runs (same three cases) -----------------
if __name__ == "__main__":
    # 1) toy 3-object
    toy_bounds = {0: (0,2,0,2), 1: (0,4,2,4), 2: (2,4,4,6)}
    print("\nToy →", build_and_solve_milp_short(
        obj_bounds=toy_bounds,
        laplace_pairs=[(0,1)],
        proximity_pairs=[(1,2)],
        tile_size=1.0,
        n_layers=4)[:4])

    # 2) micro-heater (stub)
    heater_bounds = {0:(0,30,0,5), 1:(0,30,5.5,10.5),
                     2:(0,30,11,16), 3:(0,30,16.5,21.5)}
    print("Heater →", build_and_solve_milp_short(
        obj_bounds=heater_bounds,
        laplace_pairs=[],
        proximity_pairs=[],
        tile_size=math.sqrt(30*22/1e4),  # ~0.26 mm
        n_layers=6)[:4])

    # 3) digital micro-fluidics (stub)
    dmf_bounds = {0:(0,74.9,0,3.3), 1:(0,74.9,3.9,7.2), 2:(0,74.9,7.8,11.1)}
    print("DMF →", build_and_solve_milp_short(
        obj_bounds=dmf_bounds,
        laplace_pairs=[],
        proximity_pairs=[],
        tile_size=0.66,
        n_layers=8)[:4])
