

import math
from collections import defaultdict
from typing import Optional

from docplex.mp.model import Model

# Gaussian drying kernel  –Eq.(8) constant ∆s
def gaussian_pdf(dist_mm: float, sigma: float = 0.5) -> float:
    """fσ(d) = (1/σ√(2π)) · exp(−d²/(2σ²))   (paper Eq.(7) kernel)"""
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))
            * math.exp(-dist_mm ** 2 / (2.0 * sigma ** 2)))

def build_and_solve_milp_short(*,  # force keyword args
                               object_bounds_mm: dict,  # {id:(xmin,xmax,ymin,ymax)}
                               laplace_pairs: list,  # [(small,big), …]
                               proximity_pairs: list,  # [(a,b), …]
                               tile_size_mm: float,
                               max_layers: int = 10,
                               c_print: Optional[float] = None,
                               c_dry: float = 512.0):
    # 1 ── constants & helper ranges
    c_print = 24.0 * tile_size_mm if c_print is None else c_print
    sigma = 0.5
    objs = list(object_bounds_mm)
    L   = range(max_layers)

    # tiling dimensions
    max_x = max(b[1] for b in object_bounds_mm.values())
    max_y = max(b[3] for b in object_bounds_mm.values())
    NX = math.ceil(max_x / tile_size_mm)
    NY = math.ceil(max_y / tile_size_mm)
    X = range(NX)
    Y = range(NY)

    # 2 ── pre-compute helpers  p_mi  and  Δs_{oi→Tm,n}
    # pmi = 1 if object i intersects row m
    pmi = defaultdict(int)              # key (m,i) → 0/1
    Δs  = defaultdict(float)            # key (i,m,n) → gaussian weight

    for i, (xmin,xmax,ymin,ymax) in object_bounds_mm.items():
        cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
        for m in range(int(ymin//tile_size_mm), int(ymax//tile_size_mm)+1):
            pmi[(m,i)] = 1
            for n in range(int(xmin//tile_size_mm), int(xmax//tile_size_mm)+1):
                tcx = (n+0.5)*tile_size_mm
                tcy = (m+0.5)*tile_size_mm
                Δs[(i,m,n)] = gaussian_pdf(math.hypot(tcx-cx, tcy-cy), sigma)

    BIG_M = len(objs)  # “sufficiently large” for Eq.(5)

    # 3 ── model & decision variables
    mdl = Model(name="MfgCycleTime_Short", log_output=False)

    # Eq.(2)  q_{i,j}
    q = mdl.binary_var_dict([(i,j) for i in objs for j in L], name="q")

    # Eq.(5)  q_row_{m,j}
    q_row = mdl.binary_var_dict([(m,j) for m in Y for j in L], name="qRow")

    # Eq.(6),(7)  v_print_{m,j}
    v_print = mdl.integer_var_dict([(m,j) for m in Y for j in L],
                                   lb=0, name="vPrint")

    # Eq.(8)  v_dry_{m,n,j}
    v_dry = mdl.continuous_var_dict([(m,n,j) for m in Y for n in X for j in L],
                                    lb=0, name="vDry")

    # Eq.(9)  v_row_{m,j}
    v_row = mdl.continuous_var_dict([(m,j) for m in Y for j in L],
                                    lb=0, name="vRow")

    # Eq.(10) v_layer_j
    v_layer = mdl.continuous_var_dict(L, lb=0, name="vLayer")

    # 4 ── constraints -
    # (2) assignment – each object printed exactly once
    for i in objs:
        mdl.add_constraint(mdl.sum(q[i,j] for j in L) == 1,
                           ctname=f"Eq2_once_{i}")

    # (3) Laplace precedence
    for small,big in laplace_pairs:
        mdl.add_constraint(
            mdl.sum((j+1)*q[small,j] for j in L) + 1
            <= mdl.sum((j+1)*q[big  ,j] for j in L),
            ctname=f"Eq3_Laplace_{small}_{big}")

    # (4) proximity exclusion
    for a,b in proximity_pairs:
        for j in L:
            mdl.add_constraint(q[a,j] + q[b,j] <= 1,
                               ctname=f"Eq4_prox_{a}_{b}_{j}")

    # (5) row-printing indicator  BIG-M
    for j in L:
        for m in Y:
            mdl.add_constraint(
                q_row[m,j] * BIG_M >=
                mdl.sum(pmi.get((m,i),0) * q[i,j] for i in objs),
                ctname=f"Eq5_rowFlag_{m}_{j}")

    # (6) first-row printing score
    for j in L:
        mdl.add_constraint(v_print[0,j] == q_row[0,j],
                           ctname=f"Eq6_init_{j}")

    # (7) printing-score recursion
    for j in L:
        for m in range(1, NY):
            mdl.add_constraint(
                v_print[m,j] == v_print[m-1,j] + q_row[m,j],
                ctname=f"Eq7_rec_{m}_{j}")

    # (8) drying score per tile
    for j in L:
        for m in Y:
            for n in X:
                mdl.add_constraint(
                    v_dry[m,n,j] ==
                    mdl.sum(Δs.get((i,m,n),0.0) * q[i,j] for i in objs),
                    ctname=f"Eq8_dry_{m}_{n}_{j}")

    # (9) manufacturing score per row
    for j in L:
        for m in Y:
            for n in X:
                mdl.add_constraint(
                    v_row[m,j] >=
                    c_print * v_print[m,j] + c_dry * v_dry[m,n,j],
                    ctname=f"Eq9_rowScore_{m}_{n}_{j}")

    for j in L:
        for m in Y:
            mdl.add_constraint(v_layer[j] >= v_row[m,j],
                               ctname=f"Eq10_layerMax_{m}_{j}")

    # 5 ── objective  minimise Σ v_layer_j  (Eq.(1))
    mdl.minimize(mdl.sum(v_layer[j] for j in L))

    # 6 ── solve
    mdl.solve()

    assignment   = {i: next(j for j in L if q[i,j].solution_value > .5)
                    for i in objs}
    layer_scores = [v_layer[j].solution_value or 0.0 for j in L]
    used_layers  = [j for j in L if layer_scores[j] > 1e-6]
    total_score  = sum(layer_scores)

    return assignment, layer_scores, used_layers, total_score
if __name__ == "__main__":

    toy_bounds = {0: (0, 2, 0, 2),
                  1: (0, 4, 2, 4),
                  2: (2, 4, 4, 6)}
    toy_out = build_and_solve_milp_short(
        object_bounds_mm=toy_bounds,
        laplace_pairs=[(0, 1)],
        proximity_pairs=[(1, 2)],
        tile_size_mm=1.0,
        max_layers=4)
    print("\nToy example  →", toy_out[:4])

    heater_bounds = {0: (0, 30, 0, 5), 1: (0, 30, 5.5, 10.5),
                     2: (0, 30, 11, 16), 3: (0, 30, 16.5, 21.5)}
    heater_out = build_and_solve_milp_short(
        object_bounds_mm=heater_bounds,
        laplace_pairs=[],
        proximity_pairs=[],
        tile_size_mm=math.sqrt(30*22/1e4),
        max_layers=6)
    print("Micro-heater  →", heater_out[:4])


    dmf_bounds = {0: (0, 74.9, 0, 3.3),
                  1: (0, 74.9, 3.9, 7.2),
                  2: (0, 74.9, 7.8, 11.1)}
    dmf_out = build_and_solve_milp_short(
        object_bounds_mm=dmf_bounds,
        laplace_pairs=[],
        proximity_pairs=[],
        tile_size_mm=0.66,
        max_layers=8)
    print("Dig.-µfluidics →", dmf_out[:4])
