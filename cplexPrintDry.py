
# ───────────────────────────── Imports ────────────────────────────────────────
import math  # √, exp, hypot, π              » std‑math
from docplex.mp.model import Model  # DOcplex modelling front‑end   » CPLEX API


# ───────────────── Gaussian drying kernel  g(d)  –Eqs.16–19 ────────────────
def gaussian_pdf(distance_mm: float, sigma_mm: float = 0.5) -> float:
    """g(d)=1/(σ√(2π))·exp(−d²/(2σ²))» Eq.(17)"""
    return (1.0 / (sigma_mm * math.sqrt(2.0 * math.pi))  # 1/σ√2π
            * math.exp(-distance_mm ** 2 / (2.0 * sigma_mm ** 2)))  # ·exp(−d²/2σ²)


# ─────────────────────── Build & solve MILP (Eqs.22‑29) ─────────────────────
def build_and_solve_milp(
        *,  # force keyword args
        object_bounds_mm: dict,  # {id:(xmin,xmax,ymin,ymax)}
        laplace_order_pairs: list,  # precedence edges  (small,big)
        proximity_exclusion_pairs: list,  # proximity edges   (id_a,id_b)
        tile_size_mm: float,  # l_T
        max_num_layers: int = 10  # NL (upper bound)
):
    # 1 ── Fixed coefficients from SectionVI (“Experimental setup”) ──────────
    sigma_mm = 0.5  # σ=0.5mm               » Eq.(17)
    printing_cost_per_row = 24.0 * tile_size_mm  # c_print=24·l_T         » Eq.(28)
    drying_cost_per_tile = 512.0  # c_dry=512             » Eq.(28)

    # 2 ── Derive tiling grid sizes  n_rows × n_cols  (index ranges for Eq.22)
    max_x = max(bb[1] for bb in object_bounds_mm.values())  # right‑mostx
    max_y = max(bb[3] for bb in object_bounds_mm.values())  # top‑mosty
    n_cols = math.ceil(max_x / tile_size_mm)  # n_x=⌈W/l_T⌉
    n_rows = math.ceil(max_y / tile_size_mm)  # n_y=⌈H/l_T⌉
    col_ids, row_ids, layer_ids = range(n_cols), range(n_rows), range(max_num_layers)
    object_ids = list(object_bounds_mm.keys())  # i∈objects
    big_M = len(object_ids)  # ≥ objects/row  (Eq.25)

    # 3 ── Pre‑compute  rows(o)  &  Gaussian weights s_{i,m,n}  (Eq.27) ─────
    rows_per_object = {o: set() for o in object_ids}  # helper for Eq.25
    weight_s = {}  # (i,m,n)→s_{i,m,n}
    for obj, (x_min, x_max, y_min, y_max) in object_bounds_mm.items():
        obj_cx, obj_cy = (x_min + x_max) / 2, (y_min + y_max) / 2  # centre of object
        for row in range(int(y_min // tile_size_mm), int(y_max // tile_size_mm) + 1):
            rows_per_object[obj].add(row)  # rows(o)
            for col in range(int(x_min // tile_size_mm), int(x_max // tile_size_mm) + 1):
                tile_cx = (col + 0.5) * tile_size_mm  # tile centrex
                tile_cy = (row + 0.5) * tile_size_mm  # tile centrey
                dist = math.hypot(tile_cx - obj_cx, tile_cy - obj_cy)  # d in Eq.17
                weight_s[(obj, row, col)] = gaussian_pdf(dist, sigma_mm)  # s_{i,m,n}

    # 4 ── Create CPLEX model & declare variables (symbols of Eqs.22‑29) ────
    mdl = Model(name="InkjetPrintScheduling", log_output=False)  # silent solve

    # q_{i,ℓ}∈{0,1}: “objecti printed on layerℓ”     » Eq.(22)
    var_object_on_layer = mdl.binary_var_dict(
        [(o, l) for o in object_ids for l in layer_ids], name="q")

    # r_{m,ℓ}∈{0,1}: “rowm has any printing on layerℓ”  » Eq.(25)
    var_row_active = mdl.binary_var_dict(
        [(m, l) for m in row_ids for l in layer_ids], name="r")

    # y_ℓ∈{0,1}: helper flag –layer ℓ used by any object           (not in paper)
    var_layer_used = mdl.binary_var_dict(layer_ids, name="y")

    # v_print_{m,ℓ}≥0: cumulative printed rows up tom     » Eq.(26)
    var_cumulative_rows = mdl.continuous_var_dict(
        [(m, l) for m in row_ids for l in layer_ids], name="vPrint")

    # v_dry_{m,n,ℓ}≥0: Gaussian drying load on tile (m,n)  » Eq.(27)
    var_tile_drying_load = mdl.continuous_var_dict(
        [(m, n, l) for m in row_ids for n in col_ids for l in layer_ids], name="vDry")

    # v_row_{m,ℓ}≥0: manufacturing score for rowm         » Eq.(28)
    var_row_score = mdl.continuous_var_dict(
        [(m, l) for m in row_ids for l in layer_ids], name="vRow")

    # v_layer_ℓ≥0: layer score = maxrowscores            » Eq.(29)
    var_layer_score = mdl.continuous_var_dict(layer_ids, name="vLayer")

    # 5 ── Constraints (each bullet = paper equation) ─────────────────────────

    # • Eq.22Σ_ℓq_{i,ℓ}=1   (each object exactly once)
    for obj in object_ids:
        mdl.add_constraint(mdl.sum(var_object_on_layer[obj, l] for l in layer_ids) == 1,
                           ctname=f"Eq22_once_{obj}")

    # • helper: q_{i,ℓ}≤y_ℓ   (activate layer flag if any object uses it)
    for l in layer_ids:
        for obj in object_ids:
            mdl.add_constraint(var_object_on_layer[obj, l] <= var_layer_used[l],
                               ctname=f"gate_q→y_{obj}_{l}")

    # • Eq.23Σ_ℓ (ℓ+1)·q_{small,ℓ} +1≤Σ_ℓ (ℓ+1)·q_{large,ℓ}
    for small, large in laplace_order_pairs:
        mdl.add_constraint(
            mdl.sum((l + 1) * var_object_on_layer[small, l] for l in layer_ids) + 1
            <= mdl.sum((l + 1) * var_object_on_layer[large, l] for l in layer_ids),
            ctname=f"Eq23_Laplace_{small}->{large}")

    # • Eq.24q_{a,ℓ}+q_{b,ℓ}≤1   (objects a & b cannot share layer)
    for a, b in proximity_exclusion_pairs:
        for l in layer_ids:
            mdl.add_constraint(var_object_on_layer[a, l] + var_object_on_layer[b, l] <= 1,
                               ctname=f"Eq24_prox_{a}_{b}_{l}")

    # • Eq.25row indicator definition
    for m in row_ids:
        for l in layer_ids:
            cover_m_l = mdl.sum(var_object_on_layer[o, l]  # Σ_{i∈rows(m)} q_{i,l}
                                for o in object_ids if m in rows_per_object[o])
            mdl.add_constraint(cover_m_l <= big_M * var_row_active[m, l],  # cover ⇒ r=1
                               ctname=f"Eq25a_rowFlagUb_{m}_{l}")
            mdl.add_constraint(var_row_active[m, l] <= cover_m_l,  # no cover ⇒ r=0
                               ctname=f"Eq25b_rowFlagLb_{m}_{l}")

    # • Eq.26v_print recursion: v_print_{0,ℓ}=r_{0,ℓ};  v_print_{m,ℓ}=v_print_{m-1,ℓ}+r_{m,ℓ}
    for l in layer_ids:
        mdl.add_constraint(var_cumulative_rows[0, l] == var_row_active[0, l],
                           ctname=f"Eq26_init_{l}")
        for m in range(1, n_rows):
            mdl.add_constraint(
                var_cumulative_rows[m, l] ==
                var_cumulative_rows[m - 1, l] + var_row_active[m, l],
                ctname=f"Eq26_rec_{m}_{l}")

    # • Eq.27v_dry_{m,n,ℓ}=Σ_is_{i,m,n}·q_{i,ℓ}
    for l in layer_ids:
        for m in row_ids:
            for n in col_ids:
                mdl.add_constraint(
                    var_tile_drying_load[m, n, l] ==
                    mdl.sum(weight_s.get((o, m, n), 0.0) * var_object_on_layer[o, l]
                            for o in object_ids),
                    ctname=f"Eq27_dry_{m}_{n}_{l}")

    # • Eq.28v_row_{m,ℓ}≥c_print·v_print_{m,ℓ}+c_dry·v_dry_{m,n,ℓ}
    for l in layer_ids:
        for m in row_ids:
            for n in col_ids:
                mdl.add_constraint(
                    var_row_score[m, l] >=
                    printing_cost_per_row * var_cumulative_rows[m, l] +
                    drying_cost_per_tile * var_tile_drying_load[m, n, l],
                    ctname=f"Eq28_rowScore_{m}_{n}_{l}")

    # • Eq.29v_layer_ℓ≥v_row_{m,ℓ}  ∀m   and gate unused layers
    huge_M = 1e9
    for l in layer_ids:
        for m in row_ids:
            mdl.add_constraint(var_layer_score[l] >= var_row_score[m, l],
                               ctname=f"Eq29_max_{m}_{l}")
        mdl.add_constraint(var_layer_score[l] <= huge_M * var_layer_used[l],
                           ctname=f"gate_layerUsed_{l}")

    # 6 ── Objective  minΣ_ℓv_layer_ℓ   (Eq.29 objective) ────────────────
    mdl.minimize(mdl.sum(var_layer_score[l] for l in layer_ids))

    # 7 ── Solve MILP ─────────────────────────────────────────────────────────
    mdl.solve()  # CBC/CPLEX called here

    # 8 ── Extract solution scalars for Python dict/tuple return ---------------
    assignment = {o: next(l for l in layer_ids
                          if var_object_on_layer[o, l].solution_value > 0.5)
                  for o in object_ids}
    layer_scores = [var_layer_score[l].solution_value or 0.0 for l in layer_ids]
    used_layers = [l for l in layer_ids if var_layer_used[l].solution_value > 0.5]
    total_score = sum(layer_scores)

    # Return tuple matches the original PuLP API for downstream compatibility
    return (assignment, layer_scores, used_layers,
            total_score, printing_cost_per_row, drying_cost_per_tile)


# ──────────────────────── Demonstration runs (3 cases) ────────────────────────
if __name__ == "__main__":
    # Case1 – toy 3‑object (Fig.4)
    toy_bounds = {0: (0, 2, 0, 2), 1: (0, 4, 2, 4), 2: (2, 4, 4, 6)}
    toy_output = build_and_solve_milp(
        object_bounds_mm=toy_bounds,
        laplace_order_pairs=[(0, 1)],
        proximity_exclusion_pairs=[(1, 2)],
        tile_size_mm=1.0, max_num_layers=4)
    print("\nToy example →", toy_output[:4])  # assignment, scores, layers, total

    # Case2 – micro‑heater (stub geometry)
    heater_bounds = {0: (0, 30, 0, 5), 1: (0, 30, 5.5, 10.5),
                     2: (0, 30, 11, 16), 3: (0, 30, 16.5, 21.5)}
    heater_output = build_and_solve_milp(
        object_bounds_mm=heater_bounds,
        laplace_order_pairs=[], proximity_exclusion_pairs=[],
        tile_size_mm=math.sqrt(30 * 22 / 1e4),  # ≈0.26mm to hit 10k tiles
        max_num_layers=6)
    print("Micro‑heater →", heater_output[:4])

    # Case3 – digital‑microfluidics (stub geometry)
    dmf_bounds = {0: (0, 74.9, 0, 3.3),
                  1: (0, 74.9, 3.9, 7.2),
                  2: (0, 74.9, 7.8, 11.1)}
    dmf_output = build_and_solve_milp(
        object_bounds_mm=dmf_bounds,
        laplace_order_pairs=[], proximity_exclusion_pairs=[],
        tile_size_mm=0.66, max_num_layers=8)
    print("Dig.µ‑fluidics →", dmf_output[:4])
"""
equation numbers in the paper:

Code Eq.22 → Paper Eq.(2)(assignment: each object exactly once)
Code Eq.23 → Paper Eq.(3)(Laplace precedence: small before large)
Code Eq.24 → Paper Eq.(4)(proximity exclusion: conflicting objects not on same layer)
Code Eq.25 → Paper Eq.(5)(row printing indicator using Big-M)
Code Eq.26 → Paper Eq.(6) & (7) (cumulative printed rows: base + recurrence)
Code Eq.27 → Paper Eq.(8)(Gaussian drying load)
Code Eq.28 → Paper Eq.(9)(row manufacturing time = print + dry)
Code Eq.29 → Paper Eq.(10)(layer time = max row time)"""
