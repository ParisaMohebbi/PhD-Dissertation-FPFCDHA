######################## Hybrid LBBD and Extensions ############################

import networkx as nx
import gurobipy as gp
from gurobipy import GRB, quicksum
import json
import time
import contextlib

# Use the following if the data contains age information ##
# =============================================================================
# def FindExp(data):
#     deadly_diseases = set()
#     for patient_id, value in data.items():
#         status = value[0]
#         diseases = value[2]
#         if status == 1:
#             deadly_diseases.update(diseases)
#     return deadly_diseases
# 
# def DisDict(data, anchor_age):
#     A = {}
#     X = set()
#     old_count = {}
#     for patient_id, value in data.items():
#         status = value[0]
#         age = value[1]
#         diseases = value[2]
#         if status == 1:
#             X.add(patient_id)
#         is_old = (age >= anchor_age)
#         for disease in diseases:
#             if disease not in A:
#                 A[disease] = {patient_id}
#                 old_count[disease] = 1 if is_old else 0
#             else:
#                 A[disease].add(patient_id)
#                 if is_old:
#                     old_count[disease] += 1
#     return A, X, old_count
# =============================================================================

# Use the following if cancer extension is being used ##
# ===========================================================================
#def load_cancer_codes(filename):
#    """Load cancer codes from a text file containing a Python list."""
#    try:
#        with open(filename, 'r') as f:
#            content = f.read().strip()
#        
#        # Safely parse the list
#        cancer_list = ast.literal_eval(content)
#        
#        if isinstance(cancer_list, list):
#            cancer_codes = [str(code).strip() for code in cancer_list]
#            print(f"Successfully loaded {len(cancer_codes)} cancer codes from {filename}")
#            return cancer_codes
#        else:
#            raise ValueError("File content is not a valid list")
#            
#    except Exception as e:
#        print(f"Error loading cancer codes from {filename}: {e}")
#        print("Make sure the file contains a valid Python list like: ['C50', 'C61', ...]")
#        return []
# ===========================================================================

# Otherwise
# =============================================================================
# def FindExp(data):
#     deadly_diseases = set()
#     for patient_id, (status, diseases) in data.items():
#         if status == 1:
#             valid_diseases = [n for n in diseases if n.strip() not in ('', 'ccs_number')]
#             deadly_diseases.update(valid_diseases)
#     return list(deadly_diseases)
# 
# def DisDict(data):
#     A = {}
#     X = set()
#     for patient_id, (status, diseases) in data.items():
#         if status == 1:
#             X.add(patient_id)
#         for disease in diseases:
#             if disease not in A:
#                 A[disease] = set()
#             A[disease].add(patient_id)
#     return A, X
# =============================================================================

def MakeGraph(deadly_diseases, A, edge_list, lb):
    G = nx.Graph()
    valid_nodes = {d for d in deadly_diseases if len(A.get(d, set())) >= lb}
    G.add_nodes_from(valid_nodes)
    with open(edge_list, 'r') as f:
        for line in f:
            source, target, _ = line.split()
            if source in valid_nodes and target in valid_nodes:
                if len(A[source] & A[target]) >= lb:
                    G.add_edge(source, target)
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def calculate_mortality_rate(C, A, X):
    if not C:
        return 0, 0
    patients_with_all = set.intersection(*[A[u] for u in C])
    if not patients_with_all:
        return 0, 0
    expired_count = len(X.intersection(patients_with_all))
    return expired_count, len(patients_with_all)

def calculate_parameters(graph, A, X, lb):
    gamma_v = {}
    mu_v = {}
    for v in graph.nodes():
        num = len(A[v] & X)
        den = len(A[v])
        gamma_v[v] = min(num / lb, 1.0)
        mu_v[v] = num / den if den > 0 else 0.0
    gamma_uv = {}
    mu_uv = {}
    for u, v in graph.edges():
        num = len(A[u] & A[v] & X)
        den = len(A[u] & A[v])
        gamma_uv[(u, v)] = min(num / lb, 1.0)
        mu_uv[(u, v)] = num / den if den > 0 else 0.0
    return gamma_v, mu_v, gamma_uv, mu_uv

def light_mu_star_precompute(graph, A, X, lb, mu_v, k=5):
    """Precompute exact mu* for top-k vertices ranked by individual mortality rate mu_v"""
    start = time.time()
    
    # ranking by mu_v (individual mortality rate)
    ranked_vertices = sorted(mu_v.keys(), key=lambda v: mu_v[v], reverse=True)[:k]
   
    mu_star_v = {}
    best_mu_cbb = 0.0
    total_recursions = 0

    print(f"Precomputing exact mu* for top {k} vertices by individual mortality rate...")

    for v in ranked_vertices:
        _, mu_v_star, rec = CBB_MMRCP(graph, A, set(graph.nodes()), X, lb, {v})
        mu_star_v[v] = mu_v_star
        best_mu_cbb = max(best_mu_cbb, mu_v_star)
        total_recursions += rec
        print(f"  mu*({v}) = {mu_v_star:.4f} (singleton mu={mu_v[v]:.4f}, {rec} recursions)")

    print(f"Precomputation completed. Best mu_CBB = {best_mu_cbb:.4f}")
    print(f"Total recursions: {total_recursions} | Time: {time.time() - start:.2f}s\n")
   
    return mu_star_v, best_mu_cbb

def CBB_MMRCP(graph, A, patients, X, ell, C0):
    def DfsBB(C, L, CurrentDen, CurrentNum):
        nonlocal BestSol, BestObj, recursion_count
        recursion_count += 1
        if len(L) == 0:
            return
        local_L = L.copy()
        for v in list(local_L):
            NewDen = CurrentDen & A[v]
            NewNum = CurrentNum & A[v]
            if len(NewDen) >= ell and len(NewNum) > BestObj * ell:
                if len(NewNum) > BestObj * len(NewDen):
                    BestObj = len(NewNum) / len(NewDen)
                    BestSol = C | {v}
                newL = local_L & set(graph.neighbors(v))
                DfsBB(C | {v}, newL, NewDen, NewNum)
            local_L.remove(v)

    recursion_count = 0
    X = set(X)

    if len(C0) == 0:
        best_v = max(graph.nodes(), key=lambda v: (len(A[v] & X) / len(A[v])) if len(A[v]) > 0 else 0)
        BestSol = {best_v}
        BestObj = (len(A[best_v] & X) / len(A[best_v])) if len(A[best_v]) > 0 else 0
        DfsBB(set(), set(graph.nodes()), set(patients), X)
    else:
        L = set.intersection(*(set(graph.neighbors(v)) for v in C0)) - set(C0)
        den_C0 = set.intersection(*(A[v] for v in C0))
        num_C0 = den_C0 & X
        BestSol = set(C0)
        BestObj = len(num_C0) / len(den_C0) if len(den_C0) > 0 else 0
        DfsBB(set(C0), L, den_C0, num_C0)

    return BestSol, BestObj, recursion_count

def find_minimal_cover(model, clique):
    current_clique = set(clique)
    stopflag = False
    while not stopflag:
        stopflag = True
        for u in list(current_clique):
            subset = [v for v in current_clique if v != u]
            patient_count = len(set.intersection(*[model._A[v] for v in subset])) if subset else 0
            if patient_count < model._lb:
                current_clique.remove(u)
                stopflag = False
    return list(current_clique)

def valid_F1(graph, F1, A, ell):
    if not F1:
        return False
    F1 = set(F1)
    patients = None
    for v in F1:
        if patients is None:
            patients = A[v].copy()
        else:
            patients &= A[v]
        if len(patients) < ell:
            return False
    return True


# Use the following if the data contains age information ##
# =============================================================================
# def CBMIP_model(graph, lb, gamma_v, mu_v, gamma_uv, mu_uv, mu_cbb, patients, mu_star_v, aOld_v, A_dict, X):
# =============================================================================

# Otherwise
# =============================================================================
# def CBMIP_model(data, graph, lb, gamma_v, mu_v, gamma_uv, mu_uv, mu_cbb, patients, mu_star_v):
# =============================================================================
    try:
        Comp = [c for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
        C = list(range(len(Comp)))
        master = gp.Model("Hybrid_LBBD")
        N = list(graph.nodes())
        # Comment the following line if the data contains age information
# =============================================================================
#         A_dict, X = DisDict(data)
# =============================================================================


        x = master.addVars(N, vtype=GRB.BINARY, name="x")
        z = master.addVar(vtype=GRB.CONTINUOUS, name="z")
        f = master.addVars(C, vtype=GRB.BINARY, name="f")

        master.setObjective(z, GRB.MAXIMIZE)
        # Comment the following line if we have extension constraint
# =============================================================================        
#        master.addConstr(z >= mu_cbb, name="z_lb_CBB")
# =============================================================================        
        master.addConstr(z <= 1)

        # Clique constraints
        for H in Comp:
            for i in H:
                for j in H:
                    if j not in graph.neighbors(i) and i < j:
                        master.addConstr(x[i] + x[j] <= 1)

        master.addConstr(quicksum(f[i] for i in C) == 1)
        master.addConstrs(x[i] <= f[j] for j in C for i in Comp[j])
        master.addConstr(quicksum(x[j] for j in N) >= 1)

        # GAMMA CONSTRAINTS (root relaxation)
        for v in N:
            if gamma_v[v] < 1:
                master.addConstr(z <= gamma_v[v] * x[v] + (1 - x[v]))
            else:
                hamming_v = quicksum(x[u] for u in N if u != v) + (1 - x[v])
                master.addConstr(z <= mu_v[v] + (1 - mu_v[v]) * hamming_v)

        for u, v in graph.edges():
            if gamma_uv.get((u, v), 1.0) < 1:
                master.addConstr(z <= gamma_uv[(u, v)] * (x[u] + x[v] - 1) + (2 - x[u] - x[v]))
            else:
                hamming_uv = (2 - x[u] - x[v]) + quicksum(x[w] for w in N if w not in {u, v})
                master.addConstr(z <= mu_uv[(u, v)] + (1 - mu_uv[(u, v)]) * hamming_uv)

        # CBB Hybrid Cuts for top-k singletons
        print("Adding CBB hybrid cuts for top-k singletons...")
        for v in mu_star_v:
            mu_star = mu_star_v[v]
            master.addConstr(z <= mu_star + (1 - mu_star) * (1 - x[v]), name=f"CBB_singleton_{v}")


        # Extension
# =============================================================================
#         # For data conbtaining age
#         # Average aOld(v) <= 50%, IF ONLY data contains age information
#         master.addConstr(quicksum(aOld_v[v] * x[v] for v in N) <= 0.5 * quicksum(x[v] for v in N), name="age_composition_upper_bound")
# =============================================================================

# =============================================================================
#         # For cancer constraints
#         # === CONSTRAINT: At least one cancer code ===
#         cancer_in_graph = [c for c in cancer_codes if c in N]
#         master.addConstr(quicksum(x[c] for c in cancer_in_graph) >= 1, name="at_least_one_cancer")
#         # === CONSTRAINT: Fix one cancer code ===
#         master.addConstr(x[cancer code] == 1)
# =============================================================================
                
        
        # Callback data
        master._A = A_dict
        master._patients = patients
        master._X = X
        master._N = N
        master._x = x
        master._z = z
        master._lb = lb
        master._graph = graph
        master._opt_cuts = 0          
        master._cbb_cuts = 0          
        master._feas_cuts = 0
        master._cbb_calls = 0
        master._total_recursions = 0
        master._bc_nodes = 0

        master.Params.timeLimit = 7200
        master.Params.lazyConstraints = 1
        master.Params.PreCrush = 1

        master.optimize(FOcuts_callback)

        print("\n--- Hybrid LBBD Statistics ---")
        print(f"Total standard no-good optimality cuts : {master._opt_cuts}")
        print(f"Total hybrid CBB cuts                  : {master._cbb_cuts}")
        print(f"Total feasibility cuts                 : {master._feas_cuts}")
        print(f"Total CBB calls                        : {master._cbb_calls}")
        print(f"Total recursions in CBB                : {master._total_recursions}")
        print(f"Total BC nodes processed               : {master._bc_nodes}")

        if master.SolCount > 0:
            x_sol = master.getAttr('X', x)
            solution = [j for j in N if x_sol[j] > 0.9]
            num, den = calculate_mortality_rate(solution, A_dict, X)
            print(f"\nBest solution: {solution}")
            print(f"Mortality rate: {num}/{den} = {(num/den):.4f}")
            print(f"z = {master.ObjVal:.6f}")

    except Exception as e:
        print(f"Error: {e}")

# Callback 
def FOcuts_callback(model, where):
    if not hasattr(model, '_cbb_calls'):
        model._cbb_calls = 0
        model._total_recursions = 0
        model._bc_nodes = 0

    model._bc_nodes += 1
    tau = 4

    if where == GRB.Callback.MIPSOL:
        xT = model.cbGetSolution(model._x)
        zT = model.cbGetSolution(model._z)
        CT = [u for u in model._N if xT[u] >= 0.5]

        if not hasattr(model, '_memo_mu'):
            model._memo_mu = {}

        CT_key = frozenset(CT)
        if CT_key in model._memo_mu:
            NumCT, DenCT, muCT = model._memo_mu[CT_key]
        else:
            NumCT, DenCT = calculate_mortality_rate(CT, model._A, model._X)
            muCT = NumCT / DenCT if DenCT > 0 else 0.0
            model._memo_mu[CT_key] = (NumCT, DenCT, muCT)

        if DenCT >= model._lb:
            if zT <= muCT + 1e-6:
                return

            # Add standard no-good cut 
            J0 = [u for u in model._N if xT[u] < 0.5]
            hamming_dist = (quicksum(1 - model._x[v] for v in CT) + quicksum(model._x[v] for v in J0))
            opt_cut = model._z <= muCT + (1 - muCT) * hamming_dist
            model.cbLazy(opt_cut)
            model._opt_cuts += 1
            print(f"Added no-good optimality cut #{model._opt_cuts} (mu={muCT:.4f})")

            # Add hybrid CBB cut if clique is large enough
            if len(CT) >= tau:
                model._cbb_calls += 1
                best_clique, best_mu, recurs = CBB_MMRCP(model._graph, model._A, model._patients, model._X, model._lb, CT)
                model._total_recursions += recurs

                if best_clique is not None and best_mu > muCT + 1e-6:
                    hybrid_cut = model._z <= best_mu + (1 - best_mu) * quicksum(1 - model._x[v] for v in CT)
                    model.cbLazy(hybrid_cut)
                    model._cbb_cuts += 1
                    print(f"Added hybrid CBB cut #{model._cbb_cuts} (mu={best_mu:.4f}, recursions={recurs})")
        else:
            minimal_cover = find_minimal_cover(model, CT)
            feas_cut = quicksum(model._x[v] for v in minimal_cover) <= len(minimal_cover) - 1
            model.cbLazy(feas_cut)
            model._feas_cuts += 1
            print(f"Added feasibility cut #{model._feas_cuts}")


# Use the following if the data contains age information ## 
# =============================================================================
# # Use the following if the data contains age information ##            
# def main():
#     lb = 100
#     anchor_age = 60
#     name = 'sample_5000Age_seed13'
#     # output_filename = f'Cap older_HybridLBBD_{name}_lb{lb}.txt'             # If using "older_skewing_diseases_cap" constraint
#     # output_filename = f'Average aOld(v)_HybridLBBD_{name}_lb{lb}.txt'        # If using "age_composition_upper_bound" constraint
# 
#     with open(output_filename, 'w', encoding='utf-8') as f:
#         with contextlib.redirect_stdout(f):
#             with open(f'{name}.json', 'r') as fp:
#                 data = json.load(fp)
# 
#             # Preprocessing
#             A_dict, X, old_count = DisDict(data, anchor_age)
#             deadly_diseases = FindExp(data)
#             graph = MakeGraph(deadly_diseases, A_dict, "ochiai_network.edgelist", lb)
# 
#             gamma_v, mu_v, gamma_uv, mu_uv = calculate_parameters(graph, A_dict, X, lb)
#             patients = set().union(*(A_dict[v] for v in graph.nodes()))
# 
#             # Precompute mu* 
#             mu_star_v, mu_cbb = light_mu_star_precompute(graph, A_dict, X, lb, mu_v, k=5)
# 
#             # Precompute aOld_v OUTSIDE the model
#             N = list(graph.nodes())
#             aOld_v = {}
#             for v in N:
#                 den = len(A_dict.get(v, set()))
#                 num_old = old_count.get(v, 0)
#                 aOld_v[v] = num_old / den if den > 0 else 0.0
# 
#             print(f"Using mu_CBB = {mu_cbb:.4f} as lower bound for z")
# 
#             # Solve the model
#             start_solve = time.time()
#             CBMIP_model(graph, lb, gamma_v, mu_v, gamma_uv, mu_uv, mu_cbb, patients, mu_star_v, aOld_v, A_dict, X)
#             print(f"Solving Time: {time.time() - start_solve:.2f}s")            
# =============================================================================

# Use the following if cancer constraint is considered ## 
# =============================================================================
#def main():
#    lb = 100
#    name = 'sample_5000_seed13'
#    cancer_codes_file = 'CancerCodes.txt'      
#    
#    output_filename = f'Cancer_HybridLBBD_{name}_lb{lb}.txt'
#    
#    with open(output_filename, 'w', encoding='utf-8') as f:
#        with contextlib.redirect_stdout(f):
#            with open(f'{name}.json', 'r') as fp:
#                data = json.load(fp)
#
#            A, X = DisDict(data)
#            deadly_diseases = FindExp(data)
#            graph = MakeGraph(deadly_diseases, A, "ochiai_network.edgelist", lb)
#            
#            # Load cancer codes
#            cancer_codes = load_cancer_codes(cancer_codes_file)
#
#            gamma_v, mu_v, gamma_uv, mu_uv = calculate_parameters(graph, A, X, lb)
#            patients = set().union(*(A[v] for v in graph.nodes()))
#            mu_star_v, mu_cbb = light_mu_star_precompute(graph, A, X, lb, mu_v, k=5)
#
#            print(f"Using mu_CBB = {mu_cbb:.4f} as lower bound for z")
#            
#            CBMIP_model(data, graph, lb, gamma_v, mu_v, gamma_uv, mu_uv, 
#                       mu_cbb, patients, mu_star_v, cancer_codes)
# =============================================================================

# Otherwise
# =============================================================================
# # Main 
# def main():
#     lb = 100
#     name = 'sample_5000_seed13'
#     # output_filename = f'Average-MR_HybridLBBD_{name}_lb{lb}.txt'            # If using "avg_mr_reliability" constraint
#     # output_filename = f'Average-MR-Bounded_HybridLBBD_{name}_lb{lb}.txt'    # If using "avg_mr_reliability_bounded" constraint
# 
#     with open(output_filename, 'w', encoding='utf-8') as f:
#         with contextlib.redirect_stdout(f):
#             with open(f'{name}.json', 'r') as fp:
#                 data = json.load(fp)
# 
#             start_pp = time.time()
#             A, X = DisDict(data)
#             deadly_diseases = FindExp(data)
#             graph = MakeGraph(deadly_diseases, A, "ochiai_network.edgelist", lb)
# 
#             gamma_v, mu_v, gamma_uv, mu_uv = calculate_parameters(graph, A, X, lb)
#             patients = set().union(*(A[v] for v in graph.nodes()))
# 
#             mu_star_v, mu_cbb = light_mu_star_precompute(graph, A, X, lb, mu_v, k=5)
# 
#             print(f"Using mu_CBB = {mu_cbb:.4f} as lower bound for z")
# 
#             start_solve = time.time()
#             CBMIP_model(data, graph, lb, gamma_v, mu_v, gamma_uv, mu_uv, mu_cbb, patients, mu_star_v)
#             print(f"Solving Time: {time.time() - start_solve:.2f}s")
# =============================================================================

if __name__ == "__main__":
    main()    
