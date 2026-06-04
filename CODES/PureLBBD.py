################ Pure LBBD ###################

import copy
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import pandas as pd
import os.path
import csv 
import json
from networkx.readwrite import json_graph
import time
import contextlib


### Preprocessing ###
## Identify deadly diseases from expired patients
def FindExp(data):
    deadly_diseases = set()
    for patient_id, (status, diseases) in data.items():
        if status == 1:  # Expired patients
            valid_diseases = [n for n in diseases if n.strip() not in ('', 'ccs_number')]
            deadly_diseases.update(valid_diseases)
    return list(deadly_diseases)

## Create disease-patient dictionary and set of expired patients
def DisDict(data):
    A = {}
    X = set()  # Expired patients
    for patient_id, (status, diseases) in data.items():
        if status == 1:
            X.add(patient_id)
        for disease in diseases:
            if disease not in A:
                A[disease] = set()
            A[disease].add(patient_id)
    return A, X

## Construct comorbidity graph from edge list
def MakeGraph(deadly_diseases, A, edge_list, lb):
    G = nx.Graph()    
    # Filter deadly diseases with at least lb patients
    valid_nodes = {d for d in deadly_diseases if len(A.get(d, set())) >= lb}
    G.add_nodes_from(valid_nodes)
    
    # Add valid edges based on patient intersection
    with open(edge_list, 'r') as f:
        for line in f:
            source, target, _ = line.split()
            if source in valid_nodes and target in valid_nodes:
                if len(A[source] & A[target]) >= lb:
                    G.add_edge(source, target)
    return G

## Calculate the mortality rate for a clique 
def calculate_mortality_rate(C, A, X):
    if not C:
        return 0, 0   
    patients_with_all = set.intersection(*[A[u] for u in C])
    if not patients_with_all:
        return 0, 0  
    expired_count = len(X.intersection(patients_with_all))
    return expired_count, len(patients_with_all)

## Calculate Gamma and Mu parameters
def calculate_parameters(graph, A, X, lb):
    # Calculate gamma_v and mu_v for all nodes
    gamma_v = {}
    mu_v = {}
    for v in graph.nodes():
        # Calculate gamma_v
        intersection_size = len(A[v] & X)
        gamma_v[v] = min(intersection_size / lb, 1.0)
        
        # Calculate mu_v
        num, den = calculate_mortality_rate([v], A, X)
        mu_v[v] = num / den if den > 0 else 0.0
    
    # Calculate gamma_uv and mu_uv for all edges
    gamma_uv = {}
    mu_uv = {}
    for u, v in graph.edges():
        # Calculate gamma_uv
        intersection_size = len(A[u] & A[v] & X)
        gamma_uv[(u, v)] = min(intersection_size / lb, 1.0)
        
        # Calculate mu_uv
        num, den = calculate_mortality_rate([u, v], A, X)
        mu_uv[(u, v)] = num / den if den > 0 else 0.0
    
    # Find maximum values for lower bound
    max_gamma_v = max(gamma_v.values()) if gamma_v else 0
    max_mu_v = max(mu_v.values()) if mu_v else 0
    max_gamma_uv = max(gamma_uv.values()) if gamma_uv else 0
    max_mu_uv = max(mu_uv.values()) if mu_uv else 0
    
    z_lb = min(max_gamma_v, max_mu_v, max_gamma_uv, max_mu_uv)

    return gamma_v, mu_v, gamma_uv, mu_uv, z_lb

## Minimal Infeasible sub-clique
def find_minimal_cover(model, clique):
    current_clique = clique.copy()
    stopflag = False
    
    while not stopflag:
        stopflag = True
        # Check each node for removal
        for u in current_clique:  # Iterate over copy to allow removal
            # Create subset without u
            subset = [v for v in current_clique if v != u]
            # Calculate intersection for subset
            if subset:
                patients = set.intersection(*[model._A[v] for v in subset])
                patient_count = len(patients)
            else:
                patient_count = 0
            # If removing u still leaves us infeasible, remove it
            if patient_count < model._lb:
                current_clique.remove(u)
                stopflag = False  # Need to check again
    
    return current_clique
###############################################################################

### Master Model ###
def CBMIP_model(data, graph, lb, gamma_v, mu_v, gamma_uv, mu_uv, z_lb):
    try:
        Comp = [c for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
        C = list(range(nx.number_connected_components(graph)))
        master = gp.Model("Master")
        N = list(graph.nodes())
        A, X = DisDict(data)

        # Variables
        x = master.addVars(N, vtype=GRB.BINARY, name="x")
        z = master.addVar(vtype=GRB.CONTINUOUS, name="z")
        f = master.addVars(C, vtype=GRB.BINARY, name="f")
        
        master.addConstr(z >= z_lb)
        master.addConstr(z <= 1)
            
        # Objective
        master.setObjective(z, GRB.MAXIMIZE)

        ## Clique
        for H in Comp:
            for i in H:
                for j in H:
                    if j not in graph.neighbors(i) and (i) < (j):
                        master.addConstr(x[i] + x[j] <= 1, f"clique_constr{i},{j}")                        
        ## Add constraint: \sum f_i = 1
        master.addConstr(quicksum(f[i] for i in C) == 1)
        ## Add constraint: x_i <= f_j
        master.addConstrs(x[i] <= f[j] for j in C for i in Comp[j])   
        
        # Size constraints
        master.addConstr(quicksum(x[j] for j in N) >= 1)
        
        # Add single-node constraints
        for v in N:
            if gamma_v[v] < 1:
                master.addConstr(z <= gamma_v[v] * x[v] + (1 - x[v]))
            else:
                hamming_v = (quicksum(x[u] for u in N if u != v) + 1 - x[v])
                master.addConstr(z <= mu_v[v] + (1 - mu_v[v]) * hamming_v)
    
        # Add edge constraints
        for u, v in graph.edges():
            if gamma_uv.get((u, v), 1.0) < 1:
                master.addConstr(z <= gamma_uv[(u, v)] * (x[u] + x[v] - 1) + (2 - x[u] - x[v]))
            else:
                hamming_uv = (2 - x[u] - x[v]) + quicksum(x[w] for w in N if w not in {u, v})
                master.addConstr(z <= mu_uv[(u, v)] + (1 - mu_uv[(u, v)]) * hamming_uv)
            
        # Store additional data for callback
        master._x = x
        master._z = z
        master._A = A
        master._X = X
        master._N = N
        master._lb = lb
        master._opt_cuts = 0
        master._feas_cuts = 0
        master._graph = graph
        master._z_lb = z_lb
        
        # Optimization parameters
        master.Params.timeLimit = 7200
        master.Params.lazyConstraints = 1
        master.Params.PreCrush = 1

        # Solve with callback
        master.optimize(FOcuts_callback)

        # Print cut statistics
        print("\n--- Cut Statistics ---")
        print(f"Total optimality cuts: {master._opt_cuts}")
        print(f"Total feasibility cuts: {master._feas_cuts}")

        # Process results
        status = master.Status
        if status == GRB.Status.OPTIMAL:
            print('\nOptimal solution found:')
            x_sol = master.getAttr('x', x)
            solution = [j for j in N if x_sol[j] > 0.9]
            num, den = calculate_mortality_rate(solution, A, X)
            mortality_rate = (num / den) if den > 0 else 0.0
            print(f"Solution diseases: {solution}")
            print(f"Mortality rate: {num}/{den} - {mortality_rate:.4f}")
            print(f"z value: {master.ObjVal:.6f}")
        elif status == GRB.Status.TIME_LIMIT:
            print('\nTime limit reached - best solution:')
            if master.SolCount > 0:
                x_sol = master.getAttr('x', x)
                solution = [j for j in N if x_sol[j] > 0.9]
                num, den = calculate_mortality_rate(solution, A, X)
                mortality_rate = (num / den) if den > 0 else 0.0
                print(f"Solution diseases: {solution}")
                print(f"Mortality rate: {num}/{den} - {mortality_rate:.4f}")
                print(f"z value: {master.ObjVal:.6f}")
        else:
            print(f'\nOptimization stopped with status {status}')

    except gp.GurobiError as e:
        print(f'Gurobi error {e.errno}: {e}')
    except Exception as e:
        print(f'Unexpected error: {str(e)}')

### Callback Function ###
def FOcuts_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        xT = model.cbGetSolution(model._x)
        zT = model.cbGetSolution(model._z)
        CT = [u for u in model._N if xT[u] >= 0.5]

        # Initialize memoization dictionary if it doesn't exist
        if not hasattr(model, '_memo_mu'):
            model._memo_mu = {}
            
        J1 = CT
        J0 = [u for u in model._N if xT[u] < 0.5]
        
        # Check memoization for full clique first
        CT_key = frozenset(CT)
        if CT_key in model._memo_mu:
            NumCT, DenCT, muCT = model._memo_mu[CT_key]
        else:
            NumCT, DenCT = calculate_mortality_rate(CT, model._A, model._X)
            muCT = NumCT / DenCT if DenCT > 0 else 0.0
            model._memo_mu[CT_key] = (NumCT, DenCT, muCT)
        
        # Case 1: Feasible solution
        if DenCT >= model._lb:
            if zT <= muCT + 1e-6:
                return

            # Add main optimality cut
            hamming_dist = (quicksum(1 - model._x[v] for v in J1) + quicksum(model._x[v] for v in J0))
            opt_cut = model._z <= muCT + (1 - muCT) * hamming_dist
            model.cbLazy(opt_cut)
            model._opt_cuts += 1
            print(f"Added optimality cut #{model._opt_cuts}")
        
        # Case 2: Infeasible solution              
        else:
            # Add the feasibility cut first
            # Use the separate function to find minimal cover
            minimal_cover = find_minimal_cover(model, CT)
            
            model.cbLazy(quicksum(model._x[v] for v in minimal_cover) <= len(minimal_cover) - 1)
            model._feas_cuts += 1
            print(f"Added improved feasibility cut #{model._feas_cuts} (mu={muCT:.4f})")
###############################################################################            
            
### Main ###        
def main():
    # Parameters
    lb = 100  # Minimum patient count threshold
    
    name = f'sample_1000_seed13'
    output_filename = f'GammaLBBD_{name}_lb{lb}.txt'
    
    # Redirect stdout to file
    with open(output_filename, 'w') as f:
        with contextlib.redirect_stdout(f):
            
            with open(f'{name}.json', 'r') as f_json:                             
                data = json.load(f_json)

            # Preprocessing
            start_pp = time.time()
            A, X = DisDict(data)
            deadly_diseases = FindExp(data)
            graph = MakeGraph(deadly_diseases, A, "ochiai_network.edgelist", lb)
            
            # Calculate parameters and lower bound for z
            gamma_v, mu_v, gamma_uv, mu_uv, z_lb = calculate_parameters(graph, A, X, lb)
            print(f"Calculated lower bound for z: {z_lb:.4f}")
            
            print("Preprocessing Time:", time.time()-start_pp)                  
            
            start_solve = time.time()
            
            CBMIP_model(data, graph, lb, gamma_v, mu_v, gamma_uv, mu_uv, z_lb)
            
            print("Solving Time:", time.time()-start_solve)

if __name__ == "__main__":
    main()
