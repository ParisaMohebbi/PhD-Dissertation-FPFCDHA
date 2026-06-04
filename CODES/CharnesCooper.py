########### Charnes-Cooper Transformation with Delayed Constraint Generation ###############

## Import packages
import copy
import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import *
import pandas as pd
import os.path
import csv 
import json
from networkx.readwrite import json_graph
import time
import contextlib

### Accessories ###
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

def FindExp(data):      
    deadly_diseases = set()                           
    for patient_id, patient_data in data.items():
        status, diseases = patient_data
        if status == 1:
            deadly_diseases.update(diseases)
    return deadly_diseases

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

## Callback function to add lazy constraints
def add_violated_constraints(model, where):
    if where == GRB.Callback.MIPSOL:
        # Get the current solution
        xb_val = model.cbGetSolution(model._xb)
        x_val = model.cbGetSolution(model._x)
        yb_val = model.cbGetSolution(model._yb)
        t_val = model.cbGetSolution(model._t)
        
        # Add constraints for yb[i] <= t - xb[j]
        for i in model._M:
            for j in model._N:
                if j not in model._data[i][1]:
                    if yb_val[i] > t_val - xb_val[j]:
                        model.cbLazy(model._yb[i] <= model._t - model._xb[j])
        
        # Add constraints for yb[i] >= t - exp_1
        for i in model._M:
            exp_1 = 0
            for j in model._N:
                if j not in model._data[i][1]:
                    exp_1 += xb_val[j]
            if yb_val[i] < t_val - exp_1:
                model.cbLazy(model._yb[i] >= model._t - quicksum(model._xb[j] for j in model._N if j not in model._data[i][1]))

## Function to solve the model
def MIP_model(data, graph, lb):
    try:
        Comp = [c for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
        C = list(range(nx.number_connected_components(graph)))
        model = gp.Model("clique_mortality")
        M = list(data.keys())
        N = graph.nodes()
        
        ## Create variables
        x  = model.addVars(N, vtype=GRB.BINARY, name="x")
        xb = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="xbar")
        yb = model.addVars(M, vtype=GRB.CONTINUOUS, lb=0, ub=1/lb, name="ybar")
        t  = model.addVar(vtype=GRB.CONTINUOUS, lb=1/(len(M)), ub=1/lb, name="t")
        f = model.addVars(C, vtype=GRB.BINARY, name="f")
        
        ## Set objective
        model.setObjective(quicksum(yb[i] for i in M if data[i][0] == 1), GRB.MAXIMIZE)

        ## Add constraints
        ## Clique
        for H in Comp:
            for i in H:
                for j in H:
                    if j not in graph.neighbors(i) and (i) < (j):
                        model.addConstr(x[i] + x[j] <= 1, f"clique_constr{i},{j}")           
        ## Add constraint: \sum f_i = 1
        model.addConstr(quicksum(f[i] for i in C) == 1)
        ## Add constraint: x_i <= f_j
        model.addConstrs(x[i] <= f[j] for j in C for i in Comp[j])         
                        
        ## Linearization of xb[j] = x[j] * t using indicator constraints
        for j in N:
            model.addGenConstrIndicator(x[j], 1, xb[j] == t)  # If x[j] == 1, then xb[j] == t
            model.addGenConstrIndicator(x[j], 0, xb[j] == 0)  # If x[j] == 0, then xb[j] == 0            
        
        ## Charnes-Cooper constraints
        model.addConstr(yb.sum("*") == 1)

        # Store data and variables in the model for use in the callback
        model._data = data
        model._M = M
        model._N = N
        model._xb = xb
        model._yb = yb
        model._t = t
        model._x = x

        # Set the callback function
        model.Params.timeLimit = 7200    
        model.Params.lazyConstraints = 1
        model.optimize(add_violated_constraints)

        status = model.Status

        if status == GRB.Status.TIME_LIMIT:
            print('Time limit reached.')
            print('Best feasible solution found:')
            print('Objective value: %.2f' % model.getAttr('ObjVal'))
            print('Printing non-zero variables...')
            t_val = t.x
            yb_val = model.getAttr('x', yb)
            x_val = model.getAttr('x', x)
            print('t = ', t_val)

            for i in N:
                if x_val[i] > 0:
                    print('x_' + str(i) + ' = ' + str(x_val[i]))
            dead = 0
            total = 0
            y_val = {}
            for i in M:
                if yb_val[i] > 0:
                    y_val[i] = yb_val[i] / t_val
                    print('y_' + str(i) + ' = ' + str(y_val[i]))
                    dead += data[i][0] * y_val[i] 
                    total += y_val[i]
            print('#dead= ', dead)
            print('#total= ', total)
            print('Rate= ', dead / total) 
            
        if status == GRB.Status.OPTIMAL:
            print('Printing non-zero variables...')
            t_val = t.x
            yb_val = model.getAttr('x', yb)
            x_val = model.getAttr('x', x)
            print('t = ', t_val)

            for i in N:
                if x_val[i] > 0:
                    print('x_' + str(i) + ' = ' + str(x_val[i]))
            dead = 0
            total = 0
            y_val = {}
            for i in M:
                if yb_val[i] > 0:
                    y_val[i] = yb_val[i] / t_val
                    print('y_' + str(i) + ' = ' + str(y_val[i]))
                    dead += data[i][0] * y_val[i] 
                    total += y_val[i]
            print('#dead= ', dead)
            print('#total= ', total)
            print('Rate= ', dead / total)  
            
        elif status == GRB.Status.INF_OR_UNBD or status == GRB.Status.INFEASIBLE or status == GRB.Status.UNBOUNDED:
            print('Model is infeasible or unbounded')
            # model.computeIIS()
            # model.write('Infeasible.ilp')  # Write conflicting constraints
        else:
            print('Optimization stopped with status ' + str(status))
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')
    
## Main function
def main():
    # Parameters
    lb = 100  # Minimum patient count threshold
    
    name = f'sample_1000_seed13'
    output_filename = f'CharnesCooper_{name}_lb{lb}.txt'
    
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
            print("Preprocessing Time:", time.time()-start_pp)                  
            
            start_solve = time.time()
            
            MIP_model(data,graph,lb)
            
            print("Solving Time:", time.time()-start_solve)

## Solve an instance
if __name__ == "__main__":
    main()
