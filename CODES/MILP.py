########################################### Module for Solving the MIP Formulation ###################################################

##import packages
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

### Accessories ###
## Graph Preprocessing
def FindExp(data):      
    deadly_diseases = set()                           
    for patient_id, patient_data in data.items():
        status, diseases = patient_data
        if status == 1:
            deadly_diseases.update(diseases)
    return deadly_diseases

## Function for generating a graph from deadly diseases
def MakeGraph(nodes_set, edge_list):
    ## Create an empty graph
    G = nx.Graph()
    ## Add nodes to the graph
    G.add_nodes_from(nodes_set)    
    ## Read the edge list from the file and add edges to the graph
    with open(edge_list, 'r') as f:
        for line in f:
            source, target, attributes = line.split()
            if source in nodes_set and target in nodes_set:
                G.add_edge(source, target)
    return G


## Function to solve the model
def MIP_model(data,graph,lb,b):
    try:
        ## Create a new model
        ## Get connected components in graph
        Comp = [c for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
        ## Number of connected components
        numComp = nx.number_connected_components(graph)
        print("The graph has ", graph.number_of_nodes(), "nodes and ", graph.number_of_edges(), " edges, and ", numComp, "connencted components." )
        model = gp.Model("clique_mortality")
        C = list(range(numComp))
        M = list(data.keys())
        N = graph.nodes()
        ## Create variables
        x = model.addVars(N, vtype=GRB.BINARY, name="x")
        y = model.addVars(M, vtype=GRB.BINARY, name="y")
        z = model.addVar(vtype=GRB.CONTINUOUS,lb=0,ub=1, name="z")
        w = model.addVars(M, vtype=GRB.CONTINUOUS,lb=0,ub=1, name="w")
        f = model.addVars(C, vtype=GRB.BINARY, name="f")
        
        ## Set objective
        model.setObjective(z, GRB.MAXIMIZE)
        
        ## Add constraint: \sum y_i = \sum w_i
        model.addConstr(quicksum(data[i][0]*y[i] for i in M) == w.sum("*"))

        ## Add clique constraint for each component
        for H in Comp:
            for i in H:
                for j in H:
                    if j not in graph.neighbors(i) and int(i) < int(j):
                        model.addConstr(x[i] + x[j] <=1, "clique_constr"+str(i)+","+str(j))                                        

        ## Add constraint: \sum f_i = 1
        model.addConstr(quicksum(f[i] for i in C) == 1)

        ## Add constraint: x_i <= f_j
        model.addConstrs(x[i] <= f[j] for j in C for i in Comp[j])                    

        ## Add constraint: \sum x_j <= b
        model.addConstr(x.sum("*") <=b)       

        ## Add the aggregated constraint, |V\A_i|*y_i <= |V\A_i| - \sum_j in V\A_i x_j and constraint: y_i >= 1- \sum x_j
        for i in M:
            exp_1 = 0
            for j in N:
                if j not in data[i][1]:
                    exp_1 +=x[j]
            model.addConstr(len(N-data[i][1])*y[i] <= len(N-data[i][1]) - exp_1)
            model.addConstr(y[i] >= 1 - exp_1) 
        
        ## Add constraint: \sum y_i >= l
        model.addConstr(y.sum("*") >= lb)

        # Optimize model
        model.Params.timeLimit = 7200    
        
        model._M = M
        model._z = z
        model._y = y
        model._w = w
                
        ## Using Callback
        model.Params.lazyConstraints = 1
        model.optimize(lcut_callback)        

        status = model.Status

        if status == GRB.Status.TIME_LIMIT:
            print('Time limit reached.')
            print('Best feasible solution found:')
            print('Objective value: %.2f' % model.getAttr('ObjVal'))
            print('Printing non-zero variables...')
            x = model.getAttr('x', x)
            y = model.getAttr('x', y)
            w = model.getAttr('x', w)
            f = model.getAttr('x',f)
            for i in N:
                if x[i]>0.9:
                    print('x_'+str(i)+' = '+ str(x[i]))
            
            dead = 0
            total = 0
            for i in M:
                if y[i]>0.9:
                    print('y_'+str(i)+' = '+ str(y[i]))
                    dead += data[i][0]*y[i] 
                    total += y[i]
            print('#dead= ', dead)
            print('#total= ', total)
            print('z' + ' = ' + str(z.x))
            model.write('CliqueSolution.sol')
            
        if status == GRB.Status.OPTIMAL:
            print('Printing non-zero variables...')
            x = model.getAttr('x', x)
            y = model.getAttr('x', y)
            w = model.getAttr('x', w)
            f = model.getAttr('x',f)
            for i in N:
                if x[i]>0.9:
                    print('x_'+str(i)+' = '+ str(x[i]))
            
            dead = 0
            total= 0
            for i in M:
                if y[i]>0.9:
                    print('y_'+str(i)+' = '+ str(y[i]))
                    dead += data[i][0]*y[i] 
                    total += y[i]
                if w[i]>0:
                    print('w_'+str(i)+' = '+ str(w[i]))
            print('#total= ', total)
            print('#dead= ', dead)
            print('z' + ' = ' + str(z.x))
            for i in C:
                if f[i]>0:
                    print('f_'+str(i)+ '= ' + str(f[i]))
            print('Obj: %g' % model.objVal)  
            # model.write('formulation.lp') #write MIP constraints
            
        elif status == GRB.Status.INF_OR_UNBD or status == GRB.Status.INFEASIBLE or status == GRB.Status.UNBOUNDED:
            print('Model is infeasible or unbounded')
            # model.computeIIS()
            # model.write('Infeasible.ilp')                                     # Write conflicting constraints
        else:
            print('Optimization stopped with status ' + str(status))
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')
        
        
## Add (violated) linearization constraints in a callback function    
def lcut_callback(m, where):    
    if where == GRB.Callback.MIPSOL:                                            # Check if LP relaxation at this BB node is integer        
        ## Retrieve the LP relaxation solution at this BB node
        yval = m.cbGetSolution(m._y)
        wval = m.cbGetSolution(m._w)
        zval = m.cbGetSolution(m._z)
        
        chosen_patients3b = [ i for i in m._M if wval[i] > yval[i] ]                    
        for i in chosen_patients3b:
            m.cbLazy(m._w[i] <= m._y[i])
        
        chosen_patients3c = [ j for j in m._M if wval[j] < zval + yval[j] -1 ]
        for i in chosen_patients3c:
            m.cbLazy(m._w[i] >= m._z + m._y[i] -1) 
        
        chosen_patients3d = [ k for k in m._M if wval[k] > zval ]
        for i in chosen_patients3d:
            m.cbLazy(m._w[i] <= m._z)                  
                     
## Main function
def main():

    with open('sample_1000_seed13.json', 'r') as f:
        data = json.load(f)

    start_pp = time.time()
    deadly_diseases = FindExp(data)
    
    ## Making the graph from the edgelist file
    nodes_set = deadly_diseases
    edge_list = "ochiai_network.edgelist"
    graph = MakeGraph(nodes_set, edge_list)
    print("Preprocessing Time:", time.time()-start_pp)
    
    ## Parameters      
    lb = 10
    b = 2                       
    
    start_solve = time.time()
    
    MIP_model(data,graph,lb,b)
    
    print("Solving Time:", time.time()-start_solve)

    ## Solve an instance
if __name__ == "__main__":
    main()