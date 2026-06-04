########################## Module for Handling All or deletion/Marginal Lethal Cliques Using the Modified Bron-Kerbosch ####################################

### Import packages ###
import copy
import networkx as nx
import pandas as pd
import math
import os.path
import csv 
import json
from networkx.readwrite import json_graph
import time
import heapq

### Global Variables ###
muStar = 0
C_max_size = 100                                                                # Set the maximum size of the priority queue
C_top = []                                                                      # The priority queue to store the top K cliques based on their mu value
calls = 0                                                                       # Number of recursive calls
#fixed_vertices = ['252']                                                       # Set of vertices to fix as C0 (for marginal clique)
#deletion = {'252','21','289','29','122'}                                       # Diseases and its corresponding records to be deleted, uncomment for deleting diseases

### Required Dictionaries and Data Preprocessing ###
## Making the Disease-Patient Dictionary and Vector of Expired Patients                                                  
def DisDict(data):                                                              # Use "DisDict(data, deletion)" header if there are diseases to omit
    A = {}
    X = set()  
    for patient_id, value in data.items():
        if value[0] == 1:
            X.add(patient_id)
        patient_diseases = value[1]
        for disease in patient_diseases:
        ## Uncomment the below lines for deleting diseases    
# =============================================================================
#             if disease in deletion:
#                 continue                                                      # Skip adding this disease to A
# =============================================================================
            if disease not in A:
                A[disease] = {patient_id}
            else:
                A[disease].add(patient_id)
    return A, X


## Data Preparation
def FindExp(data):      
    deadly_diseases = set()                           
    for patient_id, patient_data in data.items():
        status, diseases = patient_data
        if status == 1:                                                         # Check out if the Patient is expired from the set of diseases he has
            deadly_diseases.update(diseases)
    return deadly_diseases

## Function for Generating the Graph from Deadly Diseases
def MakeGraph(nodes_set, edge_list):
    G = nx.Graph()                                                              # Create an empty graph
    G.add_nodes_from(nodes_set)                                                 # Add nodes to the graph   
    with open(edge_list, 'r') as f:                                             # Read the edge list from the file and add edges to the graph
        for line in f:
            source, target, attributes = line.split()
            if source in nodes_set and target in nodes_set:
                G.add_edge(source, target)
    return G

## Function for Handling the Priority Queue
def update_priority_queue(mu, clique):
    global C_top
    if len(C_top) < C_max_size and clique not in [item[1] for item in C_top]:   # Check if the size of C_top is less than K (i.e., C_max_size) and if 'clique' is not in C_top
        heapq.heappush(C_top, (mu, clique))
    else:
        if mu > C_top[0][0] and clique not in [item[1] for item in C_top]:      # If size >= K, then compare 'mu' with the smallest element of C_top
            heapq.heappushpop(C_top, (mu, clique))

### The Modified Bron-Kerbosch ###
def solve(graph, b, data, A, X, lb, muStar):
    def XBron_Kerbosch(C, L, den):
        global fixed_vertices
        global muStar
        global calls  
        calls += 1 
        if len(C) == b or L == set():
            return
          
        for v in list(L):
            newden = den.intersection(A[v])
            if len(newden) >= lb:
                muStar = len(X.intersection(newden))/(len(newden))
                update_priority_queue(muStar, C.union({v}))
                XBron_Kerbosch(C.union({v}), L.intersection(graph.neighbors(v)), newden)
            L.remove(v)
            

    C = set()                                                                   # Initialize with "fixed_vertices"
    
    L = set(graph.nodes())
    ## Uncomment the below lines for finding lethal cliques with fixed diseases
# =============================================================================
#     for vertex in fixed_vertices[0:]:
#         L.intersection_update(graph.neighbors(vertex))
# =============================================================================
    
    den = set(data.keys())    
    ## Uncomment the below lines for finding lethal cliques with fixed diseases
# =============================================================================
#     for vertex in fixed_vertices[0:]:
#         den.intersection_update(A[vertex])
# =============================================================================
                                         
    XBron_Kerbosch(C, L, den)
    return heapq.nlargest(C_max_size, C_top) 

### Main Function ###
def main():
    with open('sample_5000_seed13.json', 'r') as f:                             # Reading the dataset
        data = json.load(f)
    
    ## Making the A, X
    A,X = DisDict(data)                                                         # Use "DisDict(data, deletion)" header if there are diseases to omit
    ## Start preparation the data
    start_pp = time.time()
    deadly_diseases = FindExp(data)
    
    ## Making the graph from the edgelist file
    nodes_set = deadly_diseases                                              # If there is a deletion set add ".difference(deletion)" to the end 
    edge_list = "Comorbidity_Network_SCI=0.05.edgelist"
    original_graph = MakeGraph(nodes_set, edge_list)
    
    pptime = time.time()-start_pp
    print("Preparation Time:", pptime)
    
    ## Start solving the problem
    ## Parameters   
    b = 3                  
    lb = 100
    
    start_solve = time.time()

    top_cliques = solve(original_graph, b, data, A, X, lb, muStar)              
    
    print("Top Cliques:", top_cliques)
    print("#Recursive Calls:", calls)    
    stime = time.time()-start_solve
    print("Solving Time:", stime)
    
    ## Writing the output to a csv file
    with open('top_marginal_cliques_.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
# =============================================================================
#         csvwriter.writerow(["Fixed Vertices: ", fixed_vertices])                # Uncomment this line for all lethal cliques with fixed diseases
# =============================================================================
        csvwriter.writerow(["b: ", b])
# =============================================================================
#         csvwriter.writerow(["delta: ", b-len(fixed_vertices)])                  # Uncomment this line for all lethal cliques with fixed diseases
# =============================================================================
        csvwriter.writerow(["Preparation Time: ", pptime])
        csvwriter.writerow(["Solving Time: ", stime])
        csvwriter.writerow(["#Recursive Calls: ", calls])
        csvwriter.writerow([]) 
        
        csvwriter.writerow(['Clique', 'Mu'])          
        for mu, clique in top_cliques:
            csvwriter.writerow([', '.join(clique), mu])
 
if __name__ == "__main__":
    main()    
