########################## Combinatorial Branch-and-Bound (CBB) ############################

import networkx as nx
import csv
import json
import time

### Global Variables ###
calls = 0
BestSol = set()
BestObj = 0.0

### Required Dictionaries and Data Preprocessing ###
def DisDict(data):
    A = {}
    X = set()
    for patient_id, value in data.items():
        if value[0] == 1:
            X.add(patient_id)
        for disease in value[1]:
            if disease not in A:
                A[disease] = {patient_id}
            else:
                A[disease].add(patient_id)
    return A, X

def FindExp(data):
    deadly_diseases = set()
    for patient_id, (status, diseases) in data.items():
        if status == 1:
            deadly_diseases.update(diseases)
    return deadly_diseases

def MakeGraph(nodes_set, edge_list, lb, A):
    """Now filters nodes exactly as in the original CBB paper (only diseases with >= lb patients)"""
    valid_nodes = {d for d in nodes_set if len(A.get(d, set())) >= lb}
    G = nx.Graph()
    G.add_nodes_from(valid_nodes)
    with open(edge_list, 'r') as f:
        for line in f:
            source, target, _ = line.split()
            if source in valid_nodes and target in valid_nodes:
                # also keep edge only if the pair is ℓ-frequent (optional but consistent with original CBB)
                if len(A[source] & A[target]) >= lb:
                    G.add_edge(source, target)
    return G

### Combinatorial Branch-and-Bound (exact match to paper pseudocode) ###
def CBB_MMRCP(graph, A, X, lb):
    global calls, BestSol, BestObj

    def DfsBB(C, L, CurrentDen, CurrentNum):
        global calls, BestSol, BestObj
        calls += 1

        if not L:
            return

        local_L = L.copy()
        for v in list(local_L):
            NewDen = CurrentDen & A.get(v, set())
            NewNum = CurrentNum & A.get(v, set())

            # Exactly the paper's pruning condition
            if len(NewDen) >= lb and len(NewNum) / lb > BestObj:
                current_mu = len(NewNum) / len(NewDen)
                if current_mu > BestObj:
                    BestObj = current_mu
                    BestSol = C | {v}

                # recurse
                newL = local_L & set(graph.neighbors(v))
                DfsBB(C | {v}, newL, NewDen, NewNum)

            local_L.remove(v)

    V = set(graph.nodes())

    # === Initialization ===
    if not V:
        return set(), 0.0

    # best singleton (now guaranteed ℓ-frequent because of filtered graph)
    best_v = max(V, key=lambda v: len(A.get(v, set()) & X) / len(A.get(v, set())))
    BestSol = {best_v}
    BestObj = len(A[best_v] & X) / len(A[best_v])

    # initial DFS call (empty clique, all vertices, all patients, expired patients)
    all_patients = set.union(*[A[v] for v in V]) if V else set()
    DfsBB(set(), V, all_patients, X)

    return BestSol, BestObj


### Main Function ###
def main():
    name = 'sample_1000_seed100'
    with open(f'{name}.json', 'r') as f:
        data = json.load(f)

    A, X = DisDict(data)

    start_pp = time.time()
    deadly_diseases = FindExp(data)
    edge_list = "ochiai_network.edgelist"
    lb = 100

    # Graph is now filtered (same as original CBB paper)
    original_graph = MakeGraph(deadly_diseases, edge_list, lb, A)

    pptime = time.time() - start_pp
    print("Preparation Time:", pptime)

    # reset globals
    global calls, BestSol, BestObj
    calls = 0
    BestSol = set()
    BestObj = 0.0

    start_solve = time.time()
    best_clique, best_mu = CBB_MMRCP(original_graph, A, X, lb)
    stime = time.time() - start_solve

    print("\n=== CBB Results ===")
    print("Best Clique :", sorted(best_clique))
    print("Best Mortality Rate :", round(best_mu, 6))
    print("#Recursive Calls (#nodes) :", calls)
    print("Solving Time :", round(stime, 4), "seconds")

    # Write output
    with open(f'CBB_{name}_lb{lb}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Preparation Time:", pptime])
        csvwriter.writerow(["Solving Time:", stime])
        csvwriter.writerow(["#Recursive Calls:", calls])
        csvwriter.writerow([])
        csvwriter.writerow(['Best Clique', 'Mu'])
        csvwriter.writerow([', '.join(sorted(best_clique)), best_mu])

if __name__ == "__main__":
    main()
