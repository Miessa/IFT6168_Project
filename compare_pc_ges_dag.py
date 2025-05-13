import os
import pandas as pd
import numpy as np
from collections import Counter
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from graphviz import Digraph

from preprocessing import preprocess_shot_data

def run_pc(data_encoded):
    for col in data_encoded.columns:
        if data_encoded[col].dtype == bool:
            data_encoded[col] = data_encoded[col].astype(int)
    data_array = csr_matrix(data_encoded.values.astype(float)).toarray()
    cg = pc(data_array, alpha=0.1, stable=True, max_cond_vars=2, verbose=True)
    return cg, data_encoded.columns.tolist()

def run_ges(data_encoded):
    for col in data_encoded.columns:
        if data_encoded[col].dtype == bool:
            data_encoded[col] = data_encoded[col].astype(int)
    data_array = data_encoded.values.astype(float)
    cg = ges(data_array, score_func='local_score_BDeu', maxP=None, parameters=None)
    return cg, data_encoded.columns.tolist()

def compare_pc_ges(df):
    required_columns = ['distance_to_net', 'shot_angle', 'emptyNetHome', 'emptyNetAway', 
                        'powerplayHome', 'powerplayAway', 'shotType', 'result']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    sample_size = 200000
    num_samples = 5
    random_seeds = [42, 123, 456, 789, 101]
    frac = sample_size / len(df)

    le_shotType = LabelEncoder()
    samples = []
    for i, seed in enumerate(random_seeds, 1):
        print(f"\nSampling ({i}) with random_state={seed}")
        sample = df[required_columns].copy()
        sample['shotType'] = le_shotType.fit_transform(sample['shotType'])
        sample = sample.groupby('result').sample(frac=frac, random_state=seed)
        samples.append(sample)

    print("\nRunning PC...")
    all_pc_edges = []
    pc_results = Parallel(n_jobs=4)(delayed(run_pc)(s) for s in samples)
    for cg, columns in pc_results:
        for edge in cg.G.get_graph_edges():
            if "-->" in str(edge):
                u, v = str(edge).split(" --> ")
                u_name, v_name = columns[int(u.replace("X", "")) - 1], columns[int(v.replace("X", "")) - 1]
                if v_name == 'result' or (v_name in ['shot_angle', 'distance_to_net'] and u_name != 'result'):
                    all_pc_edges.append((u_name, v_name))

    print("\nRunning GES...")
    all_ges_edges = []
    ges_results = Parallel(n_jobs=4)(delayed(run_ges)(s) for s in samples)
    for cg, columns in ges_results:
        graph = cg['G']
        for i in range(len(graph.nodes)):
            for j in range(len(graph.nodes)):
                if graph.graph[i, j] == 1:
                    u_name, v_name = columns[i], columns[j]
                    if v_name == 'result' or (v_name in ['shot_angle', 'distance_to_net'] and u_name != 'result'):
                        all_ges_edges.append((u_name, v_name))

    forbidden_edges = [
        ('powerplayHome', 'powerplayAway'), ('powerplayAway', 'powerplayHome'),
        ('shotType', 'powerplayHome'), ('shotType', 'powerplayAway'),
        ('emptyNetHome', 'powerplayAway'), ('emptyNetAway', 'powerplayHome'),
        ('shotType', 'distance_to_net'), ('shot_angle', 'powerplayHome'),
        ('shot_angle', 'powerplayAway'), ('powerplayHome', 'shot_angle'),
        ('powerplayAway', 'shot_angle'), ('emptyNetHome', 'shot_angle'),
        ('emptyNetAway', 'shot_angle'), ('shot_angle', 'shotType'),
        ('powerplayAway', 'shotType'), ('powerplayHome', 'shotType'),
        ('shot_angle', 'distance_to_net'), ('emptyNetAway', 'powerplayAway'),
        ('powerplayAway', 'distance_to_net'), ('distance_to_net', 'shot_angle'),
        ('distance_to_net', 'emptyNetHome'), ('powerplayAway', 'emptyNetAway'),
        ('emptyNetHome', 'powerplayHome')
    ]

    pc_counts = Counter(all_pc_edges)
    pc_consensus = [(u, v) for (u, v), c in pc_counts.items() if c >= 4 and (u, v) not in forbidden_edges]
    print("\nPC Consensus Edges:", pc_consensus)

    ges_counts = Counter(all_ges_edges)
    ges_consensus = [(u, v) for (u, v), c in ges_counts.items() if c >= 4 and (u, v) not in forbidden_edges]
    print("\nGES Consensus Edges:", ges_consensus)

    intersection = len(set(pc_consensus) & set(ges_consensus))
    union = len(set(pc_consensus) | set(ges_consensus))
    jaccard = intersection / union if union > 0 else 0
    print(f"\nJaccard Similarity PC vs GES: {jaccard:.4f}")
    print("Shared edges:", set(pc_consensus) & set(ges_consensus))
    print("Only PC:", set(pc_consensus) - set(ges_consensus))
    print("Only GES:", set(ges_consensus) - set(pc_consensus))

    hybrid = list(set([
        ('distance_to_net', 'result'), ('shot_angle', 'result'),
        ('emptyNetHome', 'result'), ('emptyNetAway', 'result'),
        ('powerplayHome', 'result'), ('powerplayAway', 'result'),
        ('shotType', 'result'), ('shotType', 'shot_angle')
    ]) & set(pc_consensus + ges_consensus))

    print("\nHybrid DAG Edges:", hybrid)

    def draw_dag(edges, name):
        dot = Digraph(comment=name)
        dot.attr(rankdir='TB', nodesep='0.8', ranksep='1.2')
        nodes = ['emptyNetHome', 'emptyNetAway', 'powerplayHome', 'powerplayAway',
                 'distance_to_net', 'shot_angle', 'shotType', 'result']
        for node in nodes:
            if node in ['distance_to_net', 'shot_angle']:
                dot.node(node, node, shape='box', style='filled', fillcolor='lightcoral')
            elif node == 'result':
                dot.node(node, 'Goal', shape='ellipse', style='filled', fillcolor='lightblue')
            else:
                dot.node(node, node, shape='box')
        for u, v in edges:
            if (u, v) in [('distance_to_net', 'result'), ('shot_angle', 'result')]:
                dot.edge(u, v, penwidth='3', color='red')
            else:
                dot.edge(u, v)
        out_path = os.path.join(os.getcwd(), name)
        dot.render(out_path, format='png', view=True)

    draw_dag(pc_consensus, 'hockey_dag_pc_consensus')
    draw_dag(ges_consensus, 'hockey_dag_ges_consensus')
    draw_dag(hybrid, 'hockey_dag_hybrid')

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = preprocess_shot_data(start_season=2016, final_season=2023)
    if df is not None:
        compare_pc_ges(df)
