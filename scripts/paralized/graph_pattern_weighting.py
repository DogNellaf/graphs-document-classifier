import os
import copy
import pickle
import shutil
from os.path import join
import networkx as nx
import ray
import base_functions as bf

@ray.remote(num_cpus=0.25)
def calculate_penalty_for_graph(graph, pattern):
    multiple_subsumption = bf.multiple_subsumption_check(graph, pattern['subgraphs'])
    
    if multiple_subsumption == 1 and len(pattern['subgraphs']) != 0:
        return (
            bf.find_size(pattern['subgraphs']),
            1 / len(graph.nodes()),
            1 / abs(bf.find_maximal_degree(pattern['subgraphs']) - bf.find_maximal_degree([graph])),
            1,
            len(pattern['subgraphs'])
        )
    return (0, 0, 0, 0, 0)

@ray.remote
def calculate_penalty(pattern, graphs):
    penalty_refs = [calculate_penalty_for_graph.remote(graph, pattern) for graph in graphs]
    penalties = ray.get(penalty_refs)
    return map(sum, zip(*penalties))

@ray.remote
def load_graph(filename):
    return nx.read_gml(filename)

@ray.remote
def save_pickle(data, filename, output_path):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.move(filename, output_path)

@ray.remote
def process_negative_class(prefix, class_name, negative_class_name, mode, graphs, weighted_graph_patterns, edge_penalties):
    filename = join(prefix, negative_class_name, negative_class_name + '_' + mode + '.pickle')
    with open(filename, 'rb') as f:
        graph_patterns = pickle.load(f)

    penalty_refs = [calculate_penalty.remote(pattern, graphs) for pattern in graph_patterns]
    penalty_results = ray.get(penalty_refs)

    for pattern, (penalty_1, penalty_2, penalty_3, penalty_5, penalty_6) in zip(graph_patterns, penalty_results):
        relevant_index = next((i for i, graph_pattern in enumerate(weighted_graph_patterns) if graph_pattern['id'] == pattern['id']), None)

        if relevant_index is None:
            weighted_graph_patterns.append({
                'id': pattern['id'],
                'supports': pattern['supports'],
                'subgraphs': pattern['subgraphs'],
                'extent': pattern['extent'],
                'baseline_penalty': 1,
                'penalty_1': penalty_1,
                'penalty_2': penalty_2,
                'penalty_3': penalty_3,
                'penalty_5': len(pattern['extent']),
                'penalty_6': penalty_6
            })
        else:
            weighted_graph_patterns[relevant_index]['penalty_1'] += penalty_1
            weighted_graph_patterns[relevant_index]['penalty_2'] += penalty_2
            weighted_graph_patterns[relevant_index]['penalty_3'] += penalty_3
            weighted_graph_patterns[relevant_index]['penalty_5'] += penalty_5
            weighted_graph_patterns[relevant_index]['penalty_6'] += penalty_6
        
        for subgraph in pattern['subgraphs']:
            for node1, node2, data in subgraph.edges(data=True):
                edge_penalties[(node1, node2, data['label'])] = edge_penalties.get((node1, node2, data['label']), 0) + penalty_5

    print(f"{mode} {negative_class_name} for class {class_name} finished.")

@ray.remote
def process_class(prefix, class_name, classes, weighted_graph_patterns, mode, edge_penalties):
    graph_data_prefix = join(prefix, class_name)
    graph_file_names = [join(graph_data_prefix, filename) for filename in os.listdir(graph_data_prefix) if '_train_' in filename and filename.endswith('.gml')]
    
    graph_refs = [load_graph.remote(filename) for filename in graph_file_names]
    graphs = ray.get(graph_refs)
    
    negative_classes = [c for c in classes if c != class_name]
    
    future_refs = [process_negative_class.remote(prefix, class_name, nc, mode, graphs, weighted_graph_patterns, edge_penalties) for nc in negative_classes]
    ray.get(future_refs)
    
    print(f"Class {class_name} training graphs finished.")

@ray.remote
def graph_pattern_weight_calculate(dataset, classes, prefix, mode):
    weighted_graph_patterns = []
    edge_penalties = {}
    
    class_refs = [process_class.remote(prefix, class_name, classes, weighted_graph_patterns, mode, edge_penalties) for class_name in classes]
    ray.get(class_refs)
    
    for class_name in classes:
        relevant_patterns = [gp for gp in weighted_graph_patterns if class_name in gp['id']]
        filename = class_name + '_weighted_' + mode + '.pickle'
        save_pickle.remote(relevant_patterns, filename, join(prefix, class_name, filename))

    save_pickle.remote(edge_penalties, dataset + '_' + mode + '_edge_penalties.pickle', join(prefix, dataset + '_' + mode + '_edge_penalties.pickle'))

@ray.remote
def weight_graph_pattern(dataset, classes, prefix, mode):
    modes = ['concepts', 'equivalence_classes', 'frequent_subgraphs'] if mode == 'all' else [mode]
    ray.get([graph_pattern_weight_calculate.remote(dataset, classes, prefix, m) for m in modes])