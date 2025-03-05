import os
from os.path import join

import pickle
import networkx as nx
import ray

import base_functions as bf

@ray.remote
def penalty_calculation(graph_pattern, graphs):
    graph_pattern_penalty_1 = 0
    graph_pattern_penalty_2 = 0
    graph_pattern_penalty_3 = 0
    graph_pattern_penalty_5 = 0
    graph_pattern_penalty_6 = 0

    for graph in graphs:
        multiple_subsumption = bf.multiple_subsumption_check(graph, graph_pattern['subgraphs'])

        if multiple_subsumption == 1 and len(graph_pattern['subgraphs']) != 0:
            graph_pattern_penalty_1 += bf.find_size(graph_pattern['subgraphs'])
            graph_pattern_penalty_2 += 1 / len(graph.nodes())
            graph_pattern_penalty_3 += 1 / abs(
                bf.find_maximal_degree(graph_pattern['subgraphs']) - bf.find_maximal_degree([graph])
            )
            graph_pattern_penalty_5 += 1
            graph_pattern_penalty_6 += len(graph_pattern['subgraphs'])

    return (
        graph_pattern_penalty_1,
        graph_pattern_penalty_2,
        graph_pattern_penalty_3,
        graph_pattern_penalty_5,
        graph_pattern_penalty_6
    )


@ray.remote
def load_graph(path):
    return nx.read_gml(path)


@ray.remote
def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def graph_pattern_weight_calculation(dataset, classes, prefix, mode):
    weighted_graph_patterns = []
    edge_penalties = {}

    for class_name in classes:
        graph_data_prefix = join(prefix, class_name)
        graph_file_names = [
            join(graph_data_prefix, filename)
            for filename in os.listdir(graph_data_prefix)
            if '_train_' in filename and filename.endswith('.gml')
        ]

        graph_futures = [
            load_graph.remote(filename)
            for filename in graph_file_names
        ]

        graphs = ray.get(graph_futures)
        negative_classes = [
            class_name
            for class_name
            in classes
            if class_name != class_name
        ]

        penalty_futures = []
        for negative_class_name in negative_classes:
            graph_patterns_file_name = join(
                prefix,
                negative_class_name,
                f"{negative_class_name}_{mode}.pickle"
            )

            with open(graph_patterns_file_name, 'rb') as f:
                graph_patterns = pickle.load(f)

            for graph_pattern in graph_patterns:
                penalty_futures.append(
                    (
                        graph_pattern,
                        penalty_calculation.remote(graph_pattern, graphs)
                    )
                )

        for graph_pattern, penalty_future in penalty_futures:
            penalty_1, penalty_2, penalty_3, penalty_5, penalty_6 = ray.get(penalty_future)
            weighted_graph_patterns.append(
                {
                    'id': graph_pattern['id'],
                    'supports': graph_pattern['supports'],
                    'subgraphs': graph_pattern['subgraphs'],
                    'extent': graph_pattern['extent'],
                    'baseline_penalty': 1,
                    'penalty_1': penalty_1,
                    'penalty_2': penalty_2,
                    'penalty_3': penalty_3,
                    'penalty_5': len(graph_pattern['extent']),
                    'penalty_6': penalty_6,
                }
            )

            subgraphs = graph_pattern['subgraphs']
            for subgraph in subgraphs:
                for node1, node2, data in subgraph.edges(data=True):
                    edge_key = (node1, node2, data['label'])
                    edge_penalties[edge_key] = edge_penalties.get(edge_key, 0) + penalty_5

        print(f"Class {class_name} training graphs finished.")

    save_weighted_patterns(dataset, classes, prefix, mode, weighted_graph_patterns, edge_penalties)

def save_weighted_patterns(dataset, classes, prefix, mode, weighted_graph_patterns, edge_penalties):
    save_futures = []

    for class_name in classes:
        class_patterns = [
            graph_pattern
            for graph_pattern
            in weighted_graph_patterns
            if class_name in graph_pattern['id']
        ]

        class_patterns_file_name = join(prefix, class_name, f"{class_name}_weighted_{mode}.pickle")
        save_futures.append(
            save_pickle.remote(class_patterns_file_name, class_patterns)
        )

    edge_penalties_path = join(prefix, f"{dataset}_{mode}_edge_penalties.pickle")
    save_futures.append(
        save_pickle.remote(edge_penalties_path, edge_penalties)
    )

    ray.get(save_futures)


def graph_pattern_weighting_iterator(dataset, classes, prefix, mode):
    if mode == 'all':
        for m in ['concepts', 'equivalence_classes', 'frequent_subgraphs']:
            graph_pattern_weight_calculation(dataset, classes, prefix, m)

    else:
        graph_pattern_weight_calculation(dataset, classes, prefix, mode)
