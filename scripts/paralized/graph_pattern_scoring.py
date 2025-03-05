import os
import pickle
import networkx as nx
import ray

from os.path import join

import base_functions as bf


@ray.remote
def penalty_calculation(graph_pattern, graphs):
    graph_pattern_penalty_1 = 0
    graph_pattern_penalty_2 = 0
    graph_pattern_penalty_3 = 0
    graph_pattern_penalty_5 = 0
    graph_pattern_penalty_6 = 0

    futures = []
    for graph in graphs:
        futures.append(
            calculate_individual_penalty.remote(graph_pattern, graph)
        )

    # Собираем все результаты параллельно
    results = ray.get(futures)

    # Суммируем все результаты
    for res in results:
        graph_pattern_penalty_1 += res[0]
        graph_pattern_penalty_2 += res[1]
        graph_pattern_penalty_3 += res[2]
        graph_pattern_penalty_5 += res[3]
        graph_pattern_penalty_6 += res[4]

    return (
        graph_pattern_penalty_1,
        graph_pattern_penalty_2,
        graph_pattern_penalty_3,
        graph_pattern_penalty_5,
        graph_pattern_penalty_6
    )

@ray.remote
def calculate_individual_penalty(graph_pattern, graph):
    multiple_subsumption = bf.multiple_subsumption_check(graph, graph_pattern['subgraphs'])

    penalty_1 = penalty_2 = penalty_3 = penalty_5 = penalty_6 = 0

    if multiple_subsumption == 1 and len(graph_pattern['subgraphs']) != 0:

        penalty_1 = bf.find_size(graph_pattern['subgraphs'])

        penalty_2 = 1 / len(graph.nodes())

        penalty_3 = 1 / abs(
            bf.find_maximal_degree(graph_pattern['subgraphs']) - bf.find_maximal_degree([graph])
        )

        penalty_5 = 1

        penalty_6 = len(graph_pattern['subgraphs'])

    return penalty_1, penalty_2, penalty_3, penalty_5, penalty_6


@ray.remote
def load_graph(graph_file_name):
    return nx.read_gml(graph_file_name)


@ray.remote
def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


@ray.remote
def process_graph_pattern(graph_pattern, graphs):
    penalty_1, penalty_2, penalty_3, penalty_5, penalty_6 = ray.get(
        penalty_calculation.remote(graph_pattern, graphs)
    )

    return {
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

@ray.remote
def process_graph_class(weighted_graph_patterns, edge_penalties, classes, prefix, mode, class_name):
    graph_data_prefix = join(prefix, class_name)
    graph_filenames = [
        join(graph_data_prefix, filename)
        for filename in os.listdir(graph_data_prefix)
        if '_train_' in filename and filename.endswith('.gml')
    ]

    # Параллельная загрузка всех графов
    graph_futures = [load_graph.remote(fname) for fname in graph_filenames]
    graphs = ray.get(graph_futures)

    negative_classes = [cls for cls in classes if cls != class_name]

    for negative_class_name in negative_classes:
        filename = f"{negative_class_name}_{mode}.pickle"
        graph_patterns_file_name = join(prefix, negative_class_name, filename)

        with open(graph_patterns_file_name, 'rb') as f:
            graph_patterns = pickle.load(f)

        # Параллельная обработка паттернов графов
        pattern_futures = [
            process_graph_pattern.remote(graph_pattern, graphs)
            for graph_pattern in graph_patterns
        ]

        weighted_patterns = ray.get(pattern_futures)
        weighted_graph_patterns.extend(weighted_patterns)

        # Параллельное обновление edge_penalties
        edge_updates = ray.get([
            update_edge_penalties.remote(graph_pattern, edge_penalties)
            for graph_pattern in graph_patterns
        ])

        for update in edge_updates:
            edge_penalties.update(update)

    print(f"Class {class_name} training graphs finished.")

@ray.remote
def update_edge_penalties(gp, edge_penalties):
    return {
        (node1, node2, data['label']):
        edge_penalties.get((node1, node2, data['label']), 0) + 1
        for subgraph in gp['subgraphs']
        for node1, node2, data in subgraph.edges(data=True)
    }

def graph_pattern_score_calculation(dataset, classes, prefix, mode):
    weighted_graph_patterns = []
    edge_penalties = {}

    # Параллельная обработка классов
    futures = [
        process_graph_class.remote(
            weighted_graph_patterns,
            edge_penalties,
            classes,
            prefix,
            mode,
            class_name
        )
        for class_name in classes
    ]

    ray.get(futures)

    save_weighted_patterns(dataset, classes, prefix, mode, weighted_graph_patterns, edge_penalties)


def save_weighted_patterns(dataset, classes, prefix, mode, weighted_graph_patterns, edge_penalties):
    # Параллельное сохранение данных
    save_futures = [
        save_pickle.remote(
            join(prefix, class_name, f"{class_name}_weighted_{mode}.pickle"),
            [
                graph_pattern
                for graph_pattern
                in weighted_graph_patterns
                if class_name in graph_pattern['id']
            ]
        )
        for class_name in classes
    ]

    save_futures.append(save_pickle.remote(
        join(prefix, f"{dataset}_{mode}_edge_penalties.pickle"),
        edge_penalties
    ))

    ray.get(save_futures)

def graph_pattern_scoring_iterator(dataset, classes, prefix, mode):
    if mode == 'all':
        for m in ['concepts', 'equivalence_classes', 'frequent_subgraphs']:
            graph_pattern_score_calculation(dataset, classes, prefix, m)
    else:
        graph_pattern_score_calculation(dataset, classes, prefix, mode)
