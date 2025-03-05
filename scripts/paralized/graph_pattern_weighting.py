import os
import copy
import pickle
import shutil

from os.path import join

import networkx as nx
import ray

import base_functions as bf

@ray.remote
def calculate_penalty(graph_pattern, graphs):
    penalty_1 = 0
    penalty_2 = 0
    penalty_3 = 0
    penalty_5 = 0
    penalty_6 = 0

    for graph in graphs:
        multiple_subsumption = bf.multiple_subsumption_check(graph, graph_pattern['subgraphs'])

        if multiple_subsumption == 1 and len(graph_pattern['subgraphs']) != 0:
            penalty_1 += bf.find_size(graph_pattern['subgraphs'])
            penalty_2 += 1/len(graph.nodes())
            penalty_3 += 1/abs(bf.find_maximal_degree(graph_pattern['subgraphs']) - bf.find_maximal_degree([graph]))
            penalty_5 += 1
            penalty_6 += len(graph_pattern['subgraphs'])

    return penalty_1, penalty_2, penalty_3, penalty_5, penalty_6


@ray.remote
def graph_pattern_weight_calculate(dataset, classes, prefix, mode):

    weighted_graph_patterns = []
    edge_penalties = {}

    for class_name in classes:
        graph_data_prefix = join(prefix, class_name)
        graph_file_names = [
            join(graph_data_prefix, filename)
            for filename in os.listdir(graph_data_prefix)
            if '_train_' in filename and filename.endswith('.gml')
        ]

        graphs = [
            nx.read_gml(filename) for filename in graph_file_names
        ]

        negative_classes = [
            negative_class_name_temp
            for negative_class_name_temp in classes
            if negative_class_name_temp != class_name
        ]

        for negative_class_name in negative_classes:
            graph_patterns_file_name = join(prefix, negative_class_name, negative_class_name + '_' + mode + '.pickle')

            with open(graph_patterns_file_name, 'rb') as f:
                graph_patterns = pickle.load(f)

            for pattern in graph_patterns:
                penalty_1, penalty_2, penalty_3, penalty_5, penalty_6 = ray.get(
                    calculate_penalty.remote(pattern, graphs)
                )

                relevant_index = [
                    i for i, graph_pattern
                    in enumerate(weighted_graph_patterns)
                    if graph_pattern['id'] == pattern['id']
                ]

                if not relevant_index:
                    weighted_graph_pattern = {
                        'id': pattern['id'],
                        'supports': pattern['supports'],
                        'subgraphs': pattern['subgraphs'],
                        'extent': pattern['extent'],
                        'baseline_penalty': 1,
                        'penalty_1': copy.deepcopy(penalty_1),
                        'penalty_2': copy.deepcopy(penalty_2),
                        'penalty_3': copy.deepcopy(penalty_3),
                        'penalty_5': copy.deepcopy(len(pattern['extent'])),
                        'penalty_6': copy.deepcopy(penalty_6)
                    }
                    weighted_graph_patterns.append(
                        copy.deepcopy(weighted_graph_pattern)
                    )

                else:
                    if len(relevant_index) == 1:
                        index = relevant_index[0]
                        weighted_graph_patterns[index]['penalty_1'] += copy.deepcopy(penalty_1)
                        weighted_graph_patterns[index]['penalty_2'] += copy.deepcopy(penalty_2)
                        weighted_graph_patterns[index]['penalty_3'] += copy.deepcopy(penalty_3)
                        weighted_graph_patterns[index]['penalty_5'] += copy.deepcopy(penalty_5)
                        weighted_graph_patterns[index]['penalty_6'] += copy.deepcopy(penalty_6)
                    else:
                        print("More than 1 relevant concept.")
                        return

                for subgraph in pattern['subgraphs']:
                    for node1, node2, data in subgraph.edges(data=True):

                        if (node1, node2, data['label']) in edge_penalties:
                            edge_penalties[(node1, node2, data['label'])] += penalty_5

                        else:
                            edge_penalties[(node1, node2, data['label'])] = 0

            print(f"{mode} {negative_class_name} for training document graphs of class {class_name} finished.")
        print(f"Class {class_name} training document graphs finished.")

    for class_name in classes:
        relevant_indexes = [
            index for index, graph
            in enumerate(weighted_graph_patterns)
            if class_name in graph['id']
        ]
        class_weighted_graph_patterns = []

        for index in relevant_indexes:
            class_weighted_graph_patterns.append(
                copy.deepcopy(weighted_graph_patterns[index])
            )

        filename = class_name + '_weighted_' + mode + '.pickle'

        with open(filename, 'wb') as handle:
            pickle.dump(class_weighted_graph_patterns, handle, protocol=pickle.HIGHEST_PROTOCOL)

        output_path = join(prefix, class_name, filename)
        shutil.move(filename, output_path)

    filename = dataset + '_' + mode + '_edge_penalties.pickle'

    with open(filename, 'wb') as handle:
        pickle.dump(edge_penalties, handle, protocol=pickle.HIGHEST_PROTOCOL)

    output_path = join(prefix, filename)
    shutil.move(filename, output_path)

@ray.remote
def weight_graph_pattern(dataset, classes, prefix, mode):

    futures = []
    if mode == 'all':
        futures.extend([
            graph_pattern_weight_calculate.remote(dataset, classes, prefix, 'concepts'),
            graph_pattern_weight_calculate.remote(dataset, classes, prefix, 'equivalence_classes'),
            graph_pattern_weight_calculate.remote(dataset, classes, prefix, 'frequent_subgraphs')
        ])

    else:
        futures = [
            graph_pattern_weight_calculate.remote(dataset, classes, prefix, mode)
        ]
    
    ray.get(futures)
