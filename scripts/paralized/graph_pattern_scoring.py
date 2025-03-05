import os
import re
import copy
import pickle
import pandas as pd
import ray
import networkx as nx

from itertools import chain
from sklearn.metrics import classification_report

import base_functions as bf

def edge_penalty_calculation(subgraphs, edge_penalties):
    edge_penalty = 0

    for subgraph in subgraphs:
        for node1, node2, data in subgraph.edges(data=True):
            edge_penalty += edge_penalties[(node1, node2, data['label'])]

    return edge_penalty


def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))


def graph_pattern_scoring(weighted_class_graph_patterns, edge_penalties, classes, prefix, penalty):

    column = []

    for class_name in classes:
        graph_data_prefix = prefix + '/' + class_name
        graph_file_names = [os.path.join(graph_data_prefix, graph_file_name) for graph_file_name in os.listdir(graph_data_prefix) if '_test_' in graph_file_name and graph_file_name.endswith('.gml')]
        graph_file_names.sort()

        graphs = [nx.read_gml(graph_file_name_temp) for graph_file_name_temp in graph_file_names]

        for graph in graphs:
            score = 0
            weighted_class_graph_pattern_counter = 0

            for weighted_class_graph_pattern in weighted_class_graph_patterns:
                weighted_class_graph_pattern_counter += 1

                multiple_subsumption = bf.multiple_subsumption_check(graph, weighted_class_graph_pattern['subgraphs'])

                if multiple_subsumption == 1:
                    weighted_class_graph_pattern_weight = bf.find_graph_pattern_weight(weighted_class_graph_pattern)

                    if penalty == 'edge_penalty':
                        weighted_class_graph_pattern_edge_penalty = edge_penalty_calculation(weighted_class_graph_pattern['subgraphs'], edge_penalties)

                        if weighted_class_graph_pattern_edge_penalty == 0:
                            score += (weighted_class_graph_pattern_weight / 1)

                        else:
                            score += (weighted_class_graph_pattern_weight / weighted_class_graph_pattern_edge_penalty)

                    else:

                        if weighted_class_graph_pattern[f'{penalty}'] == 0:
                            score += (weighted_class_graph_pattern_weight / 1)

                        else:
                            score += (weighted_class_graph_pattern_weight / weighted_class_graph_pattern[f'{penalty}'])

            column.append(copy.deepcopy(score / weighted_class_graph_pattern_counter))
    return column


def graph_pattern_classification(dataset, classes, prefix, mode):
    results = {'dataset': dataset, 'mode': mode}

    penalties = ['baseline_penalty', 'penalty_1', 'penalty_2', 'penalty_3', 'edge_penalty', 'penalty_5', 'penalty_6']

    edge_penalties_file_name = dataset + '_' + mode + '_edge_penalties.pickle'
    edge_penalties_path = prefix + '/' + edge_penalties_file_name

    with open(edge_penalties_path, 'rb') as f:
        edge_penalties = pickle.load(f)

    for penalty in penalties:

        d = {}
        y_true = []
        index_list = []

        for class_name in classes:
            weighted_class_graph_patterns_file_name = prefix + '/' + class_name + '/' + class_name + '_weighted_' + mode + '.pickle'

            with open(weighted_class_graph_patterns_file_name, 'rb') as f:
                weighted_class_graph_patterns = pickle.load(f)

            d[class_name] = graph_pattern_scoring(weighted_class_graph_patterns, edge_penalties, classes, prefix, penalty)

        for class_name in classes:
            string_pattern = fr"{class_name}_test_graph_([0-9])*"

            graph_data_prefix = prefix + '/' + class_name
            graph_file_names = [os.path.join(graph_data_prefix, graph_file_name) for graph_file_name in os.listdir(graph_data_prefix) if '_test_' in graph_file_name and graph_file_name.endswith('.gml')]
            graph_file_names.sort()

            index_list.append([re.search(string_pattern, graph_file_name_temp).group() for graph_file_name_temp in graph_file_names])

            for j in range(len(graph_file_names)):
                y_true.append(copy.deepcopy(class_name))

        index_list_2 = flatten_chain(index_list)

        classification_df = pd.DataFrame(data=d, index=index_list_2)

        max_value_index = classification_df.idxmax(axis="columns")

        y_pred = max_value_index.to_list()

        cr = classification_report(y_true=y_true, y_pred=y_pred, target_names=classes, output_dict=True)
        results[f'{penalty}_macro-averaged_f1-score'] = cr['macro avg']['f1-score']

    return results

@ray.remote
def score_graph_pattern(dataset, classes, prefix, mode):

    if mode == 'all':
        concepts_results = graph_pattern_classification(dataset, classes, prefix, 'concepts')
        print(f"Scoring for {dataset} for concepts done.")
        print(concepts_results)

        equivalence_classes_results = graph_pattern_classification(dataset, classes, prefix, 'equivalence_classes')
        print(f"Scoring for {dataset} for equivalence classes done.")
        print(equivalence_classes_results)

        frequent_subgraphs_results = graph_pattern_classification(dataset, classes, prefix, 'frequent_subgraphs')
        print(f"Scoring for {dataset} for frequent subgraphs done.")
        print(frequent_subgraphs_results)

        results = [concepts_results, equivalence_classes_results, frequent_subgraphs_results]

    else:
        results = graph_pattern_classification(dataset, classes, prefix, mode)

    return results
