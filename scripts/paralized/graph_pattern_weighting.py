import os
import copy
import pickle
import shutil
import networkx as nx
import ray

import base_functions as bf


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
            graph_pattern_penalty_2 += 1/len(graph.nodes())
            graph_pattern_penalty_3 += 1/abs(bf.find_maximal_degree(graph_pattern['subgraphs']) - bf.find_maximal_degree([graph]))
            graph_pattern_penalty_5 += 1
            graph_pattern_penalty_6 += len(graph_pattern['subgraphs'])

    return graph_pattern_penalty_1, graph_pattern_penalty_2, graph_pattern_penalty_3, graph_pattern_penalty_5, graph_pattern_penalty_6


def graph_pattern_weight_calculation(dataset, classes, prefix, mode):

    weighted_graph_patterns = []
    edge_penalties = {}

    for class_name in classes:
        graph_data_prefix = prefix + '/' + class_name
        graph_file_names = [os.path.join(graph_data_prefix, graph_file_name) for graph_file_name in os.listdir(graph_data_prefix) if '_train_' in graph_file_name and graph_file_name.endswith('.gml')]

        graphs = [nx.read_gml(graph_file_name_temp) for graph_file_name_temp in graph_file_names]

        negative_classes = [negative_class_name_temp for negative_class_name_temp in classes if negative_class_name_temp != class_name]

        for negative_class_name in negative_classes:
            graph_patterns_file_name = prefix + '/' + negative_class_name + '/' + negative_class_name + '_' + mode + '.pickle'

            with open(graph_patterns_file_name, 'rb') as f:
                graph_patterns = pickle.load(f)

            for gp_ele in graph_patterns:
                gp_ele_penalty_1, gp_ele_penalty_2, gp_ele_penalty_3, gp_ele_penalty_5, gp_ele_penalty_6 = penalty_calculation(gp_ele, graphs)

                relevant_index = [gp_i for gp_i, gp_ele_temp in enumerate(weighted_graph_patterns) if gp_ele_temp['id'] == gp_ele['id']]

                if not relevant_index:
                    weighted_graph_pattern_ele = {'id': gp_ele['id'], 'supports': gp_ele['supports'], 'subgraphs': gp_ele['subgraphs'], 'extent': gp_ele['extent'],
                                                  'baseline_penalty': 1, 'penalty_1': copy.deepcopy(gp_ele_penalty_1),
                                                  'penalty_2': copy.deepcopy(gp_ele_penalty_2), 'penalty_3': copy.deepcopy(gp_ele_penalty_3),
                                                  'penalty_5': copy.deepcopy(len(gp_ele['extent'])), 'penalty_6': copy.deepcopy(gp_ele_penalty_6)}
                    weighted_graph_patterns.append(copy.deepcopy(weighted_graph_pattern_ele))

                else:
                    if len(relevant_index) == 1:
                        weighted_graph_patterns[relevant_index[0]]['penalty_1'] += copy.deepcopy(gp_ele_penalty_1)
                        weighted_graph_patterns[relevant_index[0]]['penalty_2'] += copy.deepcopy(gp_ele_penalty_2)
                        weighted_graph_patterns[relevant_index[0]]['penalty_3'] += copy.deepcopy(gp_ele_penalty_3)
                        weighted_graph_patterns[relevant_index[0]]['penalty_5'] += copy.deepcopy(gp_ele_penalty_5)
                        weighted_graph_patterns[relevant_index[0]]['penalty_6'] += copy.deepcopy(gp_ele_penalty_6)
                    else:
                        print("More than 1 relevant concept.")
                        return

                for subgraph in gp_ele['subgraphs']:
                    for node1, node2, data in subgraph.edges(data=True):

                        if (node1, node2, data['label']) in edge_penalties:
                            edge_penalties[(node1, node2, data['label'])] += gp_ele_penalty_5

                        else:
                            edge_penalties[(node1, node2, data['label'])] = 0

            print(f"{mode} {negative_class_name} for training document graphs of class {class_name} finished.")
        print(f"Class {class_name} training document graphs finished.")

    for class_name in classes:
        relevant_indices = [gp_i for gp_i, gp_ele_temp in enumerate(weighted_graph_patterns) if class_name in gp_ele_temp['id']]
        class_weighted_graph_patterns = []

        for r_i in relevant_indices:
            class_weighted_graph_patterns.append(copy.deepcopy(weighted_graph_patterns[r_i]))

        class_weighted_graph_patterns_file_name = class_name + '_weighted_' + mode + '.pickle'

        with open(class_weighted_graph_patterns_file_name, 'wb') as handle:
            pickle.dump(class_weighted_graph_patterns, handle, protocol=pickle.HIGHEST_PROTOCOL)

        class_weighted_graph_patterns_path = prefix + '/' + class_name + '/' + class_weighted_graph_patterns_file_name

        shutil.move(class_weighted_graph_patterns_file_name, class_weighted_graph_patterns_path)

    edge_penalties_file_name = dataset + '_' + mode + '_edge_penalties.pickle'

    with open(edge_penalties_file_name, 'wb') as handle:
        pickle.dump(edge_penalties, handle, protocol=pickle.HIGHEST_PROTOCOL)

    edge_penalties_path = prefix + '/' + edge_penalties_file_name

    shutil.move(edge_penalties_file_name, edge_penalties_path)

@ray.remote
def graph_pattern_weighting_iterator(dataset, classes, prefix, mode):

    if mode == 'all':
        graph_pattern_weight_calculation(dataset, classes, prefix, 'concepts')
        graph_pattern_weight_calculation(dataset, classes, prefix, 'equivalence_classes')
        graph_pattern_weight_calculation(dataset, classes, prefix, 'frequent_subgraphs')

    else:
        graph_pattern_weight_calculation(dataset, classes, prefix, mode)
