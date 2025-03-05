import json
import copy
import pickle
import shutil

from os.path import join

import ray
import networkx as nx

from itertools import groupby, islice

import base_functions as bf

def chunked_iterator(iterable, size):
    """Разделяет итератор на чанки заданного размера."""
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk

@ray.remote
def process_subgraph_chunk(lines, vertex_labels, edge_labels):
    """Обрабатывает чанку строк для построения подграфов."""
    subgraphs = []
    g = nx.MultiDiGraph()
    frequent_subgraph = {}

    for line in lines:
        words = line.split(" ")
        if words[0] == "#":
            if g.number_of_nodes() > 0:
                frequent_subgraph['subgraphs'] = [copy.deepcopy(g)]
                subgraphs.append(copy.deepcopy(frequent_subgraph))
                g.clear()
            frequent_subgraph = {'supports': [int(words[1])]}

        elif words[0] == "v":
            g.add_node(int(words[1]), label=vertex_labels[int(words[2])])

        elif words[0] == "e":
            g.add_edge(
                int(words[1]),
                int(words[2]),
                label=edge_labels[int(words[3])]
            )

    if g.number_of_nodes() > 0:
        frequent_subgraph['subgraphs'] = [copy.deepcopy(g)]
        subgraphs.append(copy.deepcopy(frequent_subgraph))

    return subgraphs

@ray.remote
def merge_pair(g1, g2):
    """Объединяет два графа в один."""
    combined = nx.compose(g1, g2)
    return combined

def merge_graphs_parallel(graphs):
    """Иерархически объединяет графы с использованием Ray."""
    current_graphs = graphs.copy()
    while len(current_graphs) > 1:
        new_graphs = []
        futures = []
        for i in range(0, len(current_graphs), 2):
            pair = current_graphs[i:i+2]
            if len(pair) == 1:
                new_graphs.append(pair[0])
            else:
                futures.append(merge_pair.remote(pair[0], pair[1]))

        merged = ray.get(futures)
        new_graphs.extend(merged)
        current_graphs = new_graphs

    return current_graphs[0] if current_graphs else nx.MultiDiGraph()

def frequent_subgraphs_builder(class_name, prefix, chunk_size=1000):
    """Основная функция для параллельной сборки частых подграфов."""
    frequent_subgraphs_file_name = join(prefix, class_name, f"{class_name}_patterns.OUT")
    vertex_labels_file_name = join(prefix, class_name, f"{class_name}_vertex_labels.pickle")
    edge_labels_file_name = join(prefix, class_name, f"{class_name}_edge_labels.pickle")

    with open(vertex_labels_file_name, 'rb') as f:
        vertex_labels = {v: k for k, v in pickle.load(f).items()}

    with open(edge_labels_file_name, 'rb') as f:
        edge_labels = {e: k for k, e in pickle.load(f).items()}

    with open(frequent_subgraphs_file_name, 'r') as f:
        lines = f.read().splitlines()

    futures = [
        process_subgraph_chunk.remote(chunk, vertex_labels, edge_labels)
        for chunk in chunked_iterator(lines, chunk_size)
    ]

    subgraph_chunks = ray.get(futures)
    frequent_subgraphs = [sg for chunk in subgraph_chunks for sg in chunk]

    frequent_subgraphs_file_name = join(prefix, class_name, f"{class_name}_frequent_subgraphs.pickle")
    with open(frequent_subgraphs_file_name, 'wb') as handle:
        pickle.dump(frequent_subgraphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Сохранено {len(frequent_subgraphs)} частых подграфов для класса {class_name}")
    return frequent_subgraphs

@ray.remote
def check_subsumption(i, maximal_subgraph, frequent_subgraphs_ref):
    """Проверяет субсуммирование подграфов."""
    frequent_subgraphs = ray.get(frequent_subgraphs_ref)
    subgraph_i = frequent_subgraphs[i]['subgraphs'][0]
    return bf.singular_subsumption_check(maximal_subgraph, subgraph_i)

def find_maximal_subgraph_index(indices, frequent_subgraphs):
    """Находит индекс максимального подграфа."""
    maximal_subgraph_node_count = 0
    maximal_subgraph_edge_count = 0
    maximal_subgraph_index = 0

    for i in indices:
        edge_count = len(frequent_subgraphs[i]['subgraphs'][0].edges)
        node_count = len(frequent_subgraphs[i]['subgraphs'][0].nodes)

        if node_count > maximal_subgraph_node_count:
            maximal_subgraph_node_count = node_count
            maximal_subgraph_index = i

        elif node_count == maximal_subgraph_node_count:
            singular_subsumption = bf.singular_subsumption_check(
                frequent_subgraphs[maximal_subgraph_index]['subgraphs'][0],
                frequent_subgraphs[i]['subgraphs'][0]
            )


            if singular_subsumption == 0 and edge_count > maximal_subgraph_edge_count:
                maximal_subgraph_node_count = node_count
                maximal_subgraph_edge_count = edge_count
                maximal_subgraph_index = i
    return maximal_subgraph_index

def filter_frequent_subgraphs(indices, frequent_subgraphs):
    """Фильтрует подграфы с параллельной проверкой субсуммирования."""
    filtered_indices = []
    while len(indices) > 0:
        maximal_subgraph_index = find_maximal_subgraph_index(indices, frequent_subgraphs)
        filtered_indices.append(maximal_subgraph_index)
        indices.remove(maximal_subgraph_index)

        maximal_subgraph = frequent_subgraphs[maximal_subgraph_index]['subgraphs'][0]
        fs_ref = ray.put(frequent_subgraphs)
        futures = [
            check_subsumption.remote(i, maximal_subgraph, fs_ref) for i in indices
        ]
        results = ray.get(futures)

        bad_indices = [
            i for i, res in zip(indices, results) if res == 1
        ]
        indices = [
            index for index in indices if index not in bad_indices
        ]
    return filtered_indices

@ray.remote
def process_ext_node(ext, frequent_subgraphs_ref, class_name, ext_index):
    """Обрабатывает узел решетки для построения концептов."""
    # frequent_subgraphs = ray.get(frequent_subgraphs_ref)
    frequent_subgraphs = frequent_subgraphs_ref
    concept_ele = {}
    print(frequent_subgraphs)
    matching_indices = [
        f_index for f_index, fs in enumerate(frequent_subgraphs)
        if set(ext['Ext']['Inds']).issubset(list(map(int, fs['extent'])))
    ]

    filtered_indices = filter_frequent_subgraphs(matching_indices, frequent_subgraphs)

    concept_ele['subgraphs'] = [
        copy.deepcopy(frequent_subgraphs[m]['subgraphs'][0]) for m in filtered_indices
    ]

    concept_ele['supports'] = [
        copy.deepcopy(frequent_subgraphs[m]['supports'][0]) for m in filtered_indices
    ]

    concept_ele['extent'] = ext['Ext']['Inds']
    concept_ele['id'] = f"{class_name}_concept_{ext_index}"
    return concept_ele

def concepts_builder(class_name, prefix):
    """Строит концепты с использованием параллельных задач."""
    frequent_subgraphs_path = join(prefix, class_name, f"{class_name}_frequent_subgraphs.pickle")
    with open(frequent_subgraphs_path, 'rb') as f:
        frequent_subgraphs = pickle.load(f)

    lattice_path = join(prefix, class_name, f"{class_name}_lattice.json")
    with open(lattice_path, 'r') as f:
        lattice = json.load(f)

    frequent_subgraphs_ref = ray.put(frequent_subgraphs)
    futures = [
        process_ext_node.remote(ext, frequent_subgraphs_ref, class_name, ext_index)
        for ext_index, ext in enumerate(lattice[1]['Nodes'])
    ]
    concepts = ray.get(futures)

    concepts_file_name = f"{class_name}_concepts.pickle"
    with open(concepts_file_name, 'wb') as handle:
        pickle.dump(concepts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    path = join(prefix, class_name, concepts_file_name)
    shutil.move(concepts_file_name, path)

@ray.remote
def process_equivalence_group(group, class_name, group_id):
    """Обрабатывает группу эквивалентности."""
    return {
        'id': f"{class_name}_equivalence_class_{group_id}",
        'extent': group[0]['extent'],
        'subgraphs': [
            copy.deepcopy(node['subgraphs'][0]) for node in group
        ],
        'supports': [
            copy.deepcopy(node['supports'][0]) for node in group
        ]
    }

def equivalence_classes_builder(class_name, prefix):
    """Строит классы эквивалентности с параллельной обработкой."""
    path = join(prefix, class_name, f"{class_name}_frequent_subgraphs.pickle")
    with open(path, 'rb') as f:
        frequent_subgraphs = pickle.load(f)

    sorted_fs = sorted(frequent_subgraphs, key=lambda x: x['extent'])
    groups = [
        list(g) for k, g in groupby(sorted_fs, key=lambda x: x['extent'])
    ]

    futures = [
        process_equivalence_group.remote(g, class_name, i) for i, g in enumerate(groups)
    ]
    equivalence_classes = ray.get(futures)

    eq_file_name = f"{class_name}_equivalence_classes.pickle"
    with open(eq_file_name, 'wb') as handle:
        pickle.dump(equivalence_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    path = join(prefix, class_name, eq_file_name)
    shutil.move(eq_file_name, path)

@ray.remote
def process_class(class_name, prefix, mode):
    """Обрабатывает класс в параллельном режиме."""

    if mode == 'frequent_subgraphs':
        frequent_subgraphs_builder(class_name, prefix)

    elif mode == 'concepts':
        concepts_builder(class_name, prefix)

    elif mode == 'equivalence_classes':
        equivalence_classes_builder(class_name, prefix)

    elif mode == 'all':
        frequent_subgraphs_builder(class_name, prefix)
        concepts_builder(class_name, prefix)
        equivalence_classes_builder(class_name, prefix)

    return True

def graph_pattern_reconstruction_iterator(classes, prefix, mode):
    """Параллельно обрабатывает все классы."""
    futures = [
        process_class.remote(class_, prefix, mode) for class_ in classes
    ]
    ray.get(futures)
