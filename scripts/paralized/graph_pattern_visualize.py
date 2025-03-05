from os.path import join

import shutil
import pickle
import networkx as nx


def draw_graph(graph, graph_name):
    """
    Draw graph

    Args:
        graph (NetworkX graph): Graph to draw
        graph_name (str): Graph name

    Returns:
        str: New graph name
    """
    dot_graph = nx.nx_agraph.to_agraph(graph)
    dot_graph.layout(prog="dot")
    new_graph_name = str(graph_name) + '.png'
    dot_graph.draw(new_graph_name)

    return new_graph_name


def concepts_iterator(class_name, prefix):

    path = join(prefix, class_name, f'{class_name}_concepts.pickle')
    with open(path, 'rb') as f:
        concepts = pickle.load(f)

    for k, concept in enumerate(concepts):
        print(f"Concept {concept['id']}")
        for s, subgraph in enumerate(concepts[k]['subgraphs']):
            supports = concepts[k]['supports'][s]

            subgraph_name = concepts[k]['id'] + '_subgraph_' + str(s)
            print(f"Subgraph {subgraph_name} support:{supports}")
            new_subgraph_name = draw_graph(subgraph, subgraph_name)

            path = join(prefix, class_name, 'visualizations', new_subgraph_name)
            shutil.move(new_subgraph_name, path)


def equivalence_classes_iterator(class_name, prefix):

    path = join(prefix, class_name, f'{class_name}_equivalence_classes.pickle')
    with open(path, 'rb') as f:
        equivalence_classes = pickle.load(f)

    for _, equivalence_class in enumerate(equivalence_classes):
        equivalence_id = equivalence_class['id']
        extent = equivalence_class['extent']

        print(f"Equivalence class {equivalence_id} with extent:{extent}")
        for s, subgraph in enumerate(equivalence_class['subgraphs']):
            supports = equivalence_class['supports'][s]
            subgraph_name = equivalence_id + '_subgraph_' + str(s)

            print(f"Subgraph {subgraph_name} support:{supports}")
            new_subgraph_name = draw_graph(
                subgraph,
                subgraph_name
            )

            subgraph_path = join(prefix, class_name, 'visualizations', new_subgraph_name)
            shutil.move(new_subgraph_name, subgraph_path)


def frequent_subgraphs_iterator(class_name, prefix):

    path = join(prefix, class_name, f'{class_name}_frequent_subgraphs.pickle')
    with open(path, 'rb') as f:
        frequent_subgraphs = pickle.load(f)

    for _, frequent_subgraph in enumerate(frequent_subgraphs):
        frequent_subgraph_id = frequent_subgraph['id']
        old_frequent_subgraph_name = frequent_subgraph_id
        first_support = frequent_subgraph['supports'][0]
        first_subgraph = frequent_subgraph['subgraphs'][0]

        print(f"Frequent subgraph {frequent_subgraph_id} has support {first_support}")

        new_frequent_subgraph_name = draw_graph(
            first_subgraph,
            old_frequent_subgraph_name
        )

        path = join(prefix, class_name, 'visualizations', new_frequent_subgraph_name)
        shutil.move(new_frequent_subgraph_name, path)


def graph_iterator(class_name, prefix, graph_indices):
    for graph_index in graph_indices:
        graph_name = join(prefix, class_name, str(graph_index))
        filename = str(graph_name + '.gml')
        graph = nx.read_gml(filename)
        new_graph_name = draw_graph(graph, graph_name)

        path = join(prefix, class_name, 'visualizations', f'{graph_index}.png')
        shutil.move(new_graph_name, path)


def graph_pattern_visualize_iterator(classes, prefix, mode):

    for class_name in classes:

        if mode == 'frequent_subgraphs':
            frequent_subgraphs_iterator(class_name, prefix)

        elif mode == 'concepts':
            concepts_iterator(class_name, prefix)

        elif mode == 'equivalence_classes':
            equivalence_classes_iterator(class_name, prefix)

        elif mode == 'all':
            frequent_subgraphs_iterator(class_name, prefix)
            concepts_iterator(class_name, prefix)
            equivalence_classes_iterator(class_name, prefix)
            graph_iterator(class_name, prefix, ['g_1', 'g_2', 'g_3'])
