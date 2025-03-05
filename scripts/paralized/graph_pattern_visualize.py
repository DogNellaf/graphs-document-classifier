import shutil
import pickle
import networkx as nx


def draw_graph(g, graph_name):
    dot_graph = nx.nx_agraph.to_agraph(g)
    dot_graph.layout(prog="dot")
    new_graph_name = str(graph_name) + '.png'
    dot_graph.draw(new_graph_name)

    return new_graph_name


def concepts_iterator(class_name, prefix):
    concepts_file_name = prefix + '/' + class_name + '/' + class_name + '_concepts.pickle'

    with open(concepts_file_name, 'rb') as f:
        concepts = pickle.load(f)

    for k in range(len(concepts)):
        print(f"Concept {concepts[k]['id']}")
        for s in range(len(concepts[k]['subgraphs'])):
            subgraph_name = concepts[k]['id'] + '_subgraph_' + str(s)
            print(f"Subgraph {subgraph_name} support:{concepts[k]['supports'][s]}")
            new_subgraph_name = draw_graph(concepts[k]['subgraphs'][s], subgraph_name)

            subgraph_path = prefix + '/' + class_name + '/' + 'visualizations' + '/' + new_subgraph_name
            shutil.move(new_subgraph_name, subgraph_path)


def equivalence_classes_iterator(class_name, prefix):
    equivalence_classes_file_name = prefix + '/' + class_name + '/' + class_name + '_equivalence_classes.pickle'

    with open(equivalence_classes_file_name, 'rb') as f:
        equivalence_classes = pickle.load(f)

    for k in range(len(equivalence_classes)):
        print(f"Equivalence class {equivalence_classes[k]['id']} with extent:{equivalence_classes[k]['extent']}")
        for s in range(len(equivalence_classes[k]['subgraphs'])):
            subgraph_name = equivalence_classes[k]['id'] + '_subgraph_' + str(s)
            print(f"Subgraph {subgraph_name} support:{equivalence_classes[k]['supports'][s]}")
            new_subgraph_name = draw_graph(equivalence_classes[k]['subgraphs'][s], subgraph_name)

            subgraph_path = prefix + '/' + class_name + '/' + 'visualizations' + '/' + new_subgraph_name
            shutil.move(new_subgraph_name, subgraph_path)


def frequent_subgraphs_iterator(class_name, prefix):
    frequent_subgraphs_file_name = prefix + '/' + class_name + '/' + class_name + '_frequent_subgraphs.pickle'

    with open(frequent_subgraphs_file_name, 'rb') as f:
        frequent_subgraphs = pickle.load(f)

    for k in range(len(frequent_subgraphs)):
        frequent_subgraph_name = frequent_subgraphs[k]['id']
        print(f"Frequent subgraph {frequent_subgraphs[k]['id']} has support {frequent_subgraphs[k]['supports'][0]}")
        new_frequent_subgraph_name = draw_graph(frequent_subgraphs[k]['subgraphs'][0], frequent_subgraph_name)

        frequent_graph_path = prefix + '/' + class_name + '/' + 'visualizations' + '/' + new_frequent_subgraph_name
        shutil.move(new_frequent_subgraph_name, frequent_graph_path)


def graph_iterator(class_name, prefix, graph_indices):
    for graph_index in graph_indices:
        graph_name = prefix + '/' + class_name + '/' + graph_index
        graph = nx.read_gml(str(graph_name + '.gml'))
        new_graph_name = draw_graph(graph, graph_name)

        graph_path = prefix + '/' + class_name + '/' + 'visualizations' + '/' + graph_index + '.png'
        shutil.move(new_graph_name, graph_path)


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
