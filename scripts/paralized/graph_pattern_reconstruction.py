import cugraph
import cupy as cp
import pickle
import shutil
import torch
import copy
import ray

# Перенос данных на GPU
def move_to_cuda(data, dtype=torch.float32):
    return torch.tensor(data, dtype=dtype).cuda()

# Создание графа с использованием cuGraph
def graph_to_cuda_tensor(num_nodes, edges):
    # Используем cuGraph для создания графа на GPU
    g = cugraph.Graph()
    g.add_edges_from(edges)
    return g

@ray.remote
def frequent_subgraphs_builder(class_name, prefix):
    frequent_subgraphs = []
    frequent_subgraph_ele = {}
    vertex_positions = {}

    frequent_subgraphs_file_name = f"{prefix}/{class_name}/{class_name}_patterns.OUT"
    vertex_labels_file_name = f"{prefix}/{class_name}/{class_name}_vertex_labels.pickle"
    edge_labels_file_name = f"{prefix}/{class_name}/{class_name}_edge_labels.pickle"

    # Загрузка меток для вершин и рёбер
    with open(vertex_labels_file_name, 'rb') as f:
        vertex_labels_fake = pickle.load(f)
    vertex_labels = {v: k for k, v in vertex_labels_fake.items()}

    with open(edge_labels_file_name, 'rb') as f:
        edge_labels_fake = pickle.load(f)
    edge_labels = {e: k for k, e in edge_labels_fake.items()}

    # Чтение частых подграфов
    with open(frequent_subgraphs_file_name) as f:
        lines = f.read().splitlines()

    g_edges = []
    vertex_positions = {}
    frequent_subgraph_id_counter = 0
    num_nodes = 0

    for line in lines:
        words = line.split(" ")
        if words[0] == "#":
            frequent_subgraph_ele['supports'] = move_to_cuda([int(words[1])])

        if words[0] == "v":
            vertex_positions[int(words[1])] = vertex_labels[int(words[2])]
            num_nodes = max(num_nodes, int(words[1]) + 1)  # Обновляем количество вершин

        if words[0] == "e":
            edge = (edge_labels[int(words[3])][0], edge_labels[int(words[3])][1])
            g_edges.append(edge)

        if words[0] == "x":
            # Используем cuGraph для создания графа на GPU
            cuda_graph = graph_to_cuda_tensor(num_nodes, g_edges)
            frequent_subgraph_ele['subgraphs'] = [cuda_graph]
            frequent_subgraph_ele['extent'] = move_to_cuda([int(e) for e in words[1:]])
            frequent_subgraph_ele['id'] = f"{class_name}_frequent_subgraph_{frequent_subgraph_id_counter}"

            frequent_subgraphs.append(copy.deepcopy(frequent_subgraph_ele))

            g_edges = []
            vertex_positions = {}
            frequent_subgraph_ele = {}
            frequent_subgraph_id_counter += 1

    # Сохранение результатов
    frequent_subgraphs_file_name = f"{class_name}_frequent_subgraphs.pickle"
    with open(frequent_subgraphs_file_name, 'wb') as handle:
        pickle.dump(frequent_subgraphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Перемещение файла в нужную директорию
    frequent_subgraphs_path = f"{prefix}/{class_name}/{frequent_subgraphs_file_name}"
    shutil.move(frequent_subgraphs_file_name, frequent_subgraphs_path)

# Функция для восстановления графа
@ray.remote
def reconstruct_graph_pattern(classes, prefix, mode):
    for class_name in classes:
        futures = []
        if mode == 'frequent_subgraphs':
            futures = [frequent_subgraphs_builder.remote(class_name, prefix)]
        ray.get(futures)
