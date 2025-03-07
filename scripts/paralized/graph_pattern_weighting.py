import os
import pickle
import shutil
from os.path import join
from collections import defaultdict
import networkx as nx
import ray
import multiprocessing
import base_functions as bf

# Конфигурация
BATCH_SIZE = 300  # Увеличиваем размер батча
PATTERN_BATCH = 20  # Батчим паттерны
NUM_CPUS = multiprocessing.cpu_count()


def batch_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

@ray.remote(num_cpus=0.3, max_retries=2)
def mega_batch_processor(graphs_batch, patterns_batch):
    results = []
    for pattern in patterns_batch:
        p_total = [0]*5
        for graph in graphs_batch:
            multiple_subsumption = bf.multiple_subsumption_check(graph, pattern['subgraphs'])
            if multiple_subsumption and pattern['subgraphs']:
                p_total[0] += bf.find_size(pattern['subgraphs'])
                p_total[1] += 1 / len(graph.nodes())
                p_total[2] += 1 / abs(bf.find_maximal_degree(pattern['subgraphs']) - bf.find_maximal_degree([graph]))
                p_total[3] += 1
                p_total[4] += len(pattern['subgraphs'])
        
        edge_penalties = defaultdict(int)
        for subg in pattern['subgraphs']:
            for u, v, data in subg.edges(data=True):
                edge_key = (u, v, data['label'])
                edge_penalties[edge_key] += p_total[3]
        
        results.append((pattern, p_total, dict(edge_penalties)))
    return results

@ray.remote(num_cpus=1)
def class_processor(prefix, class_name, classes, mode):
    graph_dir = join(prefix, class_name)
    graph_files = [join(graph_dir, f) for f in os.listdir(graph_dir)
                 if '_train_' in f and f.endswith('.gml')]
    
    # Загрузка всех графов в одном процессе
    graphs = [nx.read_gml(f) for f in graph_files]
    
    merged_patterns = defaultdict(lambda: defaultdict(int))
    total_edges = defaultdict(int)
    
    for nc in classes:
        if nc == class_name:
            continue
            
        try:
            with open(join(prefix, nc, f"{nc}_{mode}.pickle"), 'rb') as f:
                patterns = pickle.load(f)
            
            # Обработка батчей графов и паттернов
            pattern_batches = list(batch_iterable(patterns, PATTERN_BATCH))
            graph_batches = list(batch_iterable(graphs, BATCH_SIZE))
            
            tasks = []
            for p_batch in pattern_batches:
                for g_batch in graph_batches:
                    tasks.append(mega_batch_processor.remote(g_batch, p_batch))
            
            # Параллельная обработка с ограничением количества задач
            while tasks:
                done, tasks = ray.wait(tasks, num_returns=min(50, len(tasks)))
                for result in ray.get(done):
                    for pattern, p_total, edges in result:
                        pid = pattern['id']
                        merged_patterns[pid]['id'] = pid
                        merged_patterns[pid]['supports'] = pattern['supports']
                        merged_patterns[pid]['subgraphs'] = pattern['subgraphs']
                        merged_patterns[pid]['extent'] = pattern['extent']
                        
                        merged_patterns[pid]['penalty_1'] += p_total[0]
                        merged_patterns[pid]['penalty_2'] += p_total[1]
                        merged_patterns[pid]['penalty_3'] += p_total[2]
                        merged_patterns[pid]['penalty_5'] = len(pattern['extent'])
                        merged_patterns[pid]['penalty_6'] += p_total[4]
                        
                        for edge, count in edges.items():
                            total_edges[edge] += count
        except Exception as e:
            print(f"Error processing {nc}: {str(e)}")
    
    # Сохранение результатов
    output_file = join(prefix, class_name, f"{class_name}_weighted_{mode}.pickle")
    with open(output_file, 'wb') as f:
        pickle.dump(list(merged_patterns.values()), f)
    
    return dict(total_edges)

@ray.remote
def weight_graph_pattern(dataset, classes, prefix, mode):
    modes = ['concepts', 'equivalence_classes', 'frequent_subgraphs'] if mode == 'all' else [mode]
    
    for current_mode in modes:
        print(f"Processing mode: {current_mode}")
        
        # Ограничиваем параллелизм на уровне классов
        class_tasks = [class_processor.remote(prefix, cls, classes, current_mode) for cls in classes]
        
        total_edges = defaultdict(int)
        while class_tasks:
            done, class_tasks = ray.wait(class_tasks, num_returns=1)
            for edge_data in ray.get(done):
                for edge, count in edge_data.items():
                    total_edges[edge] += count
        
        edge_file = join(prefix, f"{dataset}_{current_mode}_edge_penalties.pickle")
        with open(edge_file, 'wb') as f:
            pickle.dump(dict(total_edges), f)
    
    print("All processing completed!")