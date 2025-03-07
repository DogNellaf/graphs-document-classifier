import os
import re
import json
import logging
import pickle
import numpy as np
import pandas as pd
import ray
import networkx as nx
import torch
from itertools import chain
from sklearn.metrics import classification_report
import base_functions as bf

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

@ray.remote(num_cpus=2)
def get_test_graphs(prefix, class_name):
    """Загрузка тестовых графов с валидацией имен файлов"""
    path = os.path.join(prefix, class_name)
    if not os.path.isdir(path):
        logging.error(f"Directory not found: {path}")
        return []

    files = []
    pattern = re.compile(r'_test_graph_(\d+)\.gml$')
    
    try:
        for f in os.listdir(path):
            full_path = os.path.join(path, f)
            if os.path.isfile(full_path) and f.endswith('.gml'):
                match = pattern.match(f)
                if match:
                    try:
                        files.append((int(match.group(1)), full_path))
                    except ValueError:
                        logging.warning(f"Invalid file number: {f}")
    except Exception as e:
        logging.error(f"Directory read error: {str(e)}")
        return []

    files.sort(key=lambda x: x[0])
    return [nx.read_gml(fp) for _, fp in files]

@ray.remote(num_cpus=1)
def load_weighted_patterns(prefix, class_name, mode):
    """Загрузка весовых шаблонов"""
    path = os.path.join(prefix, class_name, f"{class_name}_weighted_{mode}.pickle")
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Pattern load error: {str(e)}")
        return []

@ray.remote(num_gpus=0.5, max_calls=100)
def compute_graph_score_batch(graph_batch, patterns, edge_penalties, penalty):
    """Вычисление оценок для батча графов"""
    try:
        device = torch.device('cuda')
        batch_scores = []
        
        preprocessed = []
        for p in patterns:
            try:
                preprocessed.append({
                    'subgraphs': p['subgraphs'],
                    'weight': bf.find_graph_pattern_weight(p),
                    'edges': [list(sg.edges(data=True)) for sg in p['subgraphs']],
                    'penalty': p.get(penalty, 1.0)
                })
            except Exception as e:
                logging.error(f"Pattern processing error: {str(e)}")

        for graph in graph_batch:
            valid = []
            for p in preprocessed:
                if bf.multiple_subsumption_check(graph, p['subgraphs']):
                    if penalty == 'edge_penalty':
                        edges = [e for sg in p['edges'] for e in sg]
                        penalty_val = sum(edge_penalties.get((e[0], e[1], e[2]['label'])), 0)
                    else:
                        penalty_val = p['penalty']
                    valid.append((p['weight'], penalty_val))
            
            if valid:
                weights = torch.tensor([v[0] for v in valid], device=device)
                penalties = torch.tensor([v[1] for v in valid], device=device).clamp(min=1e-6)
                batch_scores.append((weights / penalties).mean().item())
            else:
                batch_scores.append(0.0)
        
        return batch_scores
    except Exception as e:
        logging.error(f"Compute error: {str(e)}")
        return [0.0]*len(graph_batch)

@ray.remote
def process_class(class_name, patterns, edge_penalties, penalty, prefix, classes):
    """Обработка одного класса"""
    try:
        all_graphs = []
        for cls in classes:
            graphs = ray.get(get_test_graphs.remote(prefix, cls))
            all_graphs.extend(graphs or [])
        
        if not all_graphs:
            return []

        batch_size = min(16, len(all_graphs))
        futures = [
            compute_graph_score_batch.remote(
                all_graphs[i:i+batch_size], 
                patterns, 
                edge_penalties, 
                penalty
            )
            for i in range(0, len(all_graphs), batch_size)
        ]
        
        return [score for future in futures for score in ray.get(future)]
    except Exception as e:
        logging.error(f"Class processing error: {str(e)}")
        return []

@ray.remote
def process_single_penalty(penalty, dataset, classes, prefix, mode, edge_penalties):
    """Обработка одного типа штрафа"""
    try:
        patterns_data = {
            cls: ray.get(load_weighted_patterns.remote(prefix, cls, mode))
            for cls in classes
        }

        futures = {
            cls: process_class.remote(
                cls, 
                patterns, 
                edge_penalties, 
                penalty, 
                prefix, 
                classes
            )
            for cls, patterns in patterns_data.items() if patterns
        }

        class_scores = {
            cls: ray.get(future)
            for cls, future in futures.items()
        }

        y_true = []
        y_pred = []
        for cls, scores in class_scores.items():
            y_true.extend([cls]*len(scores))
            y_pred.extend([cls if s == max(scores) else "" for s in scores])
        
        return {
            f"{penalty}_{mode}_macro-f1": classification_report(
                y_true, y_pred, 
                target_names=class_scores.keys(), 
                output_dict=True
            )["macro avg"]["f1-score"]
        }
    except Exception as e:
        logging.error(f"Penalty processing error: {str(e)}")
        return {}

@ray.remote(num_gpus=1)
def score_graph_pattern(dataset, classes, prefix, mode="all"):
    """Стартовая функция с оригинальными параметрами"""
    try:
        results = {}

        penalties = [
            'baseline_penalty', 'penalty_1', 'penalty_2',
            'penalty_3', 'edge_penalty', 'penalty_5', 'penalty_6'
        ]

        if mode == "all":
            modes = ['concepts', 'equivalence_classes', 'frequent_subgraphs']
        else:
            modes = [mode]

        for current_mode in modes:
            path = os.path.join(prefix, f"{dataset}_{current_mode}_edge_penalties.pickle")
            with open(path, 'rb') as f:
                edge_penalties = pickle.load(f)

            futures = [
                process_single_penalty.remote(
                    penalty, dataset, classes, prefix, current_mode, edge_penalties
                )
                for penalty in penalties
            ]
            
            for future, penalty in zip(futures, penalties):
                result = ray.get(future)
                key = f"{penalty}_{current_mode}"
                results[key] = result.get(key, 0.0)

        return results

    except Exception as e:
        logging.error(f"Main error: {str(e)}")
        return {}
    finally:
        ray.shutdown()
