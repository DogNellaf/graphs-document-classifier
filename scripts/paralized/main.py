"""
Main script to start operations
"""
import time

from os.path import join
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import ray
import yaml

from utils import get_project_root
from results_aggregator import construct_results_dataframe
from graph_pattern_reconstruction import graph_pattern_reconstruction_iterator
from graph_pattern_weighting import graph_pattern_weighting_iterator
from graph_pattern_scoring import graph_pattern_scoring_iterator
from graph_pattern_visualize import graph_pattern_visualize_iterator

ray.init(num_cpus=10, num_gpus=1)

@ray.remote
def reconstruct_graph_patterns(dataset_name, classes, data_prefix, mode):
    """
    Reconstruct graph patterns

    Args:
        dataset_name (str): name of dataset
        classes (list): classes to reconstruct
        data_prefix (str): ?
        mode (str): reconstruct mode
        ('all', 'concepts', 'frequent_subgraphs', 'equivalence_classes')
    """
    print(f"Reconstructing graph patterns for {dataset_name}...")
    graph_pattern_reconstruction_iterator(classes, data_prefix, mode)


@ray.remote
def weight_graph_patterns(dataset_name, classes, data_prefix, mode):
    print(f"Weighting graph patterns for {dataset_name}...")
    graph_pattern_weighting_iterator(dataset_name, classes, data_prefix, mode)


@ray.remote
def score_graph_patterns(dataset_name, classes, data_prefix, mode):
    print(f"Scoring graph patterns for {dataset_name}...")
    return graph_pattern_scoring_iterator(dataset_name, classes, data_prefix, mode)


def do_operations(root, config, dataset='all', mode='concepts', weighting='yes', graph_pattern_reconstruction='yes'):
    futures = []

    if dataset == 'all':
        ten_newsgroups_path = join(str(root), config['ten_newsgroups_data_prefix'])
        bbcsports_path = join(str(root), config['bbcsport_data_prefix'])

        if graph_pattern_reconstruction == 'yes':
            futures.append(
                reconstruct_graph_patterns.remote(
                    'ten_newsgroups',
                    config['ten_newsgroups_classes'],
                    ten_newsgroups_path,
                    mode
                )
            )

            futures.append(
                reconstruct_graph_patterns.remote(
                    'bbcsport',
                    config['bbcsport_classes'],
                    bbcsports_path,
                    mode
                )
            )

        if weighting == 'yes':
            futures.append(
                weight_graph_patterns.remote(
                    'ten_newsgroups',
                    config['ten_newsgroups_classes'],
                    ten_newsgroups_path,
                    mode
                )
            )

            futures.append(
                weight_graph_patterns.remote(
                    'bbcsport',
                    config['bbcsport_classes'],
                    bbcsports_path,
                    mode
                )
            )

        scores = ray.get([
            score_graph_patterns.remote(
                'ten_newsgroups',
                config['ten_newsgroups_classes'],
                ten_newsgroups_path,
                mode
            ),

            score_graph_patterns.remote(
                'bbcsport',
                config['bbcsport_classes'],
                bbcsports_path,
                mode
            )
        ])

        print(scores)

        construct_results_dataframe(root, scores, config)

    ray.get(futures)
    print("Graph processing completed!")


if __name__ == '__main__':
    root = get_project_root()

    parser = ArgumentParser(
        description='Main script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=Path, default=root/'config/config.yaml',
        help='Enter the config file path.'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['all', 'ten_newsgroups', 'bbcsport'],
        help='Choose the dataset.'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'concepts', 'frequent_subgraphs', 'equivalence_classes'],
        help='Choose the operation mode.'
    )

    parser.add_argument(
        '--weighting',
        type=str,
        default='yes',
        choices=['yes', 'no'],
        help='Choose whether to weight graph patterns.'
    )

    parser.add_argument(
        '--graph_pattern_reconstruction',
        type=str,
        default='yes',
        choices=['yes', 'no'],
        help='Choose whether to reconstruct graph patterns.'
    )

    parser.add_argument(
        '--visualization',
        type=str,
        default='no',
        choices=['yes', 'no'],
        help='Choose whether to visualize graph patterns reconstruction using a toy example.'
    )

    args, unknown = parser.parse_known_args()

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    start_time = time.time()

    do_operations(root=root,
                  config=config,
                  dataset=args.dataset,
                  mode=args.mode,
                  weighting=args.weighting,
                  graph_pattern_reconstruction=args.graph_pattern_reconstruction)

    end_time = time.time()
    print("Execution time:", end_time - start_time)
