import ray
import yaml
import pickle
import time
import multiprocessing
import shutil
import logging
import torch
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from os.path import join

from utils import get_project_root
from results_aggregator import construct_results_dataframe
from graph_pattern_reconstruction import reconstruct_graph_pattern
from graph_pattern_weighting import weight_graph_pattern
from graph_pattern_scoring import score_graph_pattern
from graph_pattern_visualize import graph_pattern_visualize_iterator

ray.init(
    logging_level=logging.DEBUG,
    num_cpus=10,
    num_gpus=torch.cuda.device_count(),
    # _system_config={
    #     "max_io_workers": 20
    # },
    log_to_driver=True
)
torch.backends.cudnn.benchmark = True

print(
    ray.available_resources()
)

def save_result(root, filename, dir, results):

    filename = 'ten_newsgroups_results.pickle'

    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ten_newsgroups_results_path = join(str(root), dir, filename)
    shutil.move(filename, ten_newsgroups_results_path)

def do_operations(root, config, dataset='all', mode='concepts', weighting='yes', graph_pattern_reconstruction='yes', visualization='no'):

    start = time.time()

    if dataset == 'all':
        if graph_pattern_reconstruction == 'yes':
            print("Reconstructing graph patterns...")
            futures = [
                reconstruct_graph_pattern.remote(
                    config['ten_newsgroups_classes'],
                    join(str(root), config['ten_newsgroups_data_prefix']),
                    mode
                ),

                reconstruct_graph_pattern.remote(
                    config['bbcsport_classes'],
                    join(str(root), config['bbcsport_data_prefix']),
                    mode
                )
            ]

            ray.get(futures)

        if weighting == 'yes':
            print("Weighting graph patterns...")
            futures = [
                weight_graph_pattern.remote(
                    'ten_newsgroups',
                    config['ten_newsgroups_classes'],
                    join(str(root), config['ten_newsgroups_data_prefix']),
                    mode
                ),

                weight_graph_pattern.remote(
                    'bbcsport',
                    config['bbcsport_classes'],
                    join(str(root), config['bbcsport_data_prefix']),
                    mode
                )
            ]

            ray.get(futures)

        futures = [
            score_graph_pattern.remote(
                'ten_newsgroups',
                config['ten_newsgroups_classes'],
                join(str(root), config['ten_newsgroups_data_prefix']),
                mode
            ),

            score_graph_pattern.remote(
                'bbcsport',
                config['bbcsport_classes'],
                join(str(root), config['bbcsport_data_prefix']),
                mode
            )
        ]

        ten_newsgroups_results, bbcsport_results = ray.get(futures)

        futures = [
            save_result(
                root,
                'ten_newsgroups_results.pickle',
                config['ten_newsgroups_data_prefix'],
                ten_newsgroups_results
            ),
            save_result(
                root,
                'bbcsport_results.pickle',
                config['bbcsport_data_prefix'],
                bbcsport_results
            )
        ]

        ray.get(futures)

        construct_results_dataframe(str(root), config)

    if visualization == 'yes':
        reconstruct_graph_pattern(
            config['example_classes'],
            join(str(root), config['example_data_prefix']),
            mode
        )

        graph_pattern_visualize_iterator(
            config['example_classes'],
            join(str(root), config['example_data_prefix']),
            mode
        )
    
    end = time.time()
    print("Execution time:", end - start)

    construct_results_dataframe(root, config)

if __name__ == '__main__':
    root = get_project_root()

    parser = ArgumentParser(description='Main script', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--config',
        type=Path,
        default=root/'config/config.yaml',
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

    do_operations(
        root=root,
        config=config,
        dataset=args.dataset,
        mode=args.mode,
        weighting=args.weighting,
        graph_pattern_reconstruction=args.graph_pattern_reconstruction,
        visualization=args.visualization
    )
