import yaml
import pickle
import shutil
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from utils import get_project_root
from results_aggregator import construct_results_dataframe
from graph_pattern_reconstruction import graph_pattern_reconstruction_iterator
from graph_pattern_weighting import graph_pattern_weighting_iterator
from graph_pattern_scoring import graph_pattern_scoring_iterator
from graph_pattern_visualize import graph_pattern_visualize_iterator


def do_operations(root, config, dataset='all', mode='concepts', weighting='yes', graph_pattern_reconstruction='yes', visualization='no'):

    if dataset == 'all':
        if graph_pattern_reconstruction == 'yes':
            print("Reconstructing graph patterns...")
            graph_pattern_reconstruction_iterator(config['ten_newsgroups_classes'], str(root) + '/' + config['ten_newsgroups_data_prefix'], mode)
            graph_pattern_reconstruction_iterator(config['bbcsport_classes'], str(root) + '/' + config['bbcsport_data_prefix'], mode)

        if weighting == 'yes':
            print("Weighting graph patterns...")
            graph_pattern_weighting_iterator('ten_newsgroups', config['ten_newsgroups_classes'], str(root) + '/' + config['ten_newsgroups_data_prefix'], mode)
            graph_pattern_weighting_iterator('bbcsport', config['bbcsport_classes'], str(root) + '/' + config['bbcsport_data_prefix'], mode)

        ten_newsgroups_results = graph_pattern_scoring_iterator('ten_newsgroups', config['ten_newsgroups_classes'], str(root) + '/' + config['ten_newsgroups_data_prefix'], mode)
        bbcsport_results = graph_pattern_scoring_iterator('bbcsport', config['bbcsport_classes'], str(root) + '/' + config['bbcsport_data_prefix'], mode)

        ten_newsgroups_results_file_name = 'ten_newsgroups_results.pickle'
        bbcsport_results_file_name = 'bbcsport_results.pickle'

        with open(ten_newsgroups_results_file_name, 'wb') as handle:
            pickle.dump(ten_newsgroups_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ten_newsgroups_results_path = str(root) + '/' + config['ten_newsgroups_data_prefix'] + '/' + ten_newsgroups_results_file_name
        shutil.move(ten_newsgroups_results_file_name, ten_newsgroups_results_path)

        with open(bbcsport_results_file_name, 'wb') as handle:
            pickle.dump(bbcsport_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        bbcsport_results_path = str(root) + '/' + config['bbcsport_data_prefix'] + '/' + bbcsport_results_file_name
        shutil.move(bbcsport_results_file_name, bbcsport_results_path)

        construct_results_dataframe(str(root), config)

    if visualization == 'yes':
        graph_pattern_reconstruction_iterator(config['example_classes'], str(root) + '/' + config['example_data_prefix'], mode)
        graph_pattern_visualize_iterator(config['example_classes'], str(root) + '/' + config['example_data_prefix'], mode)


if __name__ == '__main__':
    root = get_project_root()

    parser = ArgumentParser(description='Main script', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=Path, default=root/'config/config.yaml',
                        help='Enter the config file path.')

    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'ten_newsgroups', 'bbcsport'],
                        help='Choose the dataset.')

    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'concepts', 'frequent_subgraphs', 'equivalence_classes'],
                        help='Choose the operation mode.')

    parser.add_argument('--weighting', type=str, default='yes',
                        choices=['yes', 'no'],
                        help='Choose whether to weight graph patterns.')

    parser.add_argument('--graph_pattern_reconstruction', type=str, default='yes',
                        choices=['yes', 'no'],
                        help='Choose whether to reconstruct graph patterns.')

    parser.add_argument('--visualization', type=str, default='no',
                        choices=['yes', 'no'],
                        help='Choose whether to visualize graph patterns reconstruction using a toy example.')

    args, unknown = parser.parse_known_args()

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    do_operations(root=root,
                  config=config,
                  dataset=args.dataset,
                  mode=args.mode,
                  weighting=args.weighting,
                  graph_pattern_reconstruction=args.graph_pattern_reconstruction,
                  visualization=args.visualization)
