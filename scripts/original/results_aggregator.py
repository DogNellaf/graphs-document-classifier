import pickle
import pandas as pd
from itertools import groupby


def save_latex_table(df: pd.DataFrame, file_path: str, column_format=None, float_format='%.4f'):
    column_format = '|l|'+'c|'*df.shape[1] if column_format is None else column_format

    text = df.to_latex(column_format=column_format, float_format=float_format)
    for rule in [r'\toprule', r'\midrule', r'\bottomrule']:
        text = text.replace(rule, r'\hline')
    with open(file_path, 'w') as f:
        f.write(text)


def construct_results_dataframe(root, config):

    datasets = ['ten_newsgroups', 'bbcsport']

    results = []

    for dataset in datasets:
        dataset_prefix_key = dataset + '_data_prefix'
        dataset_results_file_name = str(root) + '/' + config[dataset_prefix_key] + '/' + dataset + '_results.pickle'

        with open(dataset_results_file_name, 'rb') as f:
            dataset_results = pickle.load(f)

        for dr_ele in dataset_results:
            results.append(dr_ele)

    results.sort(key=lambda x: x['mode'])

    for i, (k, v) in enumerate(groupby(results, key=lambda x: x['mode'])):
        v = list(v)

        dataset_column = []
        baseline_penalty_column = []
        penalty_1_column = []
        penalty_2_column = []
        penalty_3_column = []
        penalty_4_column = []
        penalty_5_column = []
        penalty_6_column = []

        for ele in v:
            dataset_column.append(ele['dataset'])
            baseline_penalty_column.append(ele['baseline_penalty_macro-averaged_f1-score'])
            penalty_1_column.append(ele['penalty_1_macro-averaged_f1-score'])
            penalty_2_column.append(ele['penalty_2_macro-averaged_f1-score'])
            penalty_3_column.append(ele['penalty_3_macro-averaged_f1-score'])
            penalty_4_column.append(ele['edge_penalty_macro-averaged_f1-score'])
            penalty_5_column.append(ele['penalty_5_macro-averaged_f1-score'])
            penalty_6_column.append(ele['penalty_6_macro-averaged_f1-score'])

        d = {'Dataset': dataset_column, 'Baseline penalty': baseline_penalty_column, 'Penalty 1': penalty_1_column,
             'Penalty 2': penalty_2_column, 'Penalty 3': penalty_3_column, 'Penalty 4': penalty_4_column,
             'Penalty 5': penalty_5_column, 'Penalty 6': penalty_6_column}

        df = pd.DataFrame(data=d)
        latex_table_path = str(root) + '/' + v[0]['mode'] + '.tex'

        save_latex_table(df, latex_table_path)
