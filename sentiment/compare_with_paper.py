"""Compare Jev with the classifiers of Kastrati et al. (2021) on their validation and test comments.

uv run python -m sentiment.compare_with_paper

Report the test table. Precision, recall and F1 are weighted averages, as in the paper; macro F1 and accuracy are
added. A trained model shows the mean and standard deviation over its seeds. The paper F1 column is Table 12 of the
paper. In the published code those scores come from the validation split, except NB, DT and RF, which come from test.

The last lines give a paired bootstrap interval on test for the difference between Jev and each of the five best
models of the paper, and the agreement between pairs of annotators on the same test comments.
"""

import itertools
from argparse import ArgumentParser

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from sentiment.dataset import read_dataset

PAPER_F1 = {
    ('DNN', 'Domain'): 69.32,
    ('DNN', 'FastText'): 63.38,
    ('1D-CNN', 'Domain'): 70.76,
    ('1D-CNN', 'FastText'): 70.45,
    ('1D-CNN + Att', 'Domain'): 71.56,
    ('BiLSTM', 'Domain'): 70.71,
    ('BiLSTM', 'FastText'): 68.95,
    ('BiLSTM + Att', 'Domain'): 72.09,
    ('Hybrid', 'Domain'): 70.00,
    ('Hybrid', 'FastText'): 68.18,
    ('Hybrid + Att', 'Domain'): 71.32,
    ('BERT', 'mBERT'): 71.35,
    ('SVM', 'tf'): 69.49,
    ('SVM', 'tf*idf'): 71.27,
    ('NB', 'tf'): 70.89,
    ('NB', 'tf*idf'): 70.04,
    ('DT', 'tf'): 63.25,
    ('DT', 'tf*idf'): 64.58,
    ('RF', 'tf'): 70.49,
    ('RF', 'tf*idf'): 71.44,
}

ANNOTATORS = ['Annot 1', 'Annot 2', 'Annot 3']


def score(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return 100 * pd.Series({
        'precision': precision,
        'recall': recall,
        'weighted F1': f1,
        'macro F1': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
    })


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    per_run = runs.groupby(['model', 'embeddings', 'seed']).apply(
        lambda run: score(run['gold'], run['prediction']),
        include_groups=False,
    )
    summary = per_run.groupby(['model', 'embeddings']).agg(['mean', 'std']).round(2)
    summary.columns = [f'{metric} {statistic}' for metric, statistic in summary.columns]
    summary = summary.drop(columns=['precision std', 'recall std', 'accuracy std'])
    summary['paper F1'] = [PAPER_F1.get(key, np.nan) for key in summary.index]
    return summary.sort_values('weighted F1 mean', ascending=False)


def paired_bootstrap(
    gold: npt.NDArray[np.int64],
    jev: npt.NDArray[np.int64],
    baseline_runs: list[npt.NDArray[np.int64]],
    average: str,
    resamples: int,
) -> tuple[float, float, float]:
    generator = np.random.default_rng(0)

    def difference(rows: npt.NDArray[np.int64]) -> float:
        baseline = np.mean([f1_score(gold[rows], run[rows], average=average) for run in baseline_runs])
        return float(100 * (f1_score(gold[rows], jev[rows], average=average) - baseline))

    samples = [difference(generator.integers(0, len(gold), len(gold))) for _ in range(resamples)]
    return difference(np.arange(len(gold))), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def compare_on_test(test: pd.DataFrame, baseline: tuple[str, str], resamples: int) -> None:
    gold = test.groupby('Id')['gold'].first()
    jev = test[test['model'] == 'Jev'].set_index('Id')['prediction'].reindex(gold.index).to_numpy()
    seeds = test[(test['model'] == baseline[0]) & (test['embeddings'] == baseline[1])].groupby('seed')
    baseline_runs = [run.set_index('Id')['prediction'].reindex(gold.index).to_numpy() for _, run in seeds]

    for average in ['weighted', 'macro']:
        difference, lower, upper = paired_bootstrap(gold.to_numpy(), jev, baseline_runs, average, resamples)
        print(f'  Jev minus {baseline[0]} ({baseline[1]}), {average} F1: {difference:+.2f} points, '
              f'95% interval {lower:+.2f} to {upper:+.2f}')


def main(dataset_path: str, prediction_paths: list[str], jev_path: str, resamples: int) -> None:
    df = read_dataset(dataset_path)
    jev = pd.read_csv(jev_path, dtype={'Id': str}).assign(model='Jev', embeddings='jev-1.13.0', seed=0)
    jev = jev.merge(df[['Id', 'split']], on='Id')
    runs = pd.concat([pd.read_csv(path, dtype={'Id': str}) for path in prediction_paths] + [jev])
    runs = runs[runs['split'] != 'train'].merge(df[['Id', 'Final annotation']], on='Id')
    runs['gold'] = runs['Final annotation'].astype(int)

    for split in ['test', 'val']:
        rows = runs[runs['split'] == split]
        print(f'\n=== {split}: {rows["Id"].nunique()} comments, {rows.groupby(["model", "embeddings"]).ngroups} models')
        print(summarize(rows).to_string())

    paper_top_five = sorted(PAPER_F1, key=lambda model: PAPER_F1[model], reverse=True)[:5]
    print(f'\nPaired bootstrap on test against the five best models of the paper, {resamples} samples:')
    for baseline in paper_top_five:
        compare_on_test(runs[runs['split'] == 'test'], baseline, resamples)

    test_comments = df[df['split'] == 'test']
    print('\nAgreement between pairs of annotators on the test comments:')
    for first, second in itertools.combinations(ANNOTATORS, 2):
        agreement = score(test_comments[first].astype(int), test_comments[second].astype(int))
        print(f'  {first} against {second}: weighted F1 {agreement["weighted F1"]:.2f}, '
              f'macro F1 {agreement["macro F1"]:.2f}')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--dataset_path', default='data/paper_dataset.csv')
    parser.add_argument(
        '-p',
        '--prediction_paths',
        nargs='+',
        default=['results/baseline_predictions.csv.gz', 'results/mbert_predictions.csv'],
    )
    parser.add_argument('-j', '--jev_path', default='results/jev_predictions.csv')
    parser.add_argument('-r', '--resamples', type=int, default=1000)

    args = parser.parse_args()
    main(args.dataset_path, args.prediction_paths, args.jev_path, args.resamples)
