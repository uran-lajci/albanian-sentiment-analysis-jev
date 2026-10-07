"""Rebuild the train, validation and test split of Kastrati et al. (2021) on the dataset they published.

uv run python -m sentiment.split_paper_dataset

data/AlbAna.csv is Dataset/AlbAna.csv from https://github.com/lule-ahmedi/AlbAna, the 10,742 comments the paper
uses. Their code splits twice with train_test_split(train_size=0.7, random_state=1000): 49% train, 21% validation and
30% test. The paper says 70/15/15, but the code is what produced its numbers. The rebuild assumes that the file their
code read lists the comments in the order of AlbAna.csv. The output is separated by ';'.
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = ['Id', 'Comment', 'Annot 1', 'Annot 2', 'Annot 3', 'Final annotation', 'split']


def main(input_path: str, output_path: str, train_size: float, seed: int) -> None:
    df = pd.read_csv(input_path, encoding='utf-8-sig', dtype=str, keep_default_na=False)

    rows = np.arange(len(df))
    train_and_validation, test = train_test_split(rows, train_size=train_size, random_state=seed)
    train, validation = train_test_split(train_and_validation, train_size=train_size, random_state=seed)

    df['split'] = ''
    df.loc[train, 'split'] = 'train'
    df.loc[validation, 'split'] = 'val'
    df.loc[test, 'split'] = 'test'
    df[COLUMNS].to_csv(output_path, sep=';', index=False)

    print(f'Wrote {output_path}. Comments per split and final annotation:')
    print(pd.crosstab(df['split'], df['Final annotation'], margins=True))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_path', default='data/AlbAna.csv')
    parser.add_argument('-o', '--output_path', default='data/paper_dataset.csv')
    parser.add_argument('-t', '--train_size', type=float, default=0.7)
    parser.add_argument('-s', '--seed', type=int, default=1000)

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.train_size, args.seed)
