import pandas as pd

LABELS = {0: 'neutral', 1: 'positive', 2: 'negative'}


def read_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=';', encoding='utf-8-sig', dtype=str, keep_default_na=False)
