"""Train the classifiers of Kastrati et al. (2021) on their split and predict the validation and test comments.

uv run python -m sentiment.train_baselines -f cc.sq.300.vec.gz

cc.sq.300.vec.gz is the Albanian fastText model from https://fasttext.cc/docs/en/crawl-vectors.html, the one the paper
uses. The networks follow the paper's Figure 6 and the code in https://github.com/lule-ahmedi/AlbAna. Each network
trains for at most 35 epochs and keeps the weights of the epoch with the highest validation accuracy, the convergence
metric the paper names, because the paper and the code give different epoch counts. Stopping on the validation loss
instead keeps first-epoch Hybrid models that almost never predict positive. The conventional models use the code's
features without its Albanian stop word list, which is not published.
"""

import gzip
from argparse import ArgumentParser
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras import layers

from sentiment.dataset import read_dataset

VOCABULARY_SIZE = 20000
SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 300


def attention() -> layers.Layer:
    return SeqSelfAttention(attention_width=8, attention_activation='sigmoid')


def bilstm() -> layers.Layer:
    return layers.Bidirectional(layers.LSTM(32, return_sequences=True))


NETWORKS: dict[str, Callable[[], list[layers.Layer]]] = {
    'DNN': lambda: [
        layers.GlobalMaxPooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
    ],
    '1D-CNN': lambda: [
        layers.SpatialDropout1D(0.3),
        layers.Conv1D(512, 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
    ],
    '1D-CNN + Att': lambda: [
        attention(),
        layers.SpatialDropout1D(0.3),
        layers.Conv1D(512, 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
    ],
    'BiLSTM': lambda: [layers.SpatialDropout1D(0.3), bilstm(), layers.Flatten()],
    'BiLSTM + Att': lambda: [layers.SpatialDropout1D(0.3), bilstm(), attention(), layers.Flatten()],
    'Hybrid': lambda: [
        layers.SpatialDropout1D(0.3),
        layers.Conv1D(512, 3, activation='relu'),
        layers.MaxPooling1D(),
        bilstm(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
    ],
    'Hybrid + Att': lambda: [
        layers.SpatialDropout1D(0.3),
        layers.Conv1D(512, 3, activation='relu'),
        layers.MaxPooling1D(),
        bilstm(),
        attention(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
    ],
}

NETWORK_RUNS = [(network, 'Domain') for network in NETWORKS] + [
    (network, 'FastText') for network in ['DNN', '1D-CNN', 'BiLSTM', 'Hybrid']
]

VECTORIZERS = {
    'tf': CountVectorizer(token_pattern=r'\w{2,}', max_features=2000),
    'tf*idf': TfidfVectorizer(token_pattern=r'\w{2,}', max_features=2000),
}

CLASSIFIERS: dict[str, Callable[[int], ClassifierMixin]] = {
    'SVM': lambda seed: SVC(kernel='linear'),
    'NB': lambda seed: MultinomialNB(),
    'DT': lambda seed: DecisionTreeClassifier(random_state=seed),
    'RF': lambda seed: RandomForestClassifier(max_depth=200, random_state=seed),
}


def load_fasttext(path: str, word_index: dict[str, int]) -> npt.NDArray[np.float32]:
    matrix = np.zeros((VOCABULARY_SIZE, EMBEDDING_DIM), dtype='float32')
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        next(file)
        for line in file:
            word, _, vector = line.rstrip().partition(' ')
            index = word_index.get(word)
            if index is not None and index < VOCABULARY_SIZE:
                matrix[index] = np.array(vector.split(' '), dtype='float32')
    return matrix


def train_network(
    network: str,
    fasttext: npt.NDArray[np.float32] | None,
    seed: int,
    x: npt.NDArray[np.int32],
    y: npt.NDArray[np.int64],
    train: npt.NDArray[np.bool_],
    validation: npt.NDArray[np.bool_],
) -> npt.NDArray[np.int64]:
    keras.utils.set_random_seed(seed)
    if fasttext is None:
        embedding = layers.Embedding(VOCABULARY_SIZE, EMBEDDING_DIM)
    else:
        embedding = layers.Embedding(
            VOCABULARY_SIZE,
            EMBEDDING_DIM,
            embeddings_initializer=keras.initializers.Constant(fasttext),
            trainable=False,
        )

    model = keras.Sequential([
        keras.Input(shape=(SEQUENCE_LENGTH,)),
        embedding,
        *NETWORKS[network](),
        layers.Dense(3, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(
        x[train],
        y[train],
        batch_size=256,
        epochs=35,
        validation_data=(x[validation], y[validation]),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True),
        ],
        verbose=0,
    )
    return np.asarray(model.predict(x, batch_size=1024, verbose=0).argmax(axis=1))


def main(dataset_path: str, fasttext_path: str, output_path: str, seeds: int) -> None:
    tf.config.experimental.enable_op_determinism()
    df = read_dataset(dataset_path)
    y = df['Final annotation'].astype(int).to_numpy()
    train = (df['split'] == 'train').to_numpy()
    validation = (df['split'] == 'val').to_numpy()
    evaluated = ~train

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=VOCABULARY_SIZE, lower=True)
    tokenizer.fit_on_texts(df['Comment'])
    x = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(df['Comment']), maxlen=SEQUENCE_LENGTH)
    fasttext = load_fasttext(fasttext_path, tokenizer.word_index)
    print(f'{(fasttext[1:] == 0).all(axis=1).sum()} of {VOCABULARY_SIZE - 1} words have no fastText vector.')
    bags_of_words = {name: vectorizer.fit_transform(df['Comment']) for name, vectorizer in VECTORIZERS.items()}

    runs = []
    for seed in range(seeds):
        for network, embeddings in NETWORK_RUNS:
            pretrained = fasttext if embeddings == 'FastText' else None
            prediction = train_network(network, pretrained, seed, x, y, train, validation)
            runs.append((network, embeddings, seed, prediction))
            print(f'seed {seed}: {network} ({embeddings}), validation accuracy '
                  f'{(prediction[validation] == y[validation]).mean():.3f}')

        for features, bag_of_words in bags_of_words.items():
            for classifier, make_classifier in CLASSIFIERS.items():
                prediction = make_classifier(seed).fit(bag_of_words[train], y[train]).predict(bag_of_words)
                runs.append((classifier, features, seed, prediction))
                print(f'seed {seed}: {classifier} ({features}), validation accuracy '
                      f'{(prediction[validation] == y[validation]).mean():.3f}')

    predictions = pd.concat([
        pd.DataFrame({
            'model': model,
            'embeddings': embeddings,
            'seed': seed,
            'Id': df['Id'][evaluated],
            'split': df['split'][evaluated],
            'prediction': prediction[evaluated],
        })
        for model, embeddings, seed, prediction in runs
    ])
    predictions.to_csv(output_path, index=False)
    print(f'Wrote {len(runs)} runs of {evaluated.sum()} validation and test predictions each to {output_path}.')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--dataset_path', default='data/paper_dataset.csv')
    parser.add_argument('-f', '--fasttext_path', required=True)
    parser.add_argument('-o', '--output_path', default='results/baseline_predictions.csv.gz')
    parser.add_argument('-s', '--seeds', type=int, default=5)

    args = parser.parse_args()
    main(args.dataset_path, args.fasttext_path, args.output_path, args.seeds)
