"""Fine-tune multilingual BERT on the split of Kastrati et al. (2021) and predict the validation and test comments.

uv run python -m sentiment.train_mbert

The paper trained mBERT on the Peltarion platform and gives only the 128-token input. This run uses
bert-base-multilingual-cased with a learning rate of 2e-5 and batches of 32 for up to 4 epochs, and keeps the epoch
with the highest validation accuracy, the same rule as the other networks.
"""

import math
from argparse import ArgumentParser
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sentiment.dataset import read_dataset

MAX_TOKENS = 128


def predict(
    model: torch.nn.Module,
    encode: Callable[[npt.NDArray[np.int64]], dict[str, torch.Tensor]],
    labels: torch.Tensor,
    rows: npt.NDArray[np.int64],
) -> tuple[float, npt.NDArray[np.int64]]:
    model.eval()
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for batch in np.array_split(rows, math.ceil(len(rows) / 128)):
            logits = model(**encode(batch)).logits.float()
            batch_labels = labels[batch].to(logits.device)
            total_loss += torch.nn.functional.cross_entropy(logits, batch_labels, reduction='sum').item()
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return total_loss / len(rows), np.concatenate(predictions)


def main(
    dataset_path: str,
    output_path: str,
    model_name: str,
    seeds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    df = read_dataset(dataset_path)
    texts = df['Comment'].tolist()
    labels = torch.tensor(df['Final annotation'].astype(int).to_numpy())
    train = np.flatnonzero(df['split'] == 'train')
    validation = np.flatnonzero(df['split'] == 'val')
    evaluated = np.flatnonzero(df['split'] != 'train')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(rows: npt.NDArray[np.int64]) -> dict[str, torch.Tensor]:
        batch = tokenizer([texts[row] for row in rows], truncation=True, max_length=MAX_TOKENS, padding=True,
                          return_tensors='pt')
        return {name: tensor.to(device) for name, tensor in batch.items()}

    frames = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        generator = np.random.default_rng(seed)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        batches_per_epoch = math.ceil(len(train) / batch_size)
        steps = batches_per_epoch * epochs
        scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
            optimizer,
            num_warmup_steps=steps // 10,
            num_training_steps=steps,
        )

        best_accuracy, best_weights = -1.0, {}
        for epoch in range(epochs):
            model.train()
            for batch in np.array_split(generator.permutation(train), batches_per_epoch):
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    loss = model(**encode(batch), labels=labels[batch].to(device)).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            validation_loss, validation_prediction = predict(model, encode, labels, validation)
            validation_accuracy = float((validation_prediction == labels[validation].numpy()).mean())
            print(f'seed {seed}, epoch {epoch + 1}: validation loss {validation_loss:.4f}, '
                  f'accuracy {validation_accuracy:.4f}')
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_weights = {name: weight.detach().cpu().clone() for name, weight in model.state_dict().items()}

        model.load_state_dict(best_weights)
        _, prediction = predict(model, encode, labels, evaluated)
        frames.append(pd.DataFrame({
            'model': 'BERT',
            'embeddings': 'mBERT',
            'seed': seed,
            'Id': df['Id'].iloc[evaluated].to_numpy(),
            'split': df['split'].iloc[evaluated].to_numpy(),
            'prediction': prediction,
        }))
        del model, optimizer
        torch.cuda.empty_cache()

    pd.concat(frames).to_csv(output_path, index=False)
    print(f'Wrote {seeds} runs of {len(evaluated)} validation and test predictions each to {output_path}.')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--dataset_path', default='data/paper_dataset.csv')
    parser.add_argument('-o', '--output_path', default='results/mbert_predictions.csv')
    parser.add_argument('-m', '--model_name', default='bert-base-multilingual-cased')
    parser.add_argument('-s', '--seeds', type=int, default=5)
    parser.add_argument('-e', '--epochs', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-l', '--learning_rate', type=float, default=2e-5)

    args = parser.parse_args()
    main(
        args.dataset_path,
        args.output_path,
        args.model_name,
        args.seeds,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )
