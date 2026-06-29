import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from nonet.tokenizer import SudokuTokenizer

tokenizer = SudokuTokenizer()

_COLS = ["puzzle", "solution"]


def _collate(batch):
    """Tokenize a batch on the fly (no upfront .map over the whole dataset)."""
    return {
        "puzzle": torch.tensor([[int(c) for c in x["puzzle"]] for x in batch]),
        "solution": torch.tensor([[int(c) for c in x["solution"]] for x in batch]),
    }


def get_train_dataloader(batch_size=32, validation_split=0.1, seed=42, shuffle_buffer=10_000):
    val_every = round(1 / validation_split)
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split="train")
    ds_train = ds.filter(lambda _, idx: idx % val_every != 0, with_indices=True)
    ds_val = ds.filter(lambda _, idx: idx % val_every == 0, with_indices=True)

    
    ds_train = ds_train.shuffle(seed=seed).select_columns(_COLS)
    ds_val = ds_val.select_columns(_COLS)

    dl_train = DataLoader(ds_train, batch_size=batch_size, collate_fn=_collate)
    dl_val = DataLoader(ds_val, batch_size=batch_size, collate_fn=_collate)
    return dl_train, dl_val


def get_test_dataloader(batch_size=32):
    ds_test = load_dataset("Ritvik19/Sudoku-Dataset", split="validation")
    ds_test = ds_test.select_columns(_COLS)
    dl_test = DataLoader(ds_test, batch_size=batch_size, collate_fn=_collate)
    return dl_test


if __name__ == "__main__":
    dl_train, dl_val = get_train_dataloader()
    dl_test = get_test_dataloader()

    batch = next(iter(dl_train))
    print(batch)
