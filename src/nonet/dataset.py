from datasets import load_dataset
from torch.utils.data import DataLoader

from nonet.tokenizer import SudokuTokenizer

tokenizer = SudokuTokenizer()


def _tokenize(x):
    return {
        "puzzle": [int(c) for c in x["puzzle"]],
        "solution": [int(c) for c in x["solution"]],
    }


def get_train_dataloader(batch_size=32, validation_split=0.1, seed=42, shuffle_buffer=10_000):
    val_every = round(1 / validation_split)
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split="train", streaming=True)
    ds_train = ds.filter(lambda _, idx: idx % val_every != 0, with_indices=True)
    ds_val = ds.filter(lambda _, idx: idx % val_every == 0, with_indices=True)

    _COLS = ["puzzle", "solution"]
    ds_train = (
        ds_train.shuffle(buffer_size=shuffle_buffer, seed=seed)
        .map(_tokenize).select_columns(_COLS).with_format("torch")
    )
    ds_val = ds_val.map(_tokenize).select_columns(_COLS).with_format("torch")

    dl_train = DataLoader(ds_train, batch_size=batch_size)
    dl_val = DataLoader(ds_val, batch_size=batch_size)
    return dl_train, dl_val


def get_test_dataloader(batch_size=32):
    ds_test = load_dataset("Ritvik19/Sudoku-Dataset", split="validation", streaming=True)
    ds_test = ds_test.map(_tokenize).select_columns(["puzzle", "solution"]).with_format("torch")
    dl_test = DataLoader(ds_test, batch_size=batch_size)
    return dl_test


if __name__ == "__main__":
    dl_train, dl_val = get_train_dataloader()
    dl_test = get_test_dataloader()

    batch = next(iter(dl_train))
    print(batch)
