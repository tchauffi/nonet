from datasets import load_dataset
from torch.utils.data import DataLoader


def get_train_dataloader(batch_size=32, validation_split=0.1, seed=42):
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split="train")

    # split the dataset into training and validation sets
    train_size = int((1 - validation_split) * len(ds))
    ds_train = ds.select(range(train_size))
    ds_val = ds.select(range(train_size, len(ds)))

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return dl_train, dl_val

def get_test_dataloader(batch_size=32):
    ds_test = load_dataset("Ritvik19/Sudoku-Dataset", split="validation")
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return dl_test


if __name__ == "__main__":
    dl_train, dl_val = get_train_dataloader()
    dl_test = get_test_dataloader()

    print(f"Train batches: {len(dl_train)}")
    print(f"Validation batches: {len(dl_val)}")
    print(f"Test batches: {len(dl_test)}")

