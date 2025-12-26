from torch.utils.data import Dataset
import torch
from pathlib import Path
import urllib.request


def dowload_shakespeare_text(path_name):
    path = Path(path_name)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://homl.info/shakespeare"
        urllib.request.urlretrieve(url, path)

    return path.read_text()


class CharDataset(Dataset):
    def __init__(self, text, window_length, tokenizer):
        self.encoded_text = torch.tensor(tokenizer.encode(text).ids)
        self.window_length = window_length

    def __len__(self):
        return len(self.encoded_text) - self.window_length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("dataset index out of range")

        end = idx + self.window_length
        window = self.encoded_text[idx:end]
        target = self.encoded_text[idx + 1 : end + 1]

        return window, target
