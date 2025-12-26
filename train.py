import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tokenizers
from src.dataset import dowload_shakespeare_text
from src.dataset import CharDataset
from src.modeling import get_tokenizer
from src.modeling import train
from src.model import ShakespeareModel
import os
from pathlib import Path

VOCAB_SIZE = 1000
WINDOW_LENGTH = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.001
N_EPOCHS = 15


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = "datasets/shakespeare/shakespeare.txt"
    print("[INFO] Iniciando o dowload dos dados")
    text = dowload_shakespeare_text(path)
    print("[INFO] Dowload dos dados finalizado")

    path_tokenizer = "models/tokenizer.json"

    if not Path(path_tokenizer).is_file():
        bpe_model = tokenizers.models.BPE(unk_token="<unk>")
        bpe_trainer = tokenizers.trainers.BpeTrainer()
        print("[INFO] Realizando a criação do tokenizer")
        tokenizer = get_tokenizer(
            tokenizer_model=bpe_model,
            tokenizer_trainer=bpe_trainer,
            text=text,
            vocab_size=VOCAB_SIZE,
        )
        os.makedirs(name="models", exist_ok=True)
        print("[INFO] Iniciando o salvamento do tokenizer")
        tokenizer.save("models/tokenizer.json")
        print("[INFO] Salvamento do tokenizer finalizado")
    else:
        print("[INFO] O tokenizer já existe, iniciando o load dele")
        tokenizer = tokenizers.Tokenizer.from_file(path_tokenizer)

    train_set = CharDataset(text, WINDOW_LENGTH, tokenizer)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    model = ShakespeareModel(vocab_size=VOCAB_SIZE).to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss().to(device)

    print(f"[INFO] Iniciando o treinamento do modelo por {N_EPOCHS} épocas")
    train(model, optimizer, criterion, train_loader, N_EPOCHS, device)
    print("[INFO] Treinamento do modelo finalizado")

    print("[INFO] Iniciando o salvamento do modelo")
    torch.save(obj=model.state_dict(), f="models/char-rnn.pth")
    print("[INFO] Salvamento do modelo finalizado")


if __name__ == "__main__":
    main()
