import torch
import tokenizers
import torch.nn.functional as F


def train(model, optimizer, criterion, train_loader, n_epochs, device):
    history = {"train_losses": []}

    for epoch in range(n_epochs):
        total_loss = 0
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            total_loss += loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss.item())

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {history['train_losses'][-1]:.4f}, "
        )

    return history


def get_tokenizer(tokenizer_model, tokenizer_trainer, text, vocab_size=1000):
    tokenizer = tokenizers.Tokenizer(model=tokenizer_model)
    special_tokens = ["<unk>", "<pad>"]

    tokenizer_trainer.vocab_size = vocab_size
    tokenizer_trainer.special_tokens = special_tokens

    tokenizer.train_from_iterator([text.lower()], tokenizer_trainer)

    return tokenizer


def next_char(model, text, tokenizer, temperature, device):
    encoded_text = torch.tensor(tokenizer.encode(text).ids).unsqueeze(dim=0).to(device)

    with torch.no_grad():
        Y_logits = model(encoded_text)
        Y_probas = F.softmax(Y_logits[0, :, -1] / temperature, dim=-1)
        predicted_char_id = torch.multinomial(Y_probas, num_samples=1).item()

    return tokenizer.decode([predicted_char_id])


def extend_text(model, text, tokenizer, n_chars=80, temperature=0.4, device="cuda"):
    for _ in range(n_chars):
        char = next_char(model, text, tokenizer, temperature, device)
        text += char

    return text
