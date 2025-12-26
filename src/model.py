import torch.nn as nn


class ShakespeareModel(nn.Module):
    def __init__(
        self, vocab_size, n_layers=2, embed_dim=100, hidden_dim=128, dropout=0.1
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, X):
        embeddings = self.embed(X)
        outputs, _states = self.gru(embeddings)

        return self.output(outputs).permute(0, 2, 1)
