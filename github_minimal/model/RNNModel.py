import torch.nn as nn


class RNNModel(nn.Module):
    """Vanilla RNN baseline with direct multivariate 24-hour output."""

    def __init__(
        self,
        input_dim=3,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len * input_dim),
        )

    def forward(self, x):
        batch_size, _, num_features = x.shape
        sequence, _ = self.rnn(x)
        output = self.head(sequence[:, -1])
        return output.reshape(batch_size, self.pred_len, num_features)
