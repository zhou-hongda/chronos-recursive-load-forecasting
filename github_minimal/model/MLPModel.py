import torch.nn as nn


class MLPModel(nn.Module):
    """Flattened-window MLP baseline with direct multivariate output."""

    def __init__(
        self,
        input_dim=3,
        seq_len=96,
        pred_len=24,
        hidden_dims=(512, 512, 256, 256, 128, 128, 64),
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        layers = []
        previous_dim = seq_len * input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, pred_len * input_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, _, num_features = x.shape
        output = self.mlp(x.reshape(batch_size, -1))
        return output.reshape(batch_size, self.pred_len, num_features)
