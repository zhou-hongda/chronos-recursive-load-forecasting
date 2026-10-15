import torch.nn as nn


class TCNBlock(nn.Module):
    """Residual dilated temporal-convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)
        output = self.conv1(x)[..., :x.shape[-1]]
        output = self.dropout1(self.activation1(self.norm1(output)))
        output = self.conv2(output)[..., :x.shape[-1]]
        output = self.dropout2(self.activation2(self.norm2(output)))
        return output + residual


class TCNModel(nn.Module):
    """Dilated TCN baseline with direct multivariate 24-hour output."""

    def __init__(
        self,
        input_dim=3,
        seq_len=96,
        pred_len=24,
        num_channels=(64, 128, 256),
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        blocks = []
        in_channels = input_dim
        for level, out_channels in enumerate(num_channels):
            blocks.append(
                TCNBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=2 ** level,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1] * seq_len, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, pred_len * input_dim),
        )

    def forward(self, x):
        batch_size, _, num_features = x.shape
        features = self.tcn(x.transpose(1, 2)).reshape(batch_size, -1)
        output = self.head(features)
        return output.reshape(batch_size, self.pred_len, num_features)
