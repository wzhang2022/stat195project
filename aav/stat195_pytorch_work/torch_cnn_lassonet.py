import numpy as np
import torch.nn as nn
import torch
from einops.einops import rearrange
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


class CNNBaseline(nn.Module):
    def __init__(
            self,
            seq_encoding_length,
            residue_encoding_size,
            conv_depth,
            conv_width,
            pool_width,
            conv_depth_multiplier,
            fc_size,
            fc_size_multiplier
    ):
        super(CNNBaseline, self).__init__()
        self.seq_encoding_length = seq_encoding_length
        self.residue_encoding_size = residue_encoding_size
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=residue_encoding_size,
                out_channels=conv_depth,
                kernel_size=conv_width,
                padding=(conv_width - 1) // 2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(conv_depth),
            nn.MaxPool1d(kernel_size=pool_width, stride=pool_width)
        )
        self.conv_block_2 = nn.Sequential(
                nn.Conv1d(
                in_channels=conv_depth,
                out_channels=conv_depth * conv_depth_multiplier,
                kernel_size=conv_width,
                padding=(conv_width - 1)//2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(conv_depth * conv_depth_multiplier),
            nn.MaxPool1d(kernel_size=pool_width, stride=pool_width)
        )
        self.fc_block_1 = nn.Sequential(
            nn.Linear(336, fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_size)
        )
        self.fc_block_2 = nn.Sequential(
            nn.Linear(fc_size, int(fc_size * fc_size_multiplier)),
            nn.ReLU(),
            nn.BatchNorm1d(int(fc_size * fc_size_multiplier))
        )
        self.output_layer = nn.Linear(int(fc_size * fc_size_multiplier), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: torch tensor of shape (batch_size, seq_len, residue_encoding_size)

        Returns:
            torch tensor of shape (batch_size,) that contains floating-point values
            of sequence viability
        """
        x = rearrange(x, 'b l c -> b c l')
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc_block_1(x)
        x = self.fc_block_2(x)
        x = self.output_layer(x)
        return x


class CNNModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = CNNBaseline(
            seq_encoding_length=58,
            residue_encoding_size=20,
            fc_size=128,
            conv_depth=12,
            conv_width=7,
            pool_width=2,
            fc_size_multiplier=0.5,
            conv_depth_multiplier=2,
        ).to(self.device)

    def fit(self, X, y, batch_size=128, num_epochs=30):
        assert isinstance(X, np.ndarray), "input must be numpy array"
        assert isinstance(y, np.ndarray), "input must be numpy array"
        assert X.shape[0] == y.shape[0], "features and labels must have the same number of samples"
        assert (X.shape[1], X.shape[2]) == (58, 20), "input features must have shape (sequence_len, num_residues)"

        criterion = nn.NLLLoss()
        optimizer = optim.AdamW(self.net.parameters(), lr=0.001)
        train_set = TensorDataset(torch.as_tensor(X), torch.as_tensor(y))
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        self.net.train()
        for epoch in range(num_epochs):
            for i, (seq, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.net(seq.to(self.device))
                loss = criterion(outputs, labels.long().to(self.device))
                loss.backward()
                optimizer.step()

    def predict(self, X):
        test_loader = DataLoader(X, batch_size=128, shuffle=False)
        self.net.eval()
        with torch.no_grad():
            y_pred = []
            for i, (seq, labels) in enumerate(test_loader):
                outputs = self.net(seq.to(self.device))
                y_pred.append(outputs.argmax(dim=1).cpu().numpy())
        return np.concatenate(y_pred, axis=0)