from itertools import islice

from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
import torch
from einops.einops import rearrange
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from lassonet import LassoNetClassifier, LassoNet
from lassonet.prox import inplace_prox


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
            fc_size_multiplier,
            flattened_inputs=False
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
        self.flattened_inputs = flattened_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: torch tensor of shape (batch_size, seq_len, residue_encoding_size)

        Returns:
            torch tensor of shape (batch_size,) that contains floating-point values
            of sequence viability
        """
        if self.flattened_inputs:
            x = x.reshape(-1, self.seq_encoding_length, self.residue_encoding_size)
        x = rearrange(x, 'b l c -> b c l')
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc_block_1(x)
        x = self.fc_block_2(x)
        x = self.output_layer(x)
        return x


class CNNModel:
    def __init__(self, flattened_inputs=False):
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
            flattened_inputs=flattened_inputs
        ).to(self.device)
        self.flattened_inputs = flattened_inputs

    def path(self, X, y, batch_size=128, num_epochs=30):
        assert isinstance(X, np.ndarray), "input must be numpy array"
        assert isinstance(y, np.ndarray), "input must be numpy array"
        assert X.shape[0] == y.shape[0], "features and labels must have the same number of samples"
        if self.flattened_inputs:
            X = X.reshape(-1, 58, 20)
        assert (X.shape[1], X.shape[2]) == (58, 20), "input features must have shape (sequence_len, num_residues)"

        criterion = nn.NLLLoss()
        optimizer = optim.AdamW(self.net.parameters(), lr=0.001)
        train_set = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.net.train()
        for epoch in tqdm(range(num_epochs)):
            for i, (seq, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.net(seq.to(self.device))
                loss = criterion(outputs, labels.long().to(self.device))
                loss.backward()
                optimizer.step()

    def predict(self, X):
        test_loader = DataLoader(X.astype(np.float32), batch_size=128, shuffle=False)
        self.net.eval()
        with torch.no_grad():
            y_pred = []
            for i, seq in enumerate(test_loader):
                outputs = self.net(seq.to(self.device))
                y_pred.append(outputs.argmax(dim=1).cpu().numpy())
        return np.concatenate(y_pred, axis=0)


class CNNLassoNetModel(nn.Module):
    def __init__(self, seq_len=58, num_residues=20):
        super(CNNLassoNetModel, self).__init__()
        self.seq_len = seq_len
        self.num_residues = num_residues
        self.layers = nn.ModuleList(
            [nn.Linear(self.seq_len, self.seq_len)]
        )
        self.skip = nn.Linear(seq_len, 2)
        self.aa_embedding = nn.Parameter(torch.randn(num_residues) / num_residues ** 0.5)
        self.cnn_module = CNNBaseline(
            seq_encoding_length=58,
            residue_encoding_size=20,
            fc_size=128,
            conv_depth=12,
            conv_width=7,
            pool_width=2,
            fc_size_multiplier=0.5,
            conv_depth_multiplier=2,
            flattened_inputs=True
        )

    def forward(self, inp):
        inp = inp.reshape(-1, 58, 20)
        result = self.skip(inp @ (self.aa_embedding / self.aa_embedding.norm()))
        scaled_input = inp.transpose(1, 2) @ torch.diag(torch.diagonal(self.layers[0].weight))
        cnn_output = self.cnn_module(scaled_input.transpose(1, 2))
        return result + cnn_output

    def prox(self, *, lambda_, lambda_bar=0, M=1):
        with torch.no_grad():
            inplace_prox(
                beta=self.skip,
                theta=self.layers[0],
                lambda_=lambda_,
                lambda_bar=lambda_bar,
                M=M,
            )

    def l2_regularization(self):
        """
        L2 regulatization of the MLP without the first layer
        which is bounded by the skip connection
        """
        ans = 0
        for param in self.cnn_module.parameters():
            ans += (
                torch.norm(
                    param.data,
                    p=2,
                )
                ** 2
            )
        return ans

    def l1_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=2, dim=0).sum()

    def l2_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=2)

    def input_mask(self):
        with torch.no_grad():
            return torch.norm(self.skip.weight.data, p=2, dim=0) != 0

    def selected_count(self):
        return self.input_mask().sum().item()

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}



class CNNLassoNetClassifier(LassoNetClassifier):
    def __init__(
            self,
            *args,
            batch_size=512,
            n_iters=(30, 10),
            lambda_start=100,
            path_multiplier=1.2,
            **kwargs):
        super(CNNLassoNetClassifier, self).__init__(
            *args,
            batch_size=batch_size,
            n_iters=n_iters,
            lambda_start=lambda_start,
            path_multiplier=path_multiplier,
            **kwargs
        )

    def _init_model(self, X, y):
        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)
        self.model = CNNLassoNetModel(
            seq_len=58, num_residues=20
        ).to(self.device)
