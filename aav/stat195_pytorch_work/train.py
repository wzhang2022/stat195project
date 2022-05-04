import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

from aav.stat195_pytorch_work.torch_cnn_lassonet import CNNBaseline, CNNModel
from aav.util import config
from aav.util.mutation_encoding import MutationSequenceEncoder
from aav.util.residue_encoding import ResidueIdentityEncoder


# class SequenceDataset(Dataset):
#     def __init__(self, sequence_file: str):
#         self.mutation_encoder = MutationSequenceEncoder(
#             ResidueIdentityEncoder(config.RESIDUES),
#             config.R1_TILE21_WT_SEQ
#         )
#         df = pd.read_csv(sequence_file)
#         self.sequences = []
#         self.labels = []
#         # df = df[df['partition'] == 'rand']
#         for i, row in df.iterrows():
#             seq, label = row['sequence'], row['is_viable']
#             try:
#                 np_seq = self.mutation_encoder.encode(seq).astype(np.float32)
#                 self.sequences.append(np_seq.reshape(-1, np_seq.shape[-1]))
#                 self.labels.append(label)
#             except:
#                 pass
#
#     def __getitem__(self, idx):
#         return self.sequences[idx], self.labels[idx]
#
#     def __len__(self):
#         return len(self.sequences)


def train(net, train_loader, optimizer, criterion, num_epochs, device):
    net.train()
    for epoch in range(num_epochs):
        for i, (seq, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(seq.to(device))
            loss = criterion(outputs, labels.long().to(device))
            loss.backward()
            optimizer.step()


def evaluate(net, test_loader, device):
    net.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for i, (seq, labels) in enumerate(test_loader):
            outputs = net(seq.to(device))
            y_pred.append(outputs.argmax(dim=1).cpu().numpy())
            y_true.append(labels.numpy())
        y_pred, y_true = np.concatenate(y_pred, axis=0), np.concatenate(y_true, axis=0)
        print(f"Accuracy is {sum(y_pred == y_true) / len(y_pred)}")


if __name__ == "__main__":
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    df = pd.read_csv("../../stat_195_project_data.csv")
    # dataset = SequenceDataset("../../stat_195_project_data.csv")
    df = df[df['partition'] != 'stop'].reset_index()
    sequences = []
    mutation_encoder = MutationSequenceEncoder(
        ResidueIdentityEncoder(config.RESIDUES),
        config.R1_TILE21_WT_SEQ
    )
    for i, row in df.iterrows():
        seq = row['sequence']
        np_seq = mutation_encoder.encode(seq).astype(np.float32)
        sequences.append(np_seq.reshape(1, -1, np_seq.shape[-1]))
    sequences = np.concatenate(sequences, axis=0)


    train_set, test_set = train_test_split(df, test_size=0.2)
    model = CNNModel()
    print(sequences[train_set.index.values].shape)
    model.path(sequences[train_set.index.values], train_set['is_viable'].values)
    y_pred = model.predict(sequences[test_set.index.values])
    print(f"Accuracy is {sum(y_pred == test_set['is_viable']) / len(y_pred)}")
