import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output


class NeuralLanguageModel(nn.Module):
    def __init__(self,
                 hidden_dim,
                 vocab_size,
                 embedding_dim,
                 n_layers=1,
                 rnn_type='LSTM',
                 dropout=0,
                 train_on_gpu=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.train_on_gpu = train_on_gpu

        # Embeddings layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # RNN Layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                n_layers,
                dropout=dropout,
                batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                n_layers,
                dropout=dropout,
                batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                n_layers,
                dropout=dropout,
                batch_first=True)

        # Fully-connected layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        embedded_input = self.embeddings(input)
        rnn_output, _ = self.rnn(embedded_input, hidden)
        rnn_output = rnn_output.contiguous().view(-1, self.hidden_dim)
        output = self.fc(rnn_output)

        return output

    def init_hidden(self, batch_size):
        """
        Initializes hidden state vector
        """
        weight = next(self.parameters()).data
        if self.train_on_gpu:
            if self.rnn_type == 'LSTM':
                hidden = (weight.new(self.n_layers, batch_size,
                                     self.hidden_dim).zero_().cuda(),
                          weight.new(self.n_layers, batch_size,
                                     self.hidden_dim).zero_().cuda())
            else:
                hidden = (weight.new(self.n_layers, batch_size,
                                     self.hidden_dim).zero_().cuda())
        else:
            if self.rnn_type == 'LSTM':
                hidden = (weight.new(self.n_layers, batch_size,
                                     self.hidden_dim).zero_(),
                          weight.new(self.n_layers, batch_size,
                                     self.hidden_dim).zero_())
            else:
                hidden = (weight.new(self.n_layers, batch_size,
                                     self.hidden_dim).zero_())
        return hidden


def prepare_dataloaders(data,
                        seq_len,
                        batch_size=64,
                        validation_set=False,
                        validation_size=0.1,
                        random_seed=42):
    """
    Prepares DataLoaders for 'rnn' generation method

    args:
        data:
            type: list of tokens
            example: ['the', 'quick', 'brown', 'fox']
        ------------------------------------------------------------------

        seq_len: length of sequences for rnn
            type: int
        ------------------------------------------------------------------

        batch_size: size of batches
            type: int
        ------------------------------------------------------------------

        validation_set: True if validation set is necessary, else False
            type: bool
        ------------------------------------------------------------------

        validation_size: size of validation set
            type: float between 0 and 1
        ------------------------------------------------------------------

        random_seed:
            type: int
        ------------------------------------------------------------------
    """
    vocab = set(data)
    token2id = {k: v for v, k in enumerate(vocab)}
    id2token = {k: v for v, k in token2id.items()}
    data_range = range(0, len(data) - seq_len, seq_len)

    data = [token2id[t] for t in data]
    data = np.array([data[i:i + seq_len] for i in data_range])
    tensor_data = torch.from_numpy(data)

    if validation_set:
        np.random.seed(random_seed)
        idx = np.random.choice(
            range(len(tensor_data)), size=len(tensor_data), replace=False)
        split = int(len(idx) * (1 - validation_size))
        train_idx = idx[:split]
        valid_idx = idx[split:]

        train_data = TensorDataset(torch.LongTensor(tensor_data[train_idx]))
        valid_data = TensorDataset(torch.LongTensor(tensor_data[valid_idx]))

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        valid_loader = DataLoader(
            valid_data, shuffle=True, batch_size=batch_size)

        return train_loader, valid_loader, vocab, token2id, id2token
    else:
        train_data = TensorDataset(torch.LongTensor(tensor_data))
        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        return train_loader, vocab, token2id, id2token


# --------------------------------------------------------------------------------------------
# Train


def eval_epoch(model, eval_loader, eval_on_gpu=True):
    criterion = nn.CrossEntropyLoss()
    loss_log = []
    model.eval()
    for sequence in eval_loader:
        # init hidden
        h = model.init_hidden(sequence[0].size(0))
        h = tuple([each.data for each in h])
        # switch to gpu/cpu
        if eval_on_gpu:
            X = sequence[0][:, :-1].cuda()
            y = sequence[0][:, 1:].cuda()
        else:
            X = sequence[0][:, :-1]
            y = sequence[0][:, 1:]

        output, hidden = model(X, h)
        loss = criterion(output, y.contiguous().view(-1))
        loss_log.append(loss.item())
    return loss_log


def train_epoch(model, optimizer, train_loader, train_on_gpu=True):
    criterion = nn.CrossEntropyLoss()
    loss_log = []
    model.train()
    for sequence in tqdm(train_loader):
        optimizer.zero_grad()
        h = model.init_hidden(sequence[0].size(0))
        h = tuple([each.data for each in h])
        if train_on_gpu:
            X = sequence[0][:, :-1].cuda()
            y = sequence[0][:, 1:].cuda()
        else:
            X = sequence[0][:, :-1]
            y = sequence[0][:, 1:]
        output, hidden = model(X, h)
        loss = criterion(output, y.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())
    return loss_log


def plot_history(train_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)
    plt.xlabel('train steps')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def train(model, opt, n_epochs,
          train_loader,
          valid_loader=None,
          save_every=1,
          perplexity_break=106,
          print_every=5,
          train_on_gpu=True,
          save_to_disk=True,
          path='pretrained_model.pt'):
    train_log = []
    total_steps = 0

    if train_on_gpu:
        model.cuda()
    for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        train_loss = train_epoch(
            model, opt, train_loader, train_on_gpu=train_on_gpu)
        train_log.extend(train_loss)
        total_steps += len(train_loader)

        if valid_loader is not None:
            perplexity = 2**np.mean(
                eval_epoch(
                    eval_loader=valid_loader, eval_on_gpu=True, model=model))

            if perplexity > perplexity_break:
                print('Desired perplexity has been successfully achieved!')
                break
            print('Validation perplexity: ', perplexity)

        clear_output()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                   np.mean(train_log[-100:])))
        plot_history(train_log)

        if epoch % save_every == 0:
            if save_to_disk:
                torch.save(model.state_dict(), path)


def eval_model(model, eval_loader, eval_on_gpu=True):
    eval_log = []

    if eval_on_gpu:
        model.cuda()
    eval_loss = eval_epoch(model, eval_loader)
    eval_log.extend(eval_loss)

    clear_output()
    plot_history(eval_log)
    return eval_log
