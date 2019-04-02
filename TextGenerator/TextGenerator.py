import os
import pickle
from collections import Counter
from tqdm import tqdm
from nltk import ngrams
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

import .NeuralLanguageModel
from .utils import save_pkl, load_pkl


class TextGenerator():
    def __init__(self, method, analyzer, n_grams=None):
        self.method = method
        self.analyzer = analyzer
        self.n_grams = n_grams

        self.model = None

    def prepare_for_genetation(self,
                               data_folder_path,
                               sos='.!?',
                               encoding=None,
                               tokenizer=None,
                               seq_len=None,
                               batch_size=64,
                               validation_set=False,
                               validation_size=0.2,
                               random_seed=42):
        """
        Prepares all of .txt data which contains in data_folder_path

        args:
            data_folder_path: a path to the folder that contains all .txt data
                              for current generation session
                type: 'str'
                example: 'data/IMDB'
            ------------------------------------------------------------------

            sos: an iterable object that contains all of the tokens
                 that should be converted to <sos> token
                type: iterable object
                example: '.!?' or ['.', '\n']
            ------------------------------------------------------------------

            encoding: encodding of the .txt data inside data_folder_path
                type: str
            ------------------------------------------------------------------

            tokenizer: tokenizer function
                type: func(str)
                example: >>> tokenizer('the quick brown fox')
                     ['the', 'quick', 'brown', 'fox']
            ------------------------------------------------------------------

            seq_len: formats data to a list of lists of sizes seq_len
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

        examples:
            >>> prepare_for_genetation()
        """
        sos_token = '<sos>'
        files = list(
            filter(lambda x: '.txt' in x, os.listdir(data_folder_path)))
        data = []

        # reading and preprocessing files from data_path folder
        for file in tqdm(files):
            if encoding:
                with open(
                        data_folder_path + file, 'r', encoding=encoding) as f:
                    text = f.read()
            else:
                with open(data_folder_path + file, 'r') as f:
                    text = f.read()
            if self.analyzer == 'word':
                for p in sos:
                    text = text.replace(p, ' ' + sos_token + ' ')

                if tokenizer:
                    data.extend(tokenizer(text.lower()))
                else:
                    prog = re.compile('[А-яа-яёA-Za-z<>]+')
                    data.extend(prog.findall(text.lower()))

            elif self.analyzer == 'char':
                data.extend(text)

        # preparing data for n_gram method of text generation
        if self.method == 'n_grams':
            return data

        # preparing data for rnn method of text generation
        elif self.method == 'RNN':
            data = NeuralLanguageModel.prepare_dataloaders(data,
                                                           seq_len,
                                                           batch_size,
                                                           validation_set,
                                                           validation_size,
                                                           random_seed)
            return data

    def fit(self, data, hidden_dim,
            vocab_size, embedding_dim,
            n_layers, rnn_type='LSTM',
            dropout=0, train_on_gpu=True,
            save_path=None, learning_rate=0.001,
            n_epochs=30):
        """
        Trains a text generation model

        args:
            data: prepared text data
                type: list of tokens if self.method == 'n_grams'
                      torch.DataLoaders if self.method == 'RNN'
            ------------------------------------------------------------------

            save_path: path where to save a model in pkl format
                type: str
            ------------------------------------------------------------------

        """
        if self.method == 'n_grams':
            probs = {}
            data_grams = Counter(ngrams(data, self.n_grams))
            if self.method == 'n_grams':
                for grams in tqdm(data_grams):
                    if grams[:self.n_grams - 1] in probs:
                        probs[grams[:self.n_grams -
                                    1]][grams[-1]] = data_grams[grams]
                    else:
                        probs[grams[:self.n_grams - 1]] = {
                            grams[-1]: data_grams[grams]
                        }

            self.model = probs

        if self.method == 'rnn':
            train_loader, vocab, token2id, id2token = data

            hidden_dim = hidden_dim
            vocab_size = vocab_size
            embedding_dim = embedding_dim
            n_layers = n_layers
            rnn_type = rnn_type
            dropout = dropout
            train_on_gpu = train_on_gpu

            rnn = NeuralLanguageModel.NeuralLanguageModel(
                hidden_dim, vocab_size, embedding_dim, n_layers)
            if train_on_gpu:
                rnn.cuda()
            optimizer = optim.Adam(rnn.params(), learning_rate)
            NeuralLanguageModel.train(rnn, optimizer,
                                      n_epochs, train_loader,
                                      print_every=1, save_path='rnn_model.pt')

    def generate(self,
                 pretrained_model=None,
                 generate_len=10,
                 start_chars=None):
        """
        Generates text of generate_len length

        args:
            pretrained_model: None if the model was trained using self.fit
                              method otherwise a path to the pretrained model
                type: None or str
            ------------------------------------------------------------------

            generate_len: length of text to generate
                type: int
            ------------------------------------------------------------------

            start_chars: start of the sentence for char level text generation
                type: str
            ------------------------------------------------------------------
        """
        if not pretrained_model:
            pretrained_model = self.model
        if self.method == 'n_grams':
            if isinstance(pretrained_model, str):
                pass
            else:
                if self.analyzer == 'word':
                    sos = list(
                        filter(lambda x: x[0] == '<sos>', pretrained_model))
                    start = list(sos[np.random.choice(len(sos))])
                    temp_gram = tuple(start)
                elif self.analyzer == 'char':
                    assert len(start_chars) >= self.n_grams - 1
                    start = [c for c in start_chars]
                    temp_gram = tuple(start[-(self.n_grams - 1):])

                for i in range(generate_len - self.n_grams + 1):
                    possible_tokens = list(pretrained_model[temp_gram].items())
                    freqs = [i[1] for i in possible_tokens]
                    probs = np.array(list(freqs)) / sum(freqs)
                    generated_word = np.random.choice(
                        [i[0] for i in possible_tokens], p=probs)
                    temp_gram = (*temp_gram[1:], generated_word)
                    start.append(generated_word)
                if self.analyzer == 'word':
                    return ' '.join(start[1:])
                elif self.analyzer == 'char':
                    return ''.join(start)
