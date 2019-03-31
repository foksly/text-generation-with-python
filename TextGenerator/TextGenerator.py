import os
import pickle
from collections import Counter
from tqdm import tqdm
from nltk import ngrams
import numpy as np


class TextGenerator():
    def __init__(self, method, analyzer, n_grams):
        self.method = method
        self.analyzer = analyzer
        self.n_grams = n_grams
        
        self.model = None

    def prepare_for_genetation(self, data_folder_path,
                               sos='.!?', encoding=None,
                               tokenizer=None, seq_len=None,
                               validation_set=False, validation_size=0.2):
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

            validation_set: True if validation set is necessary, else False
                type: bool
            ------------------------------------------------------------------

            validation_size: size of validation set
                type: float between 0 and 1
            ------------------------------------------------------------------
        
        examples:
            >>> prepare_for_genetation()
        """
        sos_token = '<sos>'
        files = list(filter(lambda x: '.txt' in x,
                            os.listdir(data_folder_path)))
        data = []
        # reading files from data_path folder
        for file in tqdm(files):
            if encoding:
                with open(data_folder_path + file,
                          'r', encoding=encoding) as f:
                    text = f.read()
            else:
                with open(data_folder_path + file, 'r') as f:
                    text = f.read()
            if self.analyzer == 'word':
                # replacing eos to sos_token
                for p in eos:
                    text = text.replace(p, ' ' + sos_token + ' ')

                # tokenization
                if tokenizer:
                    data.extend(tokenizer(text.lower()))
                else:
                    prog = re.compile('[А-яа-яёA-Za-z<>]+')
                    data.extend(prog.findall(text.lower()))

            elif self.analyzer == 'char':
                data.extend(text)

        if self.method == 'n_grams':
            return data
            # data = list(ngrams(data, self.n_grams))

        elif self.method == 'RNN':
            vocab = set(data)
            word2id = {k: v for v, k in enumerate(vocab)}
            id2word = {v: k for k, v in word2id.items()}
            data = [word2id[w] for w in data]
            return data, id2word

    def fit(self, data, save_path=None):
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
        probs = {}
        data_grams = Counter(ngrams(data, self.n_grams))
        if self.method == 'n_grams':
            for grams in tqdm(data_grams):
                if grams[:self.n_grams-1] in probs:
                    probs[grams[:self.n_grams-1]][grams[-1]] = data_grams[grams]
                else:
                    probs[grams[:self.n_grams-1]] = {grams[-1]: data_grams[grams]}
        self.model = probs

    def generate(self, pretrained_model=None, 
                 generate_len=10, start_chars=None):
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
                    sos = list(filter(lambda x: x[0] == '<sos>', pretrained_model))
                    start = list(sos[np.random.choice(len(sos))])
                    temp_gram = tuple(start)
                elif self.analyzer == 'char':
                    assert len(start_chars) >= self.n_grams-1
                    start = [c for c in start_chars]
                    temp_gram = tuple(start[-(self.n_grams-1):])

                for i in range(generate_len - self.n_grams + 1):
                    possible_tokens = list(pretrained_model[temp_gram].items())
                    freqs = [i[1] for i in possible_tokens]
                    probs = np.array(list(freqs)) / sum(freqs)
                    generated_word = np.random.choice([i[0] for i in possible_tokens], p=probs)
                    temp_gram = (*temp_gram[1:], generated_word)
                    start.append(generated_word)
                if self.analyzer == 'word':
                    return ' '.join(start)
                elif self.analyzer == 'char':
                    return ''.join(start)
