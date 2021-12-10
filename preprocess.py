import pickle
from pathlib import Path

import spacy
import torch
import json

from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import FastText

torch.manual_seed(0)


def load_data(path):
    with open(path, 'r') as f:
        intents = json.load(f)
    return intents


def train_test_split(df, test=0.3):
    idx = int(df.shape[0] * (1 - test))
    return df.iloc[:idx, :], df.iloc[idx:, :]


def preprocessing(sentence):
    """
    params sentence: a str containing the sentence we want to preprocess
    return the tokens list
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(str(sentence))
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_stop]
    return tokens


def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0


def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]


def padding(list_of_indexes, max_seq_len, padding_index=1):
    output = list_of_indexes + (max_seq_len - len(list_of_indexes)) * [padding_index]
    return output[:max_seq_len]


def load_fasttext(PIK):
    if not Path(PIK).is_file():
        vec = FastText()
        vec.vectors[1] = -torch.ones(vec.vectors[1].shape[0])
        vec.vectors[0] = torch.zeros(vec.vectors[0].shape[0])
        with open(PIK, "wb") as f:
            pickle.dump(vec, f)
    with open(PIK, "rb") as f:
        vec = pickle.load(f)
    return vec


def clean_load(intents, fasttext_PIK, PIK, max_seq_len):
    if not Path(PIK).is_file():
        clean(intents, fasttext_PIK, PIK, max_seq_len)
    with open(PIK, "rb") as f:
        data = pickle.load(f)
    return data


def clean(intents, fasttext_PIK, PIK, max_seq_len):
    vec = load_fasttext(fasttext_PIK)
    x, y = [], []

    y_dic = {}
    last_index = 0

    for intent in intents['intents']:
        if intent['tag'] not in y_dic.keys():
            y_dic[intent['tag']] = last_index
            last_index += 1
        y_val = y_dic[intent['tag']]
        for pattern in intent['patterns']:
            y.append(y_val)
            sequence = padding(encoder(preprocessing(pattern), vec), max_seq_len=max_seq_len)
            x.append(sequence)

    data = {'x': x, 'y': y, 'y_dic': y_dic}

    print("Dumping file")
    with open(PIK, "wb") as f:
        pickle.dump(data, f)
    print("Done Dumping")


class ChatData(Dataset):
    def __init__(self, x, y):
        self.n_samples = len(x)
        self.x_data = x
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def data_loader(x, y, batch_size, num_workers=0):
    dataset = ChatData(x, y)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


