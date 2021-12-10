import copy
import pickle
from pathlib import Path

import numpy as np
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
    doc = nlp(str(sentence))#.replace("!", " ").replace("!", " ").replace("?", " ").replace(".", " ")
    tokens = [token.text.lower() for token in doc if not token.is_punct]# and not token.is_stop]
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


def collate(batch, vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor([item[1] for item in batch])  # Use long tensor to avoid unwanted rounding
    return inputs, target


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


def clean_load(intents, fasttext_PIK, y_keys_PIK, PIK, max_seq_len):
    if not Path(PIK).is_file():
        clean(intents, fasttext_PIK, y_keys_PIK, PIK, max_seq_len)
    with open(PIK, "rb") as f:
        data = pickle.load(f)
    return data


def clean(intents, fasttext_PIK, y_keys_PIK, PIK, max_seq_len):
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

    data = {'x': x, 'y': y, 'y_dic': y_dic, 'vec': vec}

    print("Dumping file")
    with open(PIK, "wb") as f:
        pickle.dump(data, f)
    with open(y_keys_PIK, "wb") as f:
        pickle.dump(y_dic, f)
    print("Done Dumping")


class ChatData(Dataset):
    def __init__(self, x, y, vec, max_seq_len):
        self.x_data = x
        self.y_data = y
        self.max_seq_len = max_seq_len
        self.vec = vec
        self.vectorizer = self.get_vectorize

    def __getitem__(self, index):
        assert len(self.x_data[index]) == self.max_seq_len
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

    def get_vectorize(self, x):
        return self.vec.vectors[x]


def data_loader(x, y, batch_size, vec, max_seq_len, num_workers=0):
    dataset = ChatData(x, y, vec, max_seq_len)
    data_collate = lambda batch: collate(batch, vectorizer=dataset.vectorizer)
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=data_collate, shuffle=True, num_workers=num_workers)


def train_loop(model, epochs, optimizer, criterion, train_loader, test_loader, emb_dim,
               printing_gap, saved_model_device, model_path, device, MAX_SEQ_LEN, PIK_plot_data):
    train_loss = []
    train_acc = []
    test_acc = []
    greatest_test_accu = -np.inf

    for epoch in range(epochs):
        loss_train = 0

        for sentences, labels in train_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            sentences.resize_(sentences.size()[0], MAX_SEQ_LEN * emb_dim)
            # sentences = sentences.requires_grad_()

            optimizer.zero_grad()

            output = model.forward(sentences)  # 1) Forward pass
            loss = criterion(output, labels)  # 2) Compute loss
            loss.backward()  # 3) Backward pass
            optimizer.step()  # 4) Update model

            loss_train += loss.item()

        model.eval()

        with torch.no_grad():
            train_num_correct = 0
            train_num_samples = 0
            for sentences, labels in iter(train_loader):
                sentences, labels = sentences.to(device), labels.to(device)
                sentences.resize_(sentences.size()[0], MAX_SEQ_LEN * emb_dim)

                output = model(sentences)
                _, predictions = output.max(1)

                train_num_correct += (predictions == labels).sum()
                train_num_samples += predictions.size(0)

        with torch.no_grad():
            test_num_correct = 0
            test_num_samples = 0
            for sentences, labels in iter(test_loader):
                sentences, labels = sentences.to(device), labels.to(device)
                sentences.resize_(sentences.size()[0], MAX_SEQ_LEN * emb_dim)

                output = model(sentences)
                _, predictions = output.max(1)

                test_num_correct += (predictions == labels).sum()
                test_num_samples += predictions.size(0)

        train_accu = float(train_num_correct) / train_num_samples * 100
        test_accu = float(test_num_correct) / test_num_samples * 100

        train_loss.append(loss_train / train_num_samples)
        train_acc.append(train_accu)
        test_acc.append(test_accu)

        # Save best model
        if train_accu >= greatest_test_accu:
            greatest_test_accu = train_accu

            best_model_state = copy.deepcopy(model)
            best_model_state.to(saved_model_device)
            torch.save(best_model_state, model_path)

        if epoch % printing_gap == 0:
            print('Epoch: {}/{}\t.............'.format(epoch, epochs), end=' ')
            print("Train Loss: {:.4f}".format(loss_train / train_num_samples), end=' ')
            print("Train Acc: {:.4f}".format(train_accu), end=' ')
            print("Test Acc: {:.4f}".format(test_accu), end=' ')
            print("Best Test Acc: {:.4f}".format(greatest_test_accu))

            # Save data to pickle
            data = {'train_loss': train_loss, 'train_acc': train_acc, 'test_acc': test_acc}
            with open(PIK_plot_data, "wb") as f:
                pickle.dump(data, f)

        model.train()
