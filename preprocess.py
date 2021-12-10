import copy
import pickle
import nltk
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
torch.manual_seed(0)


def load_data(path):
    with open(path, 'r') as f:
        intents = json.load(f)
    return intents


class ChatDataGram(Dataset):

    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x = X
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def load_data_gram(X, y, batch_size):
    dataset = ChatDataGram(X, y)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag


def pattern_tag_words(intents, ignore_words, all_words_PIK):
    all_words = []
    tags = []
    patterns = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            patterns.append((w, tag))

    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    print("Dumping all_words")
    with open(all_words_PIK, "wb") as f:
        pickle.dump({'all_words': all_words, 'tags': tags}, f)
    print("Done Dumping")

    return patterns, tags, all_words


def get_x_y(patterns, tags, all_words):
    X_train = []
    Y_train = []
    for (pattern_sentence, tag) in patterns:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        Y_train.append(label)
    return np.array(X_train), np.array(Y_train)


def train_loop(model, epochs, optimizer, criterion, train_loader, test_loader,
               printing_gap, saved_model_device, model_path, device, PIK_plot_data):
    train_loss = []
    train_acc = []
    test_acc = []
    greatest_test_accu = -np.inf

    for epoch in range(epochs):
        loss_train = 0

        for sentences, labels in train_loader:
            sentences, labels = sentences.to(device), labels.to(dtype=torch.long).to(device)

            output = model(sentences)  # 1) Forward pass
            loss = criterion(output, labels)  # 2) Compute loss
            optimizer.zero_grad()
            loss.backward()  # 3) Backward pass
            optimizer.step()  # 4) Update model

            loss_train += loss.item()

        model.eval()

        with torch.no_grad():
            train_num_correct = 0
            train_num_samples = 0

            for sentences, labels in iter(train_loader):
                sentences, labels = sentences.to(device), labels.to(dtype=torch.long).to(device)

                output = model(sentences)
                _, predictions = output.max(1)
                train_num_correct += (predictions == labels).sum()
                train_num_samples += predictions.size(0)

        with torch.no_grad():
            test_num_correct = 0
            test_num_samples = 0
            for sentences, labels in iter(test_loader):
                sentences, labels = sentences.to(device), labels.to(dtype=torch.long).to(device)

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
        if test_accu >= greatest_test_accu:
            greatest_test_accu = test_accu

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
