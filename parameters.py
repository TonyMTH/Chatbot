import torch
from torch import nn, optim

data_file = 'data/intents.json'
fasttext_PIK = 'data/fasttext.dat'
y_keys_PIK = 'data/y_keys.dat'
max_seq_len = 4
PIK = "data/data_pickle.dat"
batch_size = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.001
criterion = nn.CrossEntropyLoss()
Optimizer = lambda x: optim.Adam(x, lr=lr)

epochs = 1000
printing_gap = 50
model_path = 'data/best_model.pt'
saved_model_device = torch.device("cpu")

hidden1, hidden2, hidden3 = 16, 16, 16

PIK_plot_data = './data/plot_data.dat'
emb_dim = 300


ignore_words = ['?', '!', '.']
num_epoches = 1000
batch_size = 8
learning_rate = 0.001
# input_size = len(X_train[0])
hidden_size = 8
all_words_PIK = './data/all_words.dat'
