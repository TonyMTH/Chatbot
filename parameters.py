import torch
from torch import nn, optim

data_file = 'data/intents.json'
fasttext_PIK = 'data/fasttext.dat'
max_seq_len = 10
PIK = "data/data_pickle.dat"
batch_size = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.01
criterion = nn.CrossEntropyLoss()
Optimizer = lambda x: optim.Adam(x, lr=lr)

epochs = 10
printing_gap = 1
model_path = 'data/best_model.pt'
saved_model_device = torch.device("cpu")

hidden_size = 8
