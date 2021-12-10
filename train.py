import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

import preprocess as pr
from parameters import *
import model as md

torch.manual_seed(0)

# Define Processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1.\t" + str(device.type).capitalize() + " detected")

# Fetch Data
intents = pr.load_data(data_file)

# Preprocess Data
print('2.\tProcessing data.......')
data = pr.clean_load(intents, fasttext_PIK, y_keys_PIK, PIK, max_seq_len)
X, y, y_dic, vec = data['x'], data['y'], data['y_dic'], data['vec']


# Split Data
print('3.\tTrain/Test Split.......')
train_X, test_X, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)#, stratify=y
# test_X, y_test = train_X, y_train

# Oversample Train Data
# print('4.\tOversampling Train Data.......')
# ros = RandomOverSampler(random_state=0)
# train_X, y_train = ros.fit_resample(train_X, y_train)

# Data Loaders
print('5.\tData Loaders.......')
train_loader = pr.data_loader(train_X, y_train, batch_size, vec, max_seq_len)
test_loader = pr.data_loader(test_X, y_test, batch_size, vec, max_seq_len)

# Define Model
# model = md.Model(max_seq_len, emb_dim, hidden1, hidden2, hidden3, len(y_dic.keys()))
model = md.Model2(max_seq_len, emb_dim, 8, len(y_dic.keys()))
model.to(device)
print("6.\tModel defined and moved to " + str(device.__str__()))

# Parameters
optimizer = Optimizer(model.parameters())
print("7.\tCriterion set as " + str(criterion.__str__()))
print("8.\tOptimizer set as " + str(optimizer.__str__()))

# Train Model
print("9.\tTrain loop")
pr.train_loop(model, epochs, optimizer, criterion, train_loader, test_loader, emb_dim,
              printing_gap, saved_model_device, model_path, device, max_seq_len, PIK_plot_data)

if __name__ == '__main__':
    # print(y_test,y_train)
    pass