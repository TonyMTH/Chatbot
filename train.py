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
data = pr.clean_load(intents, fasttext_PIK, PIK, max_seq_len)
X, y, y_dic = data['x'], data['y'], data['y_dic']

# Split Data
print('2.\tTrain/Test Split.......')
train_X, test_X, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=0)

# Oversample Train Data
print('3.\tOversampling Train Data.......')
ros = RandomOverSampler(random_state=0)
train_X, y_train = ros.fit_resample(train_X, y_train)

# Data Loaders
print('4.\tData Loaders.......')
train_loader = pr.data_loader(train_X, y_train, batch_size)
test_loader = pr.data_loader(test_X, y_test, batch_size)

# Define Model
model = md.Model(max_seq_len, hidden_size, len(y_dic.keys()))
model.to(device)
print("2.\tModel defined and moved to " + str(device.__str__()))

# Parameters
optimizer = Optimizer(model.parameters())
print("3.\tCriterion set as " + str(criterion.__str__()))
print("4.\tOptimizer set as " + str(optimizer.__str__()))

if __name__ == '__main__':
    # print(y_test,y_train)
    pass