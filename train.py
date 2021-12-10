from sklearn.model_selection import train_test_split

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
patterns, tags, all_words = pr.pattern_tag_words(intents,ignore_words,all_words_PIK)
X, y = pr.get_x_y(patterns,tags,all_words)

# Split Data
print('3.\tTrain/Test Split.......')
train_X, test_X, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)#, stratify=y

# Data Loaders
print('5.\tData Loaders.......')
train_loader = pr.load_data_gram(train_X, y_train, batch_size)
test_loader = pr.load_data_gram(test_X, y_test, batch_size)

# Define Model
model = md.Model(len(train_X[0]), hidden_size, len(tags))
model.to(device)
print("6.\tModel defined and moved to " + str(device.__str__()))

# Parameters
optimizer = Optimizer(model.parameters())
print("7.\tCriterion set as " + str(criterion.__str__()))
print("8.\tOptimizer set as " + str(optimizer.__str__()))
#
# Train Model
print("9.\tTrain loop")
pr.train_loop(model, epochs, optimizer, criterion, train_loader, test_loader,
              printing_gap, saved_model_device, model_path, device, PIK_plot_data)


if __name__ == '__main__':
    pass