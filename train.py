from torch.utils.data import DataLoader
import torch
# import model as md
import preprocess as pr
from parameters import *
torch.manual_seed(0)

# Define Processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1.\t"+str(device.type).capitalize()+" detected")

# Fetch Data
df_train = pr.load_data(data_file)

print(df_train)