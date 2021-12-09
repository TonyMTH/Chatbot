import torch
import preprocess as pr
from parameters import *
torch.manual_seed(0)

# Define Processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1.\t"+str(device.type).capitalize()+" detected")

# Fetch Data
df_train = pr.load_data(data_file)




if __name__ == '__main__':
    print(df_train)