import torch
import json

torch.manual_seed(0)


def load_data(path):
    with open(path, 'r') as f:
        intents = json.load(f)
    return intents
