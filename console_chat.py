import pickle
import random
import json
import preprocess as pr
from parameters import *
import torch

torch.manual_seed(0)


class ChatBot:
    def __init__(self, model_path, fasttext_PIK, y_keys_PIK, max_seq_len, emb_dim, data_file):
        self.model_path = model_path
        self.vec = pr.load_fasttext(fasttext_PIK)
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        with open(y_keys_PIK, "rb") as f:
            y_keys = pickle.load(f)
        with open(data_file, 'r') as json_data:
            intents = json.load(json_data)

        self.y_keys = {v: k for k, v in y_keys.items()}
        self.intents = intents

    def predict(self, sentence):
        # Preprocess Data
        print(pr.preprocessing(sentence))
        sequence = pr.padding(pr.encoder(pr.preprocessing(sentence), self.vec), max_seq_len=self.max_seq_len)
        print(sequence)

        sequence = torch.Tensor([sequence])
        sequence.resize_(sequence.size()[0], self.max_seq_len * self.emb_dim)

        # Predict
        model = torch.load(self.model_path)
        model.eval()
        # with torch.no_grad():
        output = model(sequence)
        _, pred = torch.max(output, dim=1)

        probs = torch.softmax(output, dim=1)
        prob = probs[0][pred.item()]

        print(self.y_keys)


        return prob, pred.item()


if __name__ == '__main__':
    chatBot = ChatBot(model_path, fasttext_PIK, y_keys_PIK, max_seq_len, emb_dim, data_file)

    bot_name = "Tony"
    print("Let's have a chat! (type 'exit' to stop chatting!)")

    while True:
        sentence = input("You: ")
        if sentence == "exit":
            print("Thanks for joining me!")
            break

        prob, tag = chatBot.predict(sentence)

        print(prob,tag)
        if prob.item() > 0.75:
            for intent in chatBot.intents['intents']:
                if tag == intent['tag']:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand. Try to be more specific")
