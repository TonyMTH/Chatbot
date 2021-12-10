import pickle
import random
import json
import preprocess as pr
from parameters import *
import torch

torch.manual_seed(0)


class ChatBot:
    def __init__(self, model_path, all_words_PIK, data_file):
        self.model_path = model_path
        with open(all_words_PIK, "rb") as f:
            allwords_tags = pickle.load(f)
        with open(data_file, 'r') as json_data:
            self.intents = json.load(json_data)

        self.all_words = allwords_tags['all_words']
        self.tags = allwords_tags['tags']

    def predict(self, sentence):

        # Preprocess Data
        sentence = pr.tokenize(sentence)
        X = pr.bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        # Predict
        model = torch.load(self.model_path)
        model.eval()
        with torch.no_grad():
            output = model(X)
            _, pred = torch.max(output, dim=1)

        tag = self.tags[pred.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][pred.item()]


        return round(prob.item(),3), tag


if __name__ == '__main__':
    chatBot = ChatBot(model_path, all_words_PIK, data_file)

    bot_name = "Tony"
    print("Let's have a chat! (type 'exit' to stop chatting!)")

    while True:
        sentence = input("You: ")
        if sentence == "exit":
            print("Thanks for joining me!")
            break

        prob, tag = chatBot.predict(sentence)

        print(prob,tag)
        if prob > 0.75:
            for intent in chatBot.intents['intents']:
                if tag == intent['tag']:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand. Try to be more specific")

