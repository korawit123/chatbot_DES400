import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
#print(input_size)
hidden_size = data["hidden_size"]
#print(hidden_size)
output_size = data["output_size"]
#print(output_size)
all_words = data['all_words']
#print(all_words)
tags = data['tags']
#print(tags)
model_state = data["model_state"]
#print(model_state)

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
#print(model.eval())

bot_name = "KOYA"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    #print(sentence)
    X = bag_of_words(sentence, all_words)
    #print(X)
    X = X.reshape(1, X.shape[0])
    #print(X)
    X = torch.from_numpy(X).to(device)
    #print(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    #print(tag)
    probs = torch.softmax(output, dim=1)
    #print(probs)
    prob = probs[0][predicted.item()]
    #print(prob.item())
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                try:
                    if intent["responses2"]:
                        for res in intent["responses2"]:
                            print( f"{bot_name}: {res}")
                except:
                    pass
    else:
        print(f"{bot_name}:Sorry! I do not understand this Question.")