import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Aura"

def get_response(sentence, context):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                context["last_tag"] = tag
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."

def save_context(context):
    with open('context.pkl', 'wb') as f:
        pickle.dump(context, f)

def load_context():
    try:
        with open('context.pkl', 'rb') as f:
            context = pickle.load(f)
    except FileNotFoundError:
        context = {}
    return context
