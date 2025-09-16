import torch
import torch.nn as nn
import random
import json
import nltk
from nltk.stem import PorterStemmer
from pathlib import Path

# --- INITIAL SETUP ---
# Ensure the 'punkt' tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Ensure the 'punkt_tab' resource is downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Initialize stemmer
stemmer = PorterStemmer()

# --- NEURAL NETWORK DEFINITION ---
# The model class must be defined here so we can load the saved state
class ChatbotV1(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=256):
        super().__init__()
        self.layer_1 = nn.Linear(input_features, hidden_units)
        self.layer_2 = nn.Linear(hidden_units, hidden_units)
        self.layer_3 = nn.Linear(hidden_units, hidden_units)
        self.classifier = nn.Linear(hidden_units, output_features)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = torch.relu(self.layer_3(x))
        return self.classifier(x)

# --- UTILITY FUNCTIONS ---
def tokenize(text):
    return nltk.word_tokenize(text)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(all_words, input_words):
    bag = torch.zeros(len(all_words), dtype=torch.float32)
    stemmed_input = [stem(w) for w in input_words]
    for idx, w in enumerate(all_words):
        if w in stemmed_input:
            bag[idx] = 1.0
    return bag

# --- CORE INFERENCE FUNCTIONS ---
MODEL_FILE = Path('models/chatbotv1.pth')

def load_model(input_features, output_features, hidden_units=256, model_path=MODEL_FILE):
    """Loads the pre-trained model from disk."""
    model = ChatbotV1(input_features, output_features, hidden_units)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_response(text, model, all_words, tags, data, threshold=0.75):
    """Generates a response from the chatbot model."""
    tokens = tokenize(text)
    # Note: The original bag_of_words function already stems the input words.
    bow = bag_of_words(all_words, tokens).unsqueeze(0)  # Add batch dimension

    # Model inference
    with torch.inference_mode():
        logits = model(bow)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        tag = tags[pred_idx.item()]

        if confidence.item() < threshold:
            return "Sorry, I didn't understand that. Can you rephrase?"

        for intent in data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    
    return "I'm not sure how to respond to that."