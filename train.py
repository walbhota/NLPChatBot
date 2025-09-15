import json
import torch
import string
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm import tqdm


# Import the necessary components from your refactored ecommerce.py
from ecommerce import tokenize, stem, bag_of_words, ChatbotV1

# --- 1. DATA LOADING & PREPROCESSING ---
print("Loading intents data...")
with open('Ecommerce_FAQ_intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []  # Will hold pairs of (tokenized_pattern, tag)

print("Processing intents...")
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        # Stemming is applied here
        stemmed_tokens = [stem(w) for w in tokens if w not in string.punctuation]
        all_words.extend(stemmed_tokens)
        xy.append((stemmed_tokens, tag))

# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"\n{len(xy)} patterns processed.")
print(f"{len(tags)} unique tags: {tags}")
print(f"{len(all_words)} unique stemmed words.")


# --- 2. CREATE TRAINING DATA ---
print("\nCreating training data (Bag of Words)...")
x_train = []
y_train = []
for (pattern_words, tag) in xy:
    # Bag of words for each pattern
    bow_vec = bag_of_words(all_words, pattern_words)
    x_train.append(bow_vec)
    
    # Corresponding tag index
    label = tags.index(tag)
    y_train.append(label)

x_train = torch.stack(x_train)
y_train = torch.tensor(y_train, dtype=torch.long)


# --- 3. MODEL TRAINING ---
# Hyperparameters
BATCH_SIZE = 8
INPUT_FEATURES = len(all_words)
OUTPUT_FEATURES = len(tags)
HIDDEN_UNITS = 256
LEARNING_RATE = 0.001
EPOCHS = 50

# DataLoader
chat_dataset = TensorDataset(x_train, y_train)
chat_dataloader = DataLoader(chat_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = ChatbotV1(input_features=INPUT_FEATURES, 
                  output_features=OUTPUT_FEATURES, 
                  hidden_units=HIDDEN_UNITS)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStarting model training...")
for epoch in range(EPOCHS):
    for (words, labels) in chat_dataloader:
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

print("\nTraining complete.")


# --- 4. SAVE THE MODEL ---
MODEL_SAVE_PATH = Path('models')
MODEL_SAVE_PATH.mkdir(exist_ok=True)
MODEL_FILE = MODEL_SAVE_PATH / 'chatbotv1.pth'

torch.save(model.state_dict(), MODEL_FILE)

print(f"Model saved to: {MODEL_FILE}")