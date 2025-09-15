# Walter's E-commerce Customer Support Chatbot

![Walter's Chatbot Icon](https://img.icons8.com/color/96/000000/shopping-cart.png)

## Overview
Walter's E-commerce Customer Support Chatbot is a Python-based AI assistant designed to answer customer queries for online shopping. It uses NLP techniques and a neural network to classify user intents and provide helpful responses.

---

## Features
- Intent-based question answering using a custom JSON dataset
- PyTorch neural network for intent classification
- NLP preprocessing (tokenization, stemming, bag-of-words)
- Streamlit web UI for interactive chat
- Easy extensibility with new intents and responses

---

## Project Structure
```
NLPChatBot/
├── Ecommerce_FAQ_intents.json   # Intents dataset
├── ecommerce.py                # Model and utility code
├── app.py                      # Streamlit chatbot UI
├── train.py                    # Training code
├── models/                     # Saved PyTorch model
└── README.md                   # Project documentation
```

---

## How It Works
1. **Data Preparation**
   - Intents and patterns are defined in `Ecommerce_FAQ_intents.json`.
   - Each intent has example user queries and possible responses.

2. **Preprocessing**
   - Tokenization: Splits sentences into words.
   - Stemming: Reduces words to their root form.
   - Bag-of-words: Encodes user input for the neural network.

3. **Model Training**
   - A feedforward neural network (`ChatbotV1`) is trained to classify intents.
   - Training data is generated from the patterns in the JSON file.
   - The trained model is saved to `models/chatbotv1.pth`.

4. **Inference & UI**
   - User input is processed and classified by the model.
   - The chatbot responds with the most relevant answer.
   - Streamlit provides a modern, interactive chat interface.

---

## Setup Instructions

### 1. Install Python & Create Environment
- Install [Python 3.8+](https://www.python.org/downloads/)
- (Recommended) Install [Anaconda](https://www.anaconda.com/products/distribution) or use `venv`:
  ```powershell
  python -m venv nlpchatbot-env
  .\nlpchatbot-env\Scripts\activate
  ```

### 2. Install Required Packages
```powershell
pip install torch nltk tqdm streamlit torchinfo
```
If you have SSL issues, use:
```powershell
pip install torch nltk tqdm streamlit torchinfo --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### 3. Download NLTK Data
In Python shell or at the top of your script:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 4. Train the Model
Run the training script:
```powershell
python ecommerce.py
```
This will preprocess data, train the model, and save it to `models/chatbotv1.pth`.

### 5. Launch the Chatbot UI
Start the Streamlit app:
```powershell
streamlit run app.py
```
A browser window will open with the chatbot interface.

---

## Example Usage
- Type questions like "How do I track my order?" or "What payment methods do you accept?"
- The chatbot will respond with relevant answers from the dataset.

---

## Customization
- **Add new intents:** Edit `Ecommerce_FAQ_intents.json` and retrain the model.
- **Change UI:** Edit `app.py` for custom styles, icons, or layout.
- **Improve accuracy:** Add more patterns and responses to each intent.

---

## Troubleshooting
- **SSL Errors:** Use `--trusted-host` with pip commands.
- **Missing NLTK Data:** Run `nltk.download('punkt')` and `nltk.download('punkt_tab')`.
- **Module Not Found:** Ensure all packages are installed in your active environment.

---

## Credits
Developed by Walter. Powered by Python, PyTorch, NLTK, and Streamlit.

---

## License
MIT License
