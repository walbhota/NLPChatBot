import streamlit as st
from ecommerce import load_model, get_response, tokenize, stem
import json
import string

# --- INITIAL SETUP (No Changes Here) ---
# Load intents data
with open('Ecommerce_FAQ_intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Build allwords and tags
allwords = []
tags = []
for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenize(pattern.lower())
        tokens = [stem(w) for w in tokens if w not in string.punctuation]
        allwords.extend(tokens)
allwords = sorted(set(allwords))
tags = sorted(set(tags))

# Load model and data
model = load_model(len(allwords), len(tags))

# Custom page config and header
st.set_page_config(
    page_title="Walter's E-commerce Customer Support Chatbot",
    page_icon="🛒",
    layout="centered"
)

# Beautiful header with icon and custom style
st.markdown(
    """
    <div style='text-align: center; margin-top: 2em;'>
        <h1 style='font-family: "Segoe UI", Arial, sans-serif; font-size: 2.5em; margin-bottom: 0.2em;'>🛒 Walter's E-commerce Customer Support Chatbot</h1>
        <p style='color: #666; font-size: 1.2em;'>Your friendly assistant for all shopping questions!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize chat history and suggestion state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'suggestion_clicked' not in st.session_state:
    st.session_state.suggestion_clicked = None

# --- QUESTION SUGGESTIONS (No Changes Here) ---
suggestions = []
for intent in data['intents'][:8]: # Simplified loop
    if intent['patterns']:
        suggestions.append(intent['patterns'][0])

st.markdown("<h4 style='text-align: center; margin-top:1em; margin-bottom:1em;'>Try asking:</h4>", unsafe_allow_html=True)
cols = st.columns(4) # Using 4 columns for better layout
for i, question in enumerate(suggestions[:4]):
    if cols[i].button(question, key=f'suggest_{i}'):
        st.session_state.suggestion_clicked = question

cols = st.columns(4)
for i, question in enumerate(suggestions[4:]):
    if cols[i].button(question, key=f'suggest_{i+4}'):
        st.session_state.suggestion_clicked = question


# --- UNIFIED MESSAGE DISPLAY & HANDLING LOGIC (MAJOR CHANGES HERE) ---

# 1. Display the entire chat history from session state ONCE at the top.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Determine the prompt: either from a clicked suggestion or the chat input.
prompt = None
if st.session_state.suggestion_clicked:
    prompt = st.session_state.suggestion_clicked
    st.session_state.suggestion_clicked = None # Reset after use
else:
    prompt = st.chat_input("Type your message here...")

# 3. Process the prompt if it exists (from either source).
if prompt:
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display the assistant's response
    with st.spinner("Thinking..."):
        reply = get_response(prompt, model, allwords, tags, data)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    
    # Rerun to clear the suggestion button's state visually if needed
    if 'suggestion_clicked' in st.session_state and st.session_state.suggestion_clicked is None:
         st.rerun()