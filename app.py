import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#load lstm model
model=load_model('C:\\Users\\Hp\\OneDrive - BUET\\Desktop\\Machine learning\\Natural language processing\\LSTM_RNN\\app.py')

#load tokenizer
with open('C:\\Users\\Hp\\OneDrive - BUET\\Desktop\\Machine learning\\Natural language processing\\LSTM_RNN\\tokenizer.pickle','rb')as handle:
    tokenizer=pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_length):
    # Convert text to sequence
    token_list = tokenizer.texts_to_sequences([text])[0]   # fixed typo
    # Trim if longer than model input
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length-1):]
    # Pad sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    # Predict next word
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]   # extract int
    # Map index to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


#streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_length=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_length)   
    st.write(f'Next word:{next_word}')
