import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

#Load the lstm model
model=load_model('next_word_lstm.h5')

#load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

#predict the next word
def predict_nxt_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_idx=np.argmax(predicted,axis=1)
    for word,idx in tokenizer.word_index.items():
        if idx==predicted_word_idx:
            return word
    return None

#streanlit app
st.title("Next word prediction with LSTM")
input_text=st.text_input("Enter the word: ")
if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_nxt_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next word prediction: {next_word}")

