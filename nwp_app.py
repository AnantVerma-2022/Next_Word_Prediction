import numpy as np # for numerical computations
import pickle # for loading the tokenizer
import streamlit as st # for building the web app

from tensorflow.keras.models import load_model # for loading the trained model
from tensorflow.keras.preprocessing.sequence import pad_sequences # for padding sequences

@st.cache_resource # cache the model and tokenizer loading
def load_prediction_model(): # load the trained model and tokenizer
    model = load_model("model_next_word_predict.h5")
    with open("model_nwp_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer #` return the model and tokenizer`

model, tokenizer = load_prediction_model() #

def generate_seq(model, tokenizer, seq_length, seed_text, n_words=5):
    result = []
    in_text = seed_text
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([in_text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

        # map predicted index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                out_word = word
                break

        in_text += ' ' + out_word
        result.append(out_word)

    return ' '.join(result)


# Streamlit UI Development
st.title("Next Word Prediction App")
st.write("Enter some seed text and the model will continue the sentence!")

seed_text = st.text_input("Enter seed text:", "Once upon a time")
seq_length = st.number_input("Sequence length (same as during training)", 
                             min_value=5, max_value=50, value=10)
n_words = st.slider("How many words to generate?", 1, 20, 5)

if st.button("Generate"):
    prediction = generate_seq(model, tokenizer, seq_length, seed_text, n_words)
    st.subheader("Generated Text:")
    st.write(seed_text + " " + prediction)