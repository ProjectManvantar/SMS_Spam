import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize NLTK resources once
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Remove special characters and numbers
    y = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    y = [i for i in y if i not in stop_words and i not in string.punctuation]
    
    # Stemming
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)

# Load models safely
with open("vectorizer.pkl", 'rb') as f:
    tk = pickle.load(f)

with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

# Add a background image
st.markdown("""
<style>
.stApp {
    background-image: url("bgimg.jpg");
    background-attachment: fixed;
    background-size: cover;
    color: grey;
}

.stApp h1, .stApp h2 {
    color: black;
}

.stButton > button {
    color: white;
    background-color: #0047AB;
}
</style>
""", unsafe_allow_html=True)

# Add a title and a quote
st.title("SMS Spam Detection Model")
st.markdown('<p style="color: black;">"SMS spam is a growing problem, but with the help of machine learning, we can fight back!"</p>', unsafe_allow_html=True)

# Add input components
st.markdown('<p style="color: black;">Enter the SMS...</p>', unsafe_allow_html=True)
input_sms = st.text_input("", help="Type your SMS here...", max_chars=500, label_visibility="collapsed")

# Add prediction logic
if st.button('Predict', key='predict'):
    if input_sms.strip():
        transformed_sms = transform_text(input_sms)
        vector_input = tk.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        st.error("Spam") if result == 1 else st.success("Not Spam")
    else:
        st.warning("Please enter an SMS.")

# Footer note
st.markdown("<p style='text-align: center; color: grey;'>Beware of Spam SMS, They might empty your bank account</p>", unsafe_allow_html=True)
