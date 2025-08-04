import nltk
import os
import ssl

# Streamlit sometimes needs this fix for SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download only if not already present
nltk.download('punkt')
nltk.download('stopwords')  # If you're using stopwords
nltk.download('wordnet')
nltk.download('omw-1.4')







import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt')  # <-- Add this line
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# Streamlit UI
st.title("📨 SMS Spam Classifier")

input_sms = st.text_area("Enter the message:")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = vectorizer.transform([transformed_sms]).toarray()

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Output
    if result == 1:
        st.error("🚫 This is a Spam message")
    else:
        st.success("✅ This is a Ham message")
