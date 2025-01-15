#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# In[6]:


ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


# In[15]:


import json

with open(r"C:\Users\prana\Documents\AICTE AI chatbot\intents.json") as file:
    intents = json.load(file)
#For virtual Env 'Documents/AICTE AI chatbot/intents.json'

# In[16]:


patterns = []
responses = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)


# In[17]:


model = LogisticRegression()
model.fit(X, tags)


# In[18]:


def predict_intent(user_input):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    return prediction[0]


# In[19]:


st.title("Intent-Based Chatbot")

user_input = st.text_input("You: ")

if st.button("Send"):
    if user_input:
        predicted_tag = predict_intent(user_input)
        response = random.choice([resp for intent in intents['intents'] if intent['tag'] == predicted_tag for resp in intent['responses']])
        st.write(f"Bot: {response}")
    else:
        st.write("Please enter a message.")


# In[ ]:


