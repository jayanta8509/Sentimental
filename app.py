import json
import requests
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import HuggingFaceHub
import streamlit as st
load_dotenv(".env")
api_key = os.getenv("GOOGLE_API_KEY")
huggingface = os.getenv("huggingface")
API_URL = "https://api-inference.huggingface.co/models/lxyuan/distilbert-base-multilingual-cased-sentiments-student"
headers = {"Authorization": huggingface}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

st.title("🦜Sentiment Analysis Chat Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    response = llm.invoke(prompt)
    analysis = response.content

    if len(analysis)>499:
        analysis = analysis[:499]
    else:
        pass

    output = query({
	"inputs": analysis,
     })
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.content)
        sentiment_results = output[0]
        for item in sentiment_results:
            label = item['label']
            score = item['score']
            st.markdown(f"Label: {label}, Score: {score}")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})