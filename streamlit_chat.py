
import os
import streamlit as st
import pandas as pd
import random
import time
import google.generativeai as genai
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
from pandasai.responses.response_parser import ResponseParser
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("# Chat with your data 🐢")
if "google_gemini" not in st.session_state:
    st.session_state["google_gemini"] = "gemini-1.5-pro"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["parts"])
#if 'chat_history' not in st.session_state:
    #st.session_state['chat_history'] = []

#os.environ['GOOGLE_API_KEY'] = "AIzaSyA1e7inOhwkiNheR-Qxp1wQiEobf3iL-4o"
#genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
#model = genai.GenerativeModel(model_name='gemini-1.5-pro')
#chat = model.start_chat(history=[])
#def get_gemini_response(question):
    
    #response=chat.send_message(question,stream=True)
    #return response
if prompt := st.chat_input("Ask Something"):
    os.environ['GOOGLE_API_KEY'] = "AIzaSyA1e7inOhwkiNheR-Qxp1wQiEobf3iL-4o"
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    model = genai.GenerativeModel(model_name='gemini-1.5-pro')
    #for role, text in st.session_state['chat_history']:
        #st.write(f"{role}: {text}")
    #response=get_gemini_response(prompt)
    #st.session_state['chat_history'].append(("You", prompt))
    #for chunk in response:
        #st.write(chunk.text)
        #st.session_state['chat_history'].append(("Bot", chunk.text))
    st.session_state.messages.append({"role": "user", "parts": prompt})
    #os.environ['GOOGLE_API_KEY'] = "AIzaSyA1e7inOhwkiNheR-Qxp1wQiEobf3iL-4o"
    #llm = ChatGoogleGenerativeAI(api_key = os.environ['GOOGLE_API_KEY'], model="gemini-pro")
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = model.generate_content(prompt)
        response = st.write(stream.text)
        #contents = [{'role' : m['role'],'parts' : m['parts']} for m in st.session_state.messages],
    st.session_state.messages.append({"role": "assistant", "parts": response})



        

            

