import os
import streamlit as st
import pandas as pd
import random
import time
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
from pandasai.responses.response_parser import ResponseParser
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

st.write("# Chat with your data 🐢")
@st.cache_data
def load_data(file):
    try:
        data = pd.read_csv(file)
    except:
        data = pd.read_excel(file)
    return data
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None: 
    st.info("upload a file through config", icon= "i")
    st.stop()

df = load_data(uploaded_file)
with st.expander("🔎 Your Data Preview"):
    st.write(df.head(8))
query = st.text_area("🗣️ Type Here ")
button = st.button('Submit')
if query and button == True:
    os.environ['GOOGLE_API_KEY'] = "AIzaSyA1e7inOhwkiNheR-Qxp1wQiEobf3iL-4o"
    #genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
    #model = genai.GenerativeModel('gemini-pro')
    llm = GoogleGemini(api_key = os.environ['GOOGLE_API_KEY'])
    query_engine = SmartDataframe(df,config={"llm":llm,"response_parser": StreamlitResponse})
    answer = query_engine.chat(query)
    st.write(answer, verbose = True)
