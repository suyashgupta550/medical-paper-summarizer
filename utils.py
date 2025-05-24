import time
import os
from openai.error import RateLimitError
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # Increase chunk size to reduce number of requests
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_documents(documents)

def generate_summary(docs, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")  # safer model
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

def safe_generate_summary(docs, api_key, retries=5, delay=20):
    for attempt in range(retries):
        try:
            return generate_summary(docs, api_key)
        except RateLimitError:
            if attempt < retries - 1:
                st.warning(f"Rate limit hit, retrying in {delay} seconds... (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                st.error("Exceeded rate limit retries. Please try again later.")
                raise

def structured_summary(summary, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")

    prompt = f"""
You are a medical research assistant.
Given the following summary, extract:
1. Objective
2. Methods
3. Key Findings
4. Conclusion
5. Simplified explanation for non-medical readers.

Summary:
\"\"\" 
{summary}
\"\"\"
"""
    return llm.predict(prompt)
