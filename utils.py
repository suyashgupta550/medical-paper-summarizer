import time
from openai.error import RateLimitError
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def generate_summary(docs, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")  # safer model
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

def safe_generate_summary(docs, api_key, retries=3, delay=10):
    for attempt in range(retries):
        try:
            return generate_summary(docs, api_key)
        except RateLimitError:
            if attempt < retries - 1:
                print(f"Rate limit hit, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
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
