import streamlit as st
from utils import load_pdf, safe_generate_summary, structured_summary
import tempfile

st.set_page_config(page_title="Medical Paper Summarizer", layout="wide")
st.title("🩺 AI Medical Research Paper Summarizer")

api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")
uploaded_file = st.file_uploader("📄 Upload a medical research paper (PDF)", type="pdf")

if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing..."):
        docs = load_pdf(tmp_path)
        raw_summary = safe_generate_summary(docs, api_key)
        result = structured_summary(raw_summary, api_key)

    st.subheader("📋 Extracted Summary")
    st.markdown(result)
