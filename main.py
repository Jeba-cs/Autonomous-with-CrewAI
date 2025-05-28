# main.py

import streamlit as st
from pdf_utils import process_pdf
from crew import run_crew
import tempfile
import os

st.set_page_config(page_title="PDF Agent (CrewAI + DeepSeek)", layout="centered")
st.title("🤖 PDF Smart Agent (CrewAI + DeepSeek)")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

user_query = st.text_input("🔍 Ask a question OR type 'summarize' to summarize the PDF")

run_button = st.button("🚀 Run Agent")

if uploaded_file and user_query and run_button:
    with st.spinner("⏳ Processing PDF and initializing agents..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        chunks, index = process_pdf(tmp_path)
        result = run_crew(chunks, index, user_query)

        os.remove(tmp_path)

    st.success("✅ Task Completed!")
    st.subheader("🧠 Agent Response:")
    st.write(result)
