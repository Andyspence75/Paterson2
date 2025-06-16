import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from redactor import redact_text
import os
import shutil
from uuid import uuid4

st.set_page_config(page_title="Housing QA", layout="wide")
st.title("Housing Disrepair Q&A Assistant")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Admin/user toggle
mode = st.radio("Select mode:", ["User", "Admin"])

# Optional redaction
apply_redaction = st.checkbox("Redact personal info (Admin only)", value=True if mode == "Admin" else False)

# Allow general queries
general_question = st.text_input("Ask a general question (no file required):")
if general_question:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = llm.invoke(general_question)
    st.markdown("**Answer:** " + response.content)

# File upload
uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.info("Processing uploaded documents...")
    os.makedirs("data", exist_ok=True)
    docs = []

    for uploaded_file in uploaded_files:
        temp_path = os.path.join("data", f"{uuid4()}_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        if apply_redaction:
            with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                redacted = redact_text(f.read())
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(redacted)

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        else:
            loader = TextLoader(temp_path)

        docs.extend(loader.load())

    # Embedding and vector storage
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)

    # Retrieval chain
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    st.success("Documents processed. You can now ask questions.")

    question = st.text_input("Ask a question about the documents:")
    if question:
        response = qa_chain({qa_chain.input_keys.pop(): question})
        st.markdown("**Answer:** " + response['result'])

        with st.expander("Sources"):
            for doc in response['source_documents']:
                st.markdown(doc.page_content[:500] + "...")
