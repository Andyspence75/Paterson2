import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI
from redactor import redact_text
import tempfile

st.set_page_config(page_title="Housing QA", layout="wide")
st.title("🏠 Housing Disrepair QA Assistant")

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

uploaded_files = st.file_uploader("Upload PDF or Word documents", type=["pdf", "docx"], accept_multiple_files=True)
user_question = st.text_input("Ask a general housing disrepair question (optional):")

docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = UnstructuredWordDocumentLoader(tmp_path)

        loaded_docs = loader.load()
        redacted_docs = [redact_text(doc.page_content) for doc in loaded_docs]
        for i, doc in enumerate(loaded_docs):
            doc.page_content = redacted_docs[i]
        docs.extend(loaded_docs)

if docs or user_question:
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(splits, embeddings)
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    else:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=None)

    if user_question:
        try:
            response = qa_chain({"query": user_question})
            st.subheader("Answer:")
            st.write(response["result"])

            if "source_documents" in response:
                with st.expander("Referenced Sections"):
                    for doc in response["source_documents"]:
                        st.markdown(doc.page_content[:500] + "...")
        except Exception as e:
            st.error(f"Error generating answer: {e}")