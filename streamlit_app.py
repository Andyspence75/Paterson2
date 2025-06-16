import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from redactor import redact_text

st.set_page_config(page_title="Housing Disrepair QA", layout="wide")
st.title("Housing Disrepair QA System")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDFs or text files", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        all_docs = []
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())

            ext = uploaded_file.name.split(".")[-1].lower()
            loader = PyPDFLoader(uploaded_file.name) if ext == "pdf" else UnstructuredFileLoader(uploaded_file.name)
            docs = loader.load()
            docs = [doc for doc in docs if doc.page_content.strip()]
            docs = [doc.__class__(page_content=redact_text(doc.page_content)) for doc in docs]
            all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(splits, embeddings)
        st.session_state.vectorstore = vectordb
        st.success("Documents processed and indexed.")

if st.session_state.vectorstore:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    retriever = st.session_state.vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    query = st.text_input("Ask a question about the documents:")
    if query:
        result = qa({"query": query})
        st.write("**Answer:**", result["result"])
        with st.expander("See Sources"):
            for doc in result["source_documents"]:
                st.markdown(doc.page_content[:300] + "...")