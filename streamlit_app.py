
import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI
from redactor import redact_text

st.title("Housing Disrepair QA System")

# General query box (optional input)
general_query = st.text_input("Ask a general housing question (no document needed):")

# File uploader for PDF/DOCX
uploaded_files = st.file_uploader("Upload survey reports (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

docs = []

if uploaded_files:
    for file in uploaded_files:
        ext = file.name.split(".")[-1]
        path = os.path.join("temp", file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        if ext == "pdf":
            loader = PyPDFLoader(path)
        elif ext == "docx":
            loader = UnstructuredWordDocumentLoader(path)
        else:
            continue
        docs.extend(loader.load())

# Redact PII from documents
docs = [doc.__class__(page_content=redact_text(doc.page_content), metadata=doc.metadata) for doc in docs]

if docs or general_query:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs) if docs else []

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings) if splits else None

    retriever = vectordb.as_retriever() if vectordb else None
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    if general_query and not docs:
        response = llm.invoke(general_query)
        st.write("Response:", response.content)

    elif docs:
        question = st.text_input("Ask something about the uploaded documents:")
        if question:
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            response = qa_chain({"query": question})
            st.write("Answer:", response["result"])
            with st.expander("References"):
                for doc in response["source_documents"]:
                    st.markdown(doc.page_content[:500] + "...")
