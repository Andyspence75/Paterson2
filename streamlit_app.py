import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from redactor import redact_text

st.title("Housing Disrepair QA System")

question = st.text_input("Ask a question about housing disrepair or standards (without uploading a document):")

uploaded_file = st.file_uploader("Upload a PDF Survey Report", type="pdf")

docs = []

if uploaded_file:
    st.info("Processing uploaded document...")
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp.pdf")
    docs.extend(loader.load())

if docs or question:
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(splits, embeddings)
        retriever = vectordb.as_retriever()
    else:
        retriever = None

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    if retriever:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        response = qa_chain({"query": question})
        st.write("Answer:", response["result"])

        with st.expander("Referenced Content"):
            for doc in response["source_documents"]:
                st.markdown(redact_text(doc.page_content[:500]) + "...")
    else:
        if question:
            response = llm.invoke(question)
            st.write("Answer:", redact_text(response.content))
