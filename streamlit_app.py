
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from redactor import redact_text
import os
import pickle

st.set_page_config(page_title="Housing QA System", layout="wide")
st.title("🏠 Housing Disrepair QA System")

persist_dir = "vector_store"
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

uploaded_file = st.file_uploader("Upload a PDF Survey Report", type="pdf")

question = st.text_input("Ask a question (without uploading a file):")

docs = []

if uploaded_file:
    st.info("Processing uploaded document...")
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp.pdf")
    raw_docs = loader.load()
    redacted_docs = [doc for doc in redact_text(raw_docs)]
    docs.extend(redacted_docs)

if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load or build vector DB
    if os.path.exists(f"{persist_dir}/faiss_index"):
        db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(splits)
    else:
        db = FAISS.from_documents(splits, embeddings)
    db.save_local(persist_dir)

    retriever = db.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if question:
        response = qa_chain({qa_chain.input_keys.pop(): question})
        st.write(response["result"])
        with st.expander("Referenced content"):
            for doc in response["source_documents"]:
                st.markdown(doc.page_content[:500] + "...")
elif question:
    st.warning("Please upload a document first or wait for existing documents to load.")
