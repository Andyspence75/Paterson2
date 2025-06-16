import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from redactor import redact_text

st.set_page_config(page_title="Housing QA System", layout="wide")
st.title("Housing Disrepair QA System")

question = st.text_input("Ask a general question about housing standards or disrepair issues:")

uploaded_file = st.file_uploader("Upload a housing report (PDF or Word)", type=["pdf", "docx"])

if uploaded_file:
    st.info("Processing document...")
    with open("temp_uploaded_file", "wb") as f:
        f.write(uploaded_file.read())

    # Use appropriate loader
    if uploaded_file.name.lower().endswith(".pdf"):
        loader = PyPDFLoader("temp_uploaded_file")
    else:
        loader = UnstructuredWordDocumentLoader("temp_uploaded_file")

    try:
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        splits = splits[:20]  # Reduce memory usage
        redacted_splits = [doc for doc in splits]
        for doc in redacted_splits:
            doc.page_content = redact_text(doc.page_content)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(redacted_splits, embeddings)
        retriever = vectordb.as_retriever()

        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        if question:
            response = qa_chain({"query": question})
            st.subheader("Answer")
            st.write(response["result"])

            with st.expander("See referenced content"):
                for doc in response["source_documents"]:
                    st.markdown(doc.page_content[:300] + "...")
    except Exception as e:
        st.error(f"Failed to process the document: {e}")
elif question:
    st.warning("Please upload a document to ask context-specific questions.")
