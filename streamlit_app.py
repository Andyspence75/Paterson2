import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from redactor import redact_text

st.title("Housing Disrepair QA System")

query = st.text_input("Ask a question about housing disrepair or reports:")

uploaded_files = st.file_uploader("Upload PDF training materials or reports", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.info("Processing uploaded documents...")

    docs = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(uploaded_file.name)
        else:
            loader = UnstructuredFileLoader(uploaded_file.name)
        docs.extend(loader.load())

    # Redact documents
    redacted_docs = []
    for doc in docs:
        redacted_text = redact_text(doc.page_content)
        doc.page_content = redacted_text
        redacted_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(redacted_docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)

    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        response = qa_chain({"query": query})
        st.subheader("Answer:")
        st.write(response["result"])
