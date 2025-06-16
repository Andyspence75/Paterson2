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

st.sidebar.title("Options")
show_training_box = st.sidebar.checkbox("Enable Q&A Training")
general_query_box = st.sidebar.checkbox("General Query (No Upload)")

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

if general_query_box:
    general_question = st.text_input("Ask a general question:")
    if general_question:
        response = llm.invoke(general_question)
        st.write(response)

uploaded_file = st.file_uploader("Upload a PDF Report", type="pdf")
qa_pairs = []

if uploaded_file:
    st.info("Processing document...")
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Redact content
    docs = [doc for doc in docs if doc.page_content.strip()]
    redacted_docs = []
    for doc in docs:
        redacted_content = redact_text(doc.page_content)
        doc.page_content = redacted_content
        redacted_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(redacted_docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    question = st.text_input("Ask about the uploaded report:")
    if question:
        response = qa_chain({"query": question})
        st.write(response["result"])
        with st.expander("See referenced sections"):
            for doc in response["source_documents"]:
                st.markdown(doc.page_content[:300] + "...")

    if show_training_box:
        st.subheader("Q&A Training Data")
        question_input = st.text_input("Add a training question")
        answer_input = st.text_area("Add the answer")
        if st.button("Add Q&A Pair"):
            qa_pairs.append({"question": question_input, "answer": answer_input})
            st.success("Q&A pair added.")