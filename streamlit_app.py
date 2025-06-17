
import os
import streamlit as st


from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Connect to Qdrant Cloud
qdrant_client = QdrantClient(
    url="https://d159ceef-7eec-49cc-a25d-789140354a83.eu-west-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.BWqFWA3KpSmdPLDORh3fBHZrEWXS94W2rbG6VN4X-Xg"
)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create collection (if not exists)
qdrant_client.recreate_collection(
    collection_name="paterson_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

def upsert_documents(documents):
    points = []
    for i, doc in enumerate(documents):
        embedding = embedding_model.encode(doc)
        points.append(PointStruct(id=i, vector=embedding.tolist(), payload={"text": doc}))
    qdrant_client.upsert(collection_name="paterson_docs", points=points)

def query_qdrant(query_text, top_k=5):
    embedding = embedding_model.encode(query_text).tolist()
    results = qdrant_client.search(
        collection_name="paterson_docs",
        query_vector=embedding,
        limit=top_k
    )
    return [hit.payload["text"] for hit in results]

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from redactor import redact_text

st.set_page_config(page_title="Housing Disrepair QA", layout="wide")
st.title("Housing Disrepair QA System")

FAISS_INDEX_PATH = "vectorstore.index"

def get_loader_for_file(filename):
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return PyPDFLoader(filename)
    elif ext == "txt":
        return UnstructuredFileLoader(filename)
    elif ext in ["doc", "docx"]:
        return UnstructuredWordDocumentLoader(filename)
    elif ext in ["ppt", "pptx"]:
        return UnstructuredPowerPointLoader(filename)
    else:
        return None

def load_vectorstore():
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            return FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Failed to load previous vectorstore: {e}")
            return None
    return None

def save_vectorstore(vectorstore):
    try:
        vectorstore.save_local(FAISS_INDEX_PATH)
    except Exception as e:
        st.error(f"Could not save vectorstore: {e}")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOC, PPT, or TXT files",
        type=["pdf", "txt", "doc", "docx", "ppt", "pptx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        all_docs = []
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())

            loader = get_loader_for_file(uploaded_file.name)
            if loader is not None:
                docs = loader.load()
                docs = [doc for doc in docs if doc.page_content.strip()]
                docs = [doc.__class__(page_content=redact_text(doc.page_content)) for doc in docs]
                all_docs.extend(docs)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")

        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(all_docs)
            embeddings = OpenAIEmbeddings()
            vectordb = FAISS.from_documents(splits, embeddings)
            st.session_state.vectorstore = vectordb
            save_vectorstore(vectordb)
            st.success("Documents processed and indexed. (Saved for future use!)")

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
