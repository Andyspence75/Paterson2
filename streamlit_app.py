
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from redactor import redact_text
import tempfile
import os

st.title("Housing Disrepair QA System")

QDRANT_URL = st.secrets.get("QDRANT_URL", os.environ.get("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.environ.get("QDRANT_API_KEY"))
QDRANT_COLLECTION = "housing-disrepair"

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def process_uploaded_file(uploaded_file):
    suffix = ".pdf" if uploaded_file.name.lower().endswith(".pdf") else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = UnstructuredWordDocumentLoader(tmp_path)
    docs = loader.load()
    return docs

uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

all_docs = []
if uploaded_file:
    st.info("Processing and redacting document...")
    docs = process_uploaded_file(uploaded_file)
    # Redact content
    for doc in docs:
        doc.page_content = redact_text(doc.page_content)
    all_docs.extend(docs)

# General chat box (does not require upload)
general_query = st.text_input("General query (no upload required)")

if (uploaded_file and all_docs) or general_query:
    # Embedding and Qdrant vector DB
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(all_docs) if all_docs else []
    embeddings = OpenAIEmbeddings()
    # If new docs, re-index. If not, just use Qdrant retrieval
    if splits:
        vectorstore = Qdrant.from_documents(
            splits, embeddings, client=qdrant_client, collection_name=QDRANT_COLLECTION
        )
        st.session_state.vectorstore = vectorstore
    else:
        # Only connect to existing
        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embeddings=embeddings,
        )
        st.session_state.vectorstore = vectorstore
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    if uploaded_file and st.session_state.vectorstore:
        st.success("Documents processed, indexed, and redacted.")
        query = st.text_input("Ask a question about the uploaded documents:")
        if query:
            response = qa_chain({"query": query})
            st.write(response["result"])
            with st.expander("References"):
                for doc in response["source_documents"]:
                    st.markdown(doc.page_content[:300] + "...")
    elif general_query:
        response = qa_chain({"query": general_query})
        st.write(response["result"])
