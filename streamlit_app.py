import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from redactor import redact_text
import tempfile
import os

# Configurations
QDRANT_URL = st.secrets.get('QDRANT_URL')
QDRANT_API_KEY = st.secrets.get('QDRANT_API_KEY')
QDRANT_COLLECTION = 'housing_reports'

# Set up Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# User/admin toggle
mode = st.sidebar.selectbox('Mode', ['User', 'Admin'])

st.title('Housing Disrepair QA System')
st.write('Upload PDF or DOC survey reports and ask questions.')

uploaded_files = st.file_uploader('Upload PDF or DOC files', type=['pdf', 'doc', 'docx'], accept_multiple_files=True)

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

all_docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        suffix = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + suffix) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        if suffix == 'pdf':
            loader = PyPDFLoader(temp_path)
        elif suffix in ['doc', 'docx']:
            loader = UnstructuredWordDocumentLoader(temp_path)
        else:
            continue

        docs = loader.load()
        # Redact if enabled (admin can toggle)
        if mode == 'User' or (mode == 'Admin' and st.sidebar.checkbox('Redact personal info', value=True)):
            docs = [type(doc)(redact_text(doc.page_content), metadata=doc.metadata) for doc in docs]
        all_docs.extend(docs)
        os.unlink(temp_path)

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = Qdrant.from_documents(
        documents=splits,
        embedding=embeddings,
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
    )
    st.session_state.vectorstore = vectorstore
    st.success('Documents processed and indexed (Qdrant Cloud, persistent!).')

# General QA box (can query even if no file uploaded)
if st.session_state.vectorstore is not None:
    st.header('Ask a question:')
    question = st.text_input('Enter your query:')
    if question:
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        retriever = st.session_state.vectorstore.as_retriever()
        from langchain.chains import RetrievalQA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        response = qa_chain({'query': question})
        st.markdown('**Answer:** ' + response['result'])
        if response.get('source_documents'):
            with st.expander('See referenced content'):
                for doc in response['source_documents']:
                    st.markdown('---')
                    st.text(redact_text(doc.page_content[:500]) + '...')
else:
    st.info('No documents indexed yet. Upload to start, or ask a general query (if previous data exists).')