import os
import streamlit as st
from dotenv import load_dotenv

# -----------------------
# Load environment variables
load_dotenv()

# -----------------------
# LangChain imports
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# -----------------------
st.set_page_config(page_title="Incident Handler RAG", layout="centered")

st.title("🛡️ Incident Handler RAG System")
st.write("Ask questions based on incident handler journals")

# -----------------------
@st.cache_resource
def load_vectorstore():
    documents = []

    # TXT document
    txt_loader = TextLoader(
        "data/incident_handler_journal.txt",
        encoding="utf-8"
    )
    txt_docs = txt_loader.load()
    for doc in txt_docs:
        doc.metadata = {"file_type": "txt", "source": "incident_handler_journal.txt"}
    documents.extend(txt_docs)

    # DOCX document
    docx_loader = Docx2txtLoader(
        "data/Incident_handler_journal_correct.docx"
    )
    docx_docs = docx_loader.load()
    for doc in docx_docs:
        doc.metadata = {"file_type": "docx", "source": "Incident_handler_journal_correct.docx"}
    documents.extend(docx_docs)

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # Embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# -----------------------
vectorstore = load_vectorstore()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -----------------------
st.sidebar.header("🔎 Search Options")

doc_filter = st.sidebar.selectbox(
    "Search documents from:",
    ["All Documents", "TXT only", "DOCX only"]
)

query = st.text_input("Ask a question about incidents:")

if query:
    if doc_filter == "TXT only":
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4, "filter": {"file_type": "txt"}}
        )
    elif doc_filter == "DOCX only":
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4, "filter": {"file_type": "docx"}}
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are an incident response assistant.
Answer the question strictly using the context below.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    st.subheader("📌 Answer")
    st.write(response.content)
