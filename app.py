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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    txt_path = os.path.join(BASE_DIR, "data", "incident_handler_journal.txt")
    docx_path = os.path.join(BASE_DIR, "data", "Incident_handler_journal_correct.docx")

    documents = []

    # -------- Helper to clean metadata --------
    def clean_metadata(text):
        return str(text).strip().lower()

    # -------- TXT loader --------
    txt_loader = TextLoader(txt_path, encoding="utf-8")
    txt_docs = txt_loader.load()
    for doc in txt_docs:
        doc.metadata = {
            "file_type": clean_metadata("txt"),
            "source": clean_metadata("incident_handler_journal.txt")
        }
    documents.extend(txt_docs)

    # -------- DOCX loader --------
    docx_loader = Docx2txtLoader(docx_path)
    docx_docs = docx_loader.load()
    for doc in docx_docs:
        doc.metadata = {
            "file_type": clean_metadata("docx"),
            "source": clean_metadata("incident_handler_journal_correct.docx")
        }
    documents.extend(docx_docs)

    # -------- SPLITTING --------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # -------- EMBEDDINGS --------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------- VECTOR STORE --------
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# -----------------------
vectorstore = load_vectorstore()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")  # works with Streamlit secrets
)

# -----------------------
st.sidebar.header("🔎 Search Options")

doc_filter = st.sidebar.selectbox(
    "Search documents from:",
    ["All Documents", "TXT only", "DOCX only"]
)

query = st.text_input("Ask a question about incidents:")

if query:
    # -------- Create retriever FIRST --------
    if doc_filter == "TXT only":
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 15, "filter": {"file_type": "txt"}}
        )
    elif doc_filter == "DOCX only":
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 15, "filter": {"file_type": "docx"}}
        )
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 15}
        )

    # -------- Handle "list all incidents" queries --------
    if "list all" in query.lower() or "all incidents" in query.lower():
        # bypass retriever → load EVERYTHING
        retrieved_docs = vectorstore.similarity_search("", k=1000)
    else:
        retrieved_docs = retriever.invoke(query)

    # -------- Build context --------
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are an incident response assistant.

Using ONLY the context below:
- Identify
- Enumerate
- List EACH DISTINCT incident separately
- Do NOT merge incidents

Context:
{context}

Question:
{query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    st.subheader("📌 Answer")
    st.write(response.content)

