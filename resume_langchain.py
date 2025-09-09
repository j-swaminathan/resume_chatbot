# Streamlit PDF QA Chatbot (LangChain version)
# ---------------------------------------------------------------
# Features
# - Upload a PDF, index it locally with FAISS via LangChain
# - Ask questions; uses LangChain's RetrievalQA
# - Embeddings from sentence-transformers (local, free)
# - LLM from Hugging Face Hub (FLAN-T5)
# ---------------------------------------------------------------
# Quickstart (install):
#   pip install streamlit langchain langchain-community faiss-cpu sentence-transformers transformers torch torchvision torchaudio pypdf
# Run app:
#   streamlit run resume_langchain.py
# ---------------------------------------------------------------

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import tempfile

# -----------------------------
# Load models (cached)
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource(show_spinner=False)
def load_llm(model_name: str = "google/flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device, max_new_tokens=200)
    return HuggingFacePipeline(pipeline=pipe)


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="Ask Your PDF (LangChain)", page_icon="📄", layout="wide")

st.title("📄 Ask Your PDF (LangChain RAG)")
st.caption("Upload a PDF and ask questions using LangChain + FAISS + FLAN-T5 (open-source).")

with st.sidebar:
    st.header("Settings")
    embed_model_name = st.selectbox(
        "Embedding model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        index=0,
    )
    llm_model_name = st.selectbox(
        "LLM model",
        ["google/flan-t5-small", "google/flan-t5-base"],
        index=0,
    )
    chunk_size = st.slider("Chunk size", 400, 1500, 800, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, 10)
    k = st.slider("Top-K chunks", 2, 10, 5)

# Session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"]) 


if st.button("📚 Index Document", use_container_width=True, type="primary"):
    if not uploaded_pdf:
        st.warning("Please upload a PDF first.")
    
    else:
        
        with st.spinner("Processing document with LangChain..."):
            # Load PDF
            if uploaded_pdf is not None:
            # Save uploaded file to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.read())
                    tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path) 
            docs = loader.load()

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            split_docs = splitter.split_documents(docs)

            # Embeddings + FAISS
            embedder = load_embeddings(embed_model_name)
            vectordb = FAISS.from_documents(split_docs, embedder)

            # LLM
            llm = load_llm(llm_model_name)

            # RetrievalQA chain
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectordb.as_retriever(search_kwargs={"k": k}),
                return_source_documents=True,
            )
            st.success(f"Indexed {len(split_docs)} chunks.")

# Chat
if st.session_state.qa_chain is None:
    st.info("Upload a PDF and click **Index Document** to get started.")
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Ask a question about the PDF...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = st.session_state.qa_chain({"query": user_q})
                answer = res["result"]
                st.markdown(answer)
                with st.expander("Show sources"):
                    for i, doc in enumerate(res["source_documents"], start=1):
                        st.markdown(f"**Source {i}:**\n{doc.page_content[:500]}...")
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption("Built with Streamlit • LangChain • FAISS • Sentence-Transformers • FLAN-T5")
