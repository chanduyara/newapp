import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# --------------------------------------
# Setup & Configuration
# --------------------------------------

def load_env():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Missing GROQ_API_KEY in your .env file.")
        st.stop()
    return api_key

def configure_streamlit():
    st.set_page_config(page_title="📚 BigData Chatbot", layout="wide")
    st.title("🤖 Big Data Engineering Question & Answers")

# --------------------------------------
# Load FAISS Vectorstore
# --------------------------------------

@st.cache_resource
def load_vectorstore(index_path: str):
    if not os.path.exists(index_path):
        st.error(f"Index not found at `{index_path}`. Please run the indexing step first.")
        st.stop()
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever()

# --------------------------------------
# Load Language Model
# --------------------------------------

def load_llm(api_key: str, model_name: str = "llama3-8b-8192"):
    return ChatGroq(model=model_name, api_key=api_key)

# --------------------------------------
# Main QA Functionality
# --------------------------------------

def run_qa_interface(retriever, llm):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    st.success("Index loaded! Ask questions below 👇")

    query = st.text_input("Ask a question regarding Big Data:")
    if query:
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.run(query)
                st.markdown(f"**Answer:** {response}")
            except Exception as e:
                st.error(f"Error generating response: {e}")

# --------------------------------------
# Main App Entry Point
# --------------------------------------

def main():
    configure_streamlit()
    groq_api_key = load_env()

    INDEX_PATH = "my_faiss_index"

    with st.spinner("Loading FAISS index..."):
        retriever = load_vectorstore(INDEX_PATH)
        llm = load_llm(groq_api_key)
        run_qa_interface(retriever, llm)

if __name__ == "__main__":
    main()
