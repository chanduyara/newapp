import streamlit as st
import os
from dotenv import load_dotenv


from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatGroq
from langchain.chains import RetrievalQA

### Load environment variables
load_dotenv()

st.set_page_config(page_title="📚 BigData Chatbot", layout="wide")
st.title("🤖 Big Data Engineering Question & Answers")

# 📁 Path to your FAISS index directory
INDEX_PATH = "my_faiss_index"

if not os.path.exists(INDEX_PATH):
    st.error(f"Index not found at `{INDEX_PATH}`. Please run the indexing step first.")
else:
    with st.spinner("Loading FAISS index..."):
        try:
            embeddings = OpenAIEmbeddings()

            # ⚠️ FAISS load now requires this flag to prevent unsafe pickle loading
            vectorstore = FAISS.load_local(
                INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True  # ✅ ONLY if you trust the file
            )

            retriever = vectorstore.as_retriever()
            llm = ChatOpenAI(temperature=0)
            # llm=ChatGroq(model="llama3-8b-8192")

            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            st.success("Index loaded! Ask questions below 👇")

            # 💬 Ask user query
            query = st.text_input("Ask a question regarding Big Data:")

            if query:
                with st.spinner("Thinking..."):
                    response = qa_chain.run(query)
                st.markdown(f"**Answer:** {response}")

        except Exception as e:
            st.error(f"Something went wrong while loading the index: {e}")
