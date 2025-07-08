import streamlit as st
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from docx import Document

# Load environment variables
load_dotenv()

# Fetch values from .env file
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

FAISS_INDEX_PATH = "faiss_index"

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to extract text from Excel files
def get_excel_text(excel_docs):
    chunks = []
    for excel in excel_docs:
        df = pd.read_excel(excel, sheet_name=None, dtype=str)
        for sheet in df.values():
            for _, row in sheet.iterrows():
                row_values = row.dropna().astype(str)
                row_text = " | ".join(row_values.tolist())
                chunks.append(row_text)
    return chunks

# Function to extract text from TXT and DOCX files
def get_text_from_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# Function to split text into chunks
def get_text_chunks(text_chunks):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text("\n".join(text_chunks))

# Function to create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    st.success("✅ FAISS index created successfully! You can now ask questions.")

# Function to load conversational chain
def get_conversational_chain():
    prompt_template = """
    You are an AI assistant tasked with answering questions based only on the provided context.
    Also give all the Follow Up Questions linked to the asked question from the provided context.
    Use the provided context to answer the question with the highest possible accuracy.
    Do not use external knowledge—only rely on the given context.

    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=api_version,
        deployment_name="gpt-35-turbo",
        api_key=AZURE_OPENAI_API_KEY
    )
    return load_qa_chain(llm, prompt=prompt)

# Streamlit App
def main():
    st.set_page_config(page_title="QnA Chat with Documents")
    st.header("Ask Questions from Your Documents 💬")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar: Upload and process files
    with st.sidebar:
        st.title("Upload & Process Files")
        pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        excel_docs = st.file_uploader("Upload Excel Files", type=["xls", "xlsx"], accept_multiple_files=True)
        txt_docs = st.file_uploader("Upload TXT/DOCX Files", type=["txt", "docx"], accept_multiple_files=True)

        if st.button("Process Documents"):
            raw_text = []

            if pdf_docs:
                raw_text.append(get_pdf_text(pdf_docs))

            if excel_docs:
                raw_text.extend(get_excel_text(excel_docs))

            if txt_docs:
                for doc in txt_docs:
                    raw_text.append(get_text_from_file(doc))

            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)

    # Option to clear chat
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Ask question
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question here 👇")

    if user_question:
        if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
            st.error("❌ FAISS index not found. Please upload and process documents first.")
            return

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain()

        docs = vector_store.similarity_search(user_question, k=3)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)["output_text"]

        # Add to chat history
        st.session_state.chat_history.append((user_question, response))

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**🧑 You:** {q}")
            st.markdown(f"**🤖 AI:** {a}")

if __name__ == "__main__":
    main()
