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

# Function to extract text from Excel files (each row as a chunk)
def get_excel_text(excel_docs):
    chunks = []
    for excel in excel_docs:
        try:
            df = pd.read_excel(excel, sheet_name=None, dtype=str)  # Read as strings
            for sheet_name, sheet in df.items():
                for _, row in sheet.iterrows():
                    row_values = row.dropna().astype(str)  # Convert values to string
                    row_text = " | ".join(row_values.tolist())  # Join row values
                    chunks.append(row_text)
        except Exception as e:
            st.error(f"Error reading {excel.name}: {e}")
    return chunks

# Function to split text into smaller chunks
def get_text_chunks(text_chunks):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # 🔹 Reduced chunk size to avoid token limit
        chunk_overlap=200
    )
    return text_splitter.split_text("\n".join(text_chunks))

# Function to create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    try:
        vector_store.save_local(FAISS_INDEX_PATH)
        st.success("✅ FAISS index created successfully! You can now process your questions.")
    except Exception as e:
        st.error(f"❌ Failed to save FAISS index: {e}")

# Function to define the conversational chain
def get_conversational_chain():
    prompt_template = """
    Use the provided context to answer the question with the highest possible accuracy.
    Also give the follow up questions with answers from the document.
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

# Function to process Excel questions
def process_questions(question_file):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        st.error("No FAISS index found. Please upload and process files first.")
        return

    try:
        new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    chain = get_conversational_chain()
    df = pd.read_excel(question_file)

    if 'Questions' not in df.columns:
        st.error("The uploaded Excel file must have a column named 'Questions'.")
        return

    def get_answer(question):
        """Retrieve top 3 relevant document chunks and pass to OpenAI"""
        docs = new_db.similarity_search(question, k=3)  # 🔹 Fetch only top 3 relevant chunks
        context = "\n".join([doc.page_content for doc in docs])  # 🔹 Reduce total context size

        return chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]

    df['Answers'] = df['Questions'].apply(get_answer)

    output_file = "output_questions.xlsx"
    df.to_excel(output_file, index=False)

    with open(output_file, "rb") as f:
        st.download_button("Download Answered Questions", f, file_name="Answered_Questions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Streamlit App
def main():
    st.set_page_config(page_title="QnA with PDF & Excel 📄📊")
    st.header("QnA with Your Documents 💬")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
        excel_docs = st.file_uploader("Upload Excel Files", type=["xls", "xlsx", "csv"], accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = []

                if pdf_docs:
                    raw_text.append(get_pdf_text(pdf_docs))  # Extract PDF text

                if excel_docs:
                    raw_text.extend(get_excel_text(excel_docs))  # Extract row-wise Excel text

                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                else:
                    st.warning("No text found in uploaded files. Please check your files.")

    question_file = st.file_uploader("Upload an Excel file with questions", type=["xls", "xlsx"])
    if st.button("Process Questions") and question_file:
        process_questions(question_file)

if __name__ == "__main__":
    main()
