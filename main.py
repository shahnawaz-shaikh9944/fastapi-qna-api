from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI

# Load environment variables from .env
load_dotenv()

# Set environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
FAISS_INDEX_PATH = "faiss_index"

# Initialize app
app = FastAPI()


# ---------------------------- Utility Functions ----------------------------

def extract_pdf_text(files):
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def extract_excel_text(files):
    chunks = []
    for file in files:
        df = pd.read_excel(file, sheet_name=None, dtype=str)
        for sheet in df.values():
            for _, row in sheet.iterrows():
                row_values = row.dropna().astype(str)
                chunks.append(" | ".join(row_values.tolist()))
    return chunks


def extract_text_from_txt_docx(file):
    if file.filename.endswith(".txt"):
        return file.file.read().decode("utf-8")
    elif file.filename.endswith(".docx"):
        doc = Document(file.file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""


def get_text_chunks(text_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text("\n".join(text_list))


def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)


def load_chain():
    prompt_template = """
    You are an AI assistant tasked with answering questions based only on the provided context.
    Also give all the Follow Up Questions for the asked question ONLY from the provided context.
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


# ---------------------------- API Routes ----------------------------

@app.get("/")
def read_root():
    return {
        "message": "📄 Welcome to the Document QnA API!",
        "usage": {
            "Upload Documents": "POST /upload/",
            "Ask Question": "POST /ask/",
            "API Docs": "/docs"
        }
    }


@app.post("/upload/")
async def upload_files(
    pdf_files: List[UploadFile] = File([]),
    excel_files: List[UploadFile] = File([]),
    txt_files: List[UploadFile] = File([]),
):
    raw_text = []

    # Process PDFs
    if pdf_files:
        pdf_text = extract_pdf_text([f.file for f in pdf_files])
        raw_text.append(pdf_text)

    # Process Excel
    if excel_files:
        excel_text = extract_excel_text([f.file for f in excel_files])
        raw_text.extend(excel_text)

    # Process TXT and DOCX
    if txt_files:
        for file in txt_files:
            raw_text.append(extract_text_from_txt_docx(file))

    if raw_text:
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return {"message": "✅ Documents processed and vector store created successfully."}
    else:
        return {"message": "⚠️ No valid files uploaded."}


@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        return {"error": "❌ FAISS index not found. Please upload and process documents first."}

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    chain = load_chain()

    docs = vector_store.similarity_search(question, k=3)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    return {"response": response["output_text"]}

