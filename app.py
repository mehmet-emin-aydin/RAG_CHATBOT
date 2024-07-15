import streamlit as st
import requests
import subprocess
import atexit
import os
import signal
import os
import PyPDF2
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
from datetime import datetime
import io
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
log_data = [] 

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class User:
    def __init__(self, username):
        self.username = username
        self.llm = "llama3-8b-8192"
        self.embedder = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def upload_documents(user: User, files) -> tuple[str, int, FAISS]:
    text = _extract_text_from_document(files)
    chunks = _chunk_text(text)
    status_code , vector_store = _create_embeddings_and_save(user, chunks)
    if status_code == 200:
        return "Document uploaded successfully.", 200 , vector_store
    else:
        return "Failed to upload document.", 500


def _extract_text_from_document(files) -> str:
    text = ""
    for file in files:
        file_name = file.name
        file_extension = os.path.splitext(file_name)[1]
        if file_extension == '.txt':
            text += file.read().decode('utf-8')
        elif file_extension == '.pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                text += page.extract_text()
        elif file_extension == '.docx':
            doc = Document(io.BytesIO(file.read()))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
    return text


def _chunk_text(text: str) -> list[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=10,
        length_function=len
    )
    return text_splitter.split_text(text)


def _create_embeddings_and_save(user: User, chunks: any) -> tuple[int, FAISS]:
    embeddings = HuggingFaceEmbeddings(model_name=user.embedder)
    vector_store = FAISS.from_texts(chunks, embeddings, metadatas=[{"source": f"{user.username}:{i}"} for i in range(len(chunks))])
    
    return 200, vector_store


def ask_question(user: User, question: str, vector_store : FAISS) -> tuple[str, int]:

    docs = vector_store.similarity_search(question)
    retrieved_chunks = docs[0].page_content + docs[1].page_content + docs[2].page_content
    prompt = f'Question: "{question}"\nContext: "{retrieved_chunks}"'

    try:
        response = get_completion(prompt)
    except Exception as e:
        return f"LLM connection failed.{e}", 400

    answer = f'{response}\n\n**<Most Related Chunk>**\n\n{retrieved_chunks}'
    _log(user, question, retrieved_chunks, response)
    return answer, 200


def get_completion(prompt, model="llama3-8b-8192"):
    messages = [
        {
            "role": "system", 
            "content": "Based on the context provided, answer the question as an easy-to-understand assistant. Ensure that the answer is concise, directly addresses the question, and is in the same language as the question."
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def _log(user: User, question: str, retrieved_chunks: str, answer: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = (
        f"{timestamp}, Username: {user.username}, Question: {question}, "
        f"LLM: {user.llm}, Embedder: {user.embedder}, "
        f"Retrieved Texts: {retrieved_chunks}, Answer: {answer}\n"
    )
    log_data.append(log_message)


def main():
    st.title("Free Multilingual RAG")

    tabs = ["Upload Document", "Ask Question"]
    active_tab = st.radio("Upload documents first, ask questions later:", tabs)

    if active_tab == "Upload Document":
        upload_document()
    elif active_tab == "Ask Question":
        ask_question_ui(st.session_state.vector_store)


def upload_document():
    st.write("Several files can be uploaded, each upload crushes the old one. Depending on the number and size of files, the upload process may take a long time.")

    username = st.text_input("Enter a username (just something that represents you):")
    uploaded_files = st.file_uploader("Upload your documents (for now it only works with files that have .txt, .pdf or .docx extension):", accept_multiple_files=True)

    if uploaded_files and username:
        st.write("Number of uploaded files:", len(uploaded_files))

        for uploaded_file in uploaded_files:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            st.write(file_details)

        user = User(username=username)
        response, status_code , vector_store= upload_documents(user, uploaded_files)
        st.session_state.vector_store = vector_store
        if status_code == 200:
            st.success(response)
        else:
            st.error("Error:", response)


def ask_question_ui(vector_store : FAISS):
    username = st.text_input("Enter a username (just something that represents you):")
    question = st.text_area("Enter the question you want to ask in your document (the more detailed your question, the more accurate an answer you will get):")

    if st.button("Ask"):
        if not question:
            st.warning("Please enter a question.")
        elif not username:
            st.warning("Please enter a username.")
        else:
            user = User(username=username)
            answer, status_code = ask_question(user, question, vector_store)
            
            if status_code == 200:
                st.success("Answer: " + answer)
            else:
                st.error("Error: " + answer)

if __name__ == "__main__":
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = {}
    main()
