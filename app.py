import streamlit as st
import requests
import subprocess
import atexit
import os
import signal
import os
import PyPDF2
from docx import Document
from fastapi import UploadFile, FastAPI, File, Form, UploadFile, HTTPException
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import pickle
from datetime import datetime
import io
from dotenv import load_dotenv
class User:
    def __init__(self, username):
        self.username = username
        self.llm = "gemini-pro"
        self.embedder = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

async def upload_documents(user: User, files: list[UploadFile]) -> tuple[str, int]:
    text = await _extract_text_from_document(files)
    chunks = await _chunk_text(text)
    pkl_name, status_code = await _create_embeddings_and_save(user, chunks)
    if status_code == 200:
        return "Document uploaded successfully.", 200
    else:
        return "Failed to upload document.", 500

async def _extract_text_from_document(files: list[UploadFile]) -> str:
    text = ""
    for file in files:
        byte_object = await file.read()
        file_name = file.filename
        file_extension = os.path.splitext(file_name)[1]
        if file_extension == '.txt':
            text += byte_object.decode('utf-8')
        elif file_extension == '.pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(byte_object))
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                text += page.extract_text()
        elif file_extension == '.docx':
            doc = Document(io.BytesIO(byte_object))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
    return text

async def _chunk_text(text: str) -> list[str]:
    chunks = None
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=10,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

async def _create_embeddings_and_save(user: User, chunks: any) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=user.embedder)
    pkl_name = os.path.join(user.username + ".pkl")
    vector_store = FAISS.from_texts(chunks, embeddings, metadatas=[{"source": f"{pkl_name}:{i}"} for i in range(len(chunks))])
    with open(pkl_name, "wb") as f:
        pickle.dump(vector_store, f)
    return vector_store

async def ask_question(user: User, question: str, api_key: str) -> tuple[str, int]:
    username = user.username
    vector_store = await _get_vector_file(username)
    if vector_store is None:
        return "Document not found.", 400
    
    if api_key is not None:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        is_loaded = load_dotenv()
        if is_loaded == False:
            return "API key not found.", 400
        
    llm = ChatGoogleGenerativeAI(model=user.llm, temperature=0, max_output_tokens=256, top_k = 40, top_p = 0.8)
    docs = vector_store.similarity_search(question)
    retrieved_chunks = docs[0].page_content + docs[1].page_content + docs[2].page_content
    system_message="Figure out the answer of the question by the given information pieces. ALWAYS answer with the language of the question."
    prompt = system_message + "Question: " + question + " Context: " + retrieved_chunks
    try:
        response = llm.invoke(prompt)
    except Exception:
        return "Wrong API key.", 400
    answer = response.content + "  **<Most Related Chunk>**  " + retrieved_chunks
    await _log(user, question, system_message, retrieved_chunks, response.content)
    return answer, 200

async def _get_vector_file(username: str)-> any:
    with open(username+".pkl", "rb") as f:
        vector_store = pickle.load(f)
    return vector_store

async def _log(user: User, question: str, system_message: str, retrieved_chunks: str, answer: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = (
        f"{timestamp}, Username: {user.username}, Question: {question}, "
        f"LLM: {user.llm}, Embedder: {user.embedder}, System Message: {system_message}, "
        f"Retrieved Texts: {retrieved_chunks}, Answer: {answer}\n"
    )
    with open("log.txt", "a", encoding="utf-8") as file:
        file.write(log_message)


app = FastAPI()
@app.post("/document-uploader")
async def document_uploader(username: str = Form(...), files: list[UploadFile] = File(...)):
    user = User(username=username)
    response, status_code = await upload_documents(user, files)
    if status_code == 200:
        return {response}
    else:
        raise HTTPException(status_code=status_code, detail=response)

@app.post("/question-answerer")
async def question_answerer(username: str = Form(...), question: str = Form(...), api_key = File(None)):
    user = User(username=username)
    response, status_code = await ask_question(user, question, api_key)
    if status_code == 200:
        return {response}
    else:
        raise HTTPException(status_code=status_code, detail=response)


def main():
    st.title("Free Multilingual RAG")

    tabs = ["Upload Document", "Ask Question"]
    active_tab = st.radio("Upload documents first, ask questions later:", tabs)
    
    if active_tab == "Upload Document":
        upload_document()
    elif active_tab == "Ask Question":
        ask_question()

def upload_document():
    st.write("Several files can be uploaded, each upload crushes the old one. Depending on the number and size of files, the upload process may take a long time.")

    username = st.text_input("Enter a username (just something that represents you):")
    uploaded_files = st.file_uploader("Upload your documents (for now it only works with files that has .txt, .pdf or .docx extension):", accept_multiple_files=True)

    if uploaded_files:
        st.write("Number of uploaded files:", len(uploaded_files))
        
        for uploaded_file in uploaded_files:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            st.write(file_details)
                
        files = [("files", (uploaded_file.name, uploaded_file, uploaded_file.type)) for uploaded_file in uploaded_files]
        
        payload = {'username': username}
        
        with st.spinner('Loading...'):
            response =  requests.post("http://localhost:8000/document-uploader/", files=files, data=payload)
        
        if response.status_code == 200:
            st.success(response.text)
        else:
            st.error("Error:", response.text)


def ask_question():
    username = st.text_input("Enter a username (just something that represents you):") 
    api_key = st.text_input("Add your Google API key. It is free. Key acquisition video: [https://www.youtube.com/watch?v=brCkpzAD0gc]: (If you do not trust you can download and use the app in your local too)", type="password")
    question = st.text_area("Enter the question you want to ask in your document (the more detailed your question, the more accurate an answer you will get): ")
    
    if st.button("Ask"):
        if not question:
            st.warning("Please enter a question.")
        elif not username:
            st.warning("Please enter a username.")
        else:
            payload = {'username': username, 'question': question, 'api_key': api_key}
            
            with st.spinner('Question is getting answered...'):
                response = requests.post("http://localhost:8000/question-answerer/", data=payload)
            
            if response.status_code == 200:
                st.success("Answer: " + response.text)
            else:
                print(response)
                st.error("Error:", response.text)

uvicorn_process = None

def run_fastapi():
    global uvicorn_process
    if uvicorn_process is None:
        uvicorn_process = subprocess.Popen(["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"])
        print("FastAPI server has been started.")

def cleanup():
    global uvicorn_process
    if uvicorn_process:
        os.kill(uvicorn_process.pid, signal.SIGTERM)
        uvicorn_process.wait()
        print("FastAPI server has been closed.")

if __name__ == "__main__":
    run_fastapi()
    atexit.register(cleanup)
    main()
