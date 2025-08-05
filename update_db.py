import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
import getpass

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

DATA_PATH = "data_demo"
PERSIST_DIRECTORY = "db"
LOG_FILE = "db_log.json"

def assign_metadata_to_docs(documents, file_path):
    path_parts = os.path.normpath(file_path).split(os.sep)
    school_name = path_parts[1] if len(path_parts) > 2 else "Unknown"
    doc_type = path_parts[2] if len(path_parts) > 2 else "Unknown"
    for doc in documents:
        doc.metadata['school'] = school_name
        doc.metadata['type'] = doc_type
    return documents

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_log(log_data):
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent = 4)

def get_files_to_proceed(force_rebuild = False):
    log_data = {} if force_rebuild else load_log()

    files_to_process = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            modified_time = os.path.getmtime(file_path)

            if force_rebuild or file_path not in log_data or modified_time > log_data[file_path]:
                files_to_process.append(file_path)

    return files_to_process, log_data

def update_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model = "text-embedding-004")

    force_rebuild = not os.path.exists(PERSIST_DIRECTORY)
    if force_rebuild:
        print("Database not found. Forcing a full rebuild...")

    vector_store = Chroma(
        persist_directory = PERSIST_DIRECTORY,
        embedding_function = embeddings
    )

    files_to_process, log_data = get_files_to_proceed(force_rebuild)

    if not files_to_process:
        print("Knowledge base is already up-to-date")
        return
    
    print(f"Found {len(files_to_process)} new or updated files to process...")

    pdf_documents = []
    csv_documents = []

    for file_path in tqdm(files_to_process, desc = "Processing Files"):
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                pdf_documents.extend(assign_metadata_to_docs(documents, file_path))

            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path, encoding = 'utf-8')
                documents = loader.load()
                csv_documents.extend(assign_metadata_to_docs(documents, file_path))

            log_data[file_path] = os.path.getmtime(file_path)

        except Exception as e:
            print(f"Failed to process {file_path}. Error: {e}. Skipping file.")

    final_chunks = csv_documents

    if pdf_documents:
        print(f"Splitting {len(pdf_documents)} PDF documents into chunks...")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 150)
        split_pdf_chunks = text_splitter.split_documents(pdf_documents)
        final_chunks.extend(split_pdf_chunks)

    if final_chunks:
        print(f"Adding {len(final_chunks)} new chunks to the vector store in batches...")

        batch_size = 4000
        for i in tqdm(range(0, len(final_chunks), batch_size), desc = "Adding chunks to DB"):
            batch = final_chunks[i: i + batch_size]
            vector_store.add_documents(documents = batch)

        print("Knowledge base updated successfully!")

    save_log(log_data)

if __name__ == "__main__":
    update_vector_store()