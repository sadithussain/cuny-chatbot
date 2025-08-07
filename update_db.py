# Import required libraries
import os
import json
import sys
from tqdm import tqdm
from dotenv import load_dotenv
import getpass

# Import langchain libraries
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# load in your Google Gemini API Key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Data location
DATA_PATH = "data_demo"
# Vector store storage location
PERSIST_DIRECTORY = "db"
# Data update time storage location
LOG_FILE = "db_log.json"

# Function to assign school and category
def assign_metadata_to_docs(documents, file_path):
    # Create a list that separates the file location into multiple parts 
    # Ex: ['data', 'ccny', 'academic_calendar']
    path_parts = os.path.normpath(file_path).split(os.sep)

    # Try statement to make code safer
    try:
        # Retrieve and assign school and document type to the metadata
        school_name = path_parts[1] 
        doc_type = path_parts[2] 

    except IndexError:
        school_name = "Unknown"
        doc_type = "Unknown"
        print(f"Warning: Could not determine school/type for {file_path}. Using 'Unknown'.")

    for doc in documents:
        doc.metadata['school'] = school_name
        doc.metadata['type'] = doc_type

    return documents
    

# Function to get the log file
def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {}

# Function to save the log
def save_log(log_data):
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent = 4)

# Function to get files that will be processed
def get_files_to_proceed(force_rebuild=False, files_to_force=None):
    if files_to_force is None:
        files_to_force = []

    # If we are forcing a rebuild, create an empty log. Else, load the log
    log_data = {} if force_rebuild else load_log()
    # Make an empty list to store the files to process
    files_to_process = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            # Get the file path
            file_path = os.path.join(root, file)
            # Normalize the path
            normalized_file_path = os.path.normpath(file_path)
            
            # Get the last modified time for the file
            modified_time = os.path.getmtime(file_path)

            # If we are forcing the rebuild, the file exists in the list of files to force, the file path is not in the log, or the last modified time of the file is different from what's inisde the log, then we add this file to the list of files to be processed
            if (force_rebuild or
                normalized_file_path in files_to_force or
                normalized_file_path not in log_data or
                modified_time > log_data.get(normalized_file_path, 0)):
                files_to_process.append(normalized_file_path)

    # Return files to process and log 
    return files_to_process, log_data

# Function that updates the vector store
def update_vector_store(files_to_force = None):
    # Choose the embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model = "text-embedding-004")

    # We will force the rebuild IF and ONLY IF the database doesn't exist
    force_rebuild = not os.path.exists(PERSIST_DIRECTORY)
    if force_rebuild:
        print("Database not found. Forcing a full rebuild...")

    # Create a vector store using the Chroma library
    vector_store = Chroma(
        # Location of your database
        persist_directory = PERSIST_DIRECTORY,
        # Which embedding function you will use to allow your RAG chatbot to retrieve information easily
        embedding_function = embeddings
    )

    # Get the files to be processed and log
    files_to_process, log_data = get_files_to_proceed(force_rebuild, files_to_force)

    # --- LOGIC TO DELETE FILES FROM OUR DATABASE ---

    if not force_rebuild:
        print("Checking for file changes to synchronize database...")

        # Get all the existing chunks in the database
        existing_docs = vector_store.get(include = ["metadatas"])

        # Get the unique sources since multiple chunks may have the same source
        db_sources = set(meta['source'] for meta in existing_docs["metadatas"])

        # Create an empty set to store the files that are in our data folder
        disk_files = set()

        # Add the files that are currently in our data folder to our set of files
        for root, dirs, files in os.walk(DATA_PATH):
            for file in files:
                file_path = os.path.normpath(os.path.join(root, file))
                disk_files.add(file_path)

        # The files we need to delete are the files that are in our database but not in our data folder
        files_to_delete = db_sources - disk_files

        # Check updated files by seeing which files to be processed are in our database
        updated_files = set(f for f in files_to_process if f in db_sources)

        # Combine the files to delete set and updates files set to get all the files that must be deleted
        all_files_to_purge = files_to_delete.union(updated_files)

        if all_files_to_purge:
            # Clean the log by removing entries for any file that will be removed
            print(f"Removing {len(all_files_to_purge)} entries from the log file...")
            for file_path in all_files_to_purge:
                log_data.pop(file_path, None)

            save_log(log_data)

            # Get the ids of the chunks to delete
            ids_to_delete = [
                # Check if the chunks' source is inside the files to purse set
                doc_id for doc_id, metadata in zip(existing_docs['ids'], existing_docs['metadatas'])
                if metadata['source'] in all_files_to_purge
            ]

            if ids_to_delete:
                print(f"Removing {len(ids_to_delete)} old or removed document chunks")
                vector_store.delete(ids = ids_to_delete)

        else:
            print("Database is synchronized. No old or deleted files to remove.")

    # If there are no files to process, then the knowledge base is already up to date
    if not files_to_process:
        print("Knowledge base is already up-to-date")
        return
    
    print(f"Found {len(files_to_process)} new or updated files to process...")

    # Create a list to store pdf documents and a list to store csv documents
    pdf_documents = []
    csv_documents = []


    for file_path in tqdm(files_to_process, desc = "Processing Files"):
        try:
            # Add the pdf and csv files to their respective lists after assigning the correct metadata
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                pdf_documents.extend(assign_metadata_to_docs(documents, file_path))

            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path, encoding = 'utf-8')
                documents = loader.load()
                csv_documents.extend(assign_metadata_to_docs(documents, file_path))

            # Update the log with their last updated times
            log_data[file_path] = os.path.getmtime(file_path)

        # Make an exception if there is an error and show the error
        except Exception as e:
            print(f"Failed to process {file_path}. Error: {e}. Skipping file.")

    # Since the csv documents are already split into chunks, create a final chunks list using the csv documents as the base
    final_chunks = csv_documents

    # If there are pdf documents, then split them and add them to the final chunks list
    if pdf_documents:
        print(f"Splitting {len(pdf_documents)} PDF documents into chunks...")

        # Select the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)
        # Use the text splitter to split the pdf documents into chunks
        split_pdf_chunks = text_splitter.split_documents(pdf_documents)
        # Add these chunks to the final chunks
        final_chunks.extend(split_pdf_chunks)

    # If there are final chunks, add them to the vector store
    if final_chunks:
        print(f"Adding {len(final_chunks)} new chunks to the vector store in batches...")

        # This batch size determines how many chunks you can update to the vector store at a time. We use 4000 because it prevents the embedding function from being overloaded with API calls
        batch_size = 4000
        for i in tqdm(range(0, len(final_chunks), batch_size), desc = "Adding chunks to DB"):
            # Get the batch
            batch = final_chunks[i: i + batch_size]
            # Add the batch to the vector store
            vector_store.add_documents(documents = batch)

        print("Knowledge base updated successfully!")

    # Save the log
    save_log(log_data)

# If this file is ran in the terminal, this is what is going to be done:
if __name__ == "__main__":
    # Start with an empty list of files to force a reprocess
    files_to_force_reprocess = []
    # Check if we use the '--reprocess' tag in the terminal along with 'python update_db.py'
    if '--reprocess' in sys.argv:
        try:
            # Get the index of where '--reprocess' exists
            reprocess_index = sys.argv.index('--reprocess')
            # The files to be reprocessed exist after that index
            files_to_reprocess_raw = sys.argv[reprocess_index + 1:]
            # Check if the user actually included a file path after '--reprocess'
            if not files_to_reprocess_raw:
                print("Error: --reprocess flag used but no file paths were provided")
                sys.exit(1)
            # Normalize the path
            files_to_force_reprocess = [os.path.normpath(p) for p in files_to_reprocess_raw]
        except (ValueError, IndexError):
            pass
    
    # Execute the function to update the vector store and pass the files that will be forced, if any
    update_vector_store(files_to_force=files_to_force_reprocess)