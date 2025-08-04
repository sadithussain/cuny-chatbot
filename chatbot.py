# Import required libraries
import getpass
import os
from dotenv import load_dotenv
from tqdm import tqdm # Import the progress bar library
import re # Import the regular expression library

# Langchain libraries
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# --- Data Loading and Processing ---
def get_vector_store():
    data_path = "data_demo"
    persist_directory = "db"
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    if os.path.exists(persist_directory):
        print("✅ Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vector_store
    else:
        print("🔎 Creating new vector base...")
        all_documents = []
        file_paths = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                file_paths.append(os.path.join(root, file))

        print("Loading and processing files...")
        for file_path in tqdm(file_paths, desc="Files"):
            documents = []
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path, encoding="utf-8")
                documents = loader.load()

            if documents:
                path_parts = os.path.normpath(file_path).split(os.sep)
                if len(path_parts) > 2:
                    school_name = path_parts[1]
                    doc_type = path_parts[2] 
                    for doc in documents:
                        doc.metadata['school'] = school_name
                        doc.metadata['type'] = doc_type
                    all_documents.extend(documents)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_chunks = text_splitter.split_documents(all_documents)
        
        print("🧠 Creating new vector store (this may take several minutes)...")
        vector_store = Chroma.from_documents(
            documents=split_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print("✅ New vector store created successfully!")
        return vector_store

# --- Chain Creation ---
def create_chain(vector_store):
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=.2)

    system_prompt = """You are a helpful and friendly assistant for CUNY students named CUNYBot.

Follow these rules in order:
1.  First, analyze the user's QUESTION to determine if it is a simple conversational greeting, a thank you, or a question about you (e.g., "hello", "how are you?", "who are you?"). If it is, answer it from your own knowledge in a friendly way.
2.  If the question is not conversational, then it is a CUNY-Specific Question. You MUST answer these questions using ONLY the provided CONTEXT.
3.  If the CONTEXT contains multiple reviews about a professor, you must synthesize them into a helpful summary. Do not just list the raw comments. For example, start your response with "Students have mixed reviews about..." or "Professor [Name] is generally well-regarded..." and then summarize the key points.
4.  When answering a CUNY-Specific Question: If the user asks about a "break" or "day off," you should specifically look for terms like "College Closed" or "No classes scheduled" in the CONTEXT to find the answer.
5.  If you have searched the CONTEXT and the answer is not there, you MUST say 'I am sorry, I cannot find that information in the provided documents.' Do not make up an answer.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

IMPORTANT: Your final answer MUST be in English. Under no circumstances should you use any other language.

ANSWER:
"""

    qa_prompt = PromptTemplate(template=system_prompt, input_variables=["context", "chat_history", "question"])
    
    memory = ConversationBufferWindowMemory(k=3, return_messages=True, memory_key="chat_history", output_key="answer")
    retriever = vector_store.as_retriever(search_kwargs={'k': 8})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )
    return qa_chain

# --- School Data ---
school_names = {
    'baruch': ['baruch', 'baruch college'], 'bmcc': ['bmcc', 'borough of manhattan community college'],
    'bronxcc': ['bronxcc', 'bronx community college'], 'brooklyn': ['brooklyn', 'brooklyn college'],
    'citytech': ['city tech', 'citytech', 'new york city college of technology'], 'csi': ['csi', 'college of staten island'],
    'guttman': ['guttman', 'guttman community college'], 'hostos': ['hostos', 'hostos community college'],
    'hunter': ['hunter', 'hunter college'], 'johnjay': ['john jay', 'johnjay', 'john jay college of criminal justice'],
    'kingsborough': ['kingsborough', 'kingsborough community college'], 'laguardia': ['laguardia', 'laguardia community college'],
    'lehman': ['lehman', 'lehman college'], 'medgarevers': ['medgar evers', 'medgarevers', 'medgar evers college'],
    'queens': ['queens', 'queens college'], 'queensborough': ['queensborough', 'queensborough community college'],
    'york': ['york', 'york college'], 'ccny': ['ccny', 'city college', 'the city college of new york'],
    'cunygrad': ['cuny grad', 'cuny graduate center', 'the graduate center'], 'cunylaw': ['cuny law', 'cuny school of law'],
    'cunysph': ['cuny sph', 'cuny school of public health'], 'cunyslu': ['cuny slu', 'cuny school of labor and urban studies'],
    'cunysps': ['cuny sps', 'cuny school of professional studies']
}

# --- Terminal Interaction (Only runs when this file is executed directly) ---
if __name__ == "__main__":
    vector_store = get_vector_store()
    qa_chain = create_chain(vector_store)
    
    print("\nCUNY Chatbot is ready! Type 'quit' to exit.")
    current_school = None
    chat_history = []

    while True:
        if not current_school:
            school_input = input("Which CUNY school are you interested in? (e.g., 'ccny', 'hunter'): ").lower()
            for key, aliases in school_names.items():
                if school_input in aliases:
                    current_school = key
                    break
            if current_school:
                 print(f"Bot: Okay, I will now answer questions about {current_school.upper()}.")
                 qa_chain.retriever.search_kwargs['filter'] = {'school': current_school}
            else:
                print("Invalid school. Please try again.")
                continue

        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        result = qa_chain.invoke({"question": user_input, "chat_history": chat_history})
        chat_history.append((user_input, result["answer"]))
        print(f"Bot: {result['answer']}")
