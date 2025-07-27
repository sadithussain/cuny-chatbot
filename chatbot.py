# Import required libraries
import getpass
import os
from dotenv import load_dotenv

# Langchain libraries
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Function to laod or create information database
def get_vector_store():
    # Name of the folder where data is stored
    data_path = "data"
    # Where data vector will be fetched from/created
    persist_directory = "db"
    # Model that translates text into numerical vectors
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    # Check if the database exists
    if os.path.exists(persist_directory):
        print("Loading existing vector store")
        vector_store = Chroma(
            persist_directory = persist_directory,
            embedding_function = embeddings
        )
        return vector_store
    else:
        all_chunks = []
        print("Creating new vector base")

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    all_chunks.extend(documents)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
        split_chunks = text_splitter.split_documents(all_chunks)
        
        vector_store = Chroma.from_documents(
            documents = split_chunks,
            persist_directory = persist_directory,
            embedding = embeddings
        )

        return vector_store
                    
# Check if the Google API key exists. Otherwise, ask the user to enter theirs.
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Choose your model and tweaks
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.2
)

# Retrieve vector store
vector_store = get_vector_store()

# Create system prompt. This message will describe how the prompt will be reformatted upon submission.
system_prompt = """You are a helpful assistant for CUNY students named CUNYBot.
Your goal is to answer questions as accurately as possible based on the provided context.
You must always respond in English.
If you do not know the answer, say 'I am sorry, I cannot find that information. Could you be more specific?'
Do not try to make up an answer.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:
"""

# Create a PromptTemplate object
QA_PROMPT = PromptTemplate(
    template=system_prompt,
    input_variables=["context", "chat_history", "question"]
)

# Create empty chat history
chat_history = []

# Initialize memory
memory = ConversationBufferWindowMemory(
    k=3,
    return_messages=True,
    memory_key="chat_history" # This tells memory to store history under this key
)

# Create conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever = vector_store.as_retriever(search_kwargs = {'k': 4}),
    memory = memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)

# Print ready message in console. This will be removed when we create the actual chatbbot interface
print("CUNY Chatbot is ready! Type 'quit' to exit.")

# Now create the loop which allows the conversation to continue until 'quit' is typed.
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break

    result = qa_chain.invoke({"question": user_input, "chat_history": chat_history})
    chat_history.append((user_input, result["answer"]))

    print(f"Bot: {result['answer']}")