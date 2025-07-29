# Import required libraries
import getpass
import os
from dotenv import load_dotenv

# Langchain libraries
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
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
        # Fetch the vector store from the persist_directory
        vector_store = Chroma(
            persist_directory = persist_directory,
            embedding_function = embeddings
        )
        return vector_store
    else:
        print("Creating new vector base")
        # In this array, we will store text from the pdf files
        all_chunks = []

        # Load all pdf files into all_chunks
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()

                    # Add metadata that allows for easier identification
                    school_name = os.path.basename(root)
                    for doc in documents:
                        doc.metadata['school'] = school_name
                    
                    all_chunks.extend(documents)
        
        # text_splitter defines how exactly we want to split the pdf files
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)

        # Use text_splitter on all_chunks to get split_chunks which holds split up text
        split_chunks = text_splitter.split_documents(all_chunks)
        
        # Create a vector store from the current split_chunks
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
    model = "gemini-2.0-flash-lite",
    # Creativity
    temperature = 0.2
)

# Retrieve vector store
vector_store = get_vector_store()

# Create empty chat history
chat_history = []

# Create system prompt. This message will describe how the prompt will be reformatted upon submission.
system_prompt = """You are a helpful assistant for CUNY students named CUNYBot.
Your goal is to answer questions as accurately as possible based on the provided context for the specified school.
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

# We use the PromptTemplate object to create a prompt template
qa_prompt = PromptTemplate(
    # Assign a format
    template = system_prompt,
    # Add input variables. Should be the exact ones used inside template
    input_variables = ["context", "chat_history", "question"]
)

# Initialize memory
memory = ConversationBufferWindowMemory(
    # Chatbot's shot-term memory only remember last 3 responses and user queries
    k = 3,
    return_messages = True,
    # Where the chatbot will be accessing its short-term memory from
    memory_key = "chat_history", # This tells memory to store history under this key
    output_key = "answer",
)

# Retrives top 4 vectors related to the question
retriever = vector_store.as_retriever(search_kwargs = {'k': 4})

# Create conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    # Insert your llm you will use
    llm,
    # Set retriever
    retriever = retriever,
    # Plug in the short-term memory settings
    memory = memory,
    # Chatbot's instructions. This should be the QA_PROMPT which we made earlier
    combine_docs_chain_kwargs={"prompt": qa_prompt},

    # Return the documents for debugging. With this, we are able to see what the chatbot is referencing
    return_source_documents = True
)

# Print ready message in console. This will be removed when we create the actual chatbbot interface
print("CUNY Chatbot is ready! Type 'quit' to exit.")

# This will determine which school the chatbot and user are focusing on right now. Initially it is set to None
current_school = None

# In data, we have several folder names that holds each school's data
# To ensure we are looking in the right one, we must understand which school the user is talking about
# We will create a dictionary to store a list of all possible ways of writing out the name of a school
school_names = {
    'ccny': ['ccny', 'city college', 'the city college of new york', 'cuny city college'],
    'hunter': ['hunter', 'hunter college', 'cuny hunter'],

}

# Now create the loop which allows the conversation to continue until 'quit' is typed.
while True:
    # Collect the user's input
    user_input = input("You: ")
    # Check if the user wants to quit
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break

    # Variable to detect if a new school is mentioned in the user's input
    detected_school = None

    # Find which school the user has mentioned, if any
    for school_key, names in school_names.items():
        if any(name in user_input.lower() for name in names):
            detected_school = school_key
            break

    if detected_school:
        current_school = detected_school
        print(f"Now answering questions about {current_school.upper()}!")
        qa_chain.retriever.search_kwargs['filter'] = {'school': current_school}

    if not current_school:
        print("For the most accurate information, please give the chatbot information about what school you are attending or asking questions about. For example: 'When is the first day of classes for City College?'")
        continue

    # Otherwise, get the chatbot's response given the user's input
    result = qa_chain.invoke({"question": user_input, "chat_history": chat_history})
    chat_history.append((user_input, result["answer"]))

    # Print the response
    print(f"Bot: {result['answer']}")