# Import required libraries
import getpass
import os
from dotenv import load_dotenv
import re # Import the regular expression library

# Langchain libraries
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Function to get the vector store
def get_vector_store():
    # Vector store location
    PERSIST_DIRECTORY = "db"
    # Our embedding function
    embeddings = GoogleGenerativeAIEmbeddings(model = "text-embedding-004")

    # If the vector store folder doesn't exist, it will exit and tells us to run the updata_db script
    if not os.path.exists(PERSIST_DIRECTORY):
        print("Database not found")
        print("Please run 'python update_db.py' to build the knowledge base")
        exit()
    
    # Otherwise, load the vector store
    print("Loading existing vector store")
    # Use the Chroma library
    vector_store = Chroma(
        # Select the location of the vector store
        persist_directory = PERSIST_DIRECTORY,
        # Choose the embedding function
        embedding_function = embeddings
    )
    
    # Return the vector store
    return vector_store

# Function to create the chain that the RAG Chatbot functions on
def create_chain(vector_store):
    # Retrieve the Google Gemini API Key
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

    # Choose the LLM version and temperature (low temperature for less creativity and more factuality)
    llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-lite", temperature = .2)

    # This is the prompt that will be passed to the chatbot along with the user's question. This is very important as it defines how the chatbot should respond to the user.
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

    # Create the QA Prompt using the PromptTemplate object. We pass the system prompt as the template. We also include the input variables. These variables are variables that are used inside the system prompt/template as seen above.
    qa_prompt = PromptTemplate(template = system_prompt, input_variables = ["context", "chat_history", "question"])
    
    # This is the chatbot's short term memory. The k value defines how many back-and-forth interactions the chatbot is able to see into the past. This will be the chat_history variable that we see in the system prompt/template.
    memory = ConversationBufferWindowMemory(k = 3, return_messages = True, memory_key = "chat_history", output_key = "answer")

    # We turn the vector store into a retriever. This allows the RAG Chatbot to retrieve information. The RAG Chatbot is able to be the top k relevant chunks.
    retriever = vector_store.as_retriever(search_kwargs = {'k': 8})

    # We create the final QA Chain. This combines everything including our LLM model, our retriever, memory, and the prompt is set as the qa_prompt variable. We also return the source documents which allows us to see what chunks the RAG Chatbot used to come to it's answer.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever = retriever,
        memory = memory,
        combine_docs_chain_kwargs = {"prompt": qa_prompt},
        return_source_documents = True
    )

    # Return the final QA Chain
    return qa_chain

# List of schools in the CUNY system and a list of names that they may be called by.
school_names = {
    'baruch': ['baruch', 'baruch college'], 
    'bmcc': ['bmcc', 'borough of manhattan community college'],
    'bronxcc': ['bronxcc', 'bronx community college'], 
    'brooklyn': ['brooklyn', 'brooklyn college'],
    'citytech': ['city tech', 'citytech', 'new york city college of technology'], 
    'csi': ['csi', 'college of staten island'],
    'guttman': ['guttman', 'guttman community college'], 
    'hostos': ['hostos', 'hostos community college'],
    'hunter': ['hunter', 'hunter college'], 
    'johnjay': ['john jay', 'johnjay', 'john jay college of criminal justice'],
    'kingsborough': ['kingsborough', 'kingsborough community college'], 
    'laguardia': ['laguardia', 'laguardia community college'],
    'lehman': ['lehman', 'lehman college'], 
    'medgarevers': ['medgar evers', 'medgarevers', 'medgar evers college'],
    'queens': ['queens', 'queens college'], 
    'queensborough': ['queensborough', 'queensborough community college'],
    'york': ['york', 'york college'], 
    'ccny': ['ccny', 'city college', 'the city college of new york'],
    'cunygrad': ['cuny grad', 'cuny graduate center', 'the graduate center'], 
    'cunylaw': ['cuny law', 'cuny school of law'],
    'cunysph': ['cuny sph', 'cuny school of public health'], 
    'cunyslu': ['cuny slu', 'cuny school of labor and urban studies'],
    'cunysps': ['cuny sps', 'cuny school of professional studies']
}

# What happens when this file is run from the terminal directly. Ex: 'python chatbot.py'
if __name__ == "__main__":
    # Get the vector store
    vector_store = get_vector_store()

    # Create a QA chain using our vector store
    qa_chain = create_chain(vector_store)
    
    print("\nCUNY Chatbot is ready! Type 'quit' to exit.")

    # No school has been chosen yet so there is no current school
    current_school = None

    # The chat history is empty since no conversation has begun yet
    chat_history = []

    # This is an infinite loop that runs until the user types 'quit'
    while True:
        # If the user hasn't mentioned a school yet, we ask them which school they are interested in
        if not current_school:
            # We get the lower case version of what input they have to the question below
            school_input = input("Which CUNY school are you interested in? (e.g., 'ccny', 'hunter'): ").lower()
            # We search through the list of school names
            for key, aliases in school_names.items():
                # If the user's input is inside of any of the list of aliases that are inside the list of school names, then the current school becomes the name of the school that that alias represents
                if school_input in aliases:
                    current_school = key
                    break
            
            # If the user has successfully entered a school, the RAG Chatbot will filter information relevant to only that specific school
            if current_school:
                 print(f"Bot: Okay, I will now answer questions about {current_school.upper()}.")
                 qa_chain.retriever.search_kwargs['filter'] = {'school': current_school}
            
            # If the user still hasn't mentioned a school, they will be asked to try again
            else:
                print("Invalid school. Please try again.")
                continue

        # Get the user input
        user_input = input("You: ")

        # If the user typed 'quit', then the application closes
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Get the response by passing the user's question and the chat history
        result = qa_chain.invoke({"question": user_input, "chat_history": chat_history})

        # Add the response to the chat history
        chat_history.append((user_input, result["answer"]))

        # Print out the response for the user to see
        print(f"Bot: {result['answer']}")
