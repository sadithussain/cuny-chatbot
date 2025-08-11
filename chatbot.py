# chatbot.py

# Import required libraries
import getpass
import os
import re
from dotenv import load_dotenv

# Import langchain libraries
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables
load_dotenv()

# --- Reusable Data and Functions for the App ---

# This dictionary provides the school names and their aliases for the UI and filtering.
school_names = {
    'baruch': ['baruch', 'Baruch College'], 
    'bmcc': ['bmcc', 'Borough Of Manhattan Community College'],
    'bronxcc': ['bronxcc', 'Bronx Community College'], 
    'brooklyn': ['brooklyn', 'Brooklyn College'],
    'citytech': ['citytech', 'New York City College Of Technology'], 
    'csi': ['csi', 'College Of Staten Island'],
    'guttman': ['guttman', 'Guttman Community College'], 
    'hostos': ['hostos', 'Hostos Community College'],
    'hunter': ['hunter', 'Hunter College'], 
    'johnjay': ['johnjay', 'John Jay College Of Criminal Justice'],
    'kingsborough': ['kingsborough', 'Kingsborough Community College'], 
    'laguardia': ['laguardia', 'Laguardia Community College'],
    'lehman': ['lehman', 'Lehman College'], 
    'medgarevers': ['medgarevers', 'Medgar Evers College'],
    'queens': ['queens', 'Queens College'], 
    'queensborough': ['queensborough', 'Queensborough Community College'],
    'york': ['york', 'York College'], 
    'ccny': ['ccny', 'City College'],
    'cunygrad': ['cunygrad', 'The Graduate Center'], 
    'cunylaw': ['cunylaw', 'Cuny School Of Law'],
    'cunysph': ['cunysph', 'Cuny School Of Public Health'], 
    'cunyslu': ['cunyslu', 'Cuny School Of Labor And Urban Studies'],
    'cunysps': ['cunysps', 'Cuny School Of Professional Studies']
}


# Function to load the existing vector store.
def get_vector_store():
    # This function should ONLY load the database. The 'update_db.py' script handles creation.
    persist_directory = "db"
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    if not os.path.exists(persist_directory):
        print("Database not found. Please run 'python update_db.py' to build the knowledge base.")
        # We use st.error and st.stop() to halt the app if the DB doesn't exist.
        import streamlit as st
        st.error("Knowledge base not found. Please tell the site administrator to build it.")
        st.stop()
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_store

# Function to create the main AI chain.
def create_chain(vector_store):
    # Retrieve the Google Gemini API Key
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

    llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-lite", temperature = 0.2)

    # 1. REMOVE the MultiQueryRetriever and its specific prompt.
    #    REPLACE it with the simple, effective MMR retriever from before.
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs=  {'k': 15, 'fetch_k': 25} # Increased k and fetch_k slightly for better results
    )
    
    # 2. DEFINE the prompt for handling conversational follow-up questions.
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    # This is your main system prompt for the final answer generation.
    system_prompt = """You are a helpful and friendly assistant for CUNY students named CUNYBot.

    Follow these rules in order:
    1.  First, analyze the user's QUESTION to determine if it is a simple conversational greeting, a thank you, or a question about you (e.g., "hello", "how are you?", "who are you?"). If it is, answer it from your own knowledge in a friendly way.
    2.  If the question is not conversational, then it is a CUNY-Specific Question. You MUST answer these questions using ONLY the provided CONTEXT.
    3.  If the CONTEXT contains multiple reviews about a professor, you must synthesize them into a helpful summary. Do not just list the raw comments.
    4.  When a user asks for a 'link', 'website', or 'URL' for a specific organization, you MUST provide the value from the `page_url` field found in the CONTEXT.
    5.  When answering a CUNY-Specific Question: If the user asks about a "break" or "day off," you should specifically look for terms like "College Closed" or "No classes scheduled" in the CONTEXT to find the answer.
    6.  If you have searched the CONTEXT and the answer is not there, you MUST say 'I am sorry, I cannot find that information in the provided documents.' Do not make up an answer.

    CONTEXT:
    {context}

    CHAT HISTORY:
    {chat_history}

    QUESTION:
    {question}

    IMPORTANT: Your final answer MUST be in English. Under no circumstances should you use any other language.

    ANSWER:
    """

    qa_prompt = PromptTemplate(template = system_prompt, input_variables = ["context", "chat_history", "question"])
    
    memory = ConversationBufferWindowMemory(k = 3, return_messages = True, memory_key = "chat_history", output_key = "answer")
    
    # 3. CREATE the final chain, now with the condense_question_prompt.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever = retriever, # Using the simple MMR retriever
        memory = memory,
        condense_question_prompt = CONDENSE_QUESTION_PROMPT, # Using the new condense prompt
        combine_docs_chain_kwargs = {"prompt": qa_prompt},
        return_source_documents=True
    )
    return qa_chain