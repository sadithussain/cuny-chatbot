# app.py
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import re # --- NEW: Import the regular expression library ---
from chatbot import get_vector_store, create_chain, school_names

# --- App Configuration ---
st.set_page_config(page_title="CUNY AI Assistant", page_icon="🤖")

# --- Caching ---
# Cache the vector store and chain creation for performance.
@st.cache_resource
def load_chain():
    vector_store = get_vector_store()
    return create_chain(vector_store)

# --- App State Management ---
# Initialize session state to remember the conversation and selected school/category
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_school" not in st.session_state:
    st.session_state.selected_school = None
if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

# --- UI Rendering ---
st.title("CUNY AI Assistant 🤖")

# --- Sidebar for Selections ---
with st.sidebar:
    st.header("Search Options")
    
    # 1. School Selection Dropdown
    school_display_names = {key: aliases[1].title() for key, aliases in school_names.items()}
    selected_display_name = st.selectbox(
        "1. Choose a CUNY School:",
        options = list(school_display_names.values()),
        index=None,
        placeholder="Select a school..."
    )
    
    if selected_display_name:
        st.session_state.selected_school = [key for key, value in school_display_names.items() if value == selected_display_name][0]
    
    # 2. Category Selection Dropdown (appears after a school is selected)
    if st.session_state.selected_school:
        st.session_state.selected_category = st.selectbox(
            "2. Choose a Category:",
            options=["All", "Academic Calendar", "Professor Reviews"],
            index=0 # Default to "All"
        )
        st.success(f"Ready to answer questions about {selected_display_name} in the '{st.session_state.selected_category}' category!")

# Display existing chat messages from the history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Guardrail Logic ---
# This is the same guardrail your friend added
restricted_patterns = [
    r"my financial aid", 
    r"my aid status", 
    r"my classes",
    r"my class schedule",
    r"my schedule", 
    r"my account", 
    r"my application status", 
    r"\bssn\b",
    r"social security number", 
    r"health advice", 
    r"my social security",
    r"my student id", 
    r"my tuition"
]

def is_restricted_question(user_input):
    for pattern in restricted_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return True
    return False

# --- Chat Interaction Logic ---
if prompt := st.chat_input("Ask a question..."):
    # Add the user's message to the chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- NEW: Check if the question is restricted ---
    if is_restricted_question(prompt):
        response = "Sorry, I'm unable to access personal or sensitive information. Please contact your school's official office for help."
        with st.chat_message("assistant"):
            st.error(response) # Use an error box for restricted answers
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Check if a school has been selected
    elif not st.session_state.selected_school:
        with st.chat_message("assistant"):
            st.warning("Please select a school from the sidebar to begin.")
    else:
        # If a school is selected, get the bot's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa_chain = load_chain()
                
                # --- Robust filter logic ---
                current_filter = {'school': st.session_state.selected_school}
                
                if st.session_state.selected_category == "Academic Calendar":
                    current_filter = {"$and": [{'school': st.session_state.selected_school}, {'type': 'academic_calendar'}]}
                elif st.session_state.selected_category == "Professor Reviews":
                    current_filter = {"$and": [{'school': st.session_state.selected_school}, {'type': 'rmp_reviews'}]}
                
                qa_chain.retriever.search_kwargs['filter'] = current_filter
                
                # Get the chat history in the format the chain expects
                chat_history = [
                    (msg["content"], st.session_state.messages[i+1]["content"])
                    for i, msg in enumerate(st.session_state.messages)
                    if msg["role"] == "user" and i + 1 < len(st.session_state.messages)
                ]

                # Invoke the chain to get a result
                result = qa_chain.invoke({"question": prompt, "chat_history": chat_history})
                response = result["answer"]
                st.markdown(response)

                # Optionally display source documents in an expandable section
                if result.get("source_documents"):
                    with st.expander("View Sources"):
                        for doc in result["source_documents"]:
                            st.info(f"Source: {doc.metadata.get('source', 'N/A')}")
                            st.markdown(f"> {doc.page_content}")

        # Add the bot's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
