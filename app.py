# This import allows asynchronous code to be used
import nest_asyncio
nest_asyncio.apply()

# Import streamlit, the library used to create our web interface
import streamlit as st

# Import re, the library used to check for restricted wordss
import re

# Import the functions that data created in our chatbot.py file
from chatbot import get_vector_store, create_chain, school_names

# Set the title and icon that appear in the browser
st.set_page_config(page_title="CUNY AI Assistant", page_icon="🤖")

# Cache the vector store. This means that the 'get_vector_store' function will only be run once. This improves performance as we do not require to rerun this function every time the user interacts with the UI
@st.cache_resource
def load_chain():
    vector_store = get_vector_store()
    return create_chain(vector_store)

# This is how streamlit remembers user interactions
# If the session_state doesn't include these variables, new ones are created
# Create the message list
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create the selected_school variable
if "selected_school" not in st.session_state:
    st.session_state.selected_school = None

# Create the selected_category variable
if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

# Create the title seen at the top of the web page
st.title("CUNY AI Assistant 🤖")

# Create a sidebar
with st.sidebar:
    # Heading at the top of the sidebar
    st.header("Search Options")
    
    # Create key : value pair using the school names list that we have inside chatbot.py. This allows us to display clean school names while still having the correct key for filtering.
    school_display_names = {key: aliases[1].title() for key, aliases in school_names.items()}

    # Create a dropdown menu to allow users to select the school they have questions about
    selected_display_name = st.selectbox(
        # Title
        "1. Choose a CUNY School:",
        # Make the options the list of school display names
        options = list(school_display_names.values()),
        index = None,
        # Placeholder text
        placeholder = "Select a school..."
    )
    
    # If a school has been selected, we update the session's selected school state
    if selected_display_name:
        # This is a reverse lookup. We create a list of keys (this will only be one key) that match the key of the school that the user selected. We then get the first key which is the only key because this list would only be a length of 1.
        st.session_state.selected_school = [key for key, value in school_display_names.items() if value == selected_display_name][0]
    
    # Create a dropdown menu to allow the users to select the category of questions they have.
    if st.session_state.selected_school:
        st.session_state.selected_category = st.selectbox(
            # Title
            "2. Choose a Category:",
            # Options, we need to manually add these
            options = ["All", "Academic Calendar", "Professor Reviews", "Clubs"],
            index = 0 # Default to "All"
        )
        st.success(f"Ready to answer questions about {selected_display_name} in the '{st.session_state.selected_category}' category!")

# Display existing chat messages from the history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Phrases that are on our restricted list
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

# Fucntion to check if the user's input contains restructed phrases
def is_restricted_question(user_input):
    for pattern in restricted_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return True
    return False

# Chat interaction logic
if prompt := st.chat_input("Ask a question..."):
    # Add the user's message to the chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if the user has entered restricted phrases
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
                
                # Filter using the selected school
                current_filter = {'school': st.session_state.selected_school}
                
                if st.session_state.selected_category == "Academic Calendar":
                    current_filter = {"$and": [{'school': st.session_state.selected_school}, {'type': 'academic_calendar'}]}
                elif st.session_state.selected_category == "Professor Reviews":
                    current_filter = {"$and": [{'school': st.session_state.selected_school}, {'type': 'rmp_reviews'}]}
                elif st.session_state.selected_category == "Clubs":
                    current_filter = {"$and": [{'school': st.session_state.selected_school}, {'type': 'clubs'}]}
                
                qa_chain.retriever.search_kwargs['filter'] = current_filter

                # Invoke the chain to get a result
                result = qa_chain.invoke(prompt)
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
