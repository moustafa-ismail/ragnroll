import streamlit as st
from snowflake.snowpark.session import Session
from snowflake.snowpark.context import get_active_session
from snowflake.core import Root
import json
from typing import List, Tuple, Dict
import logging

# Configuration
NUM_CHUNKS = 3  # Number of chunks to retrieve
SLIDE_WINDOW = 7  # Number of last conversations to remember
COLUMNS = ["chunk", "relative_path", "category"]

# Logging Configuration
logging.basicConfig(level=logging.INFO)

# Snowflake Connection Parameters
connection_params = {
    "account": st.secrets["snowflake"]["account"],
    "user": st.secrets["snowflake"]["user"],
    "password": st.secrets["snowflake"]["password"],
    "warehouse": st.secrets["snowflake"]["warehouse"],
    "database": st.secrets["snowflake"]["database"],
    "schema": st.secrets["snowflake"]["schema"],
    "role": st.secrets["snowflake"]["role"],
}

# Initialize Snowflake Session
session = Session.builder.configs(connection_params).create()
root = Root(session)
svc = root.databases[st.secrets["snowflake"]["database"]].schemas[st.secrets["snowflake"]["schema"]].cortex_search_services["CC_SEARCH_SERVICE_CS"]

def validate_secrets():
    """Validate that all required secrets are provided."""
    required_keys = ["account", "user", "password", "warehouse", "database", "schema", "role"]
    for key in required_keys:
        if key not in st.secrets["snowflake"]:
            st.error(f"Missing required secret: {key}")
            return False
    return True

def configure_sidebar():
    """Set up the sidebar components."""
    categories = ["Snacks", "Beverages", "MainCourse", "Salads", "Desserts", "Appetizers"]
    st.sidebar.selectbox("Select Food Category", categories, key="food_category")
    st.sidebar.checkbox("Remember chat history?", key="use_chat_history", value=True)
    st.sidebar.button("Start Over", key="clear_conversation", on_click=init_messages)
    if "related_paths" in st.session_state:
        with st.sidebar.expander("Related Recipes"):
            for path in st.session_state.related_paths:
                st.sidebar.markdown(path)

def init_messages():
    """Initialize the chat history for the session."""
    if st.session_state.get("clear_conversation") or "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_message = (
            "Hi! I'm Ali, your personal chef friend! Tell me what ingredients you have, and I'll help you whip up something delicious! 👨‍🍳"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})

def get_chat_history() -> List[Dict[str, str]]:
    """Retrieve the recent chat history."""
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    return st.session_state.messages[start_index:-1]

def summarize_question_with_history(chat_history: List[Dict[str, str]], question: str) -> str:
    """Summarize the chat history and current question for better context."""
    prompt = f"""
        Based on the chat history below and the question, generate a query that extends the question
        with the chat history provided. The query should be in natural language.
        Answer with only the query. Do not add any explanation.
        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
    """
    cmd = "SELECT snowflake.cortex.complete(?, ?) AS response"
    df_response = session.sql(cmd, params=["mistral-large", prompt]).collect()
    return df_response[0].RESPONSE.replace("'", "")

def get_similar_chunks_search_service(query: str, category: str) -> Dict:
    """Search for similar chunks based on query and category."""
    try:
        if category == "ALL":
            response = svc.search(query, COLUMNS, limit=NUM_CHUNKS)
        else:
            filter_obj = {"@eq": {"category": category}}
            response = svc.search(query, COLUMNS, filter=filter_obj, limit=NUM_CHUNKS)
        return response.json()
    except Exception as e:
        st.error(f"An error occurred while searching: {e}")
        return {}

def create_prompt(query: str, category: str) -> Tuple[str, List[str]]:
    """Create a prompt for the LLM using chat history and search results."""
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history:
            question_summary = summarize_question_with_history(chat_history, query)
            prompt_context = get_similar_chunks_search_service(question_summary, category)
        else:
            prompt_context = get_similar_chunks_search_service(query, category)
    else:
        prompt_context = get_similar_chunks_search_service(query, category)

    prompt = f"""
    I am Ali, a friendly and witty chef who specializes in {category} recipes! I love helping people cook and finding the perfect recipes from our collection.

    Conversation Flow:
    1. When suggesting recipes:
        - Prioritize recipes that make use of all ingredients
        - First list all matching recipes as numbered options
        - Ask which recipe they'd like to know more about
    2. When user selects a recipe, provide full details in this format:
        Recipe Name:
        Quantities (for 1 person):
        Cooking Time:
        Steps:
        Cuisine:
        General Diet Type:

    <chat_history>
    {chat_history}
    </chat_history>

    <context>
    {prompt_context}
    </context>

    User Query: {query}
    Current Category: {category}

    Response (as Ali, friendly and category-aware):
    """

    relative_paths = [item.get("relative_path", "") for item in prompt_context.get("results", [])]
    return prompt, relative_paths

def complete_query(query: str, category: str) -> Tuple[str, List[str]]:
    """Complete the query using the LLM."""
    prompt, relative_paths = create_prompt(query, category)
    cmd = "SELECT snowflake.cortex.complete(?, ?) AS response"
    df_response = session.sql(cmd, params=["mistral-large", prompt]).collect()
    return df_response[0].RESPONSE, relative_paths

def main():
    """Main Streamlit application function."""
    st.title(":fork_and_knife: Food Recipe Assistant with History")

    if "previous_category" not in st.session_state:
        st.session_state.previous_category = None

    configure_sidebar()
    init_messages()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check for category change
    current_category = st.session_state.food_category
    if (st.session_state.previous_category and 
        current_category != st.session_state.previous_category):
        category_message = f"I see you've switched to {current_category}! Let me help you find some delicious {current_category} recipes! 👨‍🍳"
        st.session_state.messages.append({"role": "assistant", "content": category_message})
        with st.chat_message("assistant"):
            st.markdown(category_message)

    st.session_state.previous_category = current_category

    # Accept user input
    if query := st.chat_input("What ingredients do you have?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Generating recipe suggestions..."):
            response_text, relative_paths = complete_query(query, current_category)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

        # Save related paths for display in the sidebar
        st.session_state.related_paths = relative_paths

if __name__ == "__main__":
    if validate_secrets():
        main()
