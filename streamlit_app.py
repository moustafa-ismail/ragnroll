import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.core import Root
import pandas as pd
import json


# Configuration
NUM_CHUNKS = 3  # Number of chunks to retrieve
SLIDE_WINDOW = 7  # Number of last conversations to remember
CORTEX_SEARCH_DATABASE = "CC_QUICKSTART_CORTEX_SEARCH_DOCS_3"
CORTEX_SEARCH_SCHEMA = "DATA"
CORTEX_SEARCH_SERVICE = "CC_SEARCH_SERVICE_CS"
COLUMNS = [
    "chunk",
    "relative_path",
    "category"
]

# Get active Snowflake session
session = get_active_session()
root = Root(session)
svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]

def config_options():
    """Configure sidebar options for the application."""
    categories = ['Snacks', 'Beverages', 'MainCourse', 'Salads', 'Desserts', 'Appetizers']
    st.sidebar.selectbox('Select Food Category', categories, key="food_category")
    st.sidebar.checkbox('Remember chat history?', key="use_chat_history", value=True)
    st.sidebar.button("Start Over", key="clear_conversation", on_click=init_messages)

def init_messages():
    """Initialize chat history."""
    if st.session_state.get("clear_conversation") or "messages" not in st.session_state:
        st.session_state.messages = []
        # Add Ali's introduction message
        welcome_message = "Hi! I'm Ali, your personal chef friend! Tell me what ingredients you have, and I'll help you whip up something delicious! 👨‍🍳"
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})

def get_chat_history():
    """Retrieve recent messages from the chat history."""
    chat_history = []
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    for i in range(start_index, len(st.session_state.messages) - 1):
        chat_history.append(st.session_state.messages[i])
    return chat_history

def summarize_question_with_history(chat_history, question):
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
    cmd = "select snowflake.cortex.complete(?, ?) as response"
    df_response = session.sql(cmd, params=['mistral-large', prompt]).collect()
    summary = df_response[0].RESPONSE
    return summary.replace("'", "")

def get_similar_chunks_search_service(query, category):
    """Search for similar chunks based on query and category."""
    if category == "ALL":
        response = svc.search(query, COLUMNS, limit=NUM_CHUNKS)
    else:
        filter_obj = {"@eq": {"category": category}}
        response = svc.search(query, COLUMNS, filter=filter_obj, limit=NUM_CHUNKS)
    st.sidebar.json(response.json())
    return response.json()

def create_prompt(query, category):
    """Create a prompt for the LLM with context from search results and chat history."""
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history:
            question_summary = summarize_question_with_history(chat_history, query)
            prompt_context = get_similar_chunks_search_service(question_summary, category)
        else:
            prompt_context = get_similar_chunks_search_service(query, category)
    else:
        prompt_context = get_similar_chunks_search_service(query, category)
        chat_history = ""

    prompt = f"""
    I am Ali, a friendly and witty chef who specializes in {category} recipes! I love helping people cook and finding the perfect recipes from our collection.

    Conversation Flow:
    1. When suggesting recipes:
        - Prioritize recipes that makes use of all ingredients
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

    json_data = json.loads(prompt_context)
    relative_paths = set(item.get('relative_path', '') for item in json_data['results'])
    return prompt, relative_paths

def complete_query(query, category):
    """Complete the query using Snowflake Cortex with Mistral model."""
    prompt, relative_paths = create_prompt(query, category)
    cmd = """
        select snowflake.cortex.complete(?, ?) as response
    """
    df_response = session.sql(cmd, params=['mistral-large', prompt]).collect()
    return df_response, relative_paths

def main():
    """Main Streamlit application function."""
    st.title(":fork_and_knife: Food Recipe Assistant with History")

    # Track previous category
    if "previous_category" not in st.session_state:
        st.session_state.previous_category = None

    config_options()
    init_messages()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check for category change
    current_category = st.session_state.food_category
    if (st.session_state.previous_category is not None and 
        current_category != st.session_state.previous_category):
        category_message = f"I see you've switched to {current_category}! Let me help you find some delicious {current_category} recipes! 👨‍🍳"
        st.session_state.messages.append({"role": "assistant", "content": category_message})
        with st.chat_message("assistant"):
            st.markdown(category_message)

    st.session_state.previous_category = current_category

    # Accept user input
    if query := st.chat_input("What ingredients do you have?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        current_category = st.session_state.food_category
        response, relative_paths = complete_query(query, current_category)
        res_text = response[0].RESPONSE

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})

        # Display related documents
        if relative_paths:
            with st.sidebar.expander("Related Recipes"):
                for path in relative_paths:
                    cmd2 = f"select GET_PRESIGNED_URL(@DOCS, '{path}', 360) as URL_LINK from directory(@DOCS)"
                    df_url_link = session.sql(cmd2).to_pandas()
                    url_link = df_url_link._get_value(0, 'URL_LINK')
                    display_url = f"Recipe: [{path}]({url_link})"
                    st.sidebar.markdown(display_url)

if __name__ == "__main__":
    main()