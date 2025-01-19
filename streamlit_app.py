import streamlit as st
import requests
from streamlit_lottie import st_lottie
import logging
import snowflake.connector
import pandas as pd
import json
from typing import List, Tuple, Dict, Optional
from snowflake.snowpark.session import Session
from snowflake.core import Root

# Set page configuration
st.set_page_config(page_title="Food Recipe Assistant", page_icon="🍴", layout="wide")

# Function to load a Lottie animation from a URL with retry logic
def load_lottie_from_url(url: str, max_retries: int = 5, timeout: int = 15) -> Optional[dict]:
    """
    Load a Lottie animation from a URL with retry logic.

    Parameters:
        url (str): URL to fetch the Lottie animation from.
        max_retries (int): Maximum number of retry attempts.
        timeout (int): Timeout for each request in seconds.

    Returns:
        dict: Lottie animation JSON if successfully loaded, otherwise None.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                logging.info("Successfully loaded animation from URL.")
                return response.json()
            else:
                logging.warning(f"Attempt {attempt + 1}: Failed to fetch animation. "
                                f"Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1}: Error fetching animation from URL: {e}")
    
    logging.error(f"Failed to load animation from URL after {max_retries} attempts.")
    return None

# Lottie Animation URL
LOTTIE_URL = "https://lottie.host/e45492cc-f42c-42f0-be45-17d8ea90c568/twGE6H6R0V.json"
DEFAULT_LOTTIE_URL = "https://path-to-your-default-animation.json"

# Load Lottie animation from the URL or fallback
lottie_cooking = load_lottie_from_url(LOTTIE_URL) or load_lottie_from_url(DEFAULT_LOTTIE_URL)

# Configuration
NUM_CHUNKS = 3
SLIDE_WINDOW = 7
CORTEX_SEARCH_SERVICE = "CC_SEARCH_SERVICE_CS"
COLUMNS = ["chunk", "relative_path", "category"]

logging.basicConfig(level=logging.INFO)

# Validate Secrets
def validate_secrets() -> bool:
    required_keys = ["account", "user", "password", "warehouse", "database", "schema", "role"]
    missing_keys = [k for k in required_keys if k not in st.secrets["snowflake"]]
    if missing_keys:
        st.error(f"Missing required secrets: {missing_keys}")
        return False
    return True

# Create Snowflake Session
@st.cache_resource
def create_snowflake_session() -> Session:
    connection_params = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
        "role": st.secrets["snowflake"]["role"],
    }
    return Session.builder.configs(connection_params).create()

# Sidebar Configuration
def configure_sidebar():
    categories = ["Snacks", "Beverages", "MainCourse", "Salads", "Desserts", "Appetizers", "ALL"]
    st.sidebar.selectbox("Select Food Category", categories, key="food_category")
    st.sidebar.checkbox("Remember chat history?", key="use_chat_history", value=True)
    st.sidebar.button("Start Over", key="clear_conversation", on_click=init_messages)

    if "related_paths" in st.session_state:
        with st.sidebar.expander("Related Recipes"):
            for path in st.session_state.related_paths:
                st.sidebar.markdown(path)

# Initialize Chat Messages
def init_messages():
    if st.session_state.get("clear_conversation") or "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_message = (
            "Hi! I'm Ali, your personal chef friend! "
            "Tell me what ingredients you have, and I'll help you whip up something delicious! 👨‍🍳"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# Get Chat History
def get_chat_history() -> List[Dict[str, str]]:
    if "messages" not in st.session_state:
        return []
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    return st.session_state.messages[start_index:-1]

# Get Search Service
def get_search_service(session: Session):
    root = Root(session)
    db = root.databases[st.secrets["snowflake"]["database"]]
    schema = db.schemas[st.secrets["snowflake"]["schema"]]
    svc = schema.cortex_search_services[CORTEX_SEARCH_SERVICE]
    return svc

# Get Similar Chunks from Search Service
def get_similar_chunks_search_service(query: str, category: str, session: Session) -> Dict:
    svc = get_search_service(session)
    if category == "ALL":
        response = svc.search(query, COLUMNS, limit=NUM_CHUNKS)
    else:
        filter_obj = {"@eq": {"category": category}}
        response = svc.search(query, COLUMNS, filter=filter_obj, limit=NUM_CHUNKS)

    raw_json = response.json()
    if isinstance(raw_json, str):
        raw_json = json.loads(raw_json)
    return raw_json

# Create Prompt
def create_prompt(query: str, category: str, session: Session) -> Tuple[str, List[str]]:
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
    else:
        chat_history = []

    search_results = get_similar_chunks_search_service(query, category, session)
    relative_paths = [
        item.get("relative_path", "") for item in search_results.get("results", [])
    ]

    prompt = f"""
I am Ali, a friendly and witty chef who specializes in {category} recipes! 
I love helping people cook and find the perfect recipes from our collection.

Conversation so far:
{chat_history}

Now, the user has asked: '{query}'

You have the following search results:
{search_results}
"""
    return prompt, relative_paths

# Complete Query
def complete_query(query: str, category: str, session: Session) -> Tuple[str, List[str]]:
    prompt, relative_paths = create_prompt(query, category, session)
    cmd = "SELECT snowflake.cortex.complete(?, ?) AS response"
    df_response = session.sql(cmd, params=["mistral-large", prompt]).collect()
    response_text = df_response[0].RESPONSE
    return response_text, relative_paths

# Main Function
def main():
    if "show_animation" not in st.session_state:
        st.session_state.show_animation = True  # Show animation on the first load

    if st.session_state.show_animation and lottie_cooking:
        st_lottie(lottie_cooking, height=300, width=300, key="page_load_animation")
        st.write("Loading your personal chef assistant...")  # Temporary loading message
        # Simulate a delay for the setup process
        import time
        time.sleep(5)  # Display animation for 5 seconds
        st.session_state.show_animation = False  # Hide animation after setup
        st.experimental_rerun()  # Reload the app to show the main content

    # Main content
    st.title(":fork_and_knife: Food Recipe Assistant with History")

    if not validate_secrets():
        st.stop()

    session = create_snowflake_session()
    configure_sidebar()
    init_messages()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if "previous_category" not in st.session_state:
        st.session_state.previous_category = st.session_state.food_category

    current_category = st.session_state.food_category
    if current_category != st.session_state.previous_category:
        cat_msg = (
            f"I see you've switched to {current_category}! "
            f"Let me help you find some delicious {current_category} recipes! 👨‍🍳"
        )
        st.session_state.messages.append({"role": "assistant", "content": cat_msg})
        with st.chat_message("assistant"):
            st.markdown(cat_msg)
        st.session_state.previous_category = current_category

    if user_query := st.chat_input("What ingredients do you have?"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Generating recipe suggestions..."):
            response_text, relative_paths = complete_query(user_query, current_category, session)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

        st.session_state.related_paths = []
        if relative_paths:
            for path in relative_paths:
                try:
                    cmd2 = f"SELECT GET_PRESIGNED_URL(@DOCS, '{path}', 360) AS URL_LINK FROM directory(@DOCS)"
                    df_url_link = session.sql(cmd2).to_pandas()
                    url_link = df_url_link.at[0, 'URL_LINK']
                    display_url = f"Recipe: [{path}]({url_link})"
                except Exception:
                    display_url = path
                st.session_state.related_paths.append(display_url)

if __name__ == "__main__":
    main()
