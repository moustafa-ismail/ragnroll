import streamlit as st
from snowflake.snowpark.session import Session
from snowflake.core import Root
from snowflake.cortex import complete
import snowflake.connector
import pandas as pd
import json
import numpy as np
from trulens.apps.custom import instrument
from trulens.apps.custom import TruCustomApp
from trulens.providers.cortex.provider import Cortex
from trulens.dashboard import run_dashboard
import trulens.dashboard.streamlit as trulens_st
from trulens.core import TruSession
# from feedback import get_feedbacks
from trulens.core import Feedback
from trulens.core import Select
from trulens.providers.cortex.provider import Cortex
import numpy as np
from typing import List, Tuple, Dict
import logging

# Configuration
NUM_CHUNKS = 3  # Number of chunks to retrieve
SLIDE_WINDOW = 7  # Number of last conversations to remember
CORTEX_SEARCH_DATABASE = st.secrets["snowflake"]["database"]
CORTEX_SEARCH_SCHEMA = st.secrets["snowflake"]["schema"]
CORTEX_SEARCH_SERVICE = "CC_SEARCH_SERVICE_CS"
COLUMNS = [
    "chunk",
    "relative_path",
    "category"
]

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
svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]

# Initialize trulens session
tru_session = TruSession()

provider = Cortex(snowpark_session=session, model_engine="mistral-large2")
# Feedback function

# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)
# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on_output()
)

# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(
        provider.context_relevance_with_cot_reasons, name="Context Relevance"
    )
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(np.mean)  # choose a different aggregation method if you wish
)

feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance]


def validate_secrets():
    """Validate that all required secrets are provided."""
    required_keys = ["account", "user", "password", "warehouse", "database", "schema", "role"]
    for key in required_keys:
        if key not in st.secrets["snowflake"]:
            st.error(f"Missing required secret: {key}")
            return False
    return True

def config_options():
    """Configure sidebar options for the application."""
    categories = ['Snacks', 'Juices', 'MainCourse', 'Salads', 'Desserts', 'Appetizers']
    st.sidebar.selectbox('Select Food Category', categories, key="food_category")
    st.sidebar.checkbox('Remember chat history?', key="use_chat_history", value=True)
    st.sidebar.button("Start Over", key="clear_conversation", on_click=init_messages)
    if "related_paths" in st.session_state:
        with st.sidebar.expander("Related Recipes"):
            for path in st.session_state.related_paths:
                st.sidebar.markdown(path)

def init_messages():
    """Initialize the chat history for the session."""
    if st.session_state.get("clear_conversation") or "messages" not in st.session_state:
        st.session_state.messages = []
        # Add Ali's introduction message
        welcome_message = (
            "Hi! I'm Ali, your personal chef friend! Tell me what ingredients you have, and I'll help you whip up something delicious! 👨‍🍳"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})

def get_chat_history() -> List[Dict[str, str]]:
    """Retrieve the recent chat history."""
    chat_history = []
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    for i in range(start_index, len(st.session_state.messages) - 1):
        chat_history.append(st.session_state.messages[i])
    return chat_history

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
    return complete("mistral-large2", prompt=prompt, session=session)

class RAG_class:

    @instrument
    def retrieve(self, query, category):
        """Search for similar chunks based on query and category."""
        if category == "ALL":
            response = svc.search(query, COLUMNS, limit=NUM_CHUNKS)
        else:
            filter_obj = {"@eq": {"category": category}}
            response = svc.search(query, COLUMNS, filter=filter_obj, limit=NUM_CHUNKS)

        json_data = json.loads(response.model_dump_json())
        relative_paths = set(item.get('relative_path', '') for item in json_data['results'])

        if response.results:
            retrieved_chunks = [curr["chunk"] for curr in response.results]
            st.sidebar.json(response.model_dump_json())
            return retrieved_chunks, relative_paths
        
        else:
            st.sidebar.json({'response':'Retrieval is empty'})
            return retrieved_chunks, relative_paths


    def create_prompt(self, query, category, prompt_context, chat_history=""):
        """Create a prompt for the LLM with context from search results and chat history."""

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

        return prompt
    

    @instrument
    def generate_completion(self, query: str, prompt: str, context_rag: list) -> str:
        """Calls mistral-large2 model with the final prompt. Extra input for Trulens monitoring"""

        return complete("mistral-large2", prompt, session=session)
    

    @instrument
    def query(self, query: str, category: str):
        """Core function that handles the chatbot response. Monitored by Trulens"""

        if st.session_state.use_chat_history:
            chat_history = get_chat_history()

            if chat_history:
                question_summary = summarize_question_with_history(chat_history, question=query)
                context_rag, relative_paths = self.retrieve(query=question_summary, category=category)

                # Create prompt with chat history and new context rag with question_summary
                prompt = self.create_prompt(query=query, category=category, prompt_context=context_rag, chat_history=chat_history)

            else:
                context_rag, relative_paths = self.retrieve(query=query, category=category)

                prompt = self.create_prompt(query=query, category=category, prompt_context=context_rag)
            
        else:
            context_rag, relative_paths = self.retrieve(query=query, category=category)
            prompt = self.create_prompt(query=query, category=category, prompt_context=context_rag)

        # Call the model
        completion = self.generate_completion(
            query=query, prompt=prompt, context_rag=context_rag
        )

        return completion, relative_paths


myrag = RAG_class()

# Start Trulens recorder
tru_rag = TruCustomApp(
    myrag,
    app_name="rag-new",
    app_version="base",
    feedbacks=feedbacks,
)

def main():
    """Main Streamlit application function."""
    st.title(":fork_and_knife: Food Recipe Assistant with History")

    if "previous_category" not in st.session_state:
        st.session_state.previous_category = None

    config_options()
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

        # Generate response
        current_category = st.session_state.food_category

        # Added recording
        with tru_rag as recording:
            response, relative_paths = myrag.query(query=query, category=current_category)
        
        #get record
        record = recording.get()

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display related documents
        if relative_paths:
            with st.sidebar.expander("Related Recipes"):
                for path in relative_paths:
                    cmd2 = f"select GET_PRESIGNED_URL(@DOCS, '{path}', 360) as URL_LINK from directory(@DOCS)"
                    df_url_link = session.sql(cmd2).to_pandas()
                    url_link = df_url_link._get_value(0, 'URL_LINK')
                    display_url = f"Recipe: [{path}]({url_link})"
                    st.sidebar.markdown(display_url)

        # Save related paths for display in the sidebar
        st.session_state.related_paths = relative_paths

        with st.expander("See the trace of this record 👀"):
            trulens_st.trulens_trace(record=record)

        trulens_st.trulens_feedback(record=record)

if __name__ == "__main__":
    if validate_secrets():
        main()
