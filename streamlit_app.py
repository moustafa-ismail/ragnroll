import streamlit as st
from snowflake.snowpark.session import Session
from snowflake.core import Root
from snowflake.cortex import complete
import json
import numpy as np
import logging

# Trulens imports
from trulens.apps.custom import instrument
from trulens.apps.custom import TruCustomApp
from trulens.providers.cortex.provider import Cortex
from trulens.core import TruSession, Feedback, Select
import trulens.dashboard.streamlit as trulens_st

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
NUM_CHUNKS = 3
SLIDE_WINDOW = 7
CORTEX_SEARCH_DATABASE = st.secrets["snowflake"]["database"]
CORTEX_SEARCH_SCHEMA = st.secrets["snowflake"]["schema"]
CORTEX_SEARCH_SERVICE = "CC_SEARCH_SERVICE_CS"
COLUMNS = ["chunk", "relative_path", "category"]

logging.basicConfig(level=logging.INFO)

connection_params = {
    "account": st.secrets["snowflake"]["account"],
    "user": st.secrets["snowflake"]["user"],
    "password": st.secrets["snowflake"]["password"],
    "warehouse": st.secrets["snowflake"]["warehouse"],
    "database": st.secrets["snowflake"]["database"],
    "schema": st.secrets["snowflake"]["schema"],
    "role": st.secrets["snowflake"]["role"],
}

# ---------------------------------------------------
# Initialize Snowflake and Trulens
# ---------------------------------------------------
session = Session.builder.configs(connection_params).create()
root = Root(session)
svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]

tru_session = TruSession()
provider = Cortex(snowpark_session=session, model_engine="mistral-large2")

# ---------------------------------------------------
# Feedback Functions
# ---------------------------------------------------
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)

f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on_output()
)

f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(np.mean)
)

feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance]

# ---------------------------------------------------
# Utility Functions
# ---------------------------------------------------
def validate_secrets():
    required_keys = ["account", "user", "password", "warehouse", "database", "schema", "role"]
    for key in required_keys:
        if key not in st.secrets["snowflake"]:
            st.error(f"Missing required secret: {key}")
            return False
    return True

def init_messages():
    if st.session_state.get("clear_conversation") or "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_message = (
            "Hey there, I'm **Chef Ali**! 🍳\n\n"
            "I’m excited to help you find the perfect recipe. "
            "Tell me what ingredients you have or what you'd like to cook!"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})

def get_chat_history():
    chat_history = []
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    for i in range(start_index, len(st.session_state.messages) - 1):
        chat_history.append(st.session_state.messages[i])
    return chat_history

def summarize_question_with_history(chat_history, question):
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

# ---------------------------------------------------
# RAG Class
# ---------------------------------------------------
class RAG_class:
    @instrument
    def retrieve(self, query, category):
        """Retrieve relevant chunks from Snowflake based on category."""
        if category == "ALL":
            response = svc.search(query, COLUMNS, limit=NUM_CHUNKS)
        else:
            filter_obj = {"@eq": {"category": category}}
            response = svc.search(query, COLUMNS, filter=filter_obj, limit=NUM_CHUNKS)

        json_data = json.loads(response.model_dump_json())
        relative_paths = set(item.get('relative_path', '') for item in json_data.get('results', []))
        
        if response.results:
            retrieved_chunks = [curr["chunk"] for curr in response.results]
            st.sidebar.json(response.model_dump_json())
        else:
            retrieved_chunks = []
            st.sidebar.json({'response':'Retrieval is empty'})
        
        return retrieved_chunks, relative_paths

    def create_prompt(self, query, category, prompt_context, chat_history=""):
        """Create the final prompt for the LLM."""
        prompt = f"""
        I am Ali, a friendly and witty chef who specializes in {category} recipes!
        I love helping people cook and finding the perfect recipes from our collection.

        Conversation Flow:
        1. When suggesting recipes:
           - Prioritize recipes that make use of the given ingredients.
           - First list all matching recipes as numbered options.
           - Ask which recipe they'd like to know more about.
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
    def generate_completion(self, query, prompt, context_rag):
        """
        Show a built-in "spinner" while the model processes, removing any Lottie references.
        """
        with st.spinner("Chef Ali is cooking up a response..."):
            response = complete("mistral-large2", prompt, session=session)

        return response

    @instrument
    def query(self, query, category):
        """
        Main function to handle a user query and return model response. 
        Monitored by TruLens.
        """
        if st.session_state.use_chat_history:
            chat_history = get_chat_history()
            if chat_history:
                question_summary = summarize_question_with_history(chat_history, question=query)
                context_rag, relative_paths = self.retrieve(query=question_summary, category=category)
                prompt = self.create_prompt(
                    query=query,
                    category=category,
                    prompt_context=context_rag,
                    chat_history=chat_history
                )
            else:
                context_rag, relative_paths = self.retrieve(query=query, category=category)
                prompt = self.create_prompt(
                    query=query,
                    category=category,
                    prompt_context=context_rag
                )
        else:
            context_rag, relative_paths = self.retrieve(query=query, category=category)
            prompt = self.create_prompt(
                query=query,
                category=category,
                prompt_context=context_rag
            )

        # Generate completion (using st.spinner)
        completion = self.generate_completion(query, prompt, context_rag)
        return completion, relative_paths

# ---------------------------------------------------
# Initialize RAG and TruCustomApp
# ---------------------------------------------------
myrag = RAG_class()
tru_rag = TruCustomApp(
    myrag,
    app_name="rag-new",
    app_version="base",
    feedbacks=feedbacks,
)

# ---------------------------------------------------
# UI Config (Sidebar)
# ---------------------------------------------------
def config_options():
    with st.sidebar:
        st.title("Chef Ali Dashboard")
        st.write("Configure your chat and explore retrieved documents.")

        categories = ['ALL', 'Snacks', 'Juices', 'MainCourse', 'Salads', 'Desserts', 'Appetizers']
        st.selectbox('Select Food Category', categories, key="food_category")
        st.checkbox('Remember chat history?', key="use_chat_history", value=True)

        if st.button("Start Over"):
            st.session_state.clear_conversation = True
            init_messages()
            st.balloons()  # optional celebratory effect

        # Show retrieved recipes
        if "related_paths" in st.session_state and st.session_state.related_paths:
            st.markdown("---")
            st.subheader("Related Recipes")
            for path in st.session_state.related_paths:
                cmd2 = f"SELECT GET_PRESIGNED_URL(@DOCS, '{path}', 360) AS URL_LINK FROM DIRECTORY(@DOCS)"
                df_url_link = session.sql(cmd2).to_pandas()
                url_link = df_url_link._get_value(0, 'URL_LINK')
                display_url = f"- [{path}]({url_link})"
                st.markdown(display_url)

# ---------------------------------------------------
# Main App
# ---------------------------------------------------
def main():
    st.set_page_config(page_title="Chef Ali", layout="wide")
    st.title("👨‍🍳 Chef Ali: Your Recipe Companion")

    st.markdown(
        """
        Welcome to your personal culinary companion! Ask me for recipe ideas based on the 
        ingredients you have, or switch categories for variety. Bon appétit! 🎉
        """
    )
    st.markdown("---")

    config_options()
    init_messages()

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check if user changed category
    if "previous_category" not in st.session_state:
        st.session_state.previous_category = st.session_state.food_category

    current_category = st.session_state.food_category
    if st.session_state.previous_category and current_category != st.session_state.previous_category:
        category_message = (
            f"I see you've switched to **{current_category}**! "
            "Let's explore some new recipes!"
        )
        st.session_state.messages.append({"role": "assistant", "content": category_message})
        with st.chat_message("assistant"):
            st.markdown(category_message)

    st.session_state.previous_category = current_category

    # Chat input
    if query := st.chat_input("What ingredients do you have or what do you want to cook?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response (Trulens monitoring)
        with tru_rag as recording:
            response, relative_paths = myrag.query(query=query, category=current_category)

        record = recording.get()

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.related_paths = relative_paths

        # Expanders for seeing trace and feedback
        with st.expander("See the trace of this record 👀"):
            trulens_st.trulens_trace(record=record)
        with st.expander("Feedback Analysis"):
            trulens_st.trulens_feedback(record=record)

        # Optional star rating or quick feedback
        st.markdown("**Rate my answer** (1: poor, 5: excellent):")
        rating = st.slider("", 1, 5, 3)
        st.info(f"Your rating: {rating} ⭐")

# ---------------------------------------------------
# Run
# ---------------------------------------------------
if __name__ == "__main__":
    if validate_secrets():
        main()
