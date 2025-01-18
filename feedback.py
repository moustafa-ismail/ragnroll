from snowflake.snowpark.context import get_active_session
from trulens.core import Feedback
from trulens.core import Select
from trulens.providers.cortex.provider import Cortex
import numpy as np



def get_feedbacks(session):
    

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

    return feedbacks
