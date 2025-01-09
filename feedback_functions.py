import numpy as np
from trulens.core import Feedback
from trulens.core import Select
from trulens.providers.cortex import Cortex
from snowflake.snowpark.context import get_active_session


snowpark_session = get_active_session()

provider = Cortex(snowpark_session.connection, "mistral-large2")

f_context_relevance = (
    Feedback(provider.context_relevance, name="Context Relevance")
    .on_input_output()
    .aggregate(np.mean)
)