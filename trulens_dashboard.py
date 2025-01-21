from trulens.dashboard import run_dashboard
from streamlit_app import tru_session

tru_session.migrate_database()
run_dashboard(tru_session, force=True)