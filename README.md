# Recipe Recommendation System with RAG LLM

## Project Overview
This project implements a Recipe Retrieval-Augmented Generation (RAG) system powered by a Large Language Model (LLM). The system searches through a curated recipe database and provides users with recipe suggestions based on their ingredients, dietary needs, or other specific queries. The goal is to assist individuals, particularly those living alone or with limited cooking time, in finding quick and suitable meal options.

---

## Features
1. **User-Friendly Interface**: A Streamlit-based app for easy interaction.
2. **Recipe Suggestions**: Tailored recommendations based on user-provided queries.
3. **Document Uploading**: Upload recipe PDFs to enrich the database.
4. **RAG Architecture**: Combines retrieval techniques with LLM-based generation for highly relevant responses.

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- Streamlit
- Snowflake database (or any preferred DB setup)
- Required libraries in `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/moustafa-ismail/ragnroll
   cd ragnroll
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure database:
   - Execute the SQL scripts in the `database` directory to set up the schema, tables, and required functions.

5. Add your Streamlit secrets:
   - Create a `.streamlit/secrets.toml` file:
     ```toml
     [snowflake]
     account = "<your-account>"
     user = "<your-user>"
     password = "<your-password>"
     warehouse = "<your-warehouse>"
     database = "<your-database>"
     schema = "<your-schema>"
     role = "<your-role>"  # Optional
     ```

6. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## File Details

### Main Scripts
- **`streamlit_app.py`**:
  The main entry point for the Streamlit application.

- **`upload_documents.py`**:
  Handles the processing and ingestion of recipe PDFs into the database.

### SQL Scripts
Located in the `database` directory, these scripts define the schema, tables, and functions required to set up the database:
- `create_database_and_schema.sql`: Initializes the database and schema.
- `create_chunks_table.sql`: Creates a table to store recipe chunks.
- `create_docs_categories_table.sql`: Stores metadata about documents and their categories.
- `insert_chunks.sql`: Inserts processed chunks into the database.
- `update_categories.sql`: Updates categories for better retrieval.

### Sample Data
The `sample_data` directory contains example PDF documents to test the system. These include categories such as appetizers, desserts, juices, main dishes, salads, and snacks.

---

## Usage
1. Launch the Streamlit app.
2. Upload recipe documents via the app or ensure the `sample_data` PDFs are processed.
3. Enter a query, such as "Show me recipes with chicken and rice," or "Suggest a quick dessert."
4. Receive personalized recipe recommendations.

---

## Future Improvements
- Add support for additional dietary preferences and restrictions.
- Enhance query understanding using NLP techniques.
- Integrate a recommendation engine for better personalization.
- Deploy the application on a cloud platform for wider accessibility.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- Streamlit for the interactive interface.
- Snowflake for the database backend.
- The open-source community for libraries and resources.

