CREATE OR REPLACE TEMPORARY TABLE docs_categories AS
WITH unique_documents AS (
  SELECT DISTINCT relative_path FROM docs_chunks_table
),
docs_category_cte AS (
  SELECT
    relative_path,
    TRIM(snowflake.cortex.COMPLETE (
      'llama3-70b',
      'Given the name of the file between <file> and </file> determine if it is related to Snacks or Salads or MainCourse or Juices or Desserts or Appetizers. Use only one word <file> ' || relative_path || '</file>'
    ), '\n') AS category
  FROM unique_documents
)
SELECT * FROM docs_category_cte;
