import snowflake.connector
import os

# Snowflake connection parameters
SNOWFLAKE_ACCOUNT = "<your_account>"
SNOWFLAKE_USER = "<your_username>"
SNOWFLAKE_PASSWORD = "<your_password>"
SNOWFLAKE_DATABASE = "CC_QUICKSTART_CORTEX_SEARCH_DOCS"
SNOWFLAKE_SCHEMA = "DATA"
SNOWFLAKE_STAGE = "docs"

def upload_to_stage(local_folder, file_extensions=(".pdf",)):
    """Uploads files from a local folder to a Snowflake stage."""
    # Connect to Snowflake
    ctx = snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
    )
    cursor = ctx.cursor()
    
    try:
        # Get the list of files to upload
        files_to_upload = [
            os.path.join(local_folder, f)
            for f in os.listdir(local_folder)
            if f.endswith(file_extensions)
        ]
        
        if not files_to_upload:
            print("No files found for upload.")
            return

        # Upload each file to the stage
        for file_path in files_to_upload:
            print(f"Uploading {file_path}...")
            cursor.execute(
                f"PUT 'file://{file_path}' @{SNOWFLAKE_STAGE}"
            )
        print("Upload completed successfully.")
    finally:
        cursor.close()
        ctx.close()

# Example usage
if __name__ == "__main__":
    # Update the local folder path
    local_folder_path = "./sample_data"
    upload_to_stage(local_folder_path)
