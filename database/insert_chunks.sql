insert into docs_chunks_table (relative_path, size, file_url, scoped_file_url, chunk)
    select relative_path, 
           size,
           file_url, 
           build_scoped_file_url(@docs, relative_path) as scoped_file_url,
           func.chunk as chunk
    from 
        directory(@docs),
        TABLE(text_chunker (TO_VARCHAR(SNOWFLAKE.CORTEX.PARSE_DOCUMENT(@docs, 
                                  relative_path, {'mode': 'LAYOUT'})))) as func;