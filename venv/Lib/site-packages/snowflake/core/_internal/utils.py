from re import compile
from typing import Any, Dict, Optional, Union

from snowflake.connector.description import PLATFORM


# The following code is copied from snowpark's code /snowflake/snowpark/_internal/utils.py to avoid being broken
# when snowpark changes the code.
# We'll need to move the code to a common place.
# Another solution is to move snowpark to the mono repo so the merge gate will find the breaking changes.
# To address later.

EMPTY_STRING = ""
DOUBLE_QUOTE = '"'
ALREADY_QUOTED = compile('^(".+")$')
UNQUOTED_CASE_INSENSITIVE = compile("^([_A-Za-z]+[_A-Za-z0-9$]*)$")
# https://docs.snowflake.com/en/sql-reference/identifiers-syntax.html
SNOWFLAKE_UNQUOTED_ID_PATTERN = r"([a-zA-Z_][\w\$]{0,255})"
SNOWFLAKE_QUOTED_ID_PATTERN = '("([^"]|""){1,255}")'
SNOWFLAKE_ID_PATTERN = f"({SNOWFLAKE_UNQUOTED_ID_PATTERN}|{SNOWFLAKE_QUOTED_ID_PATTERN})"
SNOWFLAKE_OBJECT_RE_PATTERN = compile(
    f"^(({SNOWFLAKE_ID_PATTERN}\\.){{0,2}}|({SNOWFLAKE_ID_PATTERN}\\.\\.)){SNOWFLAKE_ID_PATTERN}$"
)


def normalize_datatype(datatype: str) -> str:
    """Convert equivalent datatypes.

    These are according to https://docs.snowflake.com/en/sql-reference/intro-summary-data-types
    All string types to varchar, and all numeric types to number and float.
    """
    datatype = datatype.upper()
    datatype = datatype.replace(" ", "")
    if datatype in ("INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "BYTEINT", "NUMBER"):
        return "NUMBER(38,0)"
    if datatype in ("DOUBLE", "DOUBLEPRECISION", "REAL"):
        return "FLOAT"
    if datatype in ("STRING", "TEXT", "VARCHAR"):
        return "VARCHAR(16777216)"
    if datatype in ("CHAR", "CHARACTER"):
        return "VARCHAR(1)"
    if datatype in ("VARBINARY"):
        return "BINARY"
    datatype = datatype.replace("DECIMAL", "NUMBER").replace("NUMERIC", "NUMBER")\
        .replace("STRING", "VARCHAR").replace("TEXT", "VARCHAR")
    return datatype


def validate_object_name(name: str) -> None:
    if not SNOWFLAKE_OBJECT_RE_PATTERN.match(name):
        raise ValueError(f"The object name '{name}' is invalid.")


def is_running_inside_stored_procedure() -> bool:
    """
    Check if snowpy is running inside a stored procedure.

    Returns:
        bool: True if snowpy is running inside a stored procedure, False otherwise.
    """
    return PLATFORM == "XP"


def validate_quoted_name(name: str) -> str:
    if DOUBLE_QUOTE in name[1:-1].replace(DOUBLE_QUOTE + DOUBLE_QUOTE, EMPTY_STRING):
        raise ValueError(
            f"Invalid Identifier {name}. "
            f"The inside double quotes need to be escaped when the name itself is double quoted."
        )
    else:
        return name


def escape_quotes(unescaped: str) -> str:
    return unescaped.replace(DOUBLE_QUOTE, DOUBLE_QUOTE + DOUBLE_QUOTE)


def normalize_name(name: str) -> str:
    if ALREADY_QUOTED.match(name):
        return validate_quoted_name(name)
    elif UNQUOTED_CASE_INSENSITIVE.match(name):
        return escape_quotes(name.upper())
    else:
        return DOUBLE_QUOTE + escape_quotes(name) + DOUBLE_QUOTE


def normalize_and_unquote_name(name: str) -> str:
    return unquote_name(normalize_name(name))


def unquote_name(name: str) -> str:
    if len(name) > 1 and name[0] == name[-1] == '"':
        return name[1:-1]
    return name


def try_single_quote_value(value: Any) -> str:
    """Single quote the value if the value is a string and not single quoted yet."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    if value[0] == "'" and value[-1] == "'":  # quote wrapped the string
        value = "".join(list(value)[1:-1])
    return f"""'{value.replace("'", "''")}'"""


def double_quote_name(name: str) -> str:
    return DOUBLE_QUOTE + escape_quotes(name) + DOUBLE_QUOTE if name else name


def _retrieve_parameter_value(p: Dict[str, Optional[str]]) -> Optional[Union[int, bool, str, float]]:
    datatype = p["type"]
    value = p["value"]
    if datatype is None:
        raise ValueError("Data is wrong. Datatype shouldn't be None.")
    if datatype == "NUMBER":
        return int(value) if value is not None and value != "" else None
    elif datatype.startswith("NUMBER"):
        return float(value) if value is not None and value != "" else None
    elif datatype == "BOOLEAN":
        if value in (True, "true"):
            return True
        if value in ("", None):
            return None
        return False
    elif datatype == "STRING":
        return value if value != "" else None
    raise ValueError(f"datatype {datatype} isn't processed.")
