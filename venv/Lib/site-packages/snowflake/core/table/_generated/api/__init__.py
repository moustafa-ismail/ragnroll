from __future__ import absolute_import

# flake8: noqa

# import apis into api package
from snowflake.core.table._generated.api.table_api import TableApi

__all__ = [
    "TableApi",
]
