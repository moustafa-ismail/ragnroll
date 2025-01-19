from __future__ import absolute_import

# flake8: noqa

# import apis into api package
from snowflake.core.dynamic_table._generated.api.dynamic_table_api import DynamicTableApi

__all__ = [
    "DynamicTableApi",
]
