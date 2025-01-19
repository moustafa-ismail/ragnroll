"""Manages virtual Warehouses. Resource is Experimental.

Example:
    >>> warehouse_name = "MYWAREHOUSE"


Refer to :class:`snowflake.core.Root` to create the ``root``.
"""

from ._warehouse import Warehouse, WarehouseCollection, WarehouseResource


__all__ = [
    "Warehouse",
    "WarehouseCollection",
    "WarehouseResource"
]
