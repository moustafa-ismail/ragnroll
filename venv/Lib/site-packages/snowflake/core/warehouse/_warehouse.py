from typing import TYPE_CHECKING, Iterator, Optional

from pydantic import StrictStr

from snowflake.core._common import AccountObjectCollectionParent, CreateMode, ObjectReferenceMixin
from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core.warehouse._generated.api import WarehouseApi
from snowflake.core.warehouse._generated.api_client import StoredProcApiClient
from snowflake.core.warehouse._generated.models.warehouse import WarehouseModel as Warehouse


if TYPE_CHECKING:
    from snowflake.core import Root


class WarehouseCollection(AccountObjectCollectionParent["WarehouseResource"]):
    """Represents the collection operations of the Snowflake Warehouse resource.

    With this collection, you can create or update or fetch all warehouses that you have access to.

    Args:
        root: A :class:`Root` instance.

    Example:
        Create a WarehouseCollection instance:

        >>> # after creating a root instance
        >>> warehouse_collection = root.warehouses
    """

    def __init__(self, root: "Root") -> None:
        super().__init__(root, ref_class=WarehouseResource)
        self._api = WarehouseApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def create(
        self,
        warehouse: Warehouse,
        *,
        mode: CreateMode = CreateMode.error_if_exists
    ) -> "WarehouseResource":
        """Create a warehouse in Snowflake.

        Args:
            warehouse: an instance of :class:`Warehouse`.
            mode: One of the following enum values.

                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError` if the
                    warehouse already exists in Snowflake. Equivalent to SQL ``create warehouse <name> ...``.

                CreateMode.or_replace: Replace if the warehouse already exists in Snowflake. Equivalent to SQL
                    ``create or replace warehouse <name> ...``.

                CreateMode.if_not_exists: Do nothing if the warehouse already exists in Snowflake. Equivalent to SQL
                    ``create warehouse <name> if not exists...``

                Default value is CreateMode.error_if_exists.

        Example:
            Create a warehouse on Snowflake server and get the reference to it:

            >>> from snowflake.core.warehouse import Warehouse
            >>> warehouse_parameters = Warehouse(
            ...     name="your-warehouse-name",
            ...     warehouse_size="SMALL",
            ...     auto_suspend=500,
            ...)
            >>> # Use warehouse collection created before to create a reference to warehouse resource
            >>> # in Snowflake server.
            >>> warehouse_reference = warehouse_collection.create(warehouse_parameters)

        """
        real_mode = CreateMode[mode].value
        self._api.create_warehouses(warehouse._to_model(), StrictStr(real_mode), async_req=False)
        return self[warehouse.name]

    @api_telemetry
    def iter(
            self,
            *,
            like: Optional[str] = None,
    ) -> Iterator[Warehouse]:
        """Iterate over the list of warehouses in Snowflake, filtering on any optional `like` pattern.

        Args:
            like:
                A case-insensitive :class:`string` functioning as a filter, with support for SQL
                wildcard characters (% and _).

        Example:
            Create a warehouse on Snowflake server and get the reference to it:

            >>> from snowflake.core.warehouse import Warehouse
            >>> # Show all warehouses that you have access to see.
            >>> warehouses = warehouse_collection.iter()
            >>> # Show information of the exact warehouse you want to see.
            >>> warehouses = warehouse_collection.iter(like="your-warehouse-name")
            >>> # Show warehouses starting with 'your-warehouse-name-'.
            >>> warehouses = warehouse_collection.iter(like="your-warehouse-name-%")
            >>> # Use for loop to retrieve information from iterator.
            >>> for warehouse in warehouses:
            >>>     print(warehouse.name, warehouse.warehouse_size)
        """
        warehouses = self._api.list_warehouses(
            StrictStr(like) if like is not None else None,
            async_req=False,
        )

        return map(Warehouse._from_model, iter(warehouses))


class WarehouseResource(ObjectReferenceMixin[WarehouseCollection]):
    """A reference to a Warehouse in Snowflake."""

    def __init__(self, name: str, collection: WarehouseCollection):
        self.name = name
        self.collection = collection

    @api_telemetry
    def create_or_update(
        self,
        warehouse: Warehouse
    ) -> None:
        """Create a warehouse in Snowflake or update one if it already exists.

        Args:
            warehouse: an instance of :class:`Warehouse`.

        Example:
            Create a warehouse on Snowflake server and get the reference to it:

            >>> from snowflake.core.warehouse import Warehouse
            >>> warehouse_parameters = Warehouse(
            ...     name="your-warehouse-name",
            ...     warehouse_size="SMALL",
            ...     auto_suspend=500,
            ...)
            >>> # Use warehouse collection to create a reference to warehouse resource in Snowflake server.
            >>> root.warehouses["your-warehouse-name"].create_or_update(warehouse_parameters)
        """
        self.collection._api.create_or_alter_warehouse(warehouse.name, warehouse._to_model(), async_req=False)

    @api_telemetry
    def suspend(self) -> None:
        """Suspend the warehouse.

        Example:
            Use warehouse reference to suspend a warehouse:

            >>> from snowflake.core.warehouse import WarehouseResource
            >>> warehouse_reference.suspend()
        """
        self.collection._api.suspend_warehouse(self.name, async_req=False)

    @api_telemetry
    def resume(self) -> None:
        """Resume the warehouse.

        Example:
            Use warehouse reference to resume a warehouse:

            >>> warehouse_reference.resume()
        """
        self.collection._api.resume_warehouse(self.name, async_req=False)

    @api_telemetry
    def delete(self) -> None:
        """Delete this warehouse.

        Example:
            Use warehouse reference to delete a warehouse:

            >>> warehouse_reference.delete()
        """
        self.collection._api.drop_warehouse(self.name, async_req=False)

    @api_telemetry
    def fetch(self) -> Warehouse:
        """Retrieve the warehouse resource.

        Example:
            Use warehouse reference to fetch a warehouse:

            >>> from snowflake.core.warehouse import Warehouse
            >>> warehouse = warehouse_reference.fetch()
            >>> # Access information of the warehouse with warehouse instance.
            >>> print(warehouse.name, warehouse.warehouse_size)
        """
        return Warehouse._from_model(self.collection._api.describe_warehouse(self.name, async_req=False))

    @api_telemetry
    def rename(self, new_name: str) -> None:
        """Rename this warehouse.

        This function will ignore other parameters in warehous instance, use `create_or_update()` to udpate parameters.

        Args:
            new_name: an instance of :class:`Warehouse`.

        Example:
            Use warehouse reference to renane a warehouse:

            >>> from snowflake.core.warehouse import Warehouse
            >>> new_warehouse = Warehouse(
            ...     name="new_warehouse_name"
            ...)
            >>> warehouse = warehouse_reference.rename(new_warehouse)
        """
        self.collection._api.rename_warehouse(self.name, Warehouse(new_name)._to_model(), async_req=False)

    @api_telemetry
    def abort_all_queries(self) -> None:
        """Abort all queries running or queueing on this warehouse.

        Example:
            Use warehouse reference to abort all queries:

            >>> warehouse = warehouse_reference.abort_all_queries()
        """
        self.collection._api.abort_all_queries_on_warehouse(self.name, async_req=False)

__all__ = ["Warehouse", "WarehouseCollection", "WarehouseResource"]
