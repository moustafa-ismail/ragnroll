from typing import TYPE_CHECKING, Iterator, Optional, Union

from pydantic import StrictStr

from snowflake.core._common import (
    Clone,
    CreateMode,
    PointOfTime,
    SchemaObjectCollectionParent,
    SchemaObjectReferenceMixin,
)

from .._internal.telemetry import api_telemetry
from ._generated.api import DynamicTableApi
from ._generated.api_client import StoredProcApiClient
from ._generated.models.dynamic_table import DynamicTable
from ._generated.models.dynamic_table_clone import DynamicTableClone
from ._generated.models.point_of_time import PointOfTime as TablePointOfTime


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class DynamicTableCollection(SchemaObjectCollectionParent["DynamicTableResource"]):
    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, DynamicTableResource)
        self._api = DynamicTableApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def create(
        self, table: Union[DynamicTable, DynamicTableClone, str],
        *,
        clone_table: Optional[Union[str, Clone]] = None,
        copy_grants: Optional[bool] = False,
        mode: CreateMode=CreateMode.error_if_exists,
    ) -> "DynamicTableResource":
        """Create a dynamic table.

        Args:
            table: The dynamic table object, together with the dynamic table's properties.
                It can either be a table name or a ``DynamicTableClone`` object when it's used with `clone_table`.
                It must be a ``DynamicTable`` when it's not used with this clause.
            clone_table: The `clone` clause.
            copy_grants: copy grants when `clone_table` is provided.
            mode: One of the following strings.

                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError`
                if the table already exists in Snowflake.  Equivalent to SQL ``create table <name> ...``.

                CreateMode.or_replace: Replace if the task already exists in Snowflake. Equivalent to SQL
                ``create or replace table <name> ...``.

                CreateMode.if_not_exists: Do nothing if the task already exists in Snowflake.
                Equivalent to SQL ``create table <name> if not exists...``

                Default value is CreateMode.error_if_exists.
        """
        real_mode = CreateMode[mode].value

        if clone_table:
            # create table by clone

            if isinstance(table, str):
                table = DynamicTableClone(name=table)

            pot: Optional[TablePointOfTime] = None
            if isinstance(clone_table, Clone) and isinstance(clone_table.point_of_time, PointOfTime):
                pot = TablePointOfTime.from_dict(clone_table.point_of_time.to_dict())
            real_clone = Clone(source=clone_table) if isinstance(clone_table, str) else clone_table
            req = DynamicTableClone(
                name=table.name,
                target_lag=table.target_lag,
                warehouse=table.warehouse,
                point_of_time=pot,
            )
            self._api.clone_dynamic_table(
                self.database.name, self.schema.name, real_clone.source,
                req, create_mode=StrictStr(real_mode),
                copy_grants=copy_grants,
                async_req=False
            )
        else:
            # create empty table

            if not isinstance(table, DynamicTable):
                raise ValueError("`table` must be a `DynamicTable` unless `clone_table` is used")

            self._api.create_dynamic_table(
                self.database.name, self.schema.name, table, create_mode=StrictStr(real_mode),
                async_req=False
            )
        return DynamicTableResource(table.name, self)

    @api_telemetry
    def iter(
        self,
        *,
        like: Optional[str] = None,
        starts_with: Optional[str] = None,
        limit: Optional[int] = None,
        from_name: Optional[str] = None,
        history: bool = False,
        deep: bool = False,
    ) -> Iterator[DynamicTable]:
        """Search ``Table`` objects from Snowflake.

        Args:
            like: The pattern of the Table name. Use ``%`` to represent any number of characters and ``?`` for a
                single character.
            startswith: The table name starts with this string.
            limit: limits the number of objects returned.
            from_name: enables fetching the specified number of rows following the first row whose object name matches
                the specified string.
            deep: fetch the sub-resources columns and constraints of every table if it's ``True``. Default ``False``.
            history: includes dropped tables that have not yet been purged.
        """
        tables = self._api.list_dynamic_tables(
            database=self.database.name, var_schema=self.schema.name, like=like,
            starts_with=starts_with, show_limit=limit, from_name=from_name, history=history, deep=deep,
            async_req=False
        )

        return iter(tables)

class DynamicTableResource(SchemaObjectReferenceMixin[DynamicTableCollection]):
    """Represents a reference to a Snowflake Dynamic Table resource."""

    _supports_rest_api = True

    def __init__(self, name: str, collection: DynamicTableCollection) -> None:
        self.collection = collection
        self.name = name

    def fetch(self) -> DynamicTable:
        """Fetch the details of a dynamic table."""
        return self.collection._api.fetch_dynamic_table(
            self.database.name, self.schema.name, self.name, async_req=False,
        )

    @api_telemetry
    def delete(self) -> None:
        """Delete the dynamic table."""
        self.collection._api.delete_dynamic_table(self.database.name, self.schema.name, self.name, async_req=False)

    @api_telemetry
    def suspend(self) -> None:
        self.collection._api.suspend_dynamic_table(self.database.name, self.schema.name, self.name, async_req=False)

    @api_telemetry
    def resume(self) -> None:
        self.collection._api.resume_dynamic_table(self.database.name, self.schema.name, self.name, async_req=False)

    @api_telemetry
    def refresh(self) -> None:
        self.collection._api.refresh_dynamic_table(self.database.name, self.schema.name, self.name, async_req=False)
