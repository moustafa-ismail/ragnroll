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
from ._generated.api import TableApi
from ._generated.api_client import StoredProcApiClient
from ._generated.models.point_of_time import PointOfTime as TablePointOfTime
from ._generated.models.table import Table
from ._generated.models.table_clone import TableClone


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class TableCollection(SchemaObjectCollectionParent["TableResource"]):
    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, TableResource)
        self._api = TableApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def create(
        self, table: Union[Table, str],
        *,
        as_select: Optional[str] = None,
        template: Optional[str] = None,
        like_table: Optional[str] = None,
        clone_table: Optional[Union[str, Clone]] = None,
        copy_grants: Optional[bool] = False,
        mode: CreateMode=CreateMode.error_if_exists,
    ) -> "TableResource":
        """Create a table.

        Args:
            table: The table object, together with the table's properties, object parameters, columns, and constraints.
                It can either be a table name or a ``Table`` object when it's used together with `as_select`,
                `template`, `like_table`, `clone_table`. It must be a ``Table`` when it's not used with these clauses.
            as_select: The `as select` clause.
            template: The `using template` clause.
            like_table: The `like` clause.
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

        Not currently implemented:
            - Row access policy
            - Column masking policy
            - Search optimization
            - Tags
            - Stage file format and copy options
        """
        self.validate_table_inputs(table, as_select, template, like_table, clone_table)

        if isinstance(table, str):
            table = Table(name=table)

        real_mode = CreateMode[mode].value

        if as_select:
            # create table by select
            self._api.create_table_as_select(
                self.database.name, self.schema.name, table.name,
                as_select, table, create_mode=StrictStr(real_mode),
                copy_grants=copy_grants,
                async_req=False
            )
        elif template:
            # create table by template
            self._api.create_table_using_template(
                self.database.name, self.schema.name, table.name,
                template, create_mode=StrictStr(real_mode),
                copy_grants=copy_grants,
                async_req=False
            )
        elif clone_table:
            # create table by clone
            pot: Optional[TablePointOfTime] = None
            if isinstance(clone_table, Clone) and isinstance(clone_table.point_of_time, PointOfTime):
                pot = TablePointOfTime.from_dict(clone_table.point_of_time.to_dict())
            real_clone = Clone(source=clone_table) if isinstance(clone_table, str) else clone_table
            req = TableClone(
                point_of_time = pot,
                **table.to_dict(),
            )
            self._api.clone_table(
                self.database.name, self.schema.name, real_clone.source,
                req, create_mode=StrictStr(real_mode),
                copy_grants=copy_grants,
                async_req=False
            )
        elif like_table:
            # create table by like
            self._api.create_table_like(
                self.database.name, self.schema.name, like_table,
                table.name, create_mode=StrictStr(real_mode),
                copy_grants=copy_grants,
                async_req=False
            )
        else:
            # create empty table
            self._api.create_table(
                self.database.name, self.schema.name, table, create_mode=StrictStr(real_mode),
                copy_grants=copy_grants,
                async_req=False
            )
        return TableResource(table.name, self)

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
    ) -> Iterator[Table]:
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
        tables = self._api.list_tables( database=self.database.name, var_schema=self.schema.name, like=like,
                starts_with=starts_with, show_limit=limit, from_name=from_name, history=history, deep=deep,
                async_req=False)

        return iter(tables)

    def validate_table_inputs(
        self, table: Union[Table, str],
        as_select: Optional[str] = None,
        template: Optional[str] = None,
        like_table: Optional[str] = None,
        clone_table: Optional[Union[str, Clone]] = None
    ) -> None:
        not_none_count =\
            sum(bool(x) for x in (as_select, template, like_table, clone_table))

        if not_none_count > 1:
            raise ValueError(
                "at most one of the `as_select`, `template`, `clone_table`, "
                "or `like_table` can has value"
            )

        if not_none_count == 0 and isinstance(table, str):
            raise ValueError(
                "When `table` is a str, any one of the `as_select`, `template`, `clone_table`, "
                "or `like_table` must not be empty."
            )

class TableResource(SchemaObjectReferenceMixin[TableCollection]):
    """Represents a reference to a Snowflake Table resource."""

    def __init__(self, name: str, collection: TableCollection) -> None:
        self.collection = collection
        self.name = name

    @api_telemetry
    def create_or_update(
        self, table: Table,
    ) -> None:
        """Create or update a table.

        Args:
            table: The ``Table`` object, including the table's properties, object parameters, columns, and constraints.

        Notes:
            - Not currently implemented:
                - Row access policy
                - Column masking policy
                - Search optimization
                - Tags
                - Stage file format and copy options
                - Foreign keys.
                - Rename the table.
                - If the name and table's name don't match, an error will be thrown.
                - Rename or drop a column.
            - New columns can only be added to the back of the column list.
        """
        self.collection._api.create_or_alter_table(self.database.name, self.schema.name, self.name, table)

    def fetch(self) -> Table:
        """Fetch the details of a table.

        Notes:
            Inline constraints will become Outofline constraints because Snowflake database doesn't tell whether a
            constraint is inline or out of line from Snowflake database.
        """
        return self.collection._api.fetch_table(
            self.database.name, self.schema.name, self.name, async_req=False,
        )

    @api_telemetry
    def delete(self) -> None:
        """Delete the table."""
        self.collection._api.delete_table(self.database.name, self.schema.name, self.name, async_req=False)

    @api_telemetry
    def undelete(self) -> None:
        """Undelete the previously deleted table."""
        # TODO: undelete isn't supported on the rest API
        self.collection._api.undelete_table(self.database.name, self.schema.name, self.name, async_req=False)  # type: ignore[attr-defined]

    @api_telemetry
    def swap_with(self, to_swap_table_name: str) -> None:
        """Swap the name with another table."""
        self.collection._api.swap_with(
            self.database.name, self.schema.name, self.name, to_swap_table_name, async_req=False)

    @api_telemetry
    def suspend_recluster(self) -> None:
        self.collection._api.suspend_recluster(self.database.name, self.schema.name, self.name, async_req=False)

    @api_telemetry
    def resume_recluster(self) -> None:
        self.collection._api.resume_recluster(self.database.name, self.schema.name, self.name, async_req=False)
