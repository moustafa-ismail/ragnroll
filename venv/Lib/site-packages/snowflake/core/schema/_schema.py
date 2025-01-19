from functools import cached_property
from typing import TYPE_CHECKING, Iterator, Optional, Union

from pydantic import StrictStr

from snowflake.connector import SnowflakeConnection
from snowflake.core.schema._generated.api_client import StoredProcApiClient
from snowflake.snowpark import Session

from .._common import Clone, CreateMode, ObjectCollection, ObjectReferenceMixin, PointOfTime
from .._internal.telemetry import api_telemetry
from ..cortex.search_service import CortexSearchServiceCollection
from ..dynamic_table import DynamicTableCollection
from ..function import FunctionCollection
from ..image_repository import ImageRepositoryCollection
from ..service import ServiceCollection
from ..stage import StageCollection
from ..table import TableCollection
from ..task import TaskCollection
from ._generated.api.schema_api import SchemaApi
from ._generated.models.model_schema import ModelSchemaModel as Schema
from ._generated.models.point_of_time import PointOfTime as SchemaPointOfTime
from ._generated.models.schema_clone import SchemaClone


if TYPE_CHECKING:
    from .. import Root
    from ..database import DatabaseResource


class SchemaCollection(ObjectCollection["SchemaResource"]):
    def __init__(self, database: "DatabaseResource", root: "Root") -> None:
        super().__init__(SchemaResource)
        self._database = database
        self._api = SchemaApi(
            root=root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @property
    def database(self) -> "DatabaseResource":
        return self._database

    @property
    def root(self) -> "Root":
        return self.database.collection.root

    @api_telemetry
    def create(
        self,
        schema: Schema,
        *,
        clone: Optional[Union[str, Clone]] = None,
        mode: CreateMode = CreateMode.error_if_exists,
        kind: str = "",
    ) -> "SchemaResource":
        """Create a schema in Snowflake.

        Args:
            schema: an instance of :class:`Schema`.
            mode: One of the following strings.
                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError`
                if the schem already exists in Snowflake. Equivalent to SQL ``create schema <name> ...``.

                CreateMode.or_replace: Replace if the schema already exists in Snowflake. Equivalent to SQL
                ``create or replace schema <name> ...``.

                CreateMode.if_not_exists: Do nothing if the schema already exists in Snowflake. Equivalent to SQL
                ``create schema <name> if not exists...``

                Default value is CreateMode.error_if_exists.
        """
        real_mode = CreateMode[mode].value
        if clone is not None:
            pot: Optional[SchemaPointOfTime] = None
            if isinstance(clone, Clone) and isinstance(clone.point_of_time, PointOfTime):
                pot = SchemaPointOfTime.from_dict(clone.point_of_time.to_dict())
            real_clone = Clone(source=clone) if isinstance(clone, str) else clone
            req = SchemaClone(
                point_of_time=pot,
                **schema._to_model().to_dict(),
            )
            self._api.clone_schema(
                database=self.database.name,
                name=real_clone.source,
                schema_clone=req,
                create_mode=StrictStr(real_mode),
                kind=kind,
                async_req=False,
            )
        else:
            self._api.create_schema(
                database=self.database.name,
                model_schema=schema._to_model(),
                create_mode=StrictStr(real_mode),
                kind=kind,
                async_req=False,
            )
        return self[schema.name]

    @api_telemetry
    def iter(
        self,
        *,
        like: Optional[str] = None,
        starts_with: Optional[str] = None,
        limit: Optional[int] = None,
        from_name: Optional[str] = None
    ) -> Iterator[Schema]:
        """Look up schemas in Snowflake."""
        schemas = self._api.list_schemas(
            self.database.name,
            StrictStr(like) if like is not None else None,
            StrictStr(starts_with) if starts_with else None,
            limit,
            from_name=from_name,
            async_req=False,)

        return map(Schema._from_model, iter(schemas))

    @property
    def _session(self) -> Session:
        return self.database._session

    @property
    def _connection(self) -> SnowflakeConnection:
        return self.database._connection


class SchemaResource(ObjectReferenceMixin[SchemaCollection]):



    def __init__(self, name: str, collection: SchemaCollection) -> None:
        self.name: str = name
        self.collection: SchemaCollection = collection

    @property
    def database(self) -> "DatabaseResource":
        return self.collection.database

    @property
    def _api(self) -> SchemaApi:
        return self.collection._api

    @api_telemetry
    def create_or_update(self, schema: Schema) -> "SchemaResource":
        """Create or update a schema in Snowflake."""
        self._api.create_or_alter_schema(
            self.database.name,
            schema.name,
            schema._to_model(),
            async_req=False,
        )
        return self

    @api_telemetry
    def fetch(self) -> Schema:
        return Schema._from_model(self.collection._api.fetch_schema(
            self.database.name,
            self.name,
            async_req=False,
        ))

    @api_telemetry
    def delete(self) -> None:
        """Delete this Schema."""
        self.collection._api.delete_schema(
            self.database.name,
            name=self.name,
            async_req=False,
        )

    @cached_property
    def tasks(self) -> TaskCollection:
        return TaskCollection(self)

    @cached_property
    def services(self) -> ServiceCollection:
        return ServiceCollection(self)

    @cached_property
    def image_repositories(self) -> ImageRepositoryCollection:
        return ImageRepositoryCollection(self)

    @cached_property
    def tables(self) -> TableCollection:
        return TableCollection(self)

    @cached_property
    def dynamic_tables(self) -> DynamicTableCollection:
        return DynamicTableCollection(self)

    @cached_property
    def stages(self) -> StageCollection:
        return StageCollection(self)

    @cached_property
    def cortex_search_services(self) -> CortexSearchServiceCollection:
        return CortexSearchServiceCollection(self)

    @cached_property
    def functions(self) -> FunctionCollection:
        return FunctionCollection(self)

