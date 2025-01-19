from typing import TYPE_CHECKING, Any, Dict, List, Optional

from snowflake.core._common import (
    SchemaObjectCollectionParent,
    SchemaObjectReferenceMixin,
)
from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core.cortex.search_service._generated.api import CortexSearchServiceApi
from snowflake.core.cortex.search_service._generated.api_client import (
    StoredProcApiClient,
)
from snowflake.core.cortex.search_service._generated.models import (
    QueryRequest,
    QueryResponse,
)


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class CortexSearchServiceCollection(
    SchemaObjectCollectionParent["CortexSearchServiceResource"]
):
    """Represents the collection operations of the Snowflake Cortex Search Service resource."""

    def __init__(self, schema: "SchemaResource") -> None:
        super().__init__(schema, CortexSearchServiceResource)
        self._api = CortexSearchServiceApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root),
        )

    @api_telemetry
    def search(self, name: str, query: QueryRequest) -> QueryResponse:
        print(name, self.database.name, self.schema.name)
        return self._api.query_cortex_search_service(
            self.database.name, self.schema.name, name, query_request=query
        )


class CortexSearchServiceResource(
    SchemaObjectReferenceMixin[CortexSearchServiceCollection]
):


    def __init__(self, name: str, collection: CortexSearchServiceCollection) -> None:
        self.name = name
        self.collection = collection

    @api_telemetry
    def search(
        self,
        query: str,
        columns: List[str],
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> QueryResponse:
        return self.collection._api.query_cortex_search_service(
            self.database.name,
            self.schema.name,
            self.name,
            QueryRequest.from_dict(
                {"query": query, "columns": columns, "filter": filter, "limit": limit}
            ),
        )
