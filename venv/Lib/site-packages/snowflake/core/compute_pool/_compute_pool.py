from typing import TYPE_CHECKING, Iterator, Optional

from pydantic import StrictStr

from snowflake.core._common import AccountObjectCollectionParent, CreateMode, ObjectReferenceMixin
from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core.compute_pool._generated.api import ComputePoolApi
from snowflake.core.compute_pool._generated.api_client import StoredProcApiClient
from snowflake.core.compute_pool._generated.models.compute_pool import ComputePoolModel as ComputePool


if TYPE_CHECKING:
    from snowflake.core import Root


class ComputePoolCollection(AccountObjectCollectionParent["ComputePoolResource"]):
    """Represents the collection operations of the Snowflake Compute Pool resource."""

    def __init__(self, root: "Root") -> None:
        super().__init__(root, ref_class=ComputePoolResource)
        self._api = ComputePoolApi(
            root=root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def create(
        self,
        compute_pool: ComputePool,
        *,
        mode: CreateMode = CreateMode.error_if_exists,
    ) -> "ComputePoolResource":
        """Create a compute pool in Snowflake.

        Args:
            service: an instance of :class:`ComputePool`.
            mode: One of the following strings.

                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError`
                if the compute pool already exists in Snowflake. Equivalent to SQL ``create compute pool <name> ...``.

                CreateMode.or_replace: Replace if the compute pool already exists in Snowflake. Equivalent to SQL
                ``create or replace compute pool <name> ...``.

                CreateMode.if_not_exists: Do nothing if the compute pool already exists in Snowflake. Equivalent to SQL
                ``create compute pool <name> if not exists...``

                Default value is CreateMode.error_if_exists.

        """
        real_mode = CreateMode[mode].value
        self._api.create_compute_pool(
            compute_pool._to_model(), StrictStr(real_mode), async_req=False
        )
        return self[compute_pool.name]

    @api_telemetry
    def iter(
        self,
        *,
        like: Optional[str] = None,
        starts_with: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[ComputePool]:
        """Look up compute pools in Snowflake."""
        compute_pools = self._api.fetch_compute_pools(
            StrictStr(like) if like is not None else None,
            StrictStr(starts_with) if starts_with else None,
            limit,
            async_req=False, )

        return map(ComputePool._from_model, iter(compute_pools))


class ComputePoolResource(ObjectReferenceMixin[ComputePoolCollection]):
    """A reference to a Compute Pool in Snowflake."""

    def __init__(self, name: str, collection: ComputePoolCollection) -> None:
        self.name = name
        self.collection = collection

    @property
    def _api(self) -> ComputePoolApi:
        return self.collection._api

    @api_telemetry
    def fetch(self) -> ComputePool:
        """Fetch the compute pool details from Snowflake."""
        return ComputePool._from_model(self.collection._api.fetch_compute_pool(self.name, async_req=False))

    @api_telemetry
    def suspend(self) -> None:
        """Suspend the compute pool."""
        self.collection._api.suspend_compute_pool(self.name, async_req=False)

    @api_telemetry
    def resume(self) -> None:
        """Resume the compute pool."""
        self.collection._api.resume_compute_pool(self.name, async_req=False)

    @api_telemetry
    def stop_all_services(self) -> None:
        """Stop all services that run on this compute pool."""
        self.collection._api.stop_all_services_in_compute_pool(
            self.name, async_req=False
        )

    @api_telemetry
    def delete(self) -> None:
        """Delete this compute pool."""
        self.collection._api.delete_compute_pool(self.name, async_req=False)
