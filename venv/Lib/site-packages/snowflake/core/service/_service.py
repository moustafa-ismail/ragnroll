import json
import re

from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import yaml

from pydantic import StrictInt, StrictStr

from .._common import (
    CreateMode,
    SchemaObjectCollectionParent,
    SchemaObjectReferenceMixin,
)
from .._internal.telemetry import api_telemetry


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource

from snowflake.core.service._generated import ServiceApi, ServiceEndpoint
from snowflake.core.service._generated.api_client import StoredProcApiClient
from snowflake.core.service._generated.models import Service, ServiceSpecInlineText, ServiceSpecStageFile


class ServiceCollection(SchemaObjectCollectionParent["ServiceResource"]):
    """Represents the collection operations of the Snowpark Container Service resource."""

    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, ServiceResource)
        self._api = ServiceApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def iter(
        self,
        *,
        like: Optional[str] = None,
        starts_with: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Service]:
        """Look up Snowpark Container services in Snowflake."""
        services = self._api.list_services(
            self.database.name,
            self.schema.name,
            StrictStr(like) if like is not None else None,
            StrictStr(starts_with) if starts_with else None,
            limit,
            async_req=False,)

        return iter(services)

    @api_telemetry
    def create(
        self,
        service: Service,
        *,
        mode: CreateMode = CreateMode.error_if_exists,
    ) -> "ServiceResource":
        """Create a Snowpark Container service in Snowflake.

        Args:
            service: an instance of :class:`Service`.
            mode: One of the following strings.

                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError`
                if the service already exists in Snowflake. Equivalent to SQL ``create service <name> ...``.

                CreateMode.if_not_exists: Do nothing if the service already exists in Snowflake. Equivalent to SQL
                ``create service <name> if not exists...``

                Default value is CreateMode.error_if_exists.

        """
        if mode == CreateMode.or_replace:
            raise ValueError(f"{mode} is not a valid value for this resource")
        real_mode = CreateMode[mode].value
        self._api.create_service(
            self.database.name,
            self.schema.name,
            service,
            StrictStr(real_mode),
            async_req=False,
        )
        return self[service.name]


class ServiceResource(SchemaObjectReferenceMixin[ServiceCollection]):



    def __init__(self, name: str, collection: ServiceCollection) -> None:
        self.name = name
        self.collection = collection

    @api_telemetry
    def delete(self) -> None:
        """Delete the service."""
        self.collection._api.delete_service(
            self.database.name, self.schema.name, self.name, False, async_req=False
        )

    @api_telemetry
    def fetch(self) -> Service:
        """Fetch a snapshot of the service."""
        return self.collection._api.fetch_service(
            self.database.name, self.schema.name, self.name, async_req=False
        )

    @api_telemetry
    def suspend(self) -> None:
        """Suspend the service."""
        self.collection._api.suspend_service(
            self.database.name, self.schema.name, self.name, async_req=False
        )

    @api_telemetry
    def resume(self) -> None:
        """Resumes the service."""
        self.collection._api.resume_service(
            self.database.name, self.schema.name, self.name, async_req=False
        )

    @api_telemetry
    def get_endpoints(self) -> Iterable[ServiceEndpoint]:
        """Show the endpoints corresponding to this service."""
        return self.collection._api.show_service_endpoints(
            self.database.name, self.schema.name, self.name, async_req=False
        )

    @api_telemetry
    def get_service_status(self, timeout: int = 0) -> List[Dict[str, Any]]:
        """Get the status of the service.

        Args:
            timeout: Number of seconds to wait for the service to reach a steady state (for example, READY)
              before returning the status. If the service does not reach steady state within the specified time,
              Snowflake returns the current state.

              If not specified or 0, Snowflake returns the current state immediately.

              Default: 0 seconds.
        """
        status = self.collection._api.fetch_service_status(
            self.database.name,
            self.schema.name,
            self.name,
            StrictInt(timeout),
            async_req=False,
        )
        if status.systemget_service_status is None:
            return list()
        return json.loads(status.systemget_service_status)

    @api_telemetry
    def get_service_logs(self, instance_id: str, container_name: str, num_lines: Optional[int] = None) -> str:
        """Get the service logs of the service.

        Args:
            instance_id: Service instance ID.
            container_name: Container name.
            num_lines: (Optional) Number of the most recent log lines to retrieve.

        :meth:`get_service_status` returns the ``instance_id`` and ``container_name`` as a part of its results.

        """
        logs = self.collection._api.fetch_service_logs(
            self.database.name,
            self.schema.name,
            self.name,
            StrictInt(instance_id),
            StrictStr(container_name),
            num_lines,
            async_req=False,
        )
        if logs.systemget_service_logs is None:
            return ""
        return logs.systemget_service_logs


def ServiceSpec(spec: str) -> Union[ServiceSpecInlineText, ServiceSpecStageFile]:
    """
    Infers whether a specification is a stage file or inline text.

    Any spec that starts with '@' is parsed as a stage file, otherwise it is passed as an inline text.
    """
    spec = dedent(spec).rstrip()
    if spec.startswith("@"):
        stage, spec_file = _parse_spec_path(spec[1:])
        return ServiceSpecStageFile(stage=stage, spec_file=spec_file)
    else:
        if _validate_inline_spec(spec):
            return ServiceSpecInlineText(spec_text=spec)
        else:
            raise ValueError(f"{spec} is not a valid service spec inline text")

def _validate_inline_spec(spec_str: str) -> bool:
    try:
        spec_data = yaml.safe_load(spec_str)
    except yaml.YAMLError:
        return False
    if not isinstance(spec_data, dict) or 'spec' not in spec_data:
        return False
    return True

def _parse_spec_path(spec_path: str) -> Tuple[str, str]:
    # this pattern tries to match a file path depth of at least two (needs a stage and file name at minimum)
    pattern = r'^[^<>:"|?*\/\n]+(?:\/[^<>:"|?*\/\n]+)+$'
    if not re.fullmatch(pattern, spec_path):
        raise ValueError(f"{spec_path} is not a valid stage file path")
    stage, path = spec_path.split('/', 1)
    return stage, path
