import functools
import platform

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, TypeVar, Union

from typing_extensions import Concatenate, ParamSpec

from snowflake.connector import SnowflakeConnection
from snowflake.connector.telemetry import (
    TelemetryClient,
    TelemetryData,
)
from snowflake.connector.telemetry import (
    TelemetryField as ConnectorTelemetryField,
)
from snowflake.connector.time_util import get_time_millis
from snowflake.snowpark._internal.telemetry import TelemetryField, safe_telemetry
from snowflake.snowpark._internal.utils import is_in_stored_procedure

from .._common import ObjectCollection, ObjectReferenceMixin
from ..version import __version__ as VERSION


if TYPE_CHECKING:
    from ..task.dagv1 import DAGOperation


# Constant to decide whether we are running tests
_called_from_test = False


class ApiTelemetryClient:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self.telemetry: Optional[TelemetryClient] = (
            None if is_in_stored_procedure() else conn._telemetry
        )
        self.source: str = "snowflake.core"
        self.version: str = VERSION
        self.python_version: str = platform.python_version()
        self.os: str = platform.system()

    def send(self, msg: Dict[str, Any], timestamp: Optional[int] = None) -> None:
        if self.telemetry:
            if not timestamp:
                timestamp = get_time_millis()
            telemetry_data = TelemetryData(message=msg, timestamp=timestamp)
            self.telemetry.try_add_log_to_batch(telemetry_data)

    def _create_basic_telemetry_data(self) -> Dict[str, Any]:
        message = {
            ConnectorTelemetryField.KEY_SOURCE.value: self.source,
            TelemetryField.KEY_VERSION.value: self.version,
            TelemetryField.KEY_PYTHON_VERSION.value: self.python_version,
            TelemetryField.KEY_OS.value: self.os,
            ConnectorTelemetryField.KEY_TYPE.value: "python_api",
        }
        return message

    @safe_telemetry
    def send_api_telemetry(
        self,
        class_name: str,
        func_name: str,
        client_name: Optional[str] = None,
    ) -> None:
        data = {
            "class_name": class_name,
            TelemetryField.KEY_FUNC_NAME.value: func_name,
        }
        if client_name is not None:
            data["client_name"] = client_name
        message = self._create_basic_telemetry_data()
        message[TelemetryField.KEY_DATA.value] = data
        self.send(message)

P = ParamSpec("P")
R = TypeVar("R")


def api_telemetry(func: Callable[Concatenate[Any, P], R]) -> Callable[Concatenate[Any, P], R]:
    @functools.wraps(func)
    def wrap(
        self: Union[ObjectReferenceMixin[Any], ObjectCollection[Any], "DAGOperation"],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        from ..task.dagv1 import DAGOperation
        if isinstance(self, (ObjectReferenceMixin, ObjectCollection)):
            telemetry_client = self.root._telemetry_client  # type: ignore[misc]
        elif isinstance(self, DAGOperation):
            telemetry_client = self.schema.root._telemetry_client
        else:
            raise TypeError(f"unknown type {type(self)}")
        api = None
        if hasattr(self, "_api"):
            api = self._api
        elif hasattr(self, "collection") and hasattr(self.collection, "_api"):
            api = self.collection._api
        elif _called_from_test and not isinstance(self, DAGOperation):
            # DAGOperation will not be reported when the API object cannot be extracted
            #  from them. This is okay because this class will call other APIs
            #  downstream.
            raise Exception(f"cannot determine API for {self=}")
        if api is not None:
            # Cause resolution of api client, if not done beforehand
            api.api_client  # noqa: B018
        telemetry_client.send_api_telemetry(
            class_name=self.__class__.__name__,
            func_name=func.__name__,
        )
        r = func(self, *args, **kwargs)
        return r

    return wrap
