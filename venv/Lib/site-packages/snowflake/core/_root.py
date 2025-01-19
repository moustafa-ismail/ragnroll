
from typing import Optional, Union

from snowflake.connector import SnowflakeConnection
from snowflake.snowpark.session import Session, _active_sessions

from ._common import check_env_parameter_enabled
from ._internal.snowapi_parameters import SnowApiParameters
from ._internal.telemetry import ApiTelemetryClient
from .compute_pool import ComputePoolCollection
from .database import DatabaseCollection
from .grant._grants import Grants
from .role import RoleCollection
from .session import SnowAPISession
from .user import UserCollection
from .warehouse import WarehouseCollection


class Root:
    """The entry point of the Snowflake Core Python APIs that manage the Snowflake objects.

    Args:
        connection: A ``SnowflakeConnection`` or Snowpark ``Session`` instance.

    Example:
        Create a Root instance:

        >>> from snowflake.connector import connect
        >>> from snowflake.core import Root
        >>> from snowflake.snowpark import Session
        >>> CONNECTION_PARAMETERS = {
        ...    "account": os.environ["snowflake_account_demo"],
        ...    "user": os.environ["snowflake_user_demo"],
        ...    "password": os.environ["snowflake_password_demo"],
        ...    "database": test_database,
        ...    "warehouse": test_warehouse,
        ...    "schema": test_schema,
        ... }
        >>> # create from a Snowflake Connection
        >>> connection = connect(**CONNECTION_PARAMETERS)
        >>> root = Root(connection)
        >>> # or create from a Snowpark Session
        >>> session = Session.builder.config(CONNECTION_PARAMETERS).create()
        >>> root = Root(session)

        Use the root instance to access resource management APIs:

        >>> tasks = root.databases["mydb"].schemas["myschema"].tasks
        >>> mytask = tasks["mytask"]
        >>> mytask.resume()
        >>> compute_pools = root.compute_pools
        >>> my_computepool = compute_pools["mycomputepool"]
        >>> my_computepool.delete()

    """

    def __init__(
        self,
        connection: Union[SnowflakeConnection, Session],
    ) -> None:
        if isinstance(connection, SnowflakeConnection):
            self._connection = connection
            self._session: Session = Session.builder.configs(
                {"connection": connection}
            ).create()
            _active_sessions.remove(
                self._session
            )  # This is tentatively to avoid a user has two active sessions.
        else:
            self._session = connection
            self._connection = connection._conn._conn

        self._snowapi_session = SnowAPISession(self)
        self._refresh_parameters()

        self._databases = DatabaseCollection(self)
        self._compute_pools = ComputePoolCollection(self)
        self._telemetry_client = ApiTelemetryClient(self._connection)
        self._warehouses = WarehouseCollection(self)
        self._roles = RoleCollection(self)
        self._grants = Grants(self)
        self._users = UserCollection(self)
        self._enable_long_running_polling = check_env_parameter_enabled(
            "_SNOWFLAKE_ENABLE_LONG_RUNNING_POLLING"
        )

    def effective_parameters(self, refresh: bool = True) -> SnowApiParameters:
        if refresh:
            self._refresh_parameters()

        return self._effective_parameters

    @property
    def connection(self) -> SnowflakeConnection:
        """Return the connection in use.

        This is the connection used to create this Root instance, or the
        Snowpark session's connection if this root is created from a
        session.
        """
        return self._connection

    @property
    def session(self) -> Session:
        """Returns the session that is used to create this Root instance.

        Returns ``None`` if the root wasn't created from a session but a ``SnowflakeConnection``.
        """
        return self._session

    @property
    def databases(self) -> DatabaseCollection:
        return self._databases

    @property
    def compute_pools(self) -> ComputePoolCollection:
        return self._compute_pools

    @property
    def warehouses(self) -> WarehouseCollection:
        return self._warehouses

    @property
    def roles(self) -> RoleCollection:
        return self._roles

    @property
    def grants(self) -> Grants:
        return self._grants

    @property
    def users(self) -> UserCollection:
        return self._users

    @property
    def _session_token(self) -> Optional[str]:
        # TODO: this needs to be fixed in the connector
        return self._connection.rest.token  # type: ignore[union-attr]

    @property
    def _master_token(self) -> Optional[str]:
        # TODO: this needs to be fixed in the connector
        return self._connection.rest.master_token  # type: ignore[union-attr]

    def _refresh_parameters(self) -> None:
        # In principle, self._effective_parameters should/can contain a lot of relevant parameters,
        # but for now, we don't want so many and only care about the enablement ones.
        self._effective_parameters = SnowApiParameters(
            params_map=self._snowapi_session._get_api_enablement_parameters())
