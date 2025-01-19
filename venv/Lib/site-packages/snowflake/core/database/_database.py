from functools import cached_property
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

from pydantic import StrictStr

from snowflake.core._common import AccountObjectCollectionParent, Clone, CreateMode, ObjectReferenceMixin, PointOfTime
from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core.database._generated.api import DatabaseApi
from snowflake.core.database._generated.api_client import StoredProcApiClient
from snowflake.core.database._generated.models.account_identifiers import AccountIdentifiers
from snowflake.core.database._generated.models.database import DatabaseModel as Database
from snowflake.core.database._generated.models.database_clone import DatabaseClone
from snowflake.core.database._generated.models.point_of_time import PointOfTime as DatabasePointOfTime
from snowflake.core.schema import SchemaCollection


if TYPE_CHECKING:
    from snowflake.core import Root


class DatabaseCollection(AccountObjectCollectionParent["DatabaseResource"]):
    def __init__(self, root: "Root") -> None:
        super().__init__(root, ref_class=DatabaseResource)
        self._api = DatabaseApi(
            root=root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def create(
        self,
        database: Database,
        *,
        clone: Optional[Union[str, Clone]] = None,
        mode: CreateMode = CreateMode.error_if_exists,
        kind: str = "",
    ) -> "DatabaseResource":
        """Create a database in Snowflake.

        Args:
            database: An instance of :class:`Database`, the definition of database we should create.
            clone: Whether to clone an existing database. An instance of :class:`Clone`, or ``None``
                if no cloning is necessary.
            mode: One of the following strings.

                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError`
                if the database already exists in Snowflake. Equivalent to SQL ``create database <name> ...``.

                CreateMode.or_replace: Replace if the database already exists in Snowflake. Equivalent to SQL
                ``create or replace database <name> ...``.

                CreateMode.if_not_exists: Do nothing if the database already exists in Snowflake. Equivalent to SQL
                ``create database <name> if not exists...``

                Default value is CreateMode.error_if_exists.
            kind: The kind of database to create. At the time of writing we support regular (represented by the
                empty string) and transient databases (represented by ``transient``).
        """
        real_mode = CreateMode[mode].value
        if clone is not None:
            pot: Optional[DatabasePointOfTime] = None
            if isinstance(clone, Clone) and isinstance(clone.point_of_time, PointOfTime):
                pot = DatabasePointOfTime.from_dict(clone.point_of_time.to_dict())
            real_clone = Clone(source=clone) if isinstance(clone, str) else clone
            req = DatabaseClone(
                point_of_time=pot,
                **database._to_model().to_dict(),
            )
            self._api.clone_database(
                name=real_clone.source,
                database_clone=req,
                create_mode=StrictStr(real_mode),
                kind=kind,
                async_req=False,
            )
        else:
            self._api.create_database(
                database=database._to_model(),
                create_mode=StrictStr(real_mode),
                kind=kind,
                async_req=False,
            )
        return self[database.name]

    @api_telemetry
    def _create_from_share(
        self,
        name: str,
        share: str,
        *,
        kind: str = "",
        mode: CreateMode = CreateMode.error_if_exists
    ) -> "DatabaseResource":
        """Create a Database from a share.

        Share is of the form '<provider_account>.<share_name>'.
        """
        real_mode = CreateMode[mode].value
        self._api.create_database_from_share(
            name=name,
            share=share,
            kind=kind,
            create_mode=StrictStr(real_mode),
            async_req=False,
        )
        return self[name]

    @api_telemetry
    def iter(
        self,
        *,
        like: Optional[str] = None,
        starts_with: Optional[str] = None,
        limit: Optional[int] = None,
        from_name: Optional[str] = None
    ) -> Iterator[Database]:
        """Look up databases in Snowflake."""
        databases = self._api.list_databases(
            StrictStr(like) if like is not None else None,
            StrictStr(starts_with) if starts_with else None,
            limit,
            from_name=from_name,
            async_req=False,)

        return map(Database._from_model, iter(databases))


class DatabaseResource(ObjectReferenceMixin[DatabaseCollection]):

    def __init__(self, name: str, collection: DatabaseCollection) -> None:
        self.name = name
        self.collection: DatabaseCollection = collection

    @property
    def _api(self) -> DatabaseApi:
        return self.collection._api

    @api_telemetry
    def fetch(self) -> Database:
        return Database._from_model(self.collection._api.fetch_database(
            self.name,
            async_req=False,
        ))

    @api_telemetry
    def delete(self) -> None:
        """Delete this Database."""
        self.collection._api.delete_database(
            name=self.name,
            async_req=False,
        )

    @api_telemetry
    def create_or_update(
        self,
        database: Database,
    ) -> None:
        """Create or update a database in Snowflake."""
        self._api.create_or_alter_database(
            database.name, database._to_model(), async_req=False
        )

    @api_telemetry
    def enable_replication(self, accounts: List[str], ignore_edition_check: bool = False) -> None:
        """Promotes a local database to serve as a primary database for replication.

        A primary database can be replicated in one or more accounts, allowing users
        in those accounts to query objects in each secondary (i.e. replica) database.

        Alternatively, modify an existing primary database to add to or remove from
        the list of accounts that can store a replica of the database.

        Provide a list of accounts in your organization that can store a replica of
        this database.
        """
        if len(accounts) == 0:
            raise ValueError(
                "Account list given to replication cannot be empty.",
            )
        self.collection._api.enable_database_replication(
            name=self.name,
            account_identifiers=AccountIdentifiers(accounts=accounts),
            ignore_edition_check=ignore_edition_check,
            async_req=False,
        )

    @api_telemetry
    def disable_replication(self, accounts: Optional[List[str]] = None) -> None:
        """Disables replication for this primary database.

        Which means that no replica of this database (i.e. secondary database) in
        another account can be refreshed. Any secondary databases remain linked to
        the primary database, but requests to refresh a secondary database are denied.

        Note that disabling replication for a primary database does not prevent it
        from being replicated to the same account; therefore, the database continues
        to be listed in the SHOW REPLICATION DATABASES output.

        Optionally provide a comma-separated list of accounts in your organization
        to disable replication for this database only in the specified accounts.
        """
        if accounts is None:
            accounts = []
        self.collection._api.disable_database_replication(
            name=self.name,
            account_identifiers=AccountIdentifiers(accounts=accounts),
            async_req=False,
        )

    @api_telemetry
    def refresh_replication(self) -> None:
        """Refresh a secondary database from its primary database.

        A snapshot includes changes to the objects and data.
        """
        self.collection._api.refresh_database_replication(
            name=self.name,
            async_req=False,
        )

    @api_telemetry
    def enable_failover(self, accounts: List[str]) -> None:
        """Enable a list of replicas of this database that can be promoted to primary."""
        if len(accounts) == 0:
            raise ValueError(
                "Account list given to replication cannot be empty.",
            )
        self.collection._api.enable_database_failover(
            name=self.name,
            account_identifiers=AccountIdentifiers(accounts=accounts),
            async_req=False,
        )

    @api_telemetry
    def disable_failover(self, accounts: Optional[List[str]] = None) -> None:
        """Disables failover for this primary databases.

        Which means that no replica of this database (i.e. secondary database) can be
        promoted to serve as the primary database.

        Optionally provide a comma-separated list of accounts in your organization to
        disable failover for this database only in the specified accounts.
        """
        if accounts is None:
            accounts = []
        self.collection._api.disable_database_failover(
            name=self.name,
            account_identifiers=AccountIdentifiers(accounts=accounts),
            async_req=False,
        )

    @api_telemetry
    def promote_to_primary_failover(self) -> None:
        """Promotes the specified secondary (replica) database to serve as the primary.

        When promoted, the database becomes writeable. At the same time, the previous
        primary database becomes a read-only secondary database.
        """
        self.collection._api.primary_database_failover(
            name=self.name,
            async_req=False,
        )


    @cached_property
    def schemas(self) -> SchemaCollection:
        return SchemaCollection(self, self.root)
