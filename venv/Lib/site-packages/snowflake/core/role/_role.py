from typing import TYPE_CHECKING, Iterator, Optional

from pydantic import StrictStr

from snowflake.core._common import AccountObjectCollectionParent, CreateMode, ObjectReferenceMixin
from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core.role._generated.api import RoleApi
from snowflake.core.role._generated.api_client import StoredProcApiClient
from snowflake.core.role._generated.models.role import RoleModel as Role


if TYPE_CHECKING:
    from snowflake.core import Root


class RoleCollection(AccountObjectCollectionParent["RoleResource"]):
    """Represents the collection operations on the Snowflake Role resource.

    With this collection, you can create role or update or fetch all roles that you have access to.

    Args:
        root: A :class:`Root` instance.

    Example:
        Create a RoleCollection instance:
        >>> role_collection = root.roles
        >>> role_collection.create(new_role)
    """

    def __init__(self, root: "Root") -> None:
        super().__init__(root, ref_class=RoleResource)
        self._api = RoleApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def create(self, role: Role, *, mode: CreateMode = CreateMode.error_if_exists) -> "RoleResource":
        """
        Create a role in Snowflake.

        Args:
            role: an instance of :class:`Role`.
            mode: One of the following enum values.

                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError` if the
                    role already exists in Snowflake. Equivalent to SQL ``create role <name> ...``.

                CreateMode.or_replace: Replace if the role already exists in Snowflake. Equivalent to SQL
                    ``create or replace role <name> ...``.

                CreateMode.if_not_exists: Do nothing if the role already exists in Snowflake. Equivalent to SQL
                    ``create role <name> if not exists...``

                Default value is CreateMode.error_if_exists.

        Example:
              Create a role on Snowflake and get the reference to it:
              >>> from snowflake.core.role import Role
              >>> role = Role(
              ...   name="test-role",
              ...   comment="samplecomment"
              )
              >>> # Use role collection created before to create a reference to role resource in Snowflake server.
              >>> role_ref = root.roles.create(role)
        """
        real_mode = CreateMode[mode].value
        self._api.create_role(role._to_model(), StrictStr(real_mode))
        return self[role.name]

    def iter(self,
             *,
             like: Optional[str] = None,
             limit: Optional[int] = None,
             starts_with: Optional[str] = None,
             from_name: Optional[str] = None
             ) -> Iterator[Role]:
        """List roles available in an account.

           Args:
            like: Filter the output by resource name. Uses case-insensitive pattern matching, with support for SQL
                wildcard characters.

            limit: Limit the maximum number of rows returned.

            starts_with: filter the output based on the string of
                characters that appear at the beginning of the object name. Uses case-sensitive pattern matching.

            from_name: Enable fetching roles only following the first role whose name matches the specified
                string. Case-sensitive and does not have to be the full name.

        Example:
            >>> roles = root.iter(like, starts_with, show_limit, from_name)
        """
        roles = self._api.list_roles(
            StrictStr(like) if like else None,
            StrictStr(starts_with) if starts_with else None,
            limit,
            from_name=from_name,
            async_req=False
        )

        return map(Role._from_model, iter(roles))


class RoleResource(ObjectReferenceMixin[RoleCollection]):


    def __init__(self, name: str, collection: RoleCollection) -> None:
        self.name = name
        self.collection = collection

    @api_telemetry
    def delete(self) -> None:
        """Delete this role.

        Example:
            Use role reference to delete a role.
            >>> role_reference.delete()
        """
        self.collection._api.delete_role(self.name, async_req=False)
