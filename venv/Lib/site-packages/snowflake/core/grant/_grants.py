from typing import TYPE_CHECKING

from snowflake.core.grant._generated import GrantApi
from snowflake.core.grant._generated.api_client import StoredProcApiClient
from snowflake.core.grant._generated.models import Grant as GrantModel
from snowflake.core.grant._grant import Grant


if TYPE_CHECKING:
    from snowflake.core import Root


class Grants:
    """The entry point of the Snowflake Core Python APIs to manage Snowflake Grants.

    Args:
        root: A :class:`Root` instance.
    """

    def __init__(self, root: "Root") -> None:
        self._root = root
        self._api = GrantApi(
            root=self._root,
            resource_class=None,
            sproc_client=StoredProcApiClient(root=self._root)
        )

    def grant(self, grant: Grant) -> None:
        """
        Grant the specified privilege(s) on the named securable to the named grantee.

        Args:
            grant: an instance of :class:`Grant`
        Example:
          Apply a grant to test role
          >>>  root.grants.apply(Grant(
          >>>       grantee=Grantees.role(name=role_name),
          >>>       securable=Securables.current_account,
          >>>       privileges=[Privileges.create_database]))
        """
        privileges = [privilege.value for privilege in grant.privileges] if grant.privileges else None
        request_body = GrantModel(privileges=privileges, grant_option=grant.grant_option)

        self._api.grant_privilege(
            grantee_type=grant.grantee.grantee_type,
            grantee_name=grant.grantee.name,
            securable_type=grant.securable.securable_type,
            securable_name=grant.securable.name,
            grant=request_body)
