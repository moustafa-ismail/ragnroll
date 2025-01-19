"""Manages Snowflake Privileges. Resource is Experimental.

Example:
    >>> root.grants.grant(Grant(
    >>>        grantee=Grantees.role(name=role_name),
    >>>        securable=Securables.current_account,
    >>>        privileges=[Privileges.create_database]))
"""

from public import public

from ._grant import Grant
from ._grantee import Grantee, Grantees
from ._grants import Grants
from ._privileges import Privileges
from ._securables import Securable, Securables


public(
    Grant=Grant,
    Grants=Grants,
    Grantee=Grantee,
    Grantees=Grantees,
    Privileges=Privileges,
    Securable=Securable,
    Securables=Securables,
)
