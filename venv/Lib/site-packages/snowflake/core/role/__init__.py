"""Manages Snowflake user roles. Resource is Experimental.

Example:
    >>> role_name = "test_role"
    >>> test_role = Role(name=role_name, comment="test_comment")
    >>> created_role = roles.create(test_role)
    >>> roles[role_name].delete()

"""

from public import public

from ._role import Role, RoleCollection, RoleResource


public(
    Role=Role,
    RoleCollection=RoleCollection,
    RoleResource=RoleResource
)
