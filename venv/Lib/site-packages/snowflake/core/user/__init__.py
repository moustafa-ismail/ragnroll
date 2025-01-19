"""
Manages Snowflake Users. Resource is Experimental.

Example:
    >>> user = User("test_user")
    >>> created_user = root.users.create(user)
    >>> root.users["test_user"].fetch()
    >>> root.users["test_user"].delete()

"""


from public import public

from ._user import User, UserCollection, UserResource


public(
    User=User,
    UserCollection=UserCollection,
    UserResource=UserResource
)
