from typing import TYPE_CHECKING, Iterator, Optional

from pydantic import StrictStr

from snowflake.core._common import AccountObjectCollectionParent, CreateMode, ObjectReferenceMixin
from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core.user._generated.api import UserApi
from snowflake.core.user._generated.api_client import StoredProcApiClient
from snowflake.core.user._generated.models.user import UserModel as User


if TYPE_CHECKING:
    from snowflake.core import Root


class UserCollection(AccountObjectCollectionParent["UserResource"]):
    """Represents the collection operations on the Snowflake User resource.

    With this collection, you can create or update or fetch or list all users that you have access to.

    Args:
        root: A :class:`Root` instance.

    Example:
        Create a UserCollection instance:
        >>> role_collection = root.users
        >>> role_collection.create(new_user)
    """

    def __init__(self, root: "Root") -> None:
        super().__init__(root, ref_class=UserResource)
        self._api = UserApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def create(self, user: User, *, mode: CreateMode = CreateMode.error_if_exists) -> "UserResource":
        """
        Create a user in Snowflake account.

        Args:
            user: an instance of :class:`User`.
            mode: One of the following enum values.

                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError` if the
                    user already exists in Snowflake. Equivalent to SQL ``create user <name> ...``.

                CreateMode.or_replace: Replace if the user already exists in Snowflake. Equivalent to SQL
                    ``create or replace user <name> ...``.

                CreateMode.if_not_exists: Do nothing if the user already exists in Snowflake. Equivalent to SQL
                    ``create user <name> if not exists...``

                Default value is CreateMode.error_if_exists.

        Example:
              Create a snowflake user and get the reference to it:
              >>> from snowflake.core.user import User
              >>> sample_user = User(name="test_user")
              >>> user_ref = root.users.create(sample_user)
        """
        create_mode = CreateMode[mode].value
        self._api.create_user(user._to_model(), StrictStr(create_mode))
        return self[user.name]

    @api_telemetry
    def iter(self,
             like: Optional[str] = None,
             limit: Optional[int] = None,
             starts_with: Optional[str] = None,
             from_name: Optional[str] = None) -> Iterator[User]:
        users = self._api.list_users(
            StrictStr(like) if like else None,
            StrictStr(starts_with) if starts_with else None,
            limit,
            from_name=from_name,
            async_req=False
        )
        """
        List Users available in an account:

        Args:
            like: Filter the output by resource name. Uses case-insensitive pattern matching, with support for SQL
                wildcard characters.

            limit: Limit the maximum number of rows returned.

            starts_with: filter the output based on the string of
                characters that appear at the beginning of the object name. Uses case-sensitive pattern matching.

            from_name: Enable fetching users only following the first user whose name matches the specified
                string. Case-sensitive and does not have to be the full name.
        """
        return map(User._from_model, iter(users))


class UserResource(ObjectReferenceMixin[UserCollection]):


    def __init__(self, name: str, collection: UserCollection) -> None:
        self.name = name
        self.collection = collection

    @api_telemetry
    def delete(self) -> None:
        """
        Delete this user.

        Example:
            Use user ref to delete.
            >>> user_ref.delete()
            >>> root.users["test_user"].delete()
        """
        self.collection._api.delete_user(self.name, async_req=False)

    @api_telemetry
    def fetch(self) -> User:
        """
        Fetch the user details.

        Example:
            >>> user_ref = root.users["test_user"].fetch()
            >>> user_ref.first_name
        """
        return User._from_model(self.collection._api.fetch_user(
            name=self.name,
            async_req=False
        ))
