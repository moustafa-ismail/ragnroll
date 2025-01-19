
class Securable:
    """Class to represent an snowflake entity that is being secured by a :class:`Privileges`."""

    def __init__(self, name: str, securable_type: str):
        self._name = name
        self._securable_type = securable_type

    @property
    def name(self) -> str:
        """String that specifies the name of resource that is being secured by a privilege."""
        return self._name

    @property
    def securable_type(self) -> str:
        """String that specifies the type of resource that is being secured by a privilege."""
        return self._securable_type


class Securables:
    """
    Util Class with static method to create various :class:`Securable` class instances.

    Example:
        >>> Securables.account("test-account")
        >>> Securables.database("testdb")
        >>> Securables.current_account
    """

    current_account = Securable("-", "account")

    @staticmethod
    def account(name: str) -> "Securable":
        return Securable(name, "account")

    @staticmethod
    def database(name: str) -> "Securable":
        return Securable(name, "database")

    @staticmethod
    def role(name: str) -> "Securable":
        return Securable(name, "role")

    @staticmethod
    def integration(name: str) -> "Securable":
        return Securable(name, "integration")
