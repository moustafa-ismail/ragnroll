class Grantee:
    """
    Class to represent type of resource that is the privilege grantee.

    Example:
        >>> Grantee("test-user", "user")
        >>> Grantee("test-role", "role")

    """

    def __init__(self, name: str, grantee_type: str):
        self._name = name
        self._grantee_type = grantee_type

    @property
    def name(self) -> str:
        """String that specifies the name of the privilege grantee."""
        return self._name

    @property
    def grantee_type(self) -> str:
        """String that specifies the type of resource that is the privilege grantee."""
        return self._grantee_type


class Grantees:
    """
    Util Class with static methods to create various :class:`Grantee` class instances.

    Example:
        >>> Grantees.role("test-role")
        >>> Grantees.user("test-user")
    """

    @staticmethod
    def user(name: str) -> "Grantee":
        return Grantee(name, "user")

    @staticmethod
    def role(name: str) -> "Grantee":
        return Grantee(name, "role")

    @staticmethod
    def database_role(name: str) -> "Grantee":
        return Grantee(name, "database-role")

    @staticmethod
    def share(name: str) -> "Grantee":
        return Grantee(name, "share")

    @staticmethod
    def application_role(name: str) -> "Grantee":
        return Grantee(name, "application-role")
