from typing import List, Optional

from snowflake.core.grant._grantee import Grantee
from snowflake.core.grant._privileges import Privileges
from snowflake.core.grant._securables import Securable


class Grant:
    """
    Represents Snowflake Grant Operation.

    Args:
        grantee: An instance of :class:`Grantee` class.
        securable: An instance of :class:`Securable` instance
        privileges: Acceptable :class:`Privileges` value
        grant_option: Boolean If true, grantee can pass this privilege down.

    Example:
        >>> from snowflake.core.grant._grantee import Grantees
        >>> from snowflake.core.grant._privileges import Privileges
        >>> from snowflake.core.grant._securables import Securables
        >>>
        >>> Grant(grantee=Grantees.role(name="test_role",
        >>>    securable=Securables.current_account,
        >>>    privileges=[Privileges.create_database])
    """

    def __init__(self, grantee: Grantee,
                 securable: Securable,
                 privileges: Optional[List[Privileges]] = None,
                 grant_option: bool = False):
        self._grantee = grantee
        self._securable = securable
        self._privileges = privileges
        self._grant_option = grant_option

    @property
    def grantee(self) -> Grantee:
        return self._grantee

    @property
    def securable(self) -> Securable:
        return self._securable

    @property
    def privileges(self) -> Optional[List[Privileges]]:
        return self._privileges

    @property
    def grant_option(self) -> bool:
        return self._grant_option
