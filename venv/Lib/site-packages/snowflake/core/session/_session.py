from typing import TYPE_CHECKING, Dict, Optional

from pydantic import StrictStr

from snowflake.core.session._generated.api import SessionApi
from snowflake.core.session._generated.api_client import StoredProcApiClient


if TYPE_CHECKING:
    from snowflake.core import Root


class SnowAPISession:
    def __init__(self, root: "Root") -> None:
        self.root = root
        self._api = SessionApi(
            root=root,
            # There is no Session resource that we are exposing publicly, yet.
            resource_class=None,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    def _get_api_enablement_parameters(self) -> Dict[str, str]:
        return self.get_parameters(filter_str='ENABLE_SNOW_API_FOR_%')

    def get_parameters(self, filter_str: Optional[StrictStr]) -> Dict[str, str]:
        params = self._api.get_parameters(like=filter_str)

        params_map = dict()

        for param in params:
            p = param.to_dict()
            params_map[p['name'].lower()] = p['value'].lower()

        return params_map
