from typing import Dict


BRIDGE_OVERRIDE_PARAMETER_PREFIX = 'ENABLE_SNOW_API_FOR_'


class SnowApiParameters:
    """Wrapper that abstracts away the behavior from the parsing/reading of parameters.

    Args:
        params_map: A ``dict[str,str]`` of parameter names to their values

    """

    def __init__(self, params_map: Dict[str, str]) -> None:
        self._params_map = params_map

    # remove/deprecate this when the client bridge is gone.
    def resource_should_use_client_bridge(self, resource_name: str) -> bool:

        use_rest_api_param = (BRIDGE_OVERRIDE_PARAMETER_PREFIX + resource_name).lower()

        # Note that the only value that matters here is 'bridge'; only then do we
        # use the client bridge. No other value has an effect, i.e., we will use
        # REST for all other values, even for something like 'disabled'.
        if use_rest_api_param not in self._params_map:
            return False
        else:
            return self._params_map[use_rest_api_param].lower().strip() == "bridge"
