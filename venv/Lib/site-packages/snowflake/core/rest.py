import io
import typing

from public import public


if typing.TYPE_CHECKING:
    from urllib3.response import HTTPHeaderDict, HTTPResponse  # type: ignore[attr-defined]


@public
class RESTResponse(io.IOBase):
    def __init__(self, resp: "HTTPResponse") -> None:
        self.urllib3_response = resp
        self.status = resp.status
        self.reason = resp.reason
        self.data = resp.data

    def getheaders(self) -> "HTTPHeaderDict":
        """Return a dictionary of the response headers."""
        return self.urllib3_response.headers

    def getheader(self, name: str, default: typing.Any = None) -> typing.Any:
        """Return a given response header."""
        return self.urllib3_response.headers.get(name, default)
