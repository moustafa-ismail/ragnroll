import typing

from public import public


if typing.TYPE_CHECKING:
    from urllib3.response import HTTPHeaderDict  # type: ignore[attr-defined]

    from .rest import RESTResponse


class _OpenAPIError(Exception):
    """The base exception class for all OpenAPIExceptions."""


class _APITypeError(_OpenAPIError, TypeError):
    """Raises an exception for TypeErrors.

    Args:
        msg (str): the exception message

    Keyword Args:
        path_to_item (list): a list of keys an indices to get to the
                             current_item
                             None if unset
        valid_classes (tuple): the primitive classes that current item
                               should be an instance of
                               None if unset
        key_type (bool): False if our value is a value in a dict
                         True if it is a key in a dict
                         False if our item is an item in a list
                         None if unset
    """

    def __init__(
        self,
        msg: str,
        path_to_item: typing.Optional[typing.List[typing.Union[str, int]]] = None,
        valid_classes: typing.Optional[typing.Tuple[typing.Type[typing.Any], ...]] = None,
        key_type: typing.Optional[bool] = None,
    ) -> None:
        self.path_to_item = path_to_item
        self.valid_classes = valid_classes
        self.key_type = key_type
        full_msg = msg
        if path_to_item:
            full_msg = f"{msg} at {_render_path(path_to_item)}"
        super().__init__(full_msg)


class _APIValueError(_OpenAPIError, ValueError):
    """Raises an exception for ValueErrors.

    Args:
        msg (str): the exception message.

    Keyword Args:
        path_to_item (list) the path to the exception in the
            received_data dict. None if unset
    """

    def __init__(
        self,
        msg: str,
        path_to_item: typing.Optional[typing.List[typing.Union[str, int]]] = None,
    ) -> None:
        self.path_to_item = path_to_item
        full_msg = msg
        if path_to_item:
            full_msg = f"{msg} at {_render_path(path_to_item)}"
        super().__init__(full_msg)


@public
class APIError(_OpenAPIError):
    def __init__(
        self,
        status: typing.Optional[int] = None,
        reason: typing.Optional[str] = None,
        http_resp: typing.Optional["RESTResponse"] = None,
    ) -> None:
        self.status = status
        self.reason = reason
        self.body: typing.Optional[typing.Union[bytes, typing.Any]] = None
        self.headers: typing.Optional[HTTPHeaderDict] = None
        if http_resp:
            self.status = http_resp.status
            self.reason = http_resp.reason
            self.body = http_resp.data
            self.headers = http_resp.getheaders()

    def __str__(self) -> str:
        """Provide a custom error message for the exception."""
        error_message = f"({self.status})\nReason: {self.reason}\n"
        if self.headers:
            error_message += f"HTTP response headers: {self.headers}\n"

        if self.body:
            if isinstance(self.body, bytes):
                body = self.body.decode()
            else:
                body = self.body
            error_message += f"HTTP response body: {body}\n"

        return error_message


@public
class NotFoundError(APIError):
    """Raised when we encounter an HTTP error code of 404."""

    def __init__(
        self,
        status: typing.Optional[int] = None,
        reason: typing.Optional[str] = None,
        http_resp: typing.Optional["RESTResponse"] = None,
    ) -> None:
        super().__init__(status, reason, http_resp)


@public
class UnauthorizedError(APIError):
    """Raised when we encounter an HTTP error code of 401."""

    def __init__(
        self,
        status: typing.Optional[int] = None,
        reason: typing.Optional[str] = None,
        http_resp: typing.Optional["RESTResponse"] = None,
    ) -> None:
        super().__init__(status, reason, http_resp)


@public
class ForbiddenError(APIError):
    """Raised when we encounter an HTTP error code of 403."""

    def __init__(
        self,
        status: typing.Optional[int] = None,
        reason: typing.Optional[str] = None,
        http_resp: typing.Optional["RESTResponse"] = None,
    ) -> None:
        super().__init__(status, reason, http_resp)


@public
class ServerError(APIError):
    """Raised when we encounter an HTTP error code of 5NN."""

    def __init__(
        self,
        status: typing.Optional[int] = None,
        reason: typing.Optional[str] = None,
        http_resp: typing.Optional["RESTResponse"] = None,
    ) -> None:
        super().__init__(status, reason, http_resp)


@public
class ConflictError(APIError):
    """Raised when we encounter an HTTP error code of 409."""

    def __init__(
        self,
        status: typing.Optional[int] = None,
        reason: typing.Optional[str] = None,
        http_resp: typing.Optional["RESTResponse"] = None,
    ) -> None:
        super().__init__(status, reason, http_resp)


def _render_path(
    path_to_item: typing.Optional[typing.List[typing.Union[str, int]]],
) -> str:
    """Return a string representation of a path."""
    result = ""
    if path_to_item:
        for pth in path_to_item:
            if isinstance(pth, int):
                result += f"[{pth}]"
            else:
                result += f"['{pth}']"
    return result


@public
class InvalidResponseError(Exception):
    """Raised when the api response is invalid."""

    def __init__(
        self,
        reason: typing.Optional[str] = None
    ) -> None:
        super().__init__(reason)
        self.reason = reason


@public
class LongRunningQueryTimeout(Exception):
    """Raised when long running query timeout."""

    def __init__(
        self,
        reason: typing.Optional[str] = None
    ) -> None:
        super().__init__(reason)
        self.reason = reason


@public
class FunctionArgsInvalid(Exception):
    """Raised when function args are invalid."""

    def __init__(
        self,
        reason: typing.Optional[str] = None
    ) -> None:
        super().__init__(reason)
        self.reason = reason

@public
class FunctionResultInvalid(Exception):
    """Raised when function result is invalid."""

    def __init__(
        self,
        reason: typing.Optional[str] = None
    ) -> None:
        super().__init__(reason)
        self.reason = reason
