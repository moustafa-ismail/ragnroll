# mypy: ignore-errors

from datetime import time
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

from pydantic import StrictStr

from snowflake.core._common import (
    CreateMode,
    SchemaObjectCollectionParent,
    SchemaObjectReferenceMixin,
)

from .._internal.telemetry import api_telemetry
from ..exceptions import FunctionArgsInvalid, FunctionResultInvalid
from ._generated.api import FunctionApi
from ._generated.api_client import StoredProcApiClient
from ._generated.models.function import Function
from ._generated.models.function_argument import FunctionArgument
from ._utils import get_name_with_args


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class FunctionCollection(SchemaObjectCollectionParent["FunctionResource"]):
    """Represents the collection operations on the Snowflake Function resource.

    With this collection, you can create role or iterate or fetch function that you have access to.

    Args:
        schema: A :class:schema instance.

    Example:
        Create a FunctionCollection instance:
        >>> functions = schema.functions
        >>> functions.create(new_function)
    """

    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, FunctionResource)
        self._api = FunctionApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def create(
        self, function: Function,
        mode: CreateMode=CreateMode.error_if_exists,
    ) -> "FunctionResource":
        """Create a Functionm, currently only support service function.

        Args:
            function: The service function object, together with function properties:
                name, returns, arguments, service, endpoint, path
                optional: max_batch_rows
            mode: One of the following strings.

                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError`
                if the function already exists in Snowflake.  Equivalent to SQL ``create function <name> ...``.

                CreateMode.or_replace: Replace if the task already exists in Snowflake. Equivalent to SQL
                ``create or replace function <name> ...``.

                CreateMode.if_not_exists: Do nothing if the task already exists in Snowflake.
                Equivalent to SQL ``create function <name> if not exists...``

                Default value is CreateMode.error_if_exists.
        """
        real_mode = CreateMode[mode].value

        self._api.create_function(
            self.database.name, self.schema.name, function, create_mode=StrictStr(real_mode),
            async_req=False
        )

        return FunctionResource(get_name_with_args(function), self)

    @api_telemetry
    def iter(
        self,
        *,
        like: Optional[str] = None,
    ) -> Iterator[Function]:
        """Search ``Function`` objects from Snowflake.

        Args:
            like: The pattern of the function's name. Use ``%`` to represent any number of characters and ``?`` for a
                single character.
        """
        functions = self._api.list_functions( database=self.database.name, var_schema=self.schema.name, like=like,
                                             async_req=False)
        return iter(functions)



class FunctionResource(SchemaObjectReferenceMixin[FunctionCollection]):
    """Represents a reference to a Snowflake Function resource."""

    def __init__(
            self,
            name_with_args: StrictStr,
            collection: FunctionCollection
        ) -> None:
        self.collection = collection
        self.name_with_args = name_with_args

    def _cast_result(self, result: Any, returns: StrictStr) -> Any:
        if returns in ["NUMBER", "INT", "FIXED"]:
            return int(result)
        if returns == "REAL":
            return float(result)
        if returns == "TEXT":
            return str(result)
        if returns == "TIME":
            return time(result)
        if returns == "BOOLEAN":
            return bool(int(result))
        return result

    @api_telemetry
    def fetch(self) -> Function:
        """Fetch the details of a function.

        Example:
            functions["foo(REAL)"].fetch()
        """
        return self.collection._api.fetch_function(
            self.database.name, self.schema.name, self.name_with_args, async_req=False,
        )

    @api_telemetry
    def delete(self) -> None:
        """Delete the function."""
        self.collection._api.delete_function(self.database.name, self.schema.name, self.name_with_args, async_req=False)

    @api_telemetry
    def execute(self, input_args: List[Any] = None) -> object:
        function = self.fetch()
        args_count = len(function.arguments) if function.arguments is not None else 0

        if input_args is None:
            input_args = []

        if len(input_args) != args_count:
            raise FunctionArgsInvalid(f"Function expects {args_count} arguments but received {len(input_args)}")

        function_args = []
        for i in range(args_count):
            argument = FunctionArgument()
            argument.value = input_args[i]
            argument.datatype = function.arguments[i].datatype
            function_args.append(argument)

        result = self.collection._api.execute_function(
            self.database.name,
            self.schema.name,
            function.name,
            function_args,
            async_req=False
        )

        if not isinstance(result, dict) or len(result.values()) != 1:
            raise FunctionResultInvalid(f"Function result {str(result)} is invalid or empty")

        result = list(result.values())[0]
        return self._cast_result(result, str(function.returns))

