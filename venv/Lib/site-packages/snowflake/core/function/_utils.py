from pydantic import StrictStr

from ._generated.models.function import Function


def get_name_with_args(function: Function) -> StrictStr:
    return f"{function.name}({','.join([str(argument.datatype) for argument in function.arguments])})"
