# mypy: ignore-errors
"""
This file is meant to support pydantic 2.x.

Our code is based on pydantic 1.x, some functions in it are deprecated in pydantic 2.x. So we use pydantic.v1 to make
sure that customers could use our product with pydantic 2.x.
This may be removed in the future when we are upgraded to pydantic 2.x.
"""
try:
    from pydantic.v1 import *  # noqa: F403
except ModuleNotFoundError:
    from pydantic import *  # noqa: F403
