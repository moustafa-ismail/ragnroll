"""Defines an Iterator to represent certain resource instances fetched from the Snowflake database."""

from typing import Callable, Generic, Iterable, Iterator, Optional, TypeVar, Union, overload
from functools import partial
from public import public

T = TypeVar("T")
S = TypeVar("S")


@public
class PagedIter(Iterable[T], Generic[T]):
    """A page-by-page iterator.

    Data fetched from the server is iterated over page by page, yielding items one by
    one.

    This iterator works by accepting a closure/partial function, which it will delegate to
    in order to fetch the next page, passing in only the chunk/page index it wants to get.

    Example:
        >>> from snowflake.core import Root
        >>> root = Root(connection)
        >>> tasks: TaskCollection = root.databases["mydb"].schemas["myschema"].tasks
        >>> task_iter = tasks.iter(like="my%")  # returns a PagedIter[Task]
        >>> for task_obj in task_iter:
        ...     print(task_obj.name)
    """

    @overload
    def __init__(self, data: Iterable[T]) -> None:
        ...

    @overload
    def __init__(self, data: Iterable[T], map_: None) -> None:
        ...

    @overload
    def __init__(self, data: Iterable[S], map_: Callable[[S], T]) -> None:
        ...

    def __init__(
        self,
        page_fetch_closure_,
        number_of_chunks_=1,
    ) -> None:
        self._page_fetch_closure = page_fetch_closure_
        self._number_of_chunks = number_of_chunks_
        self._iter = iter(self)

    def __iter__(self) -> Iterator[T]:
        for chunk in range(self._number_of_chunks):
            yield from self._page_fetch_closure(chunk_index=chunk)
