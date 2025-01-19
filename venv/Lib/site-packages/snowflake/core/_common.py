# Copyright (c) 2012-2023 Snowflake Computing Inc. All rights reserved.

import json
import os
import sys

from abc import ABC, abstractmethod
from enum import Enum, EnumMeta
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    ValuesView,
)

from public import public
from pydantic import StringConstraints
from typing_extensions import Annotated

from snowflake.connector import SnowflakeConnection

from ._internal.pydantic_compatibility import BaseModel, Field


if TYPE_CHECKING:
    from snowflake.core import Root
    from snowflake.core.database import DatabaseResource
    from snowflake.core.schema import SchemaResource

from snowflake.snowpark import Session


T = TypeVar("T")

UNALTERABLE_PARAMETERS = {"name", "tag"}


def check_env_parameter_enabled(param: str, default: str = "False") -> bool:
    return os.getenv(
        param,
        default,
    ).lower() in (
        "true",
        "t",
        "yes",
        "y",
        "on",
    )


def _is_an_update(parameters: Iterable[str], alterable_parameters: Iterable[str]) -> bool:
    """Decide whether the parameters contain update-relevant values.

    Parameters:
        parameters: an iterable of parameters in the request
    """
    return len(set(parameters).intersection(alterable_parameters)) > 0


@public
class ObjectCollectionBC(ABC):
    @property
    @abstractmethod
    def _session(self) -> Session: ...

    @property
    @abstractmethod
    def _connection(self) -> SnowflakeConnection: ...

    @property
    @abstractmethod
    def root(self) -> "Root": ...


@public
class ObjectCollection(ObjectCollectionBC, Generic[T]):
    def __init__(self, ref_class: Type[T]) -> None:
        self._items: Dict[str, T] = {}
        self._ref_class = ref_class

    def __getitem__(self, item: str) -> T:
        if item not in self._items:
            # Protocol doesn't support restricting __init__
            self._items[item] = self._ref_class(item, self)  # type: ignore[call-arg]
        return self._items[item]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def keys(self) -> KeysView[str]:
        return self._items.keys()

    def items(self) -> ItemsView[str, T]:
        return self._items.items()

    def values(self) -> ValuesView[T]:
        return self._items.values()


@public
class AccountObjectCollectionParent(ObjectCollection[T], Generic[T]):
    def __init__(self, root: "Root", ref_class: Type[T]) -> None:
        super().__init__(ref_class)
        self._root = root

    @property
    def _session(self) -> Session:
        return self._root.session

    @property
    def _connection(self) -> SnowflakeConnection:
        return self._root.connection

    @property
    def root(self) -> "Root":
        return self._root


@public
class SchemaObjectCollectionParent(ObjectCollection[T], Generic[T]):
    def __init__(self, schema: "SchemaResource", ref_class: Type[T]) -> None:
        super().__init__(ref_class)
        self._schema = schema

    @property
    def _session(self) -> Session:
        return self.schema._session

    @property
    def _connection(self) -> SnowflakeConnection:
        return self.schema._connection

    @property
    def schema(self) -> "SchemaResource":
        return self._schema

    @property
    def database(self) -> "DatabaseResource":
        return self.schema.collection.database

    @property
    def root(self) -> "Root":
        return self.database.collection.root


@public
class ObjectReferenceProtocol(Protocol[T]):
    @property
    def collection(self) -> ObjectCollection[T]: ...

    @property
    def root(self) -> "Root": ...


@public
class SchemaObjectReferenceProtocol(Protocol[T]):
    @property
    def collection(self) -> SchemaObjectCollectionParent[T]: ...

    @property
    def name(self) -> str: ...

    @property
    def database(self) -> "DatabaseResource": ...

    @property
    def schema(self) -> "SchemaResource": ...


@public
class ObjectReferenceMixin(Generic[T]):


    @property
    def _session(self: ObjectReferenceProtocol[T]) -> Session:
        return self.collection._session

    @property
    def _connection(self: ObjectReferenceProtocol[T]) -> SnowflakeConnection:
        return self.collection._connection

    @property
    def root(self: ObjectReferenceProtocol[T]) -> "Root":
        return self.collection.root


@public
class SchemaObjectReferenceMixin(Generic[T], ObjectReferenceMixin[SchemaObjectCollectionParent[T]]):
    @property
    def schema(self: SchemaObjectReferenceProtocol[T]) -> "SchemaResource":
        return self.collection.schema

    @property
    def database(self: SchemaObjectReferenceProtocol[T]) -> "DatabaseResource":
        return self.collection.schema.database

    @property
    def fully_qualified_name(self: SchemaObjectReferenceProtocol[T]) -> str:
        return f"{self.database.name}.{self.schema.name}.{self.name}"


class CaseInsensitiveEnumMeta(EnumMeta):
    def __init__(cls, *args, **kws) -> None:  # type: ignore
        super().__init__(*args, **kws)

        class lookup(dict):  # type: ignore
            def get(self, key, default=None):  # type: ignore
                return super().get(key.lower(), key.lower())

        cls._legacy_mode_map_ = lookup({item.value.lower(): item.name for item in cls})  # type: ignore

    def __getitem__(cls, name: str) -> Any:
        converted_name = cls._legacy_mode_map_.get(name)
        return super().__getitem__(converted_name)


@public
class CreateMode(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    error_if_exists = "errorIfExists"
    or_replace = "orReplace"
    if_not_exists = "ifNotExists"


class Clone(BaseModel):
    source: Annotated[str, StringConstraints(strict=True)] = Field(...)
    point_of_time: Optional["PointOfTime"] = None


class PointOfTime(BaseModel):
    point_of_time_type: Annotated[str, StringConstraints(strict=True)] = Field(...)
    reference: Annotated[str, StringConstraints(strict=True)] = Field(
        ..., description="The relation to the point of time. At the time of writing at and before are supported."
    )
    when: Annotated[
        str,
        StringConstraints(strict=True),
    ] = Field(..., description="The actual description of the point of time.")
    __properties = ["point_of_time_type", "reference", "when"]

    __discriminator_property_name = "point_of_time_type"

    __discriminator_value_class_map = {
        "offset": "PointOfTimeOffset",
        "statement": "PointOfTimeStatement",
        "timestamp": "PointOfTimeTimestamp",
    }

    class Config:
        allow_population_by_field_name = True
        validate_assignment = True

    def to_dict(self) -> Dict[str, str]:
        d = {p: getattr(self, p) for p in self.__properties}
        # Need to map "when" to the discriminator value as per our OAS
        d[d[self.__discriminator_property_name]] = d["when"]
        del d["when"]
        return d

    @classmethod
    def get_discriminator_value(cls, obj: Dict[str, Optional[str]]) -> str:
        """Return the discriminator value (object type) of the data."""
        discriminator_name = obj[cls.__discriminator_property_name]
        assert discriminator_name is not None
        discriminator = cls.__discriminator_value_class_map.get(discriminator_name)
        assert discriminator is not None
        return discriminator

    @classmethod
    def from_dict(
        cls,
        obj: Dict[str, Optional[str]],
    ) -> Union[
        "PointOfTimeOffset",
        "PointOfTimeStatement",
        "PointOfTimeTimestamp",
    ]:
        """Create an instance of PointOfTime from a dict."""
        object_type = cls.get_discriminator_value(obj)
        if not object_type:
            raise ValueError(
                "PointOfTime failed to lookup discriminator value from "
                + json.dumps(obj)
                + ". Discriminator property name: "
                + cls.__discriminator_property_name
                + ", mapping: "
                + json.dumps(cls.__discriminator_value_class_map)
            )
        return getattr(sys.modules[__name__], object_type).from_dict(obj)


class PointOfTimeOffset(PointOfTime):
    point_of_time_type: Annotated[str, StringConstraints(strict=True)] = "offset"


class PointOfTimeTimestamp(PointOfTime):
    point_of_time_type: Annotated[str, StringConstraints(strict=True)] = "timestamp"


class PointOfTimeStatement(PointOfTime):
    point_of_time_type: Annotated[str, StringConstraints(strict=True)] = "statement"


# Now that everything has been defined, let's resolve forward declarations!
Clone.update_forward_refs()
