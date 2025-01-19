# Copyright (c) 2012-2023 Snowflake Computing Inc. All rights reserved.

"""Manages Snowflake Stages. Resource is Experimental.

Example:
    >>> stages: StageCollection = root.databases["mydb"].schemas["myschema"].stages
    >>> mystage = stages.create(Stage("mystage"))
    >>> stage_iter = stages.iter(like="my%")
    >>> mystage = stages["mystage"]
    >>> an_existing_stage = stages["an_existing_stage"]

Refer to :class:`snowflake.core.Root` to create the ``root``.

"""

from ..stage._generated.models import (
    AwsCredentials,
    AzureCredentials,
    Credentials,
    FileTransferMaterial,
    PresignedUrlRequest,
    Stage,
    StageDirectoryTable,
    StageEncryption,
    StageFile,
)
from ._stage import StageCollection, StageResource


__all__ = [
    "Stage",
    "StageResource",
    "StageCollection",
    "AwsCredentials",
    "AzureCredentials",
    "Credentials",
    "FileTransferMaterial",
    "PresignedUrlRequest",
    "StageDirectoryTable",
    "StageEncryption",
    "StageFile",
]

