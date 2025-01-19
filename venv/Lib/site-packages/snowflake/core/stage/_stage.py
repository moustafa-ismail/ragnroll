import os
import tempfile

from typing import TYPE_CHECKING, Iterator, Optional

import requests

from pydantic import StrictStr

from snowflake.connector.file_util import SnowflakeFileUtil
from snowflake.core._common import (
    CreateMode,
    SchemaObjectCollectionParent,
    SchemaObjectReferenceMixin,
)
from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core.exceptions import APIError
from snowflake.core.stage._generated.api import StageApi
from snowflake.core.stage._generated.api_client import StoredProcApiClient
from snowflake.core.stage._generated.models.presigned_url_request import PresignedUrlRequestModel
from snowflake.core.stage._generated.models.stage import Stage
from snowflake.core.stage._generated.models.stage_file import StageFile


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class StageCollection(SchemaObjectCollectionParent["StageResource"]):
    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, StageResource)
        self._api = StageApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root),
        )

    @api_telemetry
    def create(
        self,
        stage: Stage,
        *,
        mode: CreateMode = CreateMode.error_if_exists,
    ) -> "StageResource":
        """Create a stage.

        Args:
            stage: The stage object, together with the stage's properties, object parameters.
            mode: One of the following strings.
                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError`
                if the stage already exists in Snowflake. Equivalent to SQL ``create stage <name> ...``.

                CreateMode.or_replace: Replace if the stage already exists in Snowflake. Equivalent to SQL
                ``create or replace stage <name> ...``.

                CreateMode.if_not_exists: Do nothing if the stage already exists in Snowflake. Equivalent to SQL
                ``create stage <name> if not exists...``

                Default value is CreateMode.error_if_exists.
        """
        real_mode = CreateMode[mode].value
        self._api.create_stage(
            self.database.name, self.schema.name, stage, create_mode=StrictStr(real_mode), async_req=False
        )
        return StageResource(stage.name, self)

    @api_telemetry
    def iter(
        self,
        *,
        like: Optional[str] = None,
    ) -> Iterator[Stage]:
        """Search ``Stage`` objects from Snowflake.

        Args:
            like: The pattern of the Stage name. Use ``%`` to represent any number of characters and ``?`` for a
                single character.
        """
        stages = self._api.list_stages(
            database=self.database.name, var_schema=self.schema.name, like=like, async_req=False
        )

        return iter(stages)


class StageResource(SchemaObjectReferenceMixin[StageCollection]):
    """Represents a reference to a Snowflake Stage resource."""

    def __init__(self, name: str, collection: StageCollection) -> None:
        self.collection = collection
        self.name = name

    @api_telemetry
    def fetch(self) -> Stage:
        """Fetch the details of a stage."""
        return self.collection._api.fetch_stage(
            self.database.name,
            self.schema.name,
            self.name,
            async_req=False,
        )

    @api_telemetry
    def delete(self) -> None:
        """Delete the stage."""
        self.collection._api.delete_stage(self.database.name, self.schema.name, self.name, async_req=False)

    @api_telemetry
    def list_files(
        self,
        *,
        pattern: Optional[str] = None,
    ) -> Iterator[StageFile]:
        """List files in the stage.

        Args:
            pattern: Specifies a regular expression pattern for filtering files from the output.
            The command lists all files in the specified path and applies the regular expression pattern on each
            of the files found.
        """
        files = self.collection._api.list_files(
            self.database.name, self.schema.name, self.name, pattern, async_req=False
        )
        return iter(files)

    @api_telemetry
    def upload_file(
        self,
        file_path: str,
        stage_folder_path: str,
        *,
        auto_compress: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Upload a file to a stage location.

        Currently only supports uploading files smaller than 1MB to server-side encrypted stages.

        Args:
            file_path: A string representing the location of the file on the client machine to be uploaded.
            stage_folder_path: The stage folder location to be uploaded to, e.g. /folder or /
            auto_compress: Specifies whether Snowflake uses gzip to compress files during upload:
                True: Snowflake compresses the files (if they are not already compressed).

                False: Snowflake does not compress the files.
            overwrite: Specifies whether Snowflake overwrites an existing file with the same name during upload:
                True: An existing file with the same name is overwritten.

                False: An existing file with the same name is not overwritten.

        Raise `APIError` if upload failed
        """
        if auto_compress:
            # Compress the file.
            tmp_dir = tempfile.mkdtemp()
            (
                real_src_file_path,
                _,
            ) = SnowflakeFileUtil.compress_file_with_gzip(
                file_path, tmp_dir
            )
        else:
            real_src_file_path = file_path

        if not os.path.exists(real_src_file_path):
            raise ValueError(f"File does not exists {real_src_file_path}")
        # It's difficult to check whether the file exists using the presigned url without downloading the actual file.
        # Issue another request to GS to check whether the file exists.
        stage_folder_path_norm = stage_folder_path if stage_folder_path.endswith("/") else stage_folder_path + "/"
        stage_path = stage_folder_path_norm + os.path.basename(real_src_file_path)
        list_res = list(self.list_files(pattern=stage_path))
        if len(list_res) == 1 and list_res[0].name is not None and not overwrite:
            # Skip upload if the file already exists and overwrite is False
            return None
        file_transfer_material = self.collection._api.get_presigned_url(
            self.database.name,
            self.schema.name,
            self.name,
            stage_path,
            PresignedUrlRequestModel(mode="upload")._to_model(),
        )
        presigned_url = file_transfer_material.presigned_url
        if presigned_url is None or presigned_url.strip() == "":
            raise Exception(f"presigned_url for {stage_path} is None or empty")

        if file_transfer_material.query_stage_master_key is not None:
            # Client side encryption stage
            raise NotImplementedError("Upload/download not supported for client side encrypted stages")
        else:
            with open(real_src_file_path, "rb") as src_file:
                # Tried to submit with
                # pool_manager: SFPoolManager = self.collection._api.api_client.rest_client.pool_manager
                # res = pool_manager.request_raw(method="GET", url=presigned_url)
                # but there are two issues to be solved:
                # 1. certificate verify failed: unable to get local issuer certificate
                # 2. s3 returns 501 not implemented error, could be caused by wrong header.
                res = requests.request(method="PUT", url=presigned_url, data=src_file, timeout=600)
                if not res.status_code == 200:
                    raise APIError(res.status_code, res.reason)

    @api_telemetry
    def download_file(self, stage_path: str, file_folder_path: str) -> None:
        """Download a file from a stage location.

        Currently only supports downloading files smaller than 1MB from server-side encrypted stages.

        Args:
            stage_path: The stage location of the file to be downloaded from.
            file_folder_path: A string representing the folder location of the file to be written to.
        """
        file_transfer_material = self.collection._api.get_presigned_url(
            self.database.name,
            self.schema.name,
            self.name,
            stage_path,
            PresignedUrlRequestModel(mode="download")._to_model(),
        )
        presigned_url = file_transfer_material.presigned_url
        if presigned_url is None or presigned_url.strip() == "":
            raise Exception(f"presigned_url for {stage_path} is None or empty")

        if file_transfer_material.query_stage_master_key is not None:
            # Client side encryption stage
            raise NotImplementedError("Upload/download not supported for client side encrypted stages")
        else:
            # Tried to submit with
            # pool_manager: SFPoolManager = self.collection._api.api_client.rest_client.pool_manager
            # res = pool_manager.request_raw(method="GET", url=presigned_url)
            # but there are two issues to be solved:
            # 1. certificate verify failed: unable to get local issuer certificate
            # 2. s3 returns 501 not implemented error, could be caused by wrong header.
            res = requests.request(method="GET", url=presigned_url, timeout=600)
            if not res.status_code == 200:
                raise APIError(res.status_code, res.reason)

            if not os.path.isdir(file_folder_path):
                os.makedirs(file_folder_path)
            file_path = os.path.join(file_folder_path, stage_path.split("/")[-1])
            with open(file_path, "wb") as target_file:
                target_file.write(res.content)
