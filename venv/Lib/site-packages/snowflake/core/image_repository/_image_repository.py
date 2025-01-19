from typing import TYPE_CHECKING, Iterator, Optional

from pydantic import StrictStr

from snowflake.core._common import (
    CreateMode,
    SchemaObjectCollectionParent,
    SchemaObjectReferenceMixin,
)
from snowflake.core._internal.telemetry import api_telemetry


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


from snowflake.core.image_repository._generated import ImageRepositoryApi
from snowflake.core.image_repository._generated.api_client import StoredProcApiClient
from snowflake.core.image_repository._generated.models.image_repository import (
    ImageRepositoryModel as ImageRepository,
)


class ImageRepositoryCollection(
    SchemaObjectCollectionParent["ImageRepositoryResource"]
):
    """Represents the collection operations of the Snowflake Image Repository resource."""

    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, ImageRepositoryResource)
        self._api = ImageRepositoryApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def create(
        self,
        image_repository: ImageRepository,
        mode: CreateMode = CreateMode.error_if_exists,
    ) -> "ImageRepositoryResource":
        """Create an image repository to Snowflake.

        Args:
            image_repository: an instance of :class:`ImageRepository`.
            mode: One of the following strings.

                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError`
                if the image repository already exists in Snowflake.
                Equivalent to SQL ``create image repository <name> ...``.

                CreateMode.or_replace: Replace if the image repository already exists in Snowflake. Equivalent to SQL
                ``create or replace image repository <name> ...``.

                CreateMode.if_not_exists: Do nothing if the image repository already exists in Snowflake.
                Equivalent to SQL ``create image repository <name> if not exists...``

                Default value is CreateMode.error_if_exists.

        """
        real_mode = CreateMode[mode].value
        self._api.create_image_repository(
            self.database.name,
            self.schema.name,
            image_repository._to_model(),
            StrictStr(real_mode),
            async_req=False,
        )
        return self[image_repository.name]

    @api_telemetry
    def iter(
        self,
        *,
        like: Optional[str] = None,
    ) -> Iterator[ImageRepository]:
        """Look up image repositories in Snowflake."""
        image_respositories = self._api.list_image_repositories(
            self.database.name,
            self.schema.name,
            StrictStr(like) if like is not None else None,
            async_req=False, )

        return map(ImageRepository._from_model, iter(image_respositories))


class ImageRepositoryResource(SchemaObjectReferenceMixin[ImageRepositoryCollection]):
    """A reference to an Image Repository in Snowflake."""

    def __init__(self, name: str, collection: ImageRepositoryCollection):
        self.name = name
        self.collection = collection

    @api_telemetry
    def fetch(self) -> ImageRepository:
        """Fetch the image repository details from Snowflake."""
        return ImageRepository._from_model(self.collection._api.fetch_image_repository(
            self.database.name, self.schema.name, self.name, async_req=False
        ))

    @api_telemetry
    def delete(self) -> None:
        """Delete the image repository from Snowflake."""
        self.collection._api.delete_image_repository(
            self.database.name, self.schema.name, self.name, async_req=False
        )

    @api_telemetry
    def list_images_in_repository(self) -> Iterator[str]:
        """List images in the image repository from Snowflake."""
        images = self.collection._api.list_images_in_repository(
            self.database.name, self.schema.name, self.name, async_req=False
        )
        return iter([]) if images.images is None else iter(images.images)
