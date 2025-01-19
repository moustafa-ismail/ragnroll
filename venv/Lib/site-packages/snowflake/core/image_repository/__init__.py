"""Manages Snowpark Container Image Repositories. Resource is Experimental.

Example:
    >>> new_image_repository = ImageRepository(
    ...     name="my_imagerepo",
    ... )
    >>> image_repositories = root.databases["MYDB"].schemas["MYSCHEMA"].image_repositories
    >>> my_image_repo = image_repositories.create(new_image_repository)
    >>> my_image_repo_snapshot = my_image_repo.fetch()
    >>> ir_data = image_repositories.iter(like="%my")
    >>> an_existing_repo = image_repositories["an_existing_repo"]
    >>> an_existing_repo.delete()

Refer to :class:`snowflake.core.Root` to create the ``root``.
"""

from public import public

from ._image_repository import (
    ImageRepository,
    ImageRepositoryCollection,
    ImageRepositoryResource,
)


public(
    ImageRepository=ImageRepository,
    ImageRepositoryCollection=ImageRepositoryCollection,
    ImageRepositoryResource=ImageRepositoryResource,
)
