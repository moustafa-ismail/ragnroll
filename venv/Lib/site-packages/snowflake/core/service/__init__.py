"""Manages Snowpark Container Services. Resource is Experimental.

Example:
    >>> new_service_def  = Service(
    ...     name="MYSERVICE",
    ...     compute_pool="MYCOMPUTEPOOL",
    ...     spec="@~/myservice_spec.yml",
    ...     min_instances=1,
    ...     max_instances=1,
    ... )
    >>> services = root.databases["MYDB"].schemas["MYSCHEMA"].services
    >>> myservice = services.create(new_service_def)
    >>> myservice_snapshot = myservice.fetch()
    >>> service_data = services.iter(like="%SERVICE")
    >>> myservice.suspend()
    >>> myservice.resume()
    >>> service_status = myservice.get_service_status()
    >>> logs = myservice.get_service_logs()
    >>> myservice.delete()
    >>> an_existing_service = services["an_existing_service"]
    >>> an_existing_service.suspend()

Refer to :class:`snowflake.core.Root` to create the ``root``.
"""


from public import public

from snowflake.core.service._generated.models import ServiceSpec as ServiceSpecification

from ._service import (
    Service,
    ServiceCollection,
    ServiceResource,
    ServiceSpec,
    ServiceSpecInlineText,
    ServiceSpecStageFile,
)


public(
    Service=Service,
    ServiceCollection=ServiceCollection,
    ServiceResource=ServiceResource,
    ServiceSpec=ServiceSpec,
    ServiceSpecification=ServiceSpecification,
    ServiceSpecInlineText=ServiceSpecInlineText,
    ServiceSpecStageFile=ServiceSpecStageFile,
)
