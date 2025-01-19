"""Manages Snowpark Container Compute Pools. Resource is Experimental.

Example:
    >>> new_pool_def = ComputePool(
    ...     name="MYCOMPUTEPOOL",
    ...     instance_family="STANDARD_1",
    ...     min_nodes=1,
    ...     max_nodes=1,
    ... )
    >>> new_pool = root.compute_pools.create(new_pool_def)
    >>> cp_snapshot = new_pool.fetch()
    >>> cp_data = root.compute_pools.iter(like=”%COMPUTEPOOL”)
    >>> new_pool.resume()
    >>> new_pool.stop_all_services()
    >>> new_pool.suspend()
    >>> new_pool.delete()
    >>> an_existing_pool = root.compute_pools["existing_compute_pool"]
    >>> an_existing_pool.suspend()

Refer to :class:`snowflake.core.Root` to create the ``root``.
"""

from public import public

from ._compute_pool import ComputePool, ComputePoolCollection, ComputePoolResource


public(
    ComputePool=ComputePool,
    ComputePoolCollection=ComputePoolCollection,
    ComputePoolResource=ComputePoolResource,
)
