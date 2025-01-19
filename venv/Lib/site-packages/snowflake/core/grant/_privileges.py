from enum import Enum


class Privileges(Enum):
    """Enum for Snowflake privileges that can be granted to a role or user."""

    bind_service_endpoint = "BIND SERVICE ENDPOINT"
    create_compute_pool = "CREATE COMPUTE POOL"
    create_database = "CREATE DATABASE"
    create_integration = "CREATE INTEGRATION"
    create_warehouse = "CREATE WAREHOUSE"
    imported_privileges = "IMPORTED PRIVILEGES"
    monitor_usage = "MONITOR USAGE"
    usage = "USAGE"

