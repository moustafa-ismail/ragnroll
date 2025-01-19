# Copyright (c) 2012-2023 Snowflake Computing Inc. All rights reserved.
import json

from datetime import datetime
from typing import Any, Dict, Optional, Union

from snowflake.snowpark import Session
from snowflake.snowpark._internal.utils import validate_object_name
from snowflake.snowpark.exceptions import SnowparkSQLException


class TaskContext:
    """Represents the context in a `Snowflake Task <https://docs.snowflake.com/en/user-guide/tasks-intro>`_.

    When a :class:`Task`'s ``definition`` is a :class:`StoredProcedureCall`, the handler of the Stored Procedure
    can use this class to set the return value of the Task so the immediate successor tasks can use it.
    """

    def __init__(self, session: Session) -> None:
        """Initialize a :class:`TaskContext` object.

        Args:
            session: a Snowpark session.
        """
        self._session = session
        self._current_task_name: Optional[str] = None
        self._current_root_name: Optional[str] = None

    def _get_root_prefix(self) -> Optional[str]:
        root_task_name = self.get_current_root_task_name()
        root_prefix = root_task_name[root_task_name.rfind('.')+1:] + "$"
        current_task_name = self.get_current_task_name()
        current_task_name = current_task_name[root_task_name.rfind('.')+1:]
        if current_task_name.startswith(root_prefix):
            return root_prefix
        return None

    def set_return_value(self, value: Any) -> None:
        """Explicitly sets the return value for a task.

        An immediate successor task can then use :meth:`get_predecessor_return_value` to retrieve it.
        See `SYSTEM$SET_RETURN_VALUE
        <https://docs.snowflake.com/en/sql-reference/functions/system_set_return_value>`_ for details.
        This method can only be called in a Snowflake task.

        Args:
            value: The return value for a task. It will be converted to a ``str``
                when the underlying SQL system function is called.

        Example:
            >>> def task_handler(session: Session) -> None:
            >>>     from snowflake.core.task.context import TaskContext
            >>>     context = TaskContext(session)
            >>>     # this return value can be retrieved by successor Tasks.
            >>>     context.set_return_value("predecessor_return_value")
        """
        self._session.call("system$set_return_value", str(value))

    def get_predecessor_return_value(self, task_name: Optional[str] = None) -> str:
        """Retrieve the return value from the predecessor task in a DAG of tasks.

        The return value is explicitly set by the predecessor task using :meth:`set_return_value`.
        This method can only be called in a Snowflake task.

        See `SYSTEM$GET_PREDECESSOR_RETURN_VALUE
        <https://docs.snowflake.com/en/sql-reference/functions/system_get_predecessor_return_value>`_ for details.

        Args:
            task_name: The task name of the predecessor task that sets the return value to be retrieved.

                - If the task has only one predecessor task that is enabled, the argument is optional.
                  If this argument is omitted, the function retrieves the return value for the only enabled predecessor
                  task.
                - If the task has multiple predecessor tasks that are enabled, this argument is required.

        Example:
            >>> def task_handler(session: Session) -> None:
            >>>     from snowflake.core.task.context import TaskContext
            >>>     context = TaskContext(session)
            >>>     pred_return_value = context.get_predecessor_return_value("pred_task_name")
        """
        if task_name:
            validate_object_name(task_name)
            root_prefix = self._get_root_prefix()
            task_name = task_name.upper()
            if root_prefix and not task_name.startswith(root_prefix):
                task_name = f"{root_prefix}{task_name}"
            return str(self._session.call("system$get_predecessor_return_value", task_name))
        return str(self._session.call("system$get_predecessor_return_value"))

    def get_current_task_name(self) -> str:
        """Return the name of the task currently executing.

        This method can only be called in a Snowflake task.

        Example:
            >>> def task_handler(session: Session) -> None:
            >>>     from snowflake.core.task.context import TaskContext
            >>>     context = TaskContext(session)
            >>>     task_name = context.get_current_task_name()
        """
        if not self._current_task_name:
            self._current_task_name = str(self._session.call("system$current_user_task_name"))
        return self._current_task_name

    def get_current_task_short_name(self) -> str:
        """Return the task name under the context of a DAG.

        If a :class:`snowflake.core.task.dagv1.DAGTask` is created under a :class:`snowflake.core.task.dagv1.DAG`
        object, the real task name is in format ``{dag name}${task name}``. This API returns the ``task name`` part.
        :meth:`get_current_task_name` returns the real task.
        """
        root_prefix = self._get_root_prefix()
        name = self.get_current_task_name()
        name = name[name.rfind('.')+1:]
        if root_prefix and name.startswith(root_prefix):
            return name[len(root_prefix):]
        return name

    def get_task_graph_config(self) -> Optional[Dict[str, Any]]:
        """Return the graph config of the task."""
        config = self._session.call("SYSTEM$GET_TASK_GRAPH_CONFIG")
        if config:
            return json.loads(config)
        return None

    def get_task_graph_config_property(self, property_name: str) -> Optional[Any]:
        """Return the graph config of the task."""
        dag_config = self.get_task_graph_config()
        return dag_config.get(property_name) if dag_config else None

    def get_current_root_task_name(self) -> str:
        """Return the current task's root task name."""
        if not self._current_root_name:
            self._current_root_name = self.get_runtime_info("CURRENT_ROOT_TASK_NAME")  # type: ignore
        return self._current_root_name  # type: ignore

    def get_current_root_task_uuid(self) -> str:
        """Return the current task's root task UUID."""
        return self.get_runtime_info("CURRENT_ROOT_TASK_UUID")  # type: ignore

    def get_current_task_graph_original_schedule(self) -> str:
        """Return the current task's original schedule for this run or its root task's schedule."""
        return self.get_runtime_info("CURRENT_TASK_GRAPH_ORIGINAL_SCHEDULED_TIMESTAMP")  # type: ignore

    def get_current_task_graph_run_group_id(self) -> str:
        """Return the current task's run group."""
        return self.get_runtime_info("CURRENT_TASK_GRAPH_RUN_GROUP_ID")  # type: ignore

    def get_last_successful_task_graph_original_schedule(self) -> Optional[str]:
        """Return the last successful task run schedule."""
        return self.get_runtime_info("LAST_SUCCESSFUL_TASK_GRAPH_ORIGINAL_SCHEDULED_TIMESTAMP")  # type: ignore

    def get_last_successful_task_graph_run_group_id(self) -> Optional[str]:
        """Return the last successful task run group id."""
        return self.get_runtime_info("LAST_SUCCESSFUL_TASK_GRAPH_RUN_GROUP_ID")  # type: ignore

    def get_runtime_info(self, property_name: str) -> Optional[Union[str, datetime]]:
        """Return the runtime information of the current task.

        You usually don't need to call this function. Call the other ``get_*`` functions instead.
        """
        if not property_name:
            raise ValueError("`property_name` must be an non-empty str.")
        property_name_upper = property_name.upper()

        try:
            if property_name in (
                "CURRENT_TASK_GRAPH_ORIGINAL_SCHEDULED_TIMESTAMP",
                "LAST_SUCCESSFUL_TASK_GRAPH_ORIGINAL_SCHEDULED_TIMESTAMP"
            ):
                result = self._session.sql(
                    f"select to_timestamp(system$task_runtime_info('{property_name_upper}'))").collect()[0][0]
            else:
                result = self._session.sql(
                    f"select to_char(system$task_runtime_info('{property_name_upper}'))").collect()[0][0]
            return result
        except SnowparkSQLException as sse:
            if "NULL result in a non-nullable column" in sse.message:
                return None
            raise sse


__all__ = [
    "TaskContext",
]
