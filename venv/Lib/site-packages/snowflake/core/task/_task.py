# Copyright (c) 2012-2023 Snowflake Computing Inc. All rights reserved.

import os

from datetime import datetime, timedelta
from logging import getLogger
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

from pydantic import StrictStr

from snowflake.core._common import (
    CreateMode,
    SchemaObjectCollectionParent,
    SchemaObjectReferenceMixin,
)
from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core.task._generated import (
    CronSchedule,
    MinutesSchedule,
    TaskApi,
    TaskRun,
    TaskSchedule,
)
from snowflake.core.task._generated.api_client import StoredProcApiClient
from snowflake.core.task._generated.models import (
    Task as TaskModel,
)
from snowflake.snowpark._internal.udf_utils import generate_call_python_sp_sql
from snowflake.snowpark._internal.utils import is_in_stored_procedure
from snowflake.snowpark.stored_procedure import StoredProcedure
from snowflake.snowpark.types import DataType


if TYPE_CHECKING:
    from snowflake.core.schema._schema import SchemaResource

_logger = getLogger(__name__)


TASK_CONTEXT_FILE_IMPORT = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "context.py"),
    "snowflake.core.task.context",
)


class Cron(NamedTuple):
    """Specifies a cron expression and time zone for periodically running the task.

    Supports a subset of standard cron utility syntax.

    Example:
        >>> cron1 = Cron("0 0 10-20 * TUE,THU", "America/Los_Angeles")
    """

    expr: str
    """The cron expression. The minimum interval is 1 minute.

     It consists of the following fields:

    .. code-block::

        # __________ minute (0-59)
        # | ________ hour (0-23)
        # | | ______ day of month (1-31, or L)
        # | | | ____ month (1-12, JAN-DEC)
        # | | | | _ day of week (0-6, SUN-SAT, or L)
        # | | | | |
        # | | | | |
          * * * * *

    The following special characters are supported:

        - ``*`` Wildcard. Specifies any occurrence of the field.

        - ``L`` Stands for “last”. When used in the day-of-week field, it allows you to specify constructs such as
          “the last Friday” (“5L”) of a given month. In the day-of-month field, it specifies the last day of the month.

        - ``/n`` Indicates the nth instance of a given unit of time. Each quanta of time is computed independently.
          For example, if 4/3 is specified in the month field, then the task is scheduled for April, July and October
          (i.e. every 3 months, starting with the 4th month of the year). The same schedule is maintained in subsequent
          years. That is, the task is not scheduled to run in January (3 months after the October run).

    Tasks scheduled during specific times on days when the transition from standard time to daylight saving time
    (or the reverse) occurs can have unexpected behaviors. For example:

      - During the autumn change from daylight saving time to standard time, a task scheduled to start at 1 AM in
        the America/Los_Angeles time zone (i.e. 0 1 * * * America/Los_Angeles) would run twice: once at 1 AM and then
        again when 1:59:59 AM shifts to 1:00:00 AM local time. That is, there are two points in time when the local
        time is 1 AM.
      - During the spring change from standard time to daylight saving time, a task scheduled to start at 2 AM in
        the America/Los_Angeles time zone (i.e. 0 2 * * * America/Los_Angeles) would not run at all because the
        local time shifts from 1:59:59 AM to 3:00:00 AM. That is, there is no point during that day when the local
        time is 2 AM.

    To avoid unexpected task executions due to daylight saving time, use one of the following:
      - Do not schedule tasks to run at a specific time between 1 AM and 3 AM (daily, or on days of the week that
        include Sundays), or
      - Manually adjust the cron expression for tasks scheduled during those hours twice each year to compensate for
        the time change due to daylight saving time, or
      - Use a time format that does not apply daylight savings time, such as UTC.
    """
    timezone: str
    """The timezone for the cron expression

    For a list of time zones, see the `list of tz database time zones in Wikipedia <https://en.wikipedia.org/wiki/List_of_tz_database_time_zones>`_.
    """

    def __eq__(self, other: "Cron") -> bool:  # type: ignore[override]
        return (
            self.expr.lower() == other.expr.lower()
            and self.timezone.lower() == other.timezone.lower()
        )


def _to_model_schedule(
    schedule: Optional[Union[Cron, timedelta]]
) -> Optional[TaskSchedule]:
    if schedule is None:
        return None
    if isinstance(schedule, Cron):
        return CronSchedule(cron_expr=schedule.expr, timezone=schedule.timezone)
    elif isinstance(schedule, timedelta):
        total_seconds = schedule.total_seconds()
        if (not total_seconds.is_integer() or total_seconds % 60 != 0) or not (
            60 <= total_seconds <= 11520 * 60
        ):
            raise ValueError(
                f"The schedule time delta must be a whole integer of minutes, greater than 1 minute and "
                f"less than 8 days (11520 minutes) but got {total_seconds / 60.0} minute(s)."
            )
        return MinutesSchedule(minutes=int(total_seconds / 60))
    raise TypeError("schedule should be either Cron or timedelta value")


def _from_model_schedule(
    schedule: Optional[TaskSchedule],
) -> Optional[Union[timedelta, Cron]]:
    if schedule is None:
        return None
    if isinstance(schedule, MinutesSchedule):
        return timedelta(minutes=schedule.minutes)
    elif isinstance(schedule, CronSchedule):
        return Cron(schedule.cron_expr, schedule.timezone)
    raise TypeError("schedule must be either a MinutesSchedule or CronSchedule. ")  # won't happen in reality.


class StoredProcedureCall:
    """Represents a procedure call used as a task's ``definition``.

    Args:
        func: When it's a ``Callable``, typically a function, an anonymous stored procedure will be created as
            the Task's definition by using this ``Callable``.
            Note that the first parameter of your function should be a snowpark Session.

            When it's a ``StoredProcedure``, it will be converted to a SQL to call an existing stored procedure.
            The ``StoredProcedure`` must be a permanent one instead of a temporary one because a Task will run
            in a different session than the session that creates the Task. A temporary one won't be accessible
            from that session that runs the Task.
        args: The arguments to call the stored procedure when ``func`` is a ``StoredProcedure``.
        return_type: A :class:`~snowflake.snowpark.types.DataType` representing the return data
            type of the stored procedure. Optional if type hints are provided.
        input_types: A list of :class:`~snowflake.snowpark.types.DataType`
            representing the input data types of the stored procedure. Optional if
            type hints are provided.
        stage_location: The stage location where the Python file for the anonymous stored procedure
            and its dependencies should be uploaded.
            It must be a permanent location because a Task will run in a different session than the session that
            creates the Task. A temporary one won't be accessible from that session that runs the Task.
        imports: A list of imports that only apply to this stored procedure. You can use a string to
            represent a file path (similar to the ``path`` argument in
            :meth:`~snowflake.snowpark.Session.add_import`) in this list, or a tuple of two
            strings to represent a file path and an import path (similar to the ``import_path``
            argument in :meth:`~snowflake.snowpark.Session.add_import`). These stored procedure-level imports
            will override the session-level imports added by
            :meth:`~snowflake.snowpark.Session.add_import`.
        packages: A list of packages that only apply to this stored procedure.
            These stored procedure-level packages will override the session-level packages added by
            :meth:`~snowflake.snowpark.Session.add_packages` and
            :meth:`~snowflake.snowpark.Session.add_requirements`.
    """

    def __init__(
        self,
        func: Union[Callable[..., Any], StoredProcedure],
        *,
        args: Optional[List[Any]] = None,
        return_type: Optional[DataType] = None,
        input_types: Optional[List[DataType]] = None,
        stage_location: Optional[str] = None,
        imports: Optional[List[Union[str, Tuple[str, str]]]] = None,
        packages: Optional[List[Union[str, ModuleType]]] = None,
    ) -> None:
        if (not isinstance(func, StoredProcedure)) and stage_location is None:
            raise ValueError(
                "stage_location has to be specified when func is a Python function. And it must NOT be a temp location."
            )
        self.func: Union[Callable[..., Any], StoredProcedure] = func
        self._args = args if args else []
        self._return_type = return_type
        self._input_types = input_types
        self._stage_location = stage_location
        self._imports = imports
        self._packages = packages
        self._sql: Optional[str] = None


class Task:
    """Represents a Snowflake Task."""

    def __init__(
        self,
        name: str,
        definition: Union[str, StoredProcedureCall],
        *,
        warehouse: Optional[str] = None,
        user_task_managed_initial_warehouse_size: Optional[str] = None,
        suspend_task_after_num_failures: Optional[int] = None,
        user_task_timeout_ms: Optional[int] = None,
        schedule: Optional[Union[Cron, timedelta]] = None,
        allow_overlapping_execution: Optional[bool] = None,
        error_integration: Optional[str] = None,
        comment: Optional[str] = None,
        predecessors: Optional[List[str]] = None,
        condition: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        session_parameters: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        created_on: Optional[datetime] = None,
        last_committed_on: Optional[datetime] = None,
        last_suspended_on: Optional[datetime] = None,
        state: Optional[str] = None,
        database_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        owner: Optional[str] = None,
        owner_role_type: Optional[str] = None,
    ) -> None:
        self.name: str = name  #: Name of the task.
        self.definition: Union[str, StoredProcedureCall] = definition
        """Definition of the task.

        Any one of the following:
          - A SQL Statement. It can be a single SQL statement, or a procedural logic using Snowflake Scripting.
          - A :class:`StoredProcedureCall` instance. This provides a Pythonic way to call an existing stored procedure,
            or use a Snowflake annonymous stored procedure in Python.

        """
        #: The virtual warehouse that provides compute resources for task runs. Omit this parameter if you want to use
        #: the serverless Snowflake-managed compute resources for runs of this task.
        self.warehouse = warehouse
        #: The size of the compute resources to provision for the first run of the task, before a task history is
        #: available for Snowflake to determine an ideal size. Once a task has successfully completed a few runs,
        #: Snowflake ignores this parameter setting. Note that if the task history is unavailable for a given task, the
        #: compute resources revert to this initial size.
        #:
        #: A warehouse size is the same as in `creating a virtual warehous
        #: <https://docs.snowflake.com/en/sql-reference/sql/create-warehouse>`_.
        #:
        #: If ``warehouse`` is specified for this task, then setting this parameter produces an error.
        self.user_task_managed_initial_warehouse_size = (
            user_task_managed_initial_warehouse_size
        )
        #: The schedule for periodically running the task. The minimum schedule is 1 minute.
        self.schedule = schedule
        #: Specifies the number of consecutive failed task runs after which the current task is suspended
        #: automatically. Failed task runs include runs in which task body either produces a user error or times
        #: out. Task runs that are skipped, canceled, or that fail due to a system error are considered indeterminate
        #: and are not included in the count of failed task runs.
        #:
        #: Set the parameter on a standalone task or the root task in a DAG. When the parameter is set to a value
        #: greater than 0, the following behavior applies to runs of the standalone task or DAG:
        #:
        #: Standalone tasks are automatically suspended after the specified number of consecutive task runs either fail
        #: or time out.
        #:
        #: The root task is automatically suspended after the run of any single task in a DAG fails or times out the
        #: specified number of times in consecutive runs.
        self.suspend_task_after_num_failures = suspend_task_after_num_failures
        #: Specifies the time limit on a single run of the task before it times out (in milliseconds).
        self.user_task_timeout_ms = user_task_timeout_ms
        #: Whether to allow multiple instances of the DAG to run concurrently.
        self.allow_overlapping_execution = allow_overlapping_execution
        #: Specifies the name of the notification integration used to communicate with Amazon SNS, MS Azure Event Grid,
        #: or Google Pub/Sub.
        #:
        #: For more information, refer to `Enabling Error Notifications for Tasks
        #: <https://docs.snowflake.com/en/user-guide/tasks-errors.html>`_.
        #:
        #: Required only when configuring a task to send error notifications using Amazon Simple Notification Service
        #: (SNS), Microsoft Azure Event Grid, or Google Pub/Sub.
        self.error_integration = error_integration
        #: Specifies a comment for the task.
        self.comment = comment
        #: Specifies one or more predecessor tasks for the current task. Use this option to create a DAG of tasks or add
        #: this task to an existing DAG.
        #:
        #: `A DAG <https://docs.snowflake.com/en/user-guide/tasks-intro#label-task-dag>`_ is a series of tasks that
        #: starts with a scheduled root task and is linked together by dependencies.
        self.predecessors = predecessors
        #: Specifies a Boolean SQL expression; multiple conditions joined with AND/OR are supported. When a task is
        #: triggered (based on its SCHEDULE or AFTER setting), it validates the conditions of the expression to
        #: determine whether to execute. If the conditions of the expression are not met, then the task skips the
        #: current run. Any tasks that identify this task as a predecessor also do not run.
        #:
        #: SYSTEM$STREAM_HAS_DATA is the only function supported for evaluation in the SQL expression. This function
        #: indicates whether a specified stream contains change tracking data.
        self.condition: Optional[str] = condition
        self.config = config
        """Set the configuration for the task. It can only be set on a root task then it applies to all tasks
        in the DAG.
        The parameter can be set on standalone tasks but does not affect the task behavior.
        Snowflake ensures only one instance of a standalone task is running at a given time.
        """

        self.session_parameters = session_parameters
        """Set the session parameters for the task at runtime."""

        # read-only properties

        #: Unique identifier for each task. Note that recreating a task essentially creates a new task, which has a new
        #  ID.
        self.id = id
        #: Date and time when the task was created.
        self.created_on = created_on
        #: Timestamp when a version of the task was last set. If no version has been set (i.e. if the task has not been
        #: resumed or manually executed after it was created), the value is NULL.
        self.last_committed_on = last_committed_on
        #: Timestamp when the task was last suspended. If the task has not been suspended yet, the value is NULL.
        self.last_suspended_on = last_suspended_on
        #: `"started"` or `"suspended"` based on the current state of the task.
        self.state = state
        #: Database in which the task is stored.
        self.database_name = database_name
        #: Schema in which the task is stored.
        self.schema_name = schema_name
        #: Role that owns the task (i.e. has the OWNERSHIP privilege on the task)
        self.owner = owner
        #: The type of role that owns the object, either ROLE or DATABASE_ROLE. Note that Snowflake returns NULL if you
        #: delete the object because there is no owner role for a deleted object.
        self.owner_role_type = owner_role_type

        # client only metadata
        self._partially_listed = False

    @property
    def sql_definition(self) -> str:
        """The definition of the task in SQL text.

        It's a readonly property. To set the ``definition`` of the ``Task``, use :attr:`definition`.

        If :attr:`definition` is a :class:`StoredProcedureCall`, the SQL that calls the stored procedure, or the
            anonymous stored procedure definition will be returned.
        """
        if isinstance(self.definition, str):
            return self.definition
        else:
            if self.definition._sql:
                return self.definition._sql
            else:
                raise ValueError(
                    "definition of this task can only be retrieved after creating the task"
                )

    @classmethod
    def _from_rest_model(cls, model: TaskModel) -> "Task":
        return Task(
            name=model.name,
            definition=model.definition,
            warehouse=model.warehouse,
            suspend_task_after_num_failures=model.suspend_task_after_num_failures,
            user_task_managed_initial_warehouse_size=model.user_task_managed_initial_warehouse_size,
            user_task_timeout_ms=model.user_task_timeout_ms,
            schedule=_from_model_schedule(model.schedule),
            allow_overlapping_execution=model.allow_overlapping_execution,
            error_integration=model.error_integration,
            comment=model.comment,
            predecessors=model.predecessors,
            condition=model.condition,
            config=model.config,
            session_parameters=model.session_parameters,
            id=model.id,
            created_on=model.created_on,
            last_committed_on=model.last_committed_on,
            last_suspended_on=model.last_suspended_on,
            state=model.state,
            database_name=model.database_name,
            schema_name=model.schema_name,
            owner=model.owner,
            owner_role_type=model.owner_role_type,
        )

    def _to_rest_model(self) -> TaskModel:
        for prop in ("config", "session_parameters"):
            attr_value = getattr(self, prop, None)
            if attr_value:
                for k, v in attr_value.items():
                    if not isinstance(v, (str, int, float, bool)):
                        raise TypeError(
                            f"Task.{prop} is a dict. The value of this dict must be one of str, int, float, or bool."
                            f"Found value type {type(v)} for key {k}")
        model = TaskModel(
            name=self.name,
            definition=self.sql_definition,
            warehouse=self.warehouse,
            user_task_managed_initial_warehouse_size=self.user_task_managed_initial_warehouse_size,
            suspend_task_after_num_failures=self.suspend_task_after_num_failures,
            user_task_timeout_ms=self.user_task_timeout_ms,
            schedule=_to_model_schedule(self.schedule),
            allow_overlapping_execution=self.allow_overlapping_execution,
            error_integration=self.error_integration,
            comment=self.comment,
            predecessors=self.predecessors,
            condition=self.condition,
            config=self.config,
            session_parameters=self.session_parameters,
            database_name=self.database_name,
            schema_name=self.schema_name,
            owner=self.owner,
            owner_role_type=self.owner_role_type,
            id=self.id,
            state=self.state,
            created_on=self.created_on,
            last_committed_on=self.last_committed_on,
            last_suspended_on=self.last_suspended_on,
        )
        return model


class TaskResource(SchemaObjectReferenceMixin["TaskCollection"]):
    """A reference to a specific Task resource in Snowflake.

    Example:
        >>> tasks: TaskCollection = root.databases["mydb"].schemas["myschema"].tasks
        >>> mytask = tasks["mytask"]
        >>> # Then call other APIs to manage this task.
        >>> mytask.resume()
        >>> mytask.suspend()
    """

    def __init__(self, name: str, collection: "TaskCollection") -> None:
        self.collection = collection
        self.name = name

    @api_telemetry
    def create_or_update(self, task: Task) -> None:
        """Create or update a task in the Snowflake database. Method is Experimental.

        The Snowflake task's properties will be updated to the properties of the input ``task`` if the task already
        exists.  Note that the full picture of a task is expected. If a property isn't set a value in the input
        ``task``, the property will be set to ``NULL`` in Snowflake too because it's regarded as the expected value.

        Args:
            task: an instance of :class:`Task`.

        Example:
            >>> from snowflake.core import Root
            >>> root = Root(session)
            >>> tasks: TaskCollection = root.databases["mydb"].schemas["myschema"].tasks
            >>> mytask = tasks.create(Task("mytask", definition="select 1"))
            >>> mytask_entity = mytask.fetch()
            >>> mytask_entity.definition = "select 2"
            >>> tasks["mytask"].create_or_update(mytask_entity)
        """
        if task._partially_listed:
            raise ValueError(
                "The input task is not a full picture of of the Task object. "
                "So `TaskResource.create_or_update()` can't be used on this task. Please use "
                "`TaskResource.fetch()` to get a full picture if you want to use `create_or_update` to update the task."
            )
        self.collection._extract_definition(task)
        task_model = task._to_rest_model()
        task_model.schedule = _to_model_schedule(task.schedule)
        self.collection._api.create_or_alter_task(
            database=self.database.name,
            var_schema=self.schema.name,
            task=task_model,
            name=task.name,
        )

    def delete(self) -> None:
        """Delete the task from Snowflake."""
        self.collection._api.delete_task(
            self.database.name,
            self.schema.name,
            self.name,
            False,
            async_req=False,
        )

    @api_telemetry
    def fetch(self) -> Task:
        """Fetch the task details from Snowflake."""
        rest_model = self.collection._api.fetch_task(
            self.database.name, self.schema.name, self.name, async_req=False
        )
        return Task._from_rest_model(rest_model)

    @api_telemetry
    def execute(self, *, retry_last: bool = False) -> None:
        """Execute the task immediately without waiting for the schedule.

        Args:
            retry_last: Re-execute the last failed task of the DAG.

        """
        self.collection._api.execute_task(
            self.database.name, self.schema.name, self.name, retry_last=retry_last, async_req=False
        )

    @api_telemetry
    def resume(self) -> None:
        """Resume the task then it will run on the schedule."""
        self.collection._api.resume_task(
            self.database.name, self.schema.name, self.name, async_req=False
        )

    @api_telemetry
    def suspend(self) -> None:
        """Suspend the task so it won't run again on the schedule."""
        self.collection._api.suspend_task(
            self.database.name, self.schema.name, self.name, async_req=False
        )

    @api_telemetry
    def fetch_task_dependents(self) -> List[Task]:
        """Return the list of child tasks that use this task as the root in a DAG."""
        return [
            _list_task_converter(x)
            for x in self.collection._api.fetch_task_dependents(
                self.database.name, self.schema.name, self.name, async_req=False
            )
        ]

    @api_telemetry
    def get_complete_graphs(self, *, error_only: bool = True) -> Iterable[TaskRun]:
        """Return the status of a completed graph run.

        It returns details for runs that executed successfully, failed, or were cancelled in the past 60 minutes.

        To retrieve the details for graph runs that are currently executing, or are next scheduled to run within the
        next 8 days, use :meth:`get_current_graphs`.
        """
        return self.collection._api.get_complete_graphs(
            self.database.name,
            self.schema.name,
            self.name,
            error_only=error_only,
            async_req=False,
        )

    @api_telemetry
    def get_current_graphs(self) -> Iterable[TaskRun]:
        """Return the status of a graph run that is currently scheduled or is executing.

        It returns details for graph runs that are currently executing or are next scheduled to run within the next 8
        days.  To retrieve the details for graph runs that have completed in the past 60 minutes, use
        :meth:`get_complete_graphs`.

        """
        return self.collection._api.get_current_graphs(
            self.database.name, self.schema.name, self.name, async_req=False
        )


class TaskCollection(SchemaObjectCollectionParent[TaskResource]):
    """Represents the collection operations of the Snowflake Task resource.

    Example:
        >>> tasks: TaskCollection = root.databases["mydb"].schemas["myschema"].tasks
        >>> mytask = tasks.create(Task("mytask", definition="select 1"))  # returns a TaskResource instance.
        >>> task_iter = tasks.iter(like="my%")
    """

    def __init__(self, schema: "SchemaResource") -> None:
        super().__init__(schema, TaskResource)
        self._api = TaskApi(
            root=self.root,
            resource_class=self._ref_class,
            sproc_client=StoredProcApiClient(root=self.root)
        )

    @api_telemetry
    def create(
        self,
        task: Task,
        *,
        mode: CreateMode = CreateMode.error_if_exists,
    ) -> TaskResource:
        """Create a task in the Snowflake database.

        Args:
            task: an instance of :class:`Task`.
            mode: One of the following strings.

                CreateMode.error_if_exists: Throw an :class:`snowflake.core.exceptions.ConflictError`
                if the task already exists in Snowflake.  Equivalent to SQL ``create task <name> ...``.

                CreateMode.or_replace: Replace if the task already exists in Snowflake.
                Equivalent to SQL ``create or replace task <name> ...``.

                CreateMode.if_not_exists: Do nothing if the task already exists in Snowflake.
                Equivalent to SQL ``create task <name> if not exists...``

                Default value is CreateMode.error_if_exists.
        """
        self._extract_definition(task)
        task_model = task._to_rest_model()
        real_mode = CreateMode[mode].value
        self._api.create_task(
            self.database.name, self.schema.name, task_model, real_mode, async_req=False
        )
        return self[task.name]

    @api_telemetry
    def iter(
        self,
        *,
        like: Optional[str] = None,
        starts_with: Optional[str] = None,
        limit: Optional[int] = None,
        root_only: bool = False,
    ) -> Iterator[Task]:
        """Search for tasks in Snowflake.

        Args:
            like: The pattern of the Task name. Use ``%`` to represent any number of characters and ``?`` for a
                single character.
            startswith: The task name starts with this string.
            root_only: Look for root tasks only.
        """
        tasks = self._api.list_tasks(
            self.database.name,
            self.schema.name,
            root_only,
            StrictStr(like) if like is not None else None,
            StrictStr(starts_with) if starts_with else None,
            limit,
            async_req=False,
        )

        return map(_list_task_converter, iter(tasks))

    def _extract_definition(self, task: Task) -> None:
        definition = task.definition
        if isinstance(definition, StoredProcedureCall):
            if not task.warehouse:
                raise ValueError(
                    "warehouse must be specified when creating a task with a StoredProcedureCall object"
                )
            if isinstance(definition.func, StoredProcedure):
                sproc_obj = definition.func
            else:
                imports = definition._imports if definition._imports else []
                if not is_in_stored_procedure():
                    imports.append(TASK_CONTEXT_FILE_IMPORT)
                sproc_obj = self._session.sproc.register(
                    definition.func,
                    name="task_handler_sp",
                    return_type=definition._return_type,
                    input_types=definition._input_types,
                    stage_location=definition._stage_location,
                    imports=imports,
                    packages=definition._packages,
                    anonymous=True,
                    is_permanent=True,
                )
            sp_sql = generate_call_python_sp_sql(
                self._session, sproc_obj.name, *definition._args
            )
            if sproc_obj._anonymous_sp_sql:
                sp_sql = f"{sproc_obj._anonymous_sp_sql}{sp_sql}"
            definition._sql = sp_sql


def _list_task_converter(rest_model: TaskModel) -> Task:
    task = Task._from_rest_model(rest_model)
    task._partially_listed = True
    return task
