# Copyright (c) 2012-2023 Snowflake Computing Inc. All rights reserved.

"""Manages Snowflake Tasks. Resource in Public Preview.

Example:
    >>> tasks: TaskCollection = root.databases["mydb"].schemas["myschema"].tasks
    >>> mytask = tasks.create(Task("mytask", definition="select 1"))
    >>> task_iter = tasks.iter(like="my%")
    >>> mytask = tasks["mytask"]
    >>> # Then call other APIs to manage this task.
    >>> mytask.resume()
    >>> mytask.suspend()
    >>> an_existing_task = tasks["an_existing_task"]
    >>> an_existing_task.suspend()

Refer to :class:`snowflake.core.Root` to create the ``root``.

"""

from public import public

from ._generated.models.task_run import TaskRun
from ._task import Cron, StoredProcedureCall, Task, TaskCollection, TaskResource


public(
    Cron=Cron,
    StoredProcedureCall=StoredProcedureCall,
    Task=Task,
    TaskCollection=TaskCollection,
    TaskResource=TaskResource,
    TaskRun=TaskRun,
)
