from typing import Any, Callable

import redis
from rq import Queue as _Queue
from rq import Worker
from rq.job import Job

from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class Queue:
    def __init__(self, address: str, port: int):
        self._connection = redis.from_url(address + ":" + str(port))
        self._queue = _Queue("high", connection=self._connection)

    def ready(self):
        # Check if at least one worker is in use:
        workers = Worker.all(queue=self._queue)

        if len(workers) > 0:
            return True

        return False

    def is_processed(self, job_id):
        if (
            self.is_running(job_id)
            or self.is_pending(job_id)
            or self.is_finished(job_id)
        ):
            return True

        return False

    def is_running(self, job_id):
        for id in self._queue.started_job_registry.get_job_ids():
            if job_id == id:
                return True

        return False

    def is_pending(self, job_id):
        for id in self._queue.get_job_ids():
            if job_id == id:
                return True

        return False

    def is_finished(self, job_id):
        for id in self._queue.finished_job_registry.get_job_ids():
            if job_id == id:
                return True

        return False

    def get_jobs(self, registry="running") -> list[Job]:
        if registry == "running":
            registry = self._queue.started_job_registry
        elif registry == "pending":
            registry = self._queue
        elif registry == "finished":
            registry = self._queue.finished_job_registry
        else:
            raise NotImplementedError()

        results = []
        for job_id in registry.get_job_ids():
            job = Job.fetch(job_id, connection=self._connection)
            results.append(job)

        return results

    def get_running_jobs(self) -> list[Job]:
        return self.get_jobs(registry="running")

    def get_pending_jobs(self) -> list[Job]:
        return self.get_jobs(registry="pending")

    def get_finished_jobs(self) -> list[Job]:
        return self.get_jobs(registry="finished")

    def delete_job(self, job_id: str):
        registries = [
            self._queue.finished_job_registry,
            self._queue,
            self._queue.started_job_registry,
        ]

        for r in registries:
            try:
                r.remove(job_id, delete_job=True)
            except:
                pass

    def enqueue(
        self, func: Callable[[Any], Any], args: Any, job_id: str, meta: dict[str, str]
    ):
        # First check if job_id is already in use
        if self.is_processed(job_id):
            logger.debug("Job was not added because it was processed already.")
            return

        self._queue.enqueue(
            func,
            args=args,
            job_id=job_id,
            meta=meta,
            result_ttl=-1,  # Make sure it's not automatically deleted.
        )

    def __getattr__(self, name):
        """
        If function is not found, make sure we access self._queue directly.
        """

        try:
            return self.__getattribute__(name)
        except:
            return self._queue.__getattribute__(name)
