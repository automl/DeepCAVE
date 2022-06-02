from typing import Any, Callable, Dict, List

import redis
from rq import Queue as _Queue
from rq import Worker
from rq.command import send_kill_horse_command
from rq.job import Job
from rq.registry import BaseRegistry

from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class Queue:
    def __init__(self, address: str, port: int) -> None:
        self._connection = redis.from_url(address + ":" + str(port))
        self._queue = _Queue("high", connection=self._connection, default_timeout=-1)

    def ready(self) -> bool:
        # Check if at least one worker is in use:
        workers = self.get_workers()

        if len(workers) > 0:
            return True

        return False

    def get_worker(self, worker_name) -> Worker:
        for worker in self.get_workers():
            if worker.name == worker_name:
                return worker

        raise ValueError("Worker not found.")

    def get_workers(self):
        return Worker.all(queue=self._queue)

    def is_processed(self, job_id: str) -> bool:
        if self.is_running(job_id) or self.is_pending(job_id) or self.is_finished(job_id):
            return True

        return False

    def is_running(self, job_id: str) -> bool:
        for id in self._queue.started_job_registry.get_job_ids():
            if job_id == id:
                return True

        return False

    def is_pending(self, job_id: str) -> bool:
        for id in self._queue.get_job_ids():
            if job_id == id:
                return True

        return False

    def is_finished(self, job_id: str) -> bool:
        for id in self._queue.finished_job_registry.get_job_ids():
            if job_id == id:
                return True

        return False

    def has_failed(self, job_id: str) -> bool:
        for id in self._queue.failed_job_registry.get_job_ids():
            if job_id == id:
                return True

        return False

    def get_job(self, job_id: str) -> Job:
        return Job.fetch(job_id, connection=self._connection)

    def get_jobs(self, registry: str = "running") -> List[Job]:
        if registry == "running":
            r = self._queue.started_job_registry
        elif registry == "pending":
            r = self._queue
        elif registry == "finished":
            r = self._queue.finished_job_registry
        elif registry == "failed":
            r = self._queue.failed_job_registry
        else:
            raise NotImplementedError()

        results = []
        for job_id in r.get_job_ids():
            job = self.get_job(job_id)
            results.append(job)

        return results

    def get_running_jobs(self) -> List[Job]:
        return self.get_jobs(registry="running")

    def get_pending_jobs(self) -> List[Job]:
        return self.get_jobs(registry="pending")

    def get_finished_jobs(self) -> List[Job]:
        return self.get_jobs(registry="finished")

    def delete_job(self, job_id: str = None) -> None:
        """
        Delete a job from the queue. If no job_id is given, delete all jobs.

        Parameters
        ----------
        job_id : str, optional
            Id of the job, which should be removed. By default None.
        """

        def remove_jobs(registry: BaseRegistry, job_id: str = None) -> None:
            if job_id is not None:
                try:
                    registry.remove(job_id, delete_job=True)
                except Exception:
                    pass
            else:
                # Remove all
                for job_id in registry.get_job_ids():
                    try:
                        registry.remove(job_id, delete_job=True)
                    except Exception:
                        registry.remove(job_id)

        remove_jobs(self._queue, job_id)
        remove_jobs(self._queue.finished_job_registry, job_id)
        remove_jobs(self._queue.canceled_job_registry, job_id)

        # Started jobs perform differently
        # We have to "kill" the worker
        for worker in self.get_workers():
            job_id_ = str(worker._job_id)  # b'cbece'
            job_id_ = job_id_.replace("b'", "").replace("'", "")

            if job_id is not None:
                if job_id != job_id_:
                    continue

            if worker.state == "busy":
                # This will add the job to the failed registry
                send_kill_horse_command(self._connection, worker.name)

        # Iterate again and wait till all jobs have finished
        for worker in self.get_workers():
            job_id_ = str(worker._job_id)  # b'cbece'
            job_id_ = job_id_.replace("b'", "").replace("'", "")

            if job_id is not None:
                if job_id != job_id_:
                    continue

            # Wait till the job is actually cancled
            while worker.state == "busy":
                # We need to "update" the worker every time
                worker = self.get_worker(worker.name)

        # Now we also have to remove the failed jobs caused by the kill horse command
        remove_jobs(self._queue.failed_job_registry, job_id)

    def delete_jobs(self) -> None:
        self.delete_job()

    def enqueue(
        self, func: Callable[[Any], Any], args: Any, job_id: str, meta: Dict[str, str]
    ) -> None:
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
        except Exception:
            return self._queue.__getattribute__(name)
