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
    def __init__(self, address: str, port: int) -> None:  # noqa: D107
        self._connection = redis.from_url(address + ":" + str(port))
        self._queue = _Queue("high", connection=self._connection, default_timeout=-1)

    def ready(self) -> bool:
        # Check if at least one worker is in use:
        workers = self.get_workers()

        if len(workers) > 0:
            return True

        return False

    def get_worker(self, worker_name) -> Worker:
        """
        Retrieve a Worker from a name.

        Parameters
        ----------
        worker_name
            The name of the worker.

        Returns
        -------
        Worker
            The worker object corresponding to the worker name.

        Raises
        ------
        ValueError
            If the Worker object associated with the given name cannot be found.
        """
        for worker in self.get_workers():
            if worker.name == worker_name:
                return worker

        raise ValueError("Worker not found.")

    def get_workers(self):
        """Get the workers in the queue."""
        return Worker.all(queue=self._queue)

    def is_processed(self, job_id: str) -> bool:
        """
        Check if the job is or was processed.

        This means it is either running, pending or finished.

        Parameters
        ----------
        job_id : str
            The ID of the job being checked.

        Returns
        -------
        bool
            True, if the job is or was processed.
            Otherwise False.
        """
        if self.is_running(job_id) or self.is_pending(job_id) or self.is_finished(job_id):
            return True

        return False

    def is_running(self, job_id: str) -> bool:
        """
        Check if the job is running.

        Parameters
        ----------
        job_id : str
            The ID of the job being checked.

        Returns
        -------
        bool
            True, if the job is running.
            Otherwise False.
        """
        for id in self._queue.started_job_registry.get_job_ids():
            if job_id == id:
                return True

        return False

    def is_pending(self, job_id: str) -> bool:
        """
        Check if the job is pending in a queue.

        Parameters
        ----------
        job_id : str
            The ID of the job being checked.

        Returns
        -------
        bool
            True, if the job is pending.
            Otherwise False.
        """
        for id in self._queue.get_job_ids():
            if job_id == id:
                return True

        return False

    def is_finished(self, job_id: str) -> bool:
        """
        Check if the job is finished.

        Parameters
        ----------
        job_id : str
            The ID of the job being checked.

        Returns
        -------
        bool
            True, if the job is finished.
            Otherwise False.
        """
        for id in self._queue.finished_job_registry.get_job_ids():
            if job_id == id:
                return True

        return False

    def has_failed(self, job_id: str) -> bool:
        """
        Check if a job has failed in a registry.

        Parameters
        ----------
        job_id : str
            The ID of the job being checked.

        Returns
        -------
        bool
            True, if the job has failed.
            Otherwise False.
        """
        for id in self._queue.failed_job_registry.get_job_ids():
            if job_id == id:
                return True

        return False

    def get_job(self, job_id: str) -> Job:
        """
        Get the job fitting the given job ID.

        Parameters
        ----------
        job_id : str
            The ID of the job.

        Returns
        -------
        Job
            The job Object to the job ID.
        """
        return Job.fetch(job_id, connection=self._connection)

    def get_jobs(self, registry: str = "running") -> List[Job]:
        """
        Retrieve list of jobs from the registry.

        Parameters
        ----------
        registry : str, optional
            The registry to retrieve jobs from.
            Default is "running".

        Raises
        ------
        NotImplementedError
            If the registry value is not recognized.
        """
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
        """Get the running jobs in the registry."""
        return self.get_jobs(registry="running")

    def get_pending_jobs(self) -> List[Job]:
        """Get the pending jobs in the registry."""
        return self.get_jobs(registry="pending")

    def get_finished_jobs(self) -> List[Job]:
        """Get the finished jobs in the registry."""
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
        """Delete the jobs in the queue."""
        self.delete_job()

    def enqueue(
        self, func: Callable[[Any], Any], args: Any, job_id: str, meta: Dict[str, str]
    ) -> None:
        """
        Add a job for processing to the queue.

        Logs a debug message if the job ID is already in use.

        Parameters
        ----------
        func : Callable[[Any], Any]
            A function to be executed, once the job is done.
        args : Any
            Arguments to be passed to func.
        job_id : str
            The identifier for the job.
        meta : Dict[str, str]
            A dictionary containing the jobs metadata.
        """
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
        """If function is not found, make sure we access self._queue directly."""
        try:
            return self.__getattribute__(name)
        except Exception:
            return self._queue.__getattribute__(name)
