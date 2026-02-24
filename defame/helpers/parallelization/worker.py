import traceback
from multiprocessing import Pipe, Process
from queue import Empty
from multiprocessing.connection import Connection
from pathlib import Path
from queue import Empty
from time import sleep

from defame.common import logger, Content, Claim
from defame.fact_checker import FactChecker
from defame.helpers.common import TaskState


class Worker(Process):
    """Completing tasks in parallel."""
    id: int

    def __init__(self, identifier: int, kwargs: dict):
        super().__init__(target=self.execute, kwargs=kwargs)
        self.id = identifier
        self.running = True
        self.start()

    def task_updates(self) -> dict:
        while self._connection.poll():
            yield self._connection.recv()

    def get_messages(self):
        msgs = []
        try:
            while self._connection.poll():
                msgs.append(self._connection.recv())
        except EOFError:
            msgs.append(dict(worker_id=self.id,
                             status=Status.FAILED,
                             status_message=f"Meta connection of worker {self.id} closed unexpectedly."))
            self.terminate()
        return msgs

    # Note: Do NOT override __getstate__ as it breaks Process pickling in spawn mode


class FactCheckerWorker(Worker):
    def __init__(self, identifier: int, *args, **kwargs):
        self.runner = Runner(worker_id=identifier)
        super().__init__(identifier, *args, target=self.runner.execute, **kwargs)


class Runner:
    # TODO: Refactor
    # TODO: Use multiprocessing.Manager (instead of Queues/Connection) to share data between pool & worker
    """The instance actually executing the routine inside the worker subprocess."""

    def __init__(self, worker_id: int | None = None):
        self.running = True
        self.worker_id = worker_id

    def stop(self, signum, frame):
        logger.info(f"Runner of worker {self.worker_id} received termination signal. Stopping gracefully...")
        self.running = False

    def execute(
        self,
        input_queue,
        output_queue,
        connection: Connection,
        device_id: int,
        target_dir: str | Path,
        print_log_level: str = "info",
        **kwargs
    ):
        # Register signal handler for graceful termination
        # signal.signal(signal.SIGTERM, self.stop)
        # signal.signal(signal.SIGINT, self.stop)

        def report(status_message: str, **kwargs):
            task_id = task.id if locals().get("task") else None
            connection.send(dict(worker_id=self.worker_id,
                                 task_id=task_id,
                                 status_message=status_message,
                                 **kwargs))

        try:
            device = None if device_id is None else f"cuda:{device_id}"

            logger.set_experiment_dir(target_dir)
            logger.set_log_level(print_log_level)

            # Initialize the fact-checker
            fc = FactChecker(device=device, **kwargs)

        except Exception:
            error_message = f"Worker {self.id} encountered an error during startup:\n"
            error_message += traceback.format_exc()
            logger.error(error_message)
            quit(-1)

        # Complete tasks forever
        while self.running:
            # Fetch the next task and report it
            try:
                task = input_queue.get(block=False)
            except Empty:
                sleep(0.1)
                continue

            try:
                task.assign_worker(self)
                output_queue.put(task)

                logger.set_current_fc_id(task.id)

                # Check which type of task this is
                payload = task.payload

                if isinstance(payload, Content):
                    # Task is claim extraction
                    logger.debug("Extracting claims...")
                    task.set_status(message="Extracting claims...")
                    output_queue.put(task)

                    claims = fc.extract_claims(payload)

                    task.result = dict(
                        claims=claims,
                        topic=payload.topic
                    )

                elif isinstance(payload, Claim):
                    # Task is claim verification
                    logger.debug("Verifying claim...")
                    task.set_status(message="Verifying claim...")
                    output_queue.put(task)

                    doc, meta = fc.verify_claim(payload)
                    doc.save_to(logger.target_dir)

                    task.result = (doc, meta)

                else:
                    raise ValueError(f"Invalid task type: {type(payload)}")

                task.set_status(state=TaskState.DONE, message="Finished successfully.")
                output_queue.put(task)
                logger.debug(f"Task {task.id} completed successfully.")

            except Exception:
                error_message = f"Worker {self.id} encountered an error while processing task {task.id}:\n"
                error_message += traceback.format_exc()
                logger.error(error_message)
                task.set_status(state=TaskState.FAILED, message=error_message)
                output_queue.put(task)
