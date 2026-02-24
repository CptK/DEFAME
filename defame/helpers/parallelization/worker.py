import traceback
from multiprocessing import Process
from multiprocessing.managers import SyncManager
from pathlib import Path
from queue import Empty
from time import sleep

import torch

from defame.common import logger, Content, Claim
from defame.fact_checker import FactChecker
from defame.helpers.common import TaskState


def move_to_cpu(obj, visited=None):
    """Recursively move all CUDA tensors in an object to CPU.

    This is necessary to avoid serialization errors when passing
    objects between processes via multiprocessing queues.
    """
    # Track visited objects to avoid infinite recursion
    if visited is None:
        visited = set()

    # Use id() to track objects we've already processed
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    if isinstance(obj, torch.Tensor):
        return obj.cpu() if obj.is_cuda else obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = move_to_cpu(v, visited)
        return obj
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = move_to_cpu(obj[i], visited)
        return obj
    elif isinstance(obj, tuple):
        # Tuples are immutable, so we need to create a new one
        moved = [move_to_cpu(item, visited) for item in obj]
        return type(obj)(moved)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by moving tensors in their __dict__ directly
        # This catches instance variables more reliably than dir()
        obj_dict = obj.__dict__
        for attr_name, attr_value in list(obj_dict.items()):
            if isinstance(attr_value, torch.Tensor):
                if attr_value.is_cuda:
                    obj_dict[attr_name] = attr_value.cpu()
            elif isinstance(attr_value, (dict, list, tuple)) or hasattr(attr_value, '__dict__'):
                move_to_cpu(attr_value, visited)
        return obj
    else:
        return obj


def clean_task_for_serialization(task):
    """Move all CUDA tensors in a task to CPU before queue serialization."""
    # Clean the payload
    if task.payload is not None:
        task.payload = move_to_cpu(task.payload)
    # Clean the result
    if task.result is not None:
        task.result = move_to_cpu(task.result)
    return task


class Worker(Process):
    """Completing tasks in parallel."""
    id: int

    def __init__(self, identifier: int, kwargs: dict):
        super().__init__(target=self.execute, kwargs=kwargs)
        self.id = identifier
        self.running = True
        self.start()

    def execute(self,
                input_queue: SyncManager.Queue,
                output_queue: SyncManager.Queue,
                device_id: int,
                target_dir: str | Path,
                print_log_level: str = "info",
                **kwargs):
        """Main task handling routine."""

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

            # Clean any tensors that might have come from another process
            # (This handles the case where a payload has tensors from previous processing)
            task = clean_task_for_serialization(task)

            try:
                task.assign_worker(self)
                output_queue.put(clean_task_for_serialization(task))

                logger.set_current_fc_id(task.id)

                # Check which type of task this is
                payload = task.payload

                if isinstance(payload, Content):
                    # Task is claim extraction
                    logger.debug("Extracting claims...")
                    task.set_status(message="Extracting claims...")
                    output_queue.put(clean_task_for_serialization(task))

                    claims = fc.extract_claims(payload)

                    task.result = dict(
                        claims=claims,
                        topic=payload.topic
                    )

                elif isinstance(payload, Claim):
                    # Task is claim verification
                    logger.debug("Verifying claim...")
                    task.set_status(message="Verifying claim...")
                    output_queue.put(clean_task_for_serialization(task))

                    doc, meta = fc.verify_claim(payload)
                    doc.save_to(logger.target_dir)

                    task.result = (doc, meta)

                else:
                    raise ValueError(f"Invalid task type: {type(payload)}")

                # Move any CUDA tensors to CPU before sending across process boundary
                task.set_status(state=TaskState.DONE, message="Finished successfully.")
                output_queue.put(clean_task_for_serialization(task))
                logger.debug(f"Task {task.id} completed successfully.")

            except Exception:
                error_message = f"Worker {self.id} encountered an error while processing task {task.id}:\n"
                error_message += traceback.format_exc()
                logger.error(error_message)
                task.set_status(state=TaskState.FAILED, message=error_message)
                output_queue.put(clean_task_for_serialization(task))
