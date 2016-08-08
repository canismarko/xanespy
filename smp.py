# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of scimap.
#
# Scimap is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Scimap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

"""Everything related to parallel processing using the multiprocess
module."""

from queue import Empty
import multiprocessing as mp
from time import time

from tqdm import tqdm

from utilities import prog


class Consumer(mp.Process):
    """A process that will be spawned and then start emptying the queue
    until killed.
    """
    def __init__(self, target, task_queue, result_queue, **kwargs):
        ret = super().__init__(target=target, **kwargs)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.target = target
        return ret

    def run(self):
        """Retrieve and process a frame from the queue or exit if poison pill
        None is passed."""
        while True:
            payload = self.task_queue.get()
            if payload is None:
                # Poison pill, so exit
                self.task_queue.task_done()
                break
            result = self.target(payload)
            self.result_queue.put(result)
            self.task_queue.task_done()


class Queue():
    """A joinable queue that adds objects to a queue and then spawns
    workers to process them."""

    def __init__(self, worker, totalsize, result_callback=None,
                 description="Processing data"):
        """Prepare the queue.

        Arguments
        ---------
        worker: a callable that can process an object in the queue.

        totalsize: how many objects are going to be put in the
        queue. This lets the queue know when to kill the workers and
        reclaim resources

        result_callback: A callable that will process the results. A
        worker will not have access to the parent's memoryspace, but
        result_callback will.

        description: The text to print in the progress bar during processing.
        """
        self.num_consumers = mp.cpu_count() * 2
        self.worker = worker
        self.result_queue = mp.Queue(maxsize=totalsize)
        self.result_callback = result_callback
        self.totalsize = totalsize
        self.results_left = totalsize
        self.description = description
        self.task_queue = mp.JoinableQueue(maxsize=self.num_consumers)
        self.start_time = time()
        # Create all the worker processes
        self.consumers = [Consumer(target=worker,
                                   task_queue=self.task_queue,
                                   result_queue=self.result_queue)
                          for i in range(self.num_consumers)]
        for consumer in self.consumers:
            consumer.start()

    def put(self, obj, *args, **kwargs):
        """Add an object to the queue for processing. This method first checks
        if it the results queue has an object that can be processed to
        avoid excessive memory usage.
        """
        # Check for results to take out of the queue
        try:
            result = self.result_queue.get(block=False)
        except Empty:
            pass
        else:
            self.process_result(result)
        return self.task_queue.put(obj, *args, **kwargs)

    def process_result(self, result):
        """Pull a result from the results queue and run it through the
        result_callback that was provided with the
        constructor. Returns the processed result.
        """
        if self.result_callback is not None:
            ret = self.result_callback(result)
        else:
            ret = result
        self.results_left -= 1
        curr = self.totalsize - self.results_left
        # Prepare a status bar
        if not prog.quiet:
            status = tqdm.format_meter(n=curr,
                                       total=self.totalsize,
                                       elapsed=time() - self.start_time,
                                       prefix=self.description + ": ")
            print("\r" + status, end='')
        return ret

    def join(self):
        """Wait for all the workers to finish emptying the queue and then
        return."""
        # Send poison pill to all consumers
        for i in range(self.num_consumers):
            self.task_queue.put(None)
        ret = self.task_queue.join()
        # Finish emptying the results queue
        while self.results_left > 0:
            result = self.result_queue.get()
            self.process_result(result)
        if not prog.quiet:
            # Blank line to avoid overwriting results
            print("")
        return ret


class MockQueue(Queue):
    """A joinable queue that processes the results without
    multiprocessing. Usefuly for debugging exceptions inside a worker."""
    _counter = 0

    def __init__(self, worker, totalsize, result_callback=None,
                 description="Processing data"):
        self.worker = worker
        self.totalsize = totalsize
        self.result_callback = result_callback
        self.description = description
        self.start_time = time()

    def put(self, obj):
        # Just call the worker and result callbacks
        result = self.worker(obj)
        if self.result_callback:
            self.result_callback(result)
        if not prog.quiet:
            status = tqdm.format_meter(n=self._counter,
                                       total=self.totalsize,
                                       elapsed=time() - self.start_time,
                                       prefix=self.description + ": ")
            print('\r', status, end='')
        self._counter += 1

    def join():
        pass
