import multiprocessing
import multiprocessing.connection
import threading

from random import randint

from neroRRL.environments.wrapper import wrap_environment

def worker_process(remote: multiprocessing.connection.Connection, env_seed, env_config, worker_id: int, record_video = False, expert = None):
    """Initializes the environment and executes its interface.

    Arguments:
        remote {multiprocessing.connection.Connection} -- Parent thread
        env_seed {int} -- Sampled seed for the environment worker to use
        env_config {dict} -- The configuration data of the desired environment
        worker_id {int} -- Id for the environment's process. This is necessary for Unity ML-Agents environments, because these operate on different ports.
    """
    import numpy as np
    np.random.seed(env_seed)
    import random
    random.seed(env_seed)
    random.SystemRandom().seed(env_seed)

    # Initialize and wrap the environment
    try:
        env = wrap_environment(env_config, worker_id, record_trajectory = record_video, expert=expert)
    except KeyboardInterrupt:
        pass

    # Communication interface of the environment thread
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset(data))
            elif cmd == "close":
                remote.send(env.close())
                remote.close()
                break
            elif cmd == "video":
                remote.send(env.get_episode_trajectory)
            else:
                raise NotImplementedError
        except Exception as e:
            raise WorkerException(e)

class Worker:
    """A worker that runs one thread and controls its own environment instance."""
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process
    
    def __init__(self, env_config, worker_id: int, record_video = False, expert = None):
        """
        Arguments:
            env_config {dict -- The configuration data of the desired environment
            worker_id {int} -- worker_id {int} -- Id for the environment's process. This is necessary for Unity ML-Agents environments, because these operate on different ports.
        """
        env_seed = randint(0, 2 ** 32 - 1)
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, env_seed, env_config, worker_id, record_video, expert))
        self.process.start()

    def close(self):
        self.child.send(("close", None))
        self.child.recv()
        self.process.join()
        self.process.terminate()


import tblib.pickling_support
tblib.pickling_support.install()
import sys
class WorkerException(Exception):
    def __init__(self, ee):
        self.ee = ee
        __,  __, self.tb = sys.exc_info()
        super(WorkerException, self).__init__(str(ee))

    def re_raise(self):
        raise (self.ee, None, self.tb)
    
class WorkerList:
    def __init__(self, worker_id, configs, n_workers, expert):
        """
        Arguments:
            worker_id (_type_): _description_
            configs (_type_): _description_
            expert (_type_): _description_
        """
        # The number of workers
        self.n_workers = n_workers
        # The list of environments which should be reset
        self.envs_to_be_reset = [None for w in range(self.n_workers)]
        # The condition variable to wait on
        self.condition = threading.Condition()
        # Launch workers
        self.workers = [Worker(configs, worker_id + 200 + w, expert=expert) for w in range(self.n_workers)]
        # Reset workers which should be swapped
        for worker in self.workers:
            worker.child.send(("reset", None))
        self.reset_obs = [worker.child.recv() for worker in self.workers]
        self._is_filled = False

        # Create the thread
        self.thread = threading.Thread(target=self.wait_and_reset)
        self.thread.start()
  
    @property
    def is_filled(self):
        return self._is_filled  # Adjust the condition as needed
    
    def reset(self, worker, w):
        # Set the flag
        self._is_filled = True
        # Return the buffered rested worker and observation
        with self.condition:
            # Put the worker in the list of workers to be reset
            self.envs_to_be_reset[w] = worker
            # Notify all waiting threads
            self.condition.notify_all() 
            
        return self.workers[w], self.reset_obs[w]
    
    def wait_and_reset(self):
        with self.condition:
            while True:
                while not self.is_filled:
                    self.condition.wait()  # Wait until the list is filled

                # Execute the function
                for w in range(self.n_workers):
                    if self.envs_to_be_reset[w] is not None:
                        self.envs_to_be_reset[w].child.send(("reset", None))
                        self.reset_obs[w] = self.envs_to_be_reset[w].child.recv()
                        self.workers[w] = self.envs_to_be_reset[w]
                        self.envs_to_be_reset[w] = None
                
                # Reset the condition
                self._is_filled = False
    
    def close(self):
        self.thread.join()