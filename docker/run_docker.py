import subprocess
import time
import os
import signal
from multiprocessing import Pool
import errno
from contextlib import contextmanager
IMAGE_NAME = "orr_nadav_rl_base_2"
TIME = 3 * 60  # (3 min)
from functools import wraps


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def iteration(index):

    print(index)
    # temporary - need to change by the interactions
    name_of_container = "orr_nadav_rl_{0}".format(index)

    # get correct time
    my_location = os.path.dirname(os.path.abspath(__file__))

    # stop
    command_stop = "docker rm --force {0}".format(name_of_container)
    container_stop = subprocess.Popen([command_stop],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, shell=True)
    time.sleep(10)
    command_init = "docker run -v {0}/share_folder/catkin_ws/:/root/catkin_ws -i --name {1} {2} sleep {3}; sleep 10; docker rm - f {4}".\
        format(my_location,name_of_container,IMAGE_NAME,TIME,name_of_container)
    container_init = subprocess.Popen([command_init],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True)


    print("succee to run the container {0}".format(name_of_container))
    time.sleep(10)
    command_run_simulation = "docker exec -i {0} bash ./root/catkin_ws/simulator.sh".format(name_of_container)
    container_run_simulation = subprocess.Popen([command_run_simulation],
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, shell=True)
    time.sleep(5)
    command_run_agent = "docker exec -i {0} bash ./root/catkin_ws/agent.sh {1}".format(name_of_container, index)
    container_run_agent = subprocess.Popen([command_run_agent],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, shell=True)
    time.sleep(5)
    # complete the action
    time.sleep(TIME)

    # stop
    container_stop = subprocess.Popen([command_stop],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, shell=True)


def main():
    array = [i for i in range(1,4)]
    p = Pool()
    p.map(iteration, array)

    array = [i for i in range(1, 4)]
    p = Pool()
    p.map(iteration, array)

    array = [i for i in range(1, 4)]
    p = Pool()
    p.map(iteration, array)

    array = [i for i in range(1, 4)]
    p = Pool()
    p.map(iteration, array)


if __name__ == '__main__':
    main()
    # iteration(2)

