import subprocess
import time
import os
from multiprocessing import Pool

IMAGE_NAME = "orr_nadav_rl_base"
TIME = 7 * 60  # (7 min)


def iteration(index):

    print(index)
    # temporary - need to change by the interactions
    name_of_container = "orr_nadav_rl_{0}".format(index)

    # command_init = "docker run -i --name {0} {1}".format(name_of_container, name_of_container)

    # command_init = "docker run -i --name {0} {1}".format(name_of_container, name_of_container)
    # docker run - d - -name orr_nadav orr_nadav_rl_base sleep 10; sleep 10; docker rm - f orr_nadav

    # If container run from second time
    command_init = "docker run -i --name {0} {1} sleep {2}; sleep 10; docker rm - f {3}".format(name_of_container,
                                                                                                name_of_container, TIME,
                                                                                                name_of_container)
    container_init = subprocess.Popen([command_init],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, shell=True)

    stdout, stderr = container_init.communicate()

    # convert stderr to string
    stderr_str = stderr.decode("utf-8")

    # create container from the base image (image that continue the container not exist)

    if "No" not in stderr_str.split(" "):
        print("container {0} continue from last commit".format(name_of_container))

    else:
        print("this is first time running {0}".format(name_of_container))
        command_init = "docker run -i --name {0} {1} sleep {2}; sleep 10; docker rm - f {3}".format(name_of_container,IMAGE_NAME,TIME,name_of_container)
        container_init = subprocess.Popen([command_init],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True)
        print("container {0} start first time".format(name_of_container))

    # assume container continue to work
    command_start = "docker start {0}".format(name_of_container)
    container_start = subprocess.Popen([command_start],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, shell=True)

    my_location = os.path.dirname(os.path.abspath(__file__))
    # move files from global to local container
    # c_p ="cp `ls - t model_2/* | head - 10` ./h"
    # comm_pass = "cp `ls - t {0}/model_{1}/* | head - 10` ./".format(my_location,index)

    # comm_pass_1 = "docker cp `ls - t {0}/model_{1}/* | head - 10` ./".format(my_location, index)

    # command_pass = "docker cp `ls - t {0}/model_{1}/* | head - 10` {2}:
    # /root/catkin_ws/src/deep_learning/scripts/part1/model_{2}/".format(my_location,
    #    index,name_of_container,index)

    print("move files from local to container")
    command_pass = "docker cp {0}/RL_ROS/model_{1} {2}:/root/catkin_ws/src/deep_learning/scripts/part1/".format(
        my_location, index, name_of_container, index)

    global_pass = subprocess.Popen([command_pass], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

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

    # pass files
    timeout = time.time() + TIME
    pass_index = 0
    print("move files from container to local")
    while True:
        print("is time to pass {0} in {1}".format(pass_index,name_of_container))
        command_pass = "docker cp {0}:/root/catkin_ws/src/deep_learning/scripts/part1/model_{1}/ {2}/RL_ROS/".format(
            name_of_container, index, my_location)
        container_pass = subprocess.Popen([command_pass], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        if time.time() > timeout or pass_index > 100:
            break
        pass_index += 1
        time.sleep(TIME//10)


def main():
    array = [i for i in range(1, 31)]
    p = Pool()
    p.map(iteration, array)


if __name__ == '__main__':
    # main()
    iteration(2)

