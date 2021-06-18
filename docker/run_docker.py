import subprocess
import time
from multiprocessing import Pool

IMAGE_NAME = "new_00"
from multiprocessing import Pool


def iteration(index):
    print(index)
    # temporary - need to change by the interactions
    name_of_container = "rl_learning_{0}".format(index)

    command_init = "docker run -i --name {0} {1}".format(name_of_container, name_of_container)
    container_init = subprocess.Popen([command_init],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, shell=True)

    stdout, stderr = container_init.communicate()

    # convert stderr to string
    stderr_str = stderr.decode("utf-8")

    # image that continue the container not exist create container from the base image

    if "Unable" not in stderr_str.split(" "):
        print("container {0} continue from last commit".format(name_of_container))

    else:
        command_init = "docker run -i --name {0} {1}".format(name_of_container, IMAGE_NAME)
        container_init = subprocess.Popen([command_init],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True)
        print("container {0} start firat time".format(name_of_container))

    # assume container continue to work
    command_start = "docker start {0}".format(name_of_container)
    container_start = subprocess.Popen([command_start],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, shell=True)

    command_run_simulation = "docker exec -i {0} bash ./root/catkin_ws/simulator.sh".format(name_of_container)
    container_run_simulation = subprocess.Popen([command_run_simulation],
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, shell=True)

    command_run_agent = "docker exec -i {0} bash ./root/catkin_ws/agent.sh {1}".format(name_of_container, index)
    container_run_agent = subprocess.Popen([command_run_agent],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, shell=True)
    time.sleep(50)
    # string_a = "/home/orr/RL_ROS"
    #string_b = /home/makers/rl_docker/codeRos
 
    command_pass = "docker cp {0}:/root/catkin_ws/src/deep_learning/scripts/part1/model_{1}/ /home/orr/RL_ROS".format(
        name_of_container, index)
    container_pass = subprocess.Popen([command_pass],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, shell=True)

    time.sleep(10)
    # DOCKER STOP - to stop docker from run

    command_commit = "docker commit {0} {0}".format(name_of_container)

    container_commit = subprocess.Popen([command_commit],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, shell=True)
    time.sleep(10)
    command_stop = "docker stop {0}".format(name_of_container)
    container_stop = subprocess.Popen([command_stop],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, shell=True)

    print("container {0} finished!!".format(name_of_container))


def main():
    while 1:
        array = [i for i in range(1, 15)]
        p = Pool()
        p.map(iteration, array)

        # docker image rm $(docker image ls -f 'dangling=true' -q) delete all none files


if __name__ == '__main__':
    iteration(5)
    iteration(5)
    # main()
    # command = "\"/home/docker/catkin_ws/src/Indoors_main/node_manager/scripts/indoors_inside_docker.py " + str(
    #     container_num) + " " + \
    #           str(script_name) + " " + ROS_MASTER_URI + " " + ROS_IP + " " + batch_name + "\""
    #
    # super_cmd = "docker exec -t indoors_container_" + str(container_num) + " sh -c " + command
