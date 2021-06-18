#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os


# plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(folder=None):

    if folder is None:
        folder = "saves/"
    else:
        folder += "/"
    
    '''places = []
    if os.path.exists(f"{folder}places"):
        with open(f"{folder}{name}", 'rb') as f:
            try:
                while 1:
                    places.append(np.load(f))
            except:
                places = np.concatenate(places, axis=None)'''
    
    # places = np.random.randint(1,21, size=(275,))
    places = np.random.randint(1, 3, size=(275,))

    # open files
    for name in ["reward"]:#, "distance", "loss", "loss_e", "loss_p", "loss_v"]:
        data = []
        if not os.path.exists(f"{folder}{name}"):
            continue
        with open(f"{folder}{name}", 'rb') as f:
            try:
                while 1:
                    data.append(np.load(f))
            except:
                data = np.concatenate(data, axis=None)

        episodes = np.arange(data.shape[0])



        # set title of grath
        plt.title(f"{name} graph")
        # set x lable
        plt.xlabel("episode")
        # set y lable
        plt.ylabel("reward")
        # set grath
        plt.scatter(episodes, data, color ="red")
        # show grath
        plt.show()


def test():


    d = {"year": (1971, 1939, 1941, 1996, 1975),
         "length": (121, 71, 7, 70, 71),
         "Animation": (1, 1, 0, 1, 0)}

    df = pd.DataFrame(d)
    print(df)

    colors = np.where(df["Animation"] == 1, 'y', 'k')
    df.plot.scatter(x="year", y="length", c=colors)
    plt.show()
if __name__=="__main__":
    #plot("model_aac")
    plot("/home/makers/rl_docker/codeRos/model_1")