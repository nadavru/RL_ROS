import numpy as np
import matplotlib.pyplot as plt
import os

def plot(folder=None):

    if folder is None:
        folder = "saves/"
    else:
        folder += "/"
    
    for name in ["reward", "distance", "loss", "loss_e", "loss_p", "loss_v"]:
        data = []
        if not os.path.exists(f"{folder}{name}"):
            continue
        with open(f"{folder}{name}", 'rb') as f:
            try:
                while 1:
                    data.append(np.load(f))
            except:
                data = np.concatenate(data, axis=None)
        x = np.arange(data.shape[0])
        plt.title(f"{name} graph")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.plot(x, data, color ="red")
        plt.show()

if __name__=="__main__":
    plot("model_aac")