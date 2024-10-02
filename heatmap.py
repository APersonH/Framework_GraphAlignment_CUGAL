# create a heatmap from the in the command line specified file
import matplotlib.pyplot as plt
import numpy as np  
import sys

def create_heatmap(file):
    # read
    data = np.loadtxt(file, delimiter=',')
    # plot
    fig, axis = plt.subplots()
    heatmap = axis.pcolor(data, cmap=plt.cm.turbo)
    plt.colorbar(heatmap)
    #add colorbar
    plt.savefig(file + '_heatmap.png')
    return

if __name__ == '__main__':
    create_heatmap(sys.argv[1])