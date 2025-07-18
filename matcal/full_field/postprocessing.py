import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d




def plot_waves2D(points, waves, wave_number):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], waves[:, wave_number])
    plt.show()
