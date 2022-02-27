from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from som import SOM
from helping_functions import plot_route

np.random.seed(34)

cities = np.array([[0.4,    0.4439],
                [0.2439, 0.1463],
                [0.1707, 0.2293],
                [0.2293, 0.761 ],
                [0.5171, 0.9414],
                [0.8732, 0.6536],
                [0.6878, 0.5219],
                [0.8488, 0.3609],
                [0.6683, 0.2536],
                [0.6195, 0.2634]])

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(cities[:, 0], cities[:, 1], color="red", label="Cities")
n = range(cities.shape[0])
for i, txt in enumerate(n):
    ax.annotate(txt, (cities[i, 0], cities[i, 1]))
ax.legend()
plt.show()

W_new, All_Ws = SOM(10, cities, 80, 0.08) 

# Plotting the Animation
def animate_func(num):
    ax.clear() 
    plot_route(All_Ws[num],ax)
    ax.scatter(cities[:, 0], cities[:, 1], color="red", label="Cities")

    ax.legend()
    plt.show()

numDataPoints = len(All_Ws)
fig, ax = plt.subplots(nrows=1, ncols=1)
ani = animation.FuncAnimation(fig, animate_func, interval=100,   
                                   frames=numDataPoints)
plt.show()