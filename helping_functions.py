import numpy as np

def square_initialization(nodes):
  W = np.zeros((nodes,2))
  up_row = int(nodes/2)
  down_row = nodes - up_row
  for i in range(up_row):
    W[i, :] = np.array([i / up_row,1])
  for i in range(down_row):
    W[i + up_row, :] = np.array([1 - i / down_row,0])

  return W

def plot_route(W, ax):

  points = np.zeros((W.shape[0]+1, W.shape[1]))
  points[:W.shape[0], :] = W
  points[W.shape[0], :] = W[0]

  ax.plot(points[:, 0], points[:, 1], color="blue", label="Nodes", marker='x')
  n = range(points.shape[0]-1)

  for i, txt in enumerate(n):
      ax.annotate(txt, (points[i, 0], points[i, 1]))

def decay(t, T):
  return np.exp(-t/T)

def linear_decay(t, T):
  return 1 - 0.9 * t / T