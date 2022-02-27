import numpy as np
import matplotlib.pyplot as plt
from helping_functions import *

def get_winner(distances):
  d = float('inf')
  for i in range(distances.shape[0]):
    if distances[i,0] < d:
      d = distances[i,0]
      winner = i
  
  return winner

def update_nodes(W, dist_vectors, winner, eta, neigh_size):
  W_new = np.copy(W)
  # update winner
  W_new[winner, :] += eta * dist_vectors[winner, :]

  # update neighbors  
  for rad in range(1, neigh_size+1):
    h_coeff = 0.7 / rad

    right_neigbor = (winner + rad) % W.shape[0]
    left_neighbor = (winner - rad) % W.shape[0]
    W_new[right_neigbor, :] += h_coeff * eta * dist_vectors[right_neigbor, :]
    W_new[left_neighbor, :] += h_coeff * eta * dist_vectors[left_neighbor, :]

  return W_new


def SOM(n_nodes, inputs, epochs, learning_rate):
  W = square_initialization(n_nodes)

  iterations = 0
  total_iterations = epochs * len(inputs)
  All_Ws = []
  for epoch in range(epochs):

    if epoch/epochs < 1/3:
      neighborhood_size = 2 # On 33% of epochs
    elif epoch/epochs < 2/3:
      neighborhood_size = 1 # On 33% of epochs
    else:
      neighborhood_size = 0 # On last 33% of epochs

    # for each city
    for city in inputs:
      eta = learning_rate * linear_decay(iterations, total_iterations)

      dist_vectors = []
      for node in W:
        dist_vectors.append(city - node)
      
      dist_vectors = np.array(dist_vectors).reshape(W.shape)

      distances = [np.linalg.norm(i) for i in dist_vectors]
      distances = np.array(distances).reshape((W.shape[0], 1))

      winner = get_winner(distances)

      # normalize the distance vectors  
      normalized_dist_vectors = []
      for v in dist_vectors:
        normalized_dist_vectors.append(v / np.linalg.norm(v))
      normalized_dist_vectors = np.array(normalized_dist_vectors).reshape(W.shape)

      W = update_nodes(W, normalized_dist_vectors, winner, eta, neighborhood_size)

      iterations += 1
    
    All_Ws.append(W)
    
  return W, All_Ws