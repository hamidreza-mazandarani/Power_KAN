import time

import numpy as np
import pandas as pd

from config import *
from functions import (make_new_network,
                       calculate_users_rates, calculate_proportional_fairness)

(path_losses, user_bs_associations_num, user_bs_associations,
 user_locations, bs_locations) = make_new_network()

def convert_to_relative_locations(nodes):
    # Calculate the centroid of the nodes
    centroid = np.mean(nodes, axis=0)
    
    # Subtract the centroid from each node's position to get relative positions
    relative_locations = nodes - centroid
    
    return relative_locations

nodes = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])

relative_locations = convert_to_relative_locations(nodes)
print("Relative Locations:")
print(relative_locations)

# Assume nodes move to new positions
new_nodes = np.array([
    [2, 1],
    [4, 3],
    [6, 5],
    [8, 7]
])

# Convert new positions to relative locations
new_relative_locations = convert_to_relative_locations(new_nodes)
print("New Relative Locations:")
print(new_relative_locations)


# transmission powers must be calculated using optimization solver
# user_transmission_powers = np.random.uniform(min_power, max_power, num_users)

# rates = calculate_users_rates(user_transmission_powers,
#                               path_losses, user_bs_associations_num)
# real_best_value = calculate_proportional_fairness(rates, alpha=0.5)
