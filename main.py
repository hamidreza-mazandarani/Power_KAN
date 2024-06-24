import time

import numpy as np
import pandas as pd

from config import *
from functions import (make_new_network,
                       calculate_users_rates, calculate_proportional_fairness)

(path_losses, user_bs_associations_num, user_bs_associations,
 user_locations, bs_locations) = make_new_network()

# transmission powers must be calculated using optimization solver
user_transmission_powers = np.random.uniform(min_power, max_power, num_users)

rates = calculate_users_rates(user_transmission_powers,
                              path_losses, user_bs_associations_num)
real_best_value = calculate_proportional_fairness(rates, alpha=0.5)
