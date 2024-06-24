import numpy as np

from config import *
from utils import *


def calculate_path_loss(user_loc, bs_loc, alpha_path_loss=2):
    # alpha_path_loss: path-loss exponent

    path_loss = (np.sqrt(((user_loc - bs_loc) ** 2).sum())) ** (- alpha_path_loss)

    return path_loss


def make_new_network(user_locations=None, bs_locations=None, placement_type='random',
                     num_users=None, num_base_stations=None):
    if num_users is None:
        num_users = num_users_default

    if num_base_stations is None:
        num_base_stations = num_base_stations_default

    if user_locations is None:
        if placement_type == 'random':
            user_locations = np.random.uniform(plane_size, size=(num_users, 2))
        elif placement_type == 'grid':
            user_locations = make_grid(num_users, offset=0) * plane_size

    if bs_locations is None:
        if placement_type == 'random':
            bs_locations = np.random.uniform(plane_size, size=(num_base_stations, 2))
        elif placement_type == 'grid':
            bs_locations = make_grid(num_base_stations, offset=0.2) * plane_size

    path_losses = np.zeros((num_users, num_base_stations))
    for u in range(num_users):
        for b in range(num_base_stations):
            path_losses[u, b] = calculate_path_loss(user_locations[u],
                                                    bs_locations[b])

    # # random associations
    # user_bs_associations_num = np.random.randint(num_base_stations, size=num_users)

    user_bs_associations_num = path_losses.argmax(axis=1)

    user_bs_associations = np.zeros((num_users, num_base_stations))
    user_bs_associations[np.arange(num_users), user_bs_associations_num] = 1

    return (path_losses, user_bs_associations_num, user_bs_associations,
            user_locations, bs_locations)


def calculate_users_rates(user_transmission_powers,
                          path_losses, user_bs_associations_num,
                          num_users=None, num_base_stations=None):
    if num_users is None:
        num_users = num_users_default

    if num_base_stations is None:
        num_base_stations = num_base_stations_default

    users_at_bs_powers = np.tile(np.expand_dims(user_transmission_powers, axis=1),
                                 (1, num_base_stations)) * path_losses

    bs_received_powers = users_at_bs_powers.sum(axis=0)

    users_at_selected_bs_powers = users_at_bs_powers[np.arange(num_users), user_bs_associations_num]

    interference_per_user = [bs_received_powers[b] - users_at_selected_bs_powers[u]
                             for u, b in enumerate(user_bs_associations_num)]

    users_sinr = users_at_selected_bs_powers / (interference_per_user + (noise_power * np.ones(num_users)))

    users_rates = np.clip(np.log2(1 + users_sinr), None, max_user_rate)

    return users_rates


def calculate_proportional_fairness(x, alpha):
    if alpha == 0:
        return x.sum()
    elif alpha == 1:
        return np.log(x).sum()
    elif (0 < alpha < 1) or (1 < alpha < 100):
        return ((x ** (1 - alpha)) / (1 - alpha)).sum()
    else:
        raise ValueError('alpha must be non-negative and less than 100')
  
