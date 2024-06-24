import numpy as np
from scipy.spatial import procrustes

def make_grid(num_users, offset):
    # Generate the grid of points
    x, y = np.mgrid[
           (0 + offset):(1 - offset):np.sqrt(num_users) * 1j,
           (0 + offset):(1 - offset):np.sqrt(num_users) * 1j]

    # Flatten the arrays to get a list of points
    points = np.vstack((x.ravel(), y.ravel())).T

    return points


def procrustes_analysis(X, Y, threshold=1e-5):
    mtx1, mtx2, disparity = procrustes(X, Y)
    similar = disparity < threshold

    return disparity, similar
