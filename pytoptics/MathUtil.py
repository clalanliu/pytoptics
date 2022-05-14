import numpy as np

def xy_to_direction_cosine(x, y):
    xc = np.sin(np.deg2rad(x))
    yc = np.sin(np.deg2rad(y))
    zc = np.sqrt(1 - xc*xc - yc*yc)
    return [xc, yc, zc]