import numpy as np

def eig(cov):
    return np.linalg.eig(cov) if cov.shape == (3,3) else (None, None)