import numpy as np


def dynamic_range(chunk: np.ndarray):
    return np.var(chunk)


if __name__ == '__main__':
    chunk = np.array([3, 3, 3, 3])
    print(dynamic_range(chunk))