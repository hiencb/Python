import numpy as np


def partition_indices(n: int, ratio_train=0.8, ratio_cv: float = None) -> (np.ndarray, np.ndarray, np.ndarray):
    idx = np.random.permutation(range(n))

    n_train = int(n * ratio_train)
    train_idx = idx[: n_train]

    if ratio_cv:
        n_cv = int(n * ratio_cv)
        cv_idx = idx[n_train: n_train + n_cv]
        test_idx = idx[n_train + n_cv:]
        return train_idx, cv_idx, test_idx

    test_idx = idx[n_train:]
    return train_idx, test_idx
