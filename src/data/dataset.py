from dataclasses import dataclass

from numpy import ndarray


@dataclass
class Dataset:
    x_train: ndarray
    x_test: ndarray
    y_train: ndarray
    y_test: ndarray