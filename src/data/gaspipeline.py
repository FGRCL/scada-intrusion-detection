from numpy import genfromtxt, loadtxt
from sklearn.model_selection import train_test_split

from src.config import data_file_path
from src.data.dataset import Dataset


def load_gaspipeline_dataset(train_size = 0.8, seed=1337):
    with open(data_file_path) as f:
        data = loadtxt(f, delimiter=',', skiprows=32)

    x = data[:, :-1]
    y = data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=seed)
    dataset = Dataset(x_train, x_test, y_train, y_test)

    return dataset