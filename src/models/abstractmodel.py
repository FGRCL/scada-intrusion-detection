from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score

from src.config import f_score_beta
from src.data.dataset import Dataset
from src.data.gaspipeline import load_gaspipeline_dataset


class GaspipelineClassificationModel(ABC):
    def __init__(self, dataset: Dataset):
        x_train, x_test, y_train, y_test = dataset
        x_train, x_test, y_train, y_test = self._preprocess_features(x_train, x_test, y_train, y_test)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def tune(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def _preprocess_features(self, x_train, x_test, y_train, y_test):
        pass

    def get_metrics(self):
        y_pred = self.get_model().predict(self.x_test)

        confusion = confusion_matrix(self.y_test, y_pred)
        f_score = fbeta_score(self.y_test, y_pred, beta=f_score_beta)
        accuracy = accuracy_score(self.y_test, y_pred)

        return confusion, f_score, accuracy