from abc import ABC, abstractmethod

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score

from src.config import f_score_beta
from src.data.gaspipeline import load_gaspipeline_dataset
from src.preprocess.binarylabelencoder import BinaryLabelEncoder


class GaspipelineModelTrainer(ABC):
    def __init__(self):
        x_train, x_test, y_train, y_test = load_gaspipeline_dataset()
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

    def _preprocess_features(self, x_train, x_test, y_train, y_test):
        simple_imputer = SimpleImputer()
        binary_label_encoder = BinaryLabelEncoder()

        x_train = simple_imputer.fit_transform(x_train, y_train)
        x_test = simple_imputer.fit_transform(x_test, y_test)

        y_train = binary_label_encoder.transform(y_train)
        y_test = binary_label_encoder.transform(y_test)
        return x_train, x_test, y_train, y_test

    def get_metrics(self):
        y_pred = self.get_model().predict(self.x_test)

        confusion = confusion_matrix(self.y_test, y_pred)
        f_score = fbeta_score(self.y_test, y_pred, beta=f_score_beta)
        accuracy = accuracy_score(self.y_test, y_pred)

        return confusion, f_score, accuracy