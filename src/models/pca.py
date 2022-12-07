from os import cpu_count

from numpy import concatenate, linspace, logspace, percentile, square, zeros
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from src import config
from src.models.abstractmodel import GaspipelineModelTrainer
from src.preprocess.dataset import convert_binary_labels, remove_missing_values, scale_features
from src.preprocess.featureselection import get_first_cca_feature, get_first_ica_feature, get_first_pca_feature


class PcaTrainer(GaspipelineModelTrainer):
    best_parameters = {
        'anomaly_percentile': 1e-10,
        'n_components': 19,
        'whiten': False
    }

    tuning_parameters = {
        'anomaly_percentile': logspace(-20, 1, 22),
        'n_components': linspace(5, 30, 15, dtype=int),
        'whiten': [True, False],
    }

    def __init__(self):
        super().__init__()
        self.model = PcaAnomalyDetection(**self.best_parameters)

    def train(self):
        self.model.fit(self.x_train)

    def tune(self):
        tuned_model = GridSearchCV(self.model, self.tuning_parameters, verbose=config.verbosity, cv=10, n_jobs=cpu_count() * 2)
        tuned_model.fit(self.x_train, self.y_train)

        return tuned_model.cv_results_

    def get_model(self):
        return self.model

    def _preprocess_features(self, x_train, x_test, y_train, y_test):
        x_train, y_train = remove_missing_values(x_train, y_train)
        x_test, y_test = remove_missing_values(x_test, y_test)

        x_train, scaler = scale_features(x_train)
        x_test = scaler.transform(x_test)

        y_train = convert_binary_labels(y_train)
        y_test = convert_binary_labels(y_test)
        return x_train, x_test, y_train, y_test


class PcaAnomalyDetection(BaseEstimator, ClassifierMixin):
    def __init__(self, anomaly_percentile=5, **kwargs):
        self.pca = PCA(**kwargs)
        self._threshold = None
        self.anomaly_percentile = anomaly_percentile

    def fit(self, X, y=None):
        x_pca = self.pca.fit_transform(X)
        x_reconstructed = self.pca.inverse_transform(x_pca)
        mse = square(x_reconstructed - X).mean(axis=-1)
        self._threshold = percentile(mse, self.anomaly_percentile)

    def predict(self, X):
        x_pca = self.pca.transform(X)
        x_reconstructed = self.pca.inverse_transform(x_pca)
        mse = square(x_reconstructed - X).mean(axis=-1)
        y_pred = zeros(mse.size)
        y_pred[mse > self._threshold] =1

        return y_pred

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return f1_score(y, y_pred)

    def set_params(self, **params):
        self.anomaly_percentile = params.pop('anomaly_percentile', 5)
        self.pca.set_params(**params)
        return self
