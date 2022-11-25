from os import cpu_count

from numpy import concatenate, linspace, logspace, percentile, zeros
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score
from sklearn.model_selection import GridSearchCV, KFold

from src import config
from src.config import f_score_beta
from src.models.abstractmodel import GaspipelineModelTrainer
from src.preprocess.dataset import convert_binary_labels, remove_missing_values, scale_features
from src.preprocess.featureselection import get_first_cca_feature, get_first_ica_feature, get_first_pca_feature


class KMeansTrainer(GaspipelineModelTrainer):
    best_parameters = {
        'anomaly_percentile': 10,
        'n_clusters': 5
    }

    tuning_parameters = {
        'anomaly_percentile': logspace(-6, 2, 9),
        'n_clusters': linspace(1, 10, 5, dtype=int),
        'algorithm': ['elkan']
        # 'init': ['k-means++', 'random'],
        # 'n_init': logspace(1, 3, 10, dtype=int),
    }

    def __init__(self):
        super().__init__()
        self.model = KMeansAnomalyDetection(verbose=config.verbosity, **self.best_parameters)

    def train(self):
        self.model.fit(self.x_train)

    def tune(self):
        tuned_model = GridSearchCV(self.model, self.tuning_parameters, cv=KFold(), verbose=config.verbosity, n_jobs=cpu_count()*2)
        tuned_model.fit(self.x_train, self.y_train)

        return tuned_model.cv_results_

    def get_model(self):
        return self.model

    def _preprocess_features(self, x_train, x_test, y_train, y_test):
        x_train, y_train = remove_missing_values(x_train, y_train)
        x_test, y_test = remove_missing_values(x_test, y_test)

        x_train_pca, pca = get_first_pca_feature(x_train)
        x_train_cca, cca = get_first_cca_feature(x_train, y_train)
        x_train_ica, ica = get_first_ica_feature(x_train)
        x_test_pca = pca.transform(x_test)
        x_test_cca = cca.transform(x_test)
        x_test_ica = ica.transform(x_test)
        x_train = concatenate((x_train_pca, x_train_cca, x_train_ica), axis=1)
        x_test = concatenate((x_test_pca, x_test_cca, x_test_ica), axis=1)

        x_train, scaler = scale_features(x_train)
        x_test = scaler.transform(x_test)

        y_train = convert_binary_labels(y_train)
        y_test = convert_binary_labels(y_test)
        return x_train, x_test, y_train, y_test


class KMeansAnomalyDetection(BaseEstimator, ClassifierMixin):
    def __init__(self, anomaly_percentile=5, **kwargs):
        self._threshold = None
        self.anomaly_percentile = anomaly_percentile
        self.kmeans = KMeans(**kwargs)

    def fit(self, X, y=None):
        self.kmeans.fit(X)
        distances = self.kmeans.transform(X)
        scores = distances.min(axis=1)
        self._threshold = percentile(scores, self.anomaly_percentile)

    def predict(self, X):
        distances = self.kmeans.transform(X)
        scores = distances.min(axis=1)
        y_pred = zeros(scores.size)
        y_pred[scores > self._threshold] = 1

        return y_pred

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return f1_score(y, y_pred)

    def set_params(self, **params):
        self.anomaly_percentile = params.pop('anomaly_percentile', 5)
        self.kmeans.set_params(**params)
        return self
