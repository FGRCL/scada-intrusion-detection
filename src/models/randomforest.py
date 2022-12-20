from os import cpu_count

from imblearn.over_sampling import SMOTE
from numpy import concatenate, linspace, logspace
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.fixes import loguniform

from src import config
from src.models.abstractmodel import GaspipelineModelTrainer
from src.preprocess.dataset import convert_binary_labels, remove_missing_values, scale_features
from src.preprocess.featureextraction import GasPipelineFeatureExtraction
from src.preprocess.featureselection import get_first_cca_feature, get_first_ica_feature, get_first_pca_feature


class RandomForestClassification(GaspipelineModelTrainer):
    best_parameters = {
        'n_estimators': 500,
        'criterion': 'gini',
        'min_samples_split': 8000,
        'class_weight': 'balanced_subsample',
        'balance_dataset': True,
        'feature_reduction': False,
        'scale_features': False,
    }

    tuning_parameters = {
        'n_estimators': [500],
        'criterion': ['gini'],
        'min_samples_split': [8000],
        'class_weight': ['balanced_subsample'],
        'balance_dataset': [True, False],
        'feature_reduction': [True, False],
        'scale_features': [True, False],
        'feature_n_components': linspace(1, 12, 5, dtype=int),
    }

    def __init__(self):
        super().__init__()
        self.model = GasPipelineRandomForest(verbose=config.verbosity, n_jobs=cpu_count())

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def tune(self):
        tuned_model = GridSearchCV(self.model, self.tuning_parameters, verbose=config.verbosity, n_jobs=cpu_count())
        tuned_model.fit(self.x_train, self.y_train)

        return tuned_model.cv_results_

    def get_model(self):
        return self.model


class GasPipelineRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, balance_dataset=False, feature_reduction=False, scale_features=False, feature_n_components=1, **kwargs):
        self.balance_dataset = balance_dataset
        self.feature_reduction = feature_reduction
        self.scale_features = scale_features
        self.feature_n_components = feature_n_components
        self.feature_extraction = GasPipelineFeatureExtraction(self.feature_reduction, self.scale_features, self.feature_n_components)
        self.random_forest = RandomForestClassifier(**kwargs)

    def fit(self, X, y):
        if self.balance_dataset:
            X, y = SMOTE().fit_resample(X, y)
        X = self.feature_extraction.fit_transform(X, y)
        self.random_forest.fit(X, y)

    def predict(self, X):
        X = self.feature_extraction.transform(X)
        return self.random_forest.predict(X)

    def score(self, X, y, sample_weight=None):
        X = self.feature_extraction.transform(X)
        return self.random_forest.score(X, y, sample_weight)

    def set_params(self, **params):
        self.__init__(**params)
        return self