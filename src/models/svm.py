from os import cpu_count

from imblearn.over_sampling import SMOTE
from numpy import concatenate, linspace, logspace
from scipy.stats import loguniform, norm, truncnorm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

from src import config
from src.models.abstractmodel import GaspipelineModelTrainer
from src.preprocess.binarylabelencoder import BinaryLabelEncoder
from src.preprocess.dataset import balance_dataset, convert_binary_labels, remove_missing_values, scale_features
from src.preprocess.featureextraction import GasPipelineFeatureExtraction
from src.preprocess.featureselection import get_first_cca_feature, get_first_ica_feature, get_first_pca_feature


class SvmTrainer(GaspipelineModelTrainer):
    best_parameters = {
        'balance_dataset': False,
        'feature_reduction': False,
        'scale_features': True,
        'kernel': 'rbf',
        'cache_size': 4000,
        'shrinking': False,
        'balance_dataset': True,
        'feature_reduction': False,
        'scale_features': True,
    }

    tuning_parameters = {
        'balance_dataset': [False],
        'feature_reduction': [False],
        'scale_features': [True],
        'kernel': ['rbf'],
        'cache_size': [4000],
        'shrinking': [False],
        'balance_dataset': [True, False],
        'feature_reduction': [True, False],
        'scale_features': [True, False],
        'feature_n_components': linspace(1, 12, 5, dtype=int),
    }

    def __init__(self):
        super().__init__()
        self.model = GasPipelineSvc(verbose=config.verbosity, **self.best_parameters)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def tune(self):
        tuned_model = GridSearchCV(self.model, self.tuning_parameters, cv=5, verbose=config.verbosity, n_jobs=-1)
        tuned_model = tuned_model.fit(self.x_train, self.y_train)

        return tuned_model.cv_results_

    def get_model(self):
        return self.model


class GasPipelineSvc(BaseEstimator, ClassifierMixin):
    def __init__(self, balance_dataset=False, feature_reduction=False, scale_features=False, feature_n_components=1, **kwargs):
        self.balance_dataset = balance_dataset
        self.feature_reduction = feature_reduction
        self.scale_features = scale_features
        self.feature_n_components = feature_n_components
        self.feature_extraction = GasPipelineFeatureExtraction(self.feature_reduction, self.scale_features, self.feature_n_components)
        self.svc = SVC(**kwargs)

    def fit(self, X, y):
        if self.balance_dataset:
            X, y = SMOTE().fit_resample(X, y)
        X = self.feature_extraction.fit_transform(X, y)
        self.svc.fit(X, y)

    def predict(self, X):
        X = self.feature_extraction.transform(X)
        return self.svc.predict(X)

    def score(self, X, y, sample_weight=None):
        X = self.feature_extraction.transform(X)
        return self.svc.score(X, y, sample_weight)

    def set_params(self, **params):
        self.__init__(**params)
        return self
