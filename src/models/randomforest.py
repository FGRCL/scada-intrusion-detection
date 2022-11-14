from os import cpu_count

from numpy import concatenate, linspace, logspace
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.fixes import loguniform

from src import config
from src.models.abstractmodel import GaspipelineModelTrainer
from src.preprocess.dataset import convert_binary_labels, remove_missing_values, scale_features
from src.preprocess.featureselection import get_first_cca_feature, get_first_ica_feature, get_first_pca_feature


class RandomForestClassification(GaspipelineModelTrainer):
    best_parameters = {
        'n_estimators': 500,
        'criterion': 'gini',
        'min_samples_split': 8000,
        'class_weight': 'balanced_subsample'
    }

    tuning_parameters = {
            'n_estimators': [500],
            'criterion': ['gini'],
            'min_samples_split': [8000],
            'min_samples_leaf': linspace(1, 10, 5),
            'max_features': ['sqrt', 'log2', None],
            'min_impurity_decrease': logspace(0, -5, 5),
            'class_weight': ['balanced', 'balanced_subsample'],
            'ccp_alpha': logspace(0, -5, 5),
        }

    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(verbose=config.verbosity, n_jobs=cpu_count(), **self.best_parameters)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def tune(self):
        self.model = GridSearchCV(self.model, self.tuning_parameters, verbose=config.verbosity, n_jobs=cpu_count())
        self.model.fit(self.x_train, self.y_train)

        return self.model.cv_results_

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
