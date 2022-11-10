from numpy import linspace, logspace
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.fixes import loguniform

from src import config
from src.data.dataset import Dataset
from src.data.gaspipeline import load_gaspipeline_dataset
from src.models.abstractmodel import GaspipelineClassificationModel
from src.preprocess.dataset import convert_binary_labels, remove_missing_values, scale_features


class RandomForest(GaspipelineClassificationModel):
    best_parameters = {
        'n_estimators': logspace(0, 5, 5, dtype=int),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': linspace(1, 10, 5),
        #'min_samples_leaf': linspace(1, 10, 5),
        #'max_features': ['sqrt', 'log2', None],
        #'min_impurity_decrease': logspace(0, -5, 5),
        #'class_weight': ['balanced', 'balanced_subsample'],
        #'ccp_alpha': logspace(0, -5, 5),
    }

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.model = RandomForestClassifier(verbose=config.verbosity)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def tune(self):
        param_grid = {
            'n_estimators': logspace(0, 5, 5, dtype=int),
            'criterion': ['gini', 'entropy', 'log_loss'],
            'min_samples_split': linspace(1, 10, 5),
            'min_samples_leaf': linspace(1, 10, 5),
            'max_features': ['sqrt', 'log2', None],
            'min_impurity_decrease': logspace(0, -5, 5),
            'class_weight': ['balanced', 'balanced_subsample'],
            'ccp_alpha': logspace(0, -5, 5),
        }
        self.model = GridSearchCV(self.model, param_grid, verbose=config.verbosity)
        self.model.fit(self.x_train, self.y_train)

        return self.model.cv_results_

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
