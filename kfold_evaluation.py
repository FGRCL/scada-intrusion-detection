from train_test_split import *

# Evaluate model using 10fold cross validation
# evaluate score by cross-validation
def cross_validation(model, x_train_balanced, y_train_balanced, kf):
    _cv = kf
    # '''Function to perform 10 Folds Cross-Validation
    #  Parameters
    #  ----------
    # model: Python Class, default=None
    #         This is the machine learning algorithm to be used for training.
    # _X: array
    #      This is the matrix of features.
    # _y: array
    #      This is the target variable.
    # _cv: int, default=5
    #     Determines the number of folds for cross-validation.
    #  Returns
    #  -------
    #  The function returns a dictionary containing the metrics 'accuracy', 'precision',
    #  'recall', 'f1' for both training set and validation set.
    # '''

    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=x_train_balanced,
                             y=y_train_balanced,
                             cv=_cv,
                             scoring=_scoring,
                             return_train_score=True)

    return {"Training Accuracy scores": results['train_accuracy'],
            "Mean Training Accuracy": results['train_accuracy'].mean() * 100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean() * 100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()
            }


# function to visualize the training and validation results in each fold
# Grouped Bar Chart for both training and validation data
def cv_plot_result(x_label, y_label, plot_title, train_data, val_data):
    # '''Function to plot a grouped bar chart showing the training and validation
    #   results of the ML model in each fold after applying K-fold cross-validation.
    #  Parameters
    #  ----------
    #  x_label: str,
    #     Name of the algorithm used for training e.g 'Decision Tree'
    #
    #  y_label: str,
    #     Name of metric being visualized e.g 'Accuracy'
    #  plot_title: str,
    #     This is the title of the plot e.g 'Accuracy Plot'
    #
    #  train_result: list, array
    #     This is the list containing either training precision, accuracy, or f1 score.
    #
    #  val_result: list, array
    #     This is the list containing either validation precision, accuracy, or f1 score.
    #  Returns
    #  -------
    #  The function returns a Grouped Barchart showing the training and validation result
    #  in each fold.
    # '''

    # Set size of plot
    plt.figure(figsize=(12, 6))
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold", "6th Fold", "7th Fold", "8th Fold",
              "9th Fold",
              "10th Fold"]
    X_axis = np.arange(len(labels))
    ax = plt.gca()
    plt.ylim(0.40000, 1)
    plt.bar(X_axis - 0.2, train_data, 0.4, color='blue', label='Training')
    plt.bar(X_axis + 0.2, val_data, 0.4, color='red', label='Validation')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cv_metrics(model_name, model_cv):
    cv_plot_result(model_name,
                   "Accuracy",
                   "Accuracy scores in 10 Folds",
                   model_cv["Training Accuracy scores"],
                   model_cv["Validation Accuracy scores"])
    cv_plot_result(model_name,
                   "Precision",
                   "Precision scores in 10 Folds",
                   model_cv["Training Precision scores"],
                   model_cv["Validation Precision scores"])
    cv_plot_result(model_name,
                   "Recall",
                   "Recall scores in 10 Folds",
                   model_cv["Training Recall scores"],
                   model_cv["Validation Recall scores"])
    cv_plot_result(model_name,
                   "F1-score",
                   "F1-score scores in 10 Folds",
                   model_cv["Training F1-score scores"],
                   model_cv["Validation F1-score scores"])

    return