from KNN_classifier import *

# AdaBoost classifier & evaluation
def AdaBoost_classifier(x_train_balanced, y_train_balanced, x_test_balanced, y_test_balanced):
    # prepare the cross-validation procedure
    cv_AdaBoost = KFold(n_splits=10, random_state=1, shuffle=True)

    # Create AdaBoost Classifier
    classifier_AdaBoost = AdaBoostClassifier()

    # fit model
    classifier_AdaBoost.fit(x_train_balanced, y_train_balanced)

    # evaluate model & report performance
    cross_validation(classifier_AdaBoost, x_train_balanced, y_train_balanced, cv_AdaBoost)

    # predict the response for normal list from BF
    y_pred_AdaBoost = classifier_AdaBoost.predict(np.array(x_test_balanced))

    # classification report/evaluate model on normal list from BF
    report_AdaBoost = classification_report(y_test_balanced, y_pred_AdaBoost)
    # heat_map_AdaBoost = sns.heatmap(pd.DataFrame(report_AdaBoost).iloc[:-1, :].T, annot=True)
    # figure = heat_map_AdaBoost.get_figure()
    # figure.savefig(r"\results\AdaBoost classification report.png")

    # confusion matrix//evaluate model on normal list from BF
    confusion_matrix_AdaBoost = metrics.confusion_matrix(y_pred_AdaBoost, y_test_balanced)
    cm_display_AdaBoost = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_AdaBoost,
                                                         display_labels=[False, True])

    return report_AdaBoost, cm_display_AdaBoost