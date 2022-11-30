from binary_conversion import *
from kfold_evaluation import *

# KNN classification & evaluation
def KNN_classifier(x_train_balanced, y_train_balanced, x_test_balanced, y_test_balanced):

    # prepare the cross-validation procedure
    cv_KNN = KFold(n_splits=10, random_state=1, shuffle=True)

    # create KNN Classifier/model
    classifier_KNN = KNeighborsClassifier(n_neighbors=5)

    # fit model
    classifier_KNN.fit(x_train_balanced, y_train_balanced)

    # evaluate model & report performance
    cross_validation(classifier_KNN, x_train_balanced, y_train_balanced, cv_KNN)

    # predict the response for normal list from BF
    y_pred_KNN = classifier_KNN.predict(np.array(x_test_balanced))

    # classification report/evaluate model on normal list from BF
    report_KNN = classification_report(y_test_balanced, y_pred_KNN)

    # confusion matrix//evaluate model on normal list from BF
    confusion_matrix_KNN = metrics.confusion_matrix(y_pred_KNN, y_test_balanced)
    cm_display_KNN = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_KNN, display_labels=[False, True])
    

    return report_KNN, cm_display_KNN