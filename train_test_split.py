from binary_conversion import *

def train_test_data(x_FS_balanced_binary, y_FS_balanced_binary):
    # Create training and testing sets
    # Split dataset into training set and test set
    # 70% training and 30% test
    x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced = train_test_split(x_FS_balanced_binary,
                                                                                            y_FS_balanced_binary,
                                                                                            test_size=0.3)

    return x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced