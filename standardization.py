from initialization import *

# standardization of features in df
def standardization(x, y):
    x_scaled = StandardScaler().fit_transform(x)

    return x_scaled