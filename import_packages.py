# import python packages
import dataframe_image
import numpy as np
import pandas as pd
from imblearn.under_sampling import AllKNN

# import preprocessing packages
from matplotlib.pyplot import table
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

# import class balancing package
# from imblearn.over_sampling import RandomOverSampler

# import dimension reduction packages
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

# import bloom filter
# from bloom_filter import BloomFilter
# from collections import Counter
# !pip install imbalanced-learn
# !pip install mmh3
# !pip install bitarray
import mmh3
import math
from bitarray import bitarray

# import Classifiers models
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture as GMM

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics, ensemble
from numpy import mean
from numpy import std
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

# import plotting packages
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns