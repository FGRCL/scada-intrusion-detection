# botnet-attack-detection
# botnet attack detection using anomaly detection and classification


# Import libraries

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

#Import Classifiers

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Load Dataset

names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","others"])

TrainDataset = pd.read_csv('KDDTrain+.csv', names=names, header=None)
TestDataset = pd.read_csv('KDDTest+.csv', names=names, header=None)

TrainData = TrainDataset.iloc[:, :-1]
TestData = TestDataset.iloc[:, :-1]


# Data Analysis

attributes_labels_train =  TrainData['labels'].replace({
    'normal': 0,
    
    'back': 1,
    'land': 1,
    'neptune': 1,
    'pod': 1,
    'smurf': 1,
    'teardrop': 1,
    'mailbomb': 1,
    'apache2': 1,
    'processtable': 1,
    'udpstorm': 1,
    
    'ipsweep': 2,
    'nmap': 2,
    'portsweep': 2,
    'satan': 2,
    'mscan': 2,
    'saint': 2,

    'ftp_write': 3,
    'guess_passwd': 3,
    'imap': 3,
    'multihop': 3,
    'phf': 3,
    'spy': 3,
    'warezclient': 3,
    'warezmaster': 3,
    'sendmail': 3,
    'named': 3,
    'snmpgetattack': 3,
    'snmpguess': 3,
    'xlock': 3,
    'xsnoop': 3,
    'worm': 3,
    
    'buffer_overflow': 4,
    'loadmodule': 4,
    'perl': 4,
    'rootkit': 4,
    'httptunnel': 4,
    'ps': 4,    
    'sqlattack': 4,
    'xterm': 4
})
attributes_labels_test = TestData['labels'].replace({
    'normal': 0,
    
    'back': 1,
    'land': 1,
    'neptune': 1,
    'pod': 1,
    'smurf': 1,
    'teardrop': 1,
    'mailbomb': 1,
    'apache2': 1,
    'processtable': 1,
    'udpstorm': 1,
    
    'ipsweep': 2,
    'nmap': 2,
    'portsweep': 2,
    'satan': 2,
    'mscan': 2,
    'saint': 2,

    'ftp_write': 3,
    'guess_passwd': 3,
    'imap': 3,
    'multihop': 3,
    'phf': 3,
    'spy': 3,
    'warezclient': 3,
    'warezmaster': 3,
    'sendmail': 3,
    'named': 3,
    'snmpgetattack': 3,
    'snmpguess': 3,
    'xlock': 3,
    'xsnoop': 3,
    'worm': 3,
    
    'buffer_overflow': 4,
    'loadmodule': 4,
    'perl': 4,
    'rootkit': 4,
    'httptunnel': 4,
    'ps': 4,    
    'sqlattack': 4,
    'xterm': 4
})
TrainData['labels'] = attributes_labels_train
TestData['labels'] = attributes_labels_test


# Check Train and Test Data dimensionality

TrainData.shape
TestData.shape


# Data Cleaning

categorical = [1, 2, 3]
binary = [6, 11, 13, 14, 20, 21]
numeric = list(set(range(41)).difference(categorical).difference(binary))

categorical_columns = names[categorical].tolist()
binary_columns = names[binary].tolist()
numeric_columns = names[numeric].tolist()

categorical_columns
binary_columns
numeric_columns

TrainData[binary_columns].describe().transpose()
TrainData.loc[TrainData['su_attempted'] == 2.0, 'su_attempted'] = 0.0
TestData.loc[TestData['su_attempted'] == 2.0, 'su_attempted'] = 0.0

TrainData[binary_columns].describe().transpose()
TrainData[numeric_columns].describe().transpose()

TrainData = TrainData.drop('num_outbound_cmds',axis=1)
TestData = TestData.drop('num_outbound_cmds',axis=1)
numeric_columns.remove('num_outbound_cmds')

print('protocol_type', len(TrainData['protocol_type'].value_counts().keys()))
print('service', len(TrainData['service'].value_counts().keys()))
print('flag', len(TrainData['flag'].value_counts().keys()))

print('protocol_type', len(TestData['protocol_type'].value_counts().keys()))
print('service', len(TestData['service'].value_counts().keys()))
print('flag', len(TestData['flag'].value_counts().keys()))


# Encoding Categorical features

# Scaling

# Class Balancing

# Feature Selection

# Models training and testing

# Evaluation

# Hyperparameter tuning

# Prediction
