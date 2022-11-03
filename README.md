# import numpy as np
import pandas as pd
import sklearn
import imblearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA 
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from bloom_filter import BloomFilter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Load dataset
# Reference: https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets
# read contents of csv file
data = pd.read_csv('Gaspipelinedatasetfull.csv')

# adding header
#headerList = ['address', 'function', 'length', 'setpoint', 'gain', 'reset rate', 'deadband', 'cycle time', 'rate', 'system mode', 'control scheme', 'pump', 'solenoid', 'pressure measurement', 'crc rate', 'command response', 'time', 'binary result', 'categorized result', 'specific result']
headerList = ['command_address', 'response_address', 'command_memory', 'response_memory', 'command_memory_count', 'response_memory_count', 'comm_read_function', 'comm_write_fun', 'resp_read_fun', 'resp_write_fun', 'sub_function', 'command_length', 'resp_length', 'gain', 'reset', 'deadband', 'cycletime', 'rate', 'setpoint', 'control_mode', 'control_scheme', 'pump', 'solenoid', 'crc_rate', 'measurement', 'time', 'result']

# converting data frame to csv
data.to_csv("Gaspipelinedatasetfull.csv", header=headerList, index=False)


# Separate dataset into normal, abnormal attacks and each attack type

normal_entries = data.loc[data['result'] == 0]
print("Normal entries: " + str(len(normal_entries)))

attacks_entries = data.loc[(data['result'] == 1) | (data['result'] == 2) 
                           | (data['result'] == 3) | (data['result'] == 4)
                           | (data['result'] == 5) | (data['result'] == 6)
                           | (data['result'] == 7)]
print("attacks entries: " + str(len(attacks_entries)))

NMRI_entries = data.loc[data['result'] == 1]
print("NMRI attacks entries: " + str(len(NMRI_entries)))

CMRI_entries = data.loc[data['result'] == 2]
print("CMRI attacks entries: " + str(len(CMRI_entries)))

MSCI_entries = data.loc[data['result'] == 3]
print("MSCI attacks entries: " + str(len(MSCI_entries)))

MPCI_entries = data.loc[data['result'] == 4]
print("MPCI attacks entries: " + str(len(MPCI_entries)))

MFCI_entries = data.loc[data['result'] == 5]
print("MFCI attacks entries: " + str(len(MFCI_entries)))

DoS_entries = data.loc[data['result'] == 6]
print("DoS attacks entries: " + str(len(DoS_entries)))

Recon_entries = data.loc[data['result'] == 7]
print("Recon attacks entries: " + str(len(Recon_entries)))


# Separating features and labels
# Set your x and y parameters to perform SMOTE
x_old = data.loc[ : , data.columns != 'result']
y_old = data['result']


# oversampling the train dataset using SMOTE
# finds the the nearest neighbors for its target class and 
# combines features of the target class with features of its 
# neighbors instead of extrapolating the minority class

# Set your x and y parameters to perform SMOTE

# Separating features and labels for all dataset
x_old = data.loc[ : , data.columns != 'result']
y_old = data['result']

# Separating features and labels for normal & each attack type:
x_old_normal_entries = normal_entries.loc[ : , normal_entries.columns != 'result']
y_old_normal_entries = normal_entries['result']

x_old_NMRI_entries = NMRI_entries.loc[ : , NMRI_entries.columns != 'result']
y_old_NMRI_entries = NMRI_entries['result']

x_old_CMRI_entries = CMRI_entries.loc[ : , CMRI_entries.columns != 'result']
y_old_CMRI_entries = CMRI_entries['result']

x_old_MSCI_entries = MSCI_entries.loc[ : , MSCI_entries.columns != 'result']
y_old_MSCI_entries = MSCI_entries['result']

x_old_MPCI_entries = MPCI_entries.loc[ : , MPCI_entries.columns != 'result']
y_old_MPCI_entries = MPCI_entries['result']

x_old_MFCI_entries = MFCI_entries.loc[ : , MFCI_entries.columns != 'result']
y_old_MFCI_entries = MFCI_entries['result']

x_old_DoS_entries = DoS_entries.loc[ : , DoS_entries.columns != 'result']
y_old_DoS_entries = DoS_entries['result']

x_old_Recon_entries = Recon_entries.loc[ : , Recon_entries.columns != 'result']
y_old_Recon_entries = Recon_entries['result']



# Class Balancing
# Class balancing using SMOTE technique
!pip install imbalanced-learn

# Apply SMOTE technique
smote = SMOTE()
x_sm, y_sm = smote.fit_sample(x_old, y_old)
x_balanced = x_sm
y_balanced = y_sm

# Plot data before and after balancing
list_y_old = pd.value_counts(y_old).to_numpy().tolist()
list_y_balanced = pd.value_counts(y_balanced).to_numpy().tolist()
index = ['normal', 'CMRI', 'MPCI', 'Recon', 'NMRI', 'DoS', 'MSCI', 'MFCI'] 
    
df = pd.DataFrame({'Before Balancing': list_y_old,
                   'After Balancing': list_y_balanced}, index=index)
ax = df.plot.bar(rot=0)

plt.xlabel('Entry Category')
plt.ylabel('No. of instances')
plt.title('DATASET COMPARISON BEFORE AND AFTER BALANCING')
plt.legend()

df


# Cleaning dataset
# find missing values
x_old.isnull().values.any()
x_balanced.isnull().values.any()

# find how many missing values
x_old.isnull().sum()
x_balanced.isnull().sum()

# total summation of all missing values in the DataFrame
x_old.isnull().sum().sum()
x_balanced.isnull().sum().sum()


# Feature sclaing
# standardizing the features
x_old_std = StandardScaler().fit_transform(x_old)
x_balanced_std = StandardScaler().fit_transform(x_balanced)

# Feature scaling
stndS = StandardScaler()
x_old_scaled = stndS.fit_transform(x_old_std)
x_balanced_scaled = stndS.fit_transform(x_balanced_std)


# Feature selection

# PCA technique
# create a PCA object
# make an instance of the Model
# 0.95 for the number of components parameter
# It means that scikit-learn choose the minimum number of 
# principal components such that 95% of the variance is retained
def feature_selection_pca(x,y):
    pca_95 = PCA(n_components=1, random_state=2020)
    
    # fit PCA on training set. Fitting PCA on the training set only
    pca_95.fit(x)

    # apply the mapping (transform) to both the training set and the test set
    x_pca_95 = pca_95.transform(x)
    #print(x_pca_95.shape)
    
    # create a pandas DataFrame using the values of all principal components 
    # and add the label column of the original dataset
    data_pca = pd.DataFrame(x_pca_95, columns=['feature_1'])
    data_pca['label1'] = data.result
    #data_pca.head()
    #print(data_pca.shape)
    data_pca.to_csv('HMLIDS_pca.csv', index=False)
    selectedfeature_pca = data_pca['feature_1'].tolist()
    
    return selectedfeature_pca


# CCA technique
def feature_selection_cca(x,y):
    cca = CCA(n_components=1)
    cca.fit(x, y)
    x_cca, y_cca = cca.transform(x, y)
    #print(x_cca.shape)
    # create a pandas DataFrame using the values of all principal components 
    # and add the label column of the original dataset
    data_cca = pd.DataFrame(x_cca, columns=['feature_2'])
    data_cca['label2'] = data.result
    #data_cca.head()
    #print(data_cca.shape)
    data_cca.to_csv('HMLIDS_cca.csv', index=False)
    selectedfeature_cca = data_cca['feature_2'].tolist()
    
    return selectedfeature_cca


# ICA technique
def feature_selection_ica(x,y):
    ICA = FastICA(n_components=1, random_state=12) 
    x_ica=ICA.fit_transform(x)
    #print(x_ica.shape)
    
    # create a pandas DataFrame using the values of all principal components 
    # and add the label column of the original dataset
    data_ica = pd.DataFrame(x_ica, columns=['feature_3'])
    data_ica['label3'] = data.result
    #data_ica.head()
    #print(data_ica.shape)
    data_ica.to_csv('HMLIDS_ica.csv', index=False)
    selectedfeature_ica = data_ica['feature_3'].tolist()
    
    return selectedfeature_ica


# Combine all the previous techniques and get the features that have been chosen by at least count feature selection technique
# Build estimator from PCA, CCA and ICA:
FS_pca_old = feature_selection_pca(x_old_scaled,y_old)
FS_cca_old = feature_selection_cca(x_old_scaled,y_old)
FS_ica_old = feature_selection_ica(x_old_scaled,y_old)

FS_pca_balanced = feature_selection_pca(x_balanced_scaled,y_balanced)
FS_cca_balanced = feature_selection_cca(x_balanced_scaled,y_balanced)
FS_ica_balanced = feature_selection_ica(x_balanced_scaled,y_balanced)

#combined_features = FeatureUnion([("pca", FS_pca), ("cca", FS_cca), ("ica", FS_ica)])


# Use combined features to transform dataset:
#x_features = combined_features.fit(x_scaled, y).transform(x_scaled)

df_old = pd.DataFrame()
df_old['FS_pca'] = np.array(FS_pca_old)
df_old['FS_cca'] = np.array(FS_cca_old)
df_old['FS_ica'] = np.array(FS_ica_old)
df_old['label'] = y_old
display(df_old)
df_old.to_csv('HMLIDS_old_FS.csv', index=False)


df_balanced = pd.DataFrame()
df_balanced['FS_pca'] = np.array(FS_pca_balanced)
df_balanced['FS_cca'] = np.array(FS_cca_balanced)
df_balanced['FS_ica'] = np.array(FS_ica_balanced)
df_balanced['label'] = y_balanced
display(df_balanced)
df_balanced.to_csv('HMLIDS_balanced_FS.csv', index=False)


df_old.shape

df_balanced.shape


# Splitting data into training and testing sets
df_old_features = ['FS_pca','FS_cca','FS_ica']
df_balanced_features = ['FS_pca','FS_cca','FS_ica']

# for imbalanced data
# Separating out the features
x_old_FS = df_old.loc[:, df_old_features].values

# Separating out the target
y_old_FS = df_old.loc[:,['label']].values

# split data into training and test sets
x_old_FS_train, x_old_FS_test, y_old_FS_train, y_old_FS_test = train_test_split(
    x_old_FS, y_old_FS, test_size = 0.2, random_state=0)


# for balanced data
# Separating out the features
x_balanced_FS = df_balanced.loc[:, df_balanced_features].values

# Separating out the target
y_balanced_FS = df_balanced.loc[:,['label']].values

# split data into training and test sets
x_balanced_FS_train, x_balanced_FS_test, y_balanced_FS_train, y_balanced_FS_test = train_test_split(
    x_balanced_FS, y_balanced_FS, test_size = 0.2, random_state=0)



# Reading training data frame with normal and attack instances

# ######################### TRAINING ##########################
# #################### Imbalanced data #####################
# Features 
x_old_FS_train_df = pd.DataFrame(x_old_FS_train, columns = ['pca_FCA','cca_FCA','ica_FS'])

# Labels 
y_old_FS_train_df = pd.DataFrame(y_old_FS_train, columns = ['label'])

# ######################### TRAINING ###########################
# ###################### Balanced data #########################
# Features
x_balanced_FS_train_df = pd.DataFrame(x_balanced_FS_train, columns = ['pca_FCA','cca_FCA','ica_FS'])

# Labels 
y_balanced_FS_train_df = pd.DataFrame(y_balanced_FS_train, columns = ['label'])
           
    
# ########################## TESTING ############################       
# ###################### Imbalanced data ########################                                                         
# Features
x_old_FS_test_df = pd.DataFrame(x_old_FS_test, columns = ['pca_FCA','cca_FCA','ica_FS'])

# Labels
y_old_FS_test_df = pd.DataFrame(y_old_FS_test, columns = ['label'])
        
                                                                      
# ######################### Balanced data #####################                                                                      
# Features
x_balanced_FS_test_df = pd.DataFrame(x_balanced_FS_test, columns = ['pca_FCA','cca_FCA','ica_FS'])                                                                      

# Labels
y_balanced_FS_test_df = pd.DataFrame(y_balanced_FS_test, columns = ['label'])



# Classification
# KNN Classification & Evaluation
######################### KNN Training #########################

# for imbalanced data
# Split dataset into training set and test set
# 70% training and 30% test
x_train_old, x_test_old, y_train_old, y_test_old = train_test_split(x_old_FS_train_df, y_old_FS_train_df, test_size=0.3) 

# Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

# Create KNN Classifier
knn_old = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets
knn_old.fit(x_train_old, y_train_old)

# Predict the response for test dataset
y_pred_old = knn_old.predict(x_old_FS_test_df)


# for balanced data
# Split dataset into training set and test set
# 70% training and 30% test
#x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced = train_test_split(x_balanced_FS_train_list, y_balanced_FS_train_list, test_size=0.3) 

# Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

# Create KNN Classifier
knn_balanced = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets
knn_balanced.fit(x_balanced_FS_test_df, y_balanced_FS_test_df)

# Predict the response for test dataset
y_pred_balanced = knn_balanced.predict(x_balanced_FS_test_df)


# KNN Model Evaluation for k=5
print(metrics.accuracy_score(y_old_FS_test_df, y_pred_old))
print(classification_report(y_old_FS_test_df, y_pred_old))

print(metrics.accuracy_score(y_balanced_FS_test_df, y_pred_balanced))
print(classification_report(y_balanced_FS_test_df, y_pred_balanced))