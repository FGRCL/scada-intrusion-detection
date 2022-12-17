from standardization import *


# PCA components determination
# principal components explain a part of the variance. From the Scikit-learn implementation, we can get the information
# about the explained variance and plot the cumulative variance.
def PCA_cum_variance(x_scaled):
    import matplotlib.pyplot as plt
    PCA_ = PCA().fit(x_scaled)
    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, 27, step=1)
    y = np.cumsum(PCA_.explained_variance_ratio_)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 26, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)
    ax.grid(axis='x')
    plt.tight_layout()
    plt.savefig(r"results\PCA Cumulative Variance.png")

    return


# PCA dimension reduction
def PCA_features(x_scaled):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=12)
    # fit PCA on training set. Fitting PCA on the training set only
    pca.fit(x_scaled)
    # apply the mapping (transform) to both the training set and the test set
    # FM array
    p1 = pca.transform(x_scaled)

    return p1


# CCA components determination
def CCA_features_correlation(x_scaled):
    import matplotlib.pyplot as plt
    x_scaled_df = pd.DataFrame(x_scaled,
                               columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                                        '16',
                                        '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'])
    # Correlation between different variables
    plt.figure(figsize=(10, 7))
    corr = x_scaled_df.corr()
    sns.heatmap(corr)
    plt.title('CCA Features Correlation')
    plt.savefig(r"results\CCA Features Correlation.png")

    return


# CCA dimension reduction
def CCA_features(x, y):
    cca = CCA(n_components=1)
    cca.fit(x, y)
    x_cca, y_cca = cca.transform(x, y)
    p2 = x_cca

    return p2


# dimensionality reduction
def dimensionality_reduction(x_scaled, y):
    PCA_cum_variance(x_scaled)
    p1 = PCA_features(x_scaled)
    CCA_features_correlation(x_scaled)
    p2 = CCA_features(x_scaled, y)

    return p1, p2


# Get the features that have been selected by at least by one technique
def get_fs_features(df, p1_df, p2_df, count):
    # Get combined features
    pd.set_option('display.max_rows', None)
    df = df.loc[:, df.columns != "result"]
    feature_name = list(df.columns)
    pca_index = int(list(p1_df.columns)[-1])
    cca_index = int(list(p2_df.columns)[-1])
    feature_selection_df = pd.DataFrame({'Feature': feature_name})
    pca_extracted_features = []
    cca_extracted_features = []
    feature_selection_df["PCA"] = ""
    feature_selection_df["CCA"] = ""
    feature_selection_df["Total"] = ""
    for i in range(0, pca_index):
        pca_extracted_features.append(feature_selection_df['Feature'].iloc[i])
        feature_selection_df.loc[i, 'PCA'] = "True"
        feature_selection_df['PCA'] = feature_selection_df['PCA'].replace('', 'False')

    for j in range(0, cca_index):
        cca_extracted_features.append(feature_selection_df['Feature'].iloc[j])
        feature_selection_df.loc[j, 'CCA'] = "True"
        feature_selection_df['CCA'] = feature_selection_df['CCA'].replace('', 'False')

    # Count the frequency of each feature being selected by PCA & CCA
    feature_selection_df['Total'] = feature_selection_df.apply(lambda x: x.str.contains("True").sum(), axis=1)
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False, ignore_index=True)
    dataframe_image.export(feature_selection_df, r"results\Extracted Features Occurrences.png")
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)
    f = feature_selection_df[feature_selection_df.Total >= count]
    extracted_features = list(f['Feature'])

    return feature_selection_df, extracted_features
