from AdaBoost_classifier import *


def SCADA_IDS_system(data_csv):
    # initialization
    df, x, y, headerList = initialization(data_csv)
    print("... Initialization ...")
    print("- df shape: ", df.shape)

    # standardization
    print("... Scaling dataset ...")
    x_scaled = standardization(x, y)
    df_scaled = pd.DataFrame(x_scaled, columns=headerList[:-1])
    df_scaled['label'] = y
    df_scaled.to_csv(r"results\df_scaled.csv")
    print("- df_scaled shape: ", df_scaled.shape)

    # dimension reduction
    print("... Features Reduction ...")
    p1, p2, p3 = dimensionality_reduction(x_scaled, y)
    p1_df = pd.DataFrame(p1, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    p2_df = pd.DataFrame(p2, columns=['1'])
    p3_df = pd.DataFrame(p3, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
    feature_selection_count_df, extracted_features_names = get_fs_features(df, p1_df, p2_df, p3_df, 1)

    # create a dataframe with extracted features & label
    df_FS = df_scaled[extracted_features_names]
    df_FS['label'] = y
    df_FS.to_csv(r"results\df_FS.csv")
    print("- df_FS shape: ", df_FS.shape)

    # extracting normal/attacks instances
    print("... Normal/Attacks extraction ...")
    FS_normal, FS_NMRI, FS_CMRI, FS_MSCI, FS_MPCI, FS_MFCI, FS_DoS, FS_Recon, FS_agg_attacks = \
        extracting_normal_attacks(df_FS)

    # balancing attacks to each other
    print("... Balancing attacks ...")
    x_FS_agg_attacks_balanced, y_FS_agg_attacks_balanced = balancing_attacks(FS_agg_attacks)

    # balancing attacks to normal
    print("... Balancing normal & attacks ...")
    x_FS_normal_agg_attacks_balanced, y_FS_normal_agg_attacks_balanced = balancing_all(FS_normal,
                                                                                       x_FS_agg_attacks_balanced,
                                                                                       y_FS_agg_attacks_balanced)
    # binary conversion of labels
    print("... Binary conversion ...")
    x_FS_balanced_binary, y_FS_balanced_binary = binary_conversion(x_FS_normal_agg_attacks_balanced,
                                                                   y_FS_normal_agg_attacks_balanced,
                                                                   extracted_features_names)

    # 80% train & 20% test data split
    print("... Train/Test Data ...")
    x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced = train_test_data(x_FS_balanced_binary,
                                                                                           y_FS_balanced_binary)

    print("x_train_balanced: ", len(x_train_balanced))
    print("y_train_balanced: ", len(y_train_balanced))
    print("x_test_balanced: ", len(x_test_balanced))
    print("y_test_balanced: ", len(y_test_balanced))

    # KNN classifier & evaluation
    print("... KNN classification ...")
    report_KNN, cm_display_KNN = KNN_classifier(x_train_balanced, y_train_balanced, x_test_balanced, y_test_balanced)
    print("KNN classification report")
    print(report_KNN)
    print("\n")

    plt.figure("Figure 1")
    cm_display_KNN.plot()
    plt.tight_layout()
    plt.savefig(r"results\KNN Confusion Matrix.png")

    # AdaBoost classifier & evaluation
    print("... AdaBoost classification ...")
    report_AdaBoost, cm_display_AdaBoost = AdaBoost_classifier(x_train_balanced, y_train_balanced, x_test_balanced,
                                                               y_test_balanced)
    print("AdaBoost classification report")
    print(report_AdaBoost)
    print("\n")

    plt.figure("Figure 2")
    cm_display_AdaBoost.plot()
    plt.tight_layout()
    plt.savefig(r"results\AdaBoost Confusion Matrix.png")

    return


data_csv = 'Gaspipelinedatasetfull.csv'
SCADA_IDS_system(data_csv)
