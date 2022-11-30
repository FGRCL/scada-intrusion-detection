from oversampling_balancing import *

# Binary Conversion
def binary_conversion(x_FS_normal_agg_attacks_balanced, y_FS_normal_agg_attacks_balanced, extracted_features_names):
    x_FS_normal_agg_attacks_balanced_df = pd.DataFrame(x_FS_normal_agg_attacks_balanced,
                                                       columns=extracted_features_names)
    y_FS_balanced_binary = pd.DataFrame(y_FS_normal_agg_attacks_balanced).copy()
    y_FS_balanced_binary.columns = ["label"]
    y_FS_balanced_binary.label = y_FS_balanced_binary.label.astype(int)
    y_FS_balanced_binary['label'] = y_FS_balanced_binary['label'].replace([2, 3, 4, 5, 6, 7], 1)
    df_FS_balanced_binary = pd.DataFrame(x_FS_normal_agg_attacks_balanced_df).copy()
    df_FS_balanced_binary['label'] = y_FS_balanced_binary
    df_FS_balanced_binary.to_csv(r"results\df_FS_balanced_binary.csv")
    x_FS_balanced_binary = df_FS_balanced_binary.loc[:, df_FS_balanced_binary.columns != 'label']
    y_FS_balanced_binary = df_FS_balanced_binary['label']

    return x_FS_balanced_binary, y_FS_balanced_binary
