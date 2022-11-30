from features_selection import *

# Extracting normal & each attack & aggregated attacks instances from fd_FS
def extracting_normal_attacks(df_FS):
    # Extract features & labels of normal & each attack type from df_FS
    df_FS_normal = df_FS.loc[(df_FS['label'] == 0)]  # normal instances
    x_FS_normal = df_FS_normal.loc[:, df_FS_normal.columns != 'label']  # normal features
    y_FS_normal = df_FS_normal['label']  # normal labels

    df_FS_NMRI = df_FS.loc[(df_FS['label'] == 1)]  # NMRI attacks instances
    x_FS_NMRI = df_FS_NMRI.loc[:, df_FS_NMRI.columns != 'label']  # NMRI features
    y_FS_NMRI = df_FS_NMRI['label']  # NMRI labels

    df_FS_CMRI = df_FS.loc[(df_FS['label'] == 2)]  # CMRI attacks instances
    x_FS_CMRI = df_FS_CMRI.loc[:, df_FS_CMRI.columns != 'label']  # CMRI features
    y_FS_CMRI = df_FS_CMRI['label']  # CMRI labels

    df_FS_MSCI = df_FS.loc[(df_FS['label'] == 2)]  # MSCI attacks instances
    x_FS_MSCI = df_FS_MSCI.loc[:, df_FS_MSCI.columns != 'label']  # MSCI features
    y_FS_MSCI = df_FS_MSCI['label']  # MSCI labels

    df_FS_MPCI = df_FS.loc[(df_FS['label'] == 3)]  # MPCI attacks instances
    x_FS_MPCI = df_FS_MPCI.loc[:, df_FS_MPCI.columns != 'label']  # MPCI features
    y_FS_MPCI = df_FS_MPCI['label']  # MPCI labels

    df_FS_MFCI = df_FS.loc[(df_FS['label'] == 4)]  # MFCI attacks instances
    x_FS_MFCI = df_FS_MFCI.loc[:, df_FS_MFCI.columns != 'label']  # MFCI features
    y_FS_MFCI = df_FS_MFCI['label']  # MFCI labels

    df_FS_DoS = df_FS.loc[(df_FS['label'] == 1)]  # DoS attacks instances
    x_FS_DoS = df_FS_DoS.loc[:, df_FS_DoS.columns != 'label']  # DoS features
    y_FS_DoS = df_FS_DoS['label']  # DoS labels

    df_FS_Recon = df_FS.loc[(df_FS['label'] == 1)]  # Recon attacks instances
    x_FS_Recon = df_FS_Recon.loc[:, df_FS_Recon.columns != 'label']  # Recon features
    y_FS_Recon = df_FS_Recon['label']  # Recon labels

    df_FS_agg_attacks = df_FS.loc[(df_FS['label'] != 0)]  # Aggregated Attacks
    x_FS_agg_attacks = df_FS_agg_attacks.loc[:, df_FS_agg_attacks.columns != 'label']  # agg_attacks features
    y_FS_agg_attacks = df_FS_agg_attacks['label']  # agg_attacks labels

    FS_normal = {"df": {"df_FS_normal": df_FS_normal, "df_FS_normal_length": df_FS_normal.shape},
                 "x_FS_normal": x_FS_normal, "y_FS_normal": y_FS_normal, "FS_normal_length": len(x_FS_normal)}

    FS_NMRI = {"df": {"df_FS_NMRI": df_FS_NMRI, "df_FS_NMRI_length": df_FS_NMRI.shape},
               "x_FS_NMRI": x_FS_NMRI, "y_FS_NMRI": y_FS_NMRI, "FS_NMRI_length": len(x_FS_NMRI)}

    FS_CMRI = {"df": {"df_FS_CMRI": df_FS_CMRI, "df_FS_CMRI_length": df_FS_CMRI.shape},
               "x_FS_CMRI": x_FS_CMRI, "y_FS_CMRI": y_FS_CMRI, "FS_CMRI_length": len(x_FS_CMRI)}

    FS_MSCI = {"df": {"df_FS_MSCI": df_FS_MSCI, "df_FS_MSCI_length": df_FS_MSCI.shape},
               "x_FS_MSCI": x_FS_MSCI, "y_FS_MSCI": y_FS_MSCI, "FS_MSCI_length": len(x_FS_MSCI)}

    FS_MPCI = {"df": {"df_FS_MPCI": df_FS_MPCI, "df_FS_MPCI_length": df_FS_MPCI.shape},
               "x_FS_MPCI": x_FS_MPCI, "y_FS_MPCI": y_FS_MPCI, "FS_MPCI_length": len(x_FS_MPCI)}

    FS_MFCI = {"df": {"df_FS_MFCI": df_FS_MFCI, "df_FS_MFCI_length": df_FS_MFCI.shape},
               "x_FS_MFCI": x_FS_MFCI, "y_FS_MFCI": y_FS_MFCI, "FS_MFCI_length": len(x_FS_MFCI)}

    FS_DoS = {"df": {"df_FS_DoS": df_FS_DoS, "df_FS_DoS_length": df_FS_DoS.shape},
              "x_FS_DoS": x_FS_DoS, "y_FS_DoS": y_FS_DoS, "FS_DoS_length": len(x_FS_DoS)}

    FS_Recon = {"df": {"df_FS_Recon": df_FS_Recon, "df_FS_Recon_length": df_FS_Recon.shape},
                "x_FS_Recon": x_FS_Recon, "y_FS_Recon": y_FS_Recon, "FS_Recon_length": len(x_FS_Recon)}

    FS_agg_attacks = {
        "df": {"df_FS_agg_attacks": df_FS_agg_attacks, "df_FS_agg_attacks_length": df_FS_agg_attacks.shape},
        "x_FS_agg_attacks": x_FS_agg_attacks, "y_FS_agg_attacks": y_FS_agg_attacks,
        "FS_agg_attacks_length": len(x_FS_agg_attacks)}

    return FS_normal, FS_NMRI, FS_CMRI, FS_MSCI, FS_MPCI, FS_MFCI, FS_DoS, FS_Recon, FS_agg_attacks