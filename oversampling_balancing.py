from extracting_normal_attacks import *

# Oversampling balancing function
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np

def overSamplingTech(x, y):
    o = RandomOverSampler()
    x_FS_balanced, y_FS_balanced= o.fit_resample(x, y)
    return x_FS_balanced, y_FS_balanced


# plotting balanced attacks before & after
def plotting_balancing(y_old, y_new, index):
    # initialize data of lists
    data_ = {'Before Balancing': y_old.shape[0],
             'After Balancing': y_new.shape[0]}

    # Creates pandas DataFrame
    df = pd.DataFrame(data=data_, index=index)
    df.plot.bar(title='Comparison before & after balancing')

    return

# balancing aggregated attacks to each others
def balancing_attacks(FS_agg_attacks):
    from matplotlib import pyplot as plt
    x_FS_agg_attacks_balanced, y_FS_agg_attacks_balanced = \
        overSamplingTech(FS_agg_attacks['x_FS_agg_attacks'].to_numpy(),
                          FS_agg_attacks['y_FS_agg_attacks'].to_numpy())

    # plotting normal & aggregated attacks before & after balancing
    plotting_balancing(FS_agg_attacks['y_FS_agg_attacks'], y_FS_agg_attacks_balanced, index=['attacks'])
    plt.tight_layout()
    plt.savefig(r"results\Attacks comparison before & after balancing.png")

    return x_FS_agg_attacks_balanced, y_FS_agg_attacks_balanced


# balancing normal & balanced aggregated attacks
def balancing_all(FS_normal, x_FS_agg_attacks_balanced, y_FS_agg_attacks_balanced):
    from matplotlib import pyplot as plt
    # merge x's and y's for pca, cca & ica for normal & aggregated attacks
    x_frames = np.concatenate((FS_normal['x_FS_normal'], x_FS_agg_attacks_balanced))
    y_frames = np.concatenate((FS_normal['y_FS_normal'], y_FS_agg_attacks_balanced))
    # x_merge = pd.concat(x_frames)
    # y_merge = pd.concat(y_frames)

    # balancing normal & attacks
    x_FS_normal_agg_attacks_balanced, y_FS_normal_agg_attacks_balanced = \
        overSamplingTech(x_frames, y_frames)

    # plotting normal & aggregated attacks before & after balancing
    plotting_balancing(y_FS_normal_agg_attacks_balanced, y_frames, index=['normal & attacks'])
    plt.tight_layout()
    plt.savefig(r"results\Normal & aggregated attacks comparison before & after balancing.png")

    return x_FS_normal_agg_attacks_balanced, y_FS_normal_agg_attacks_balanced


