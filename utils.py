import numpy as np
import pandas as pd

def array_to_batch(data, batch_size):
    num_batches = np.floor(len(data) / batch_size)

    if len(data) % batch_size == 0:
        batches = np.array_split(data, num_batches)
    else:
        batches = np.array_split(data[: -(len(data) % batch_size)], num_batches)

    return np.array(batches)


def normalization(data):
    data_normalize = data.copy()
    cols = data_normalize.columns
    i = data.shape[-1] - 2
    data_normalize.loc[:, cols[i]] = (data.loc[:, cols[i]] - data.loc[:, cols[i]].min()) / (
            data.loc[:, cols[i]].max() - data.loc[:, cols[i]].min())
    # data_normalize.loc[:, cols[i]].apply(lambda x: x - x.mean() / x.std())

    return data_normalize


def normalize(data):
    data_normalize = data.copy()
    cols = data_normalize.columns
    for i in range(data.shape[-1]):
        data_normalize.loc[:, cols[i]] = (data.loc[:, cols[i]] - data.loc[:, cols[i]].mean()) / data.loc[:, cols[i]].std()
        # data_normalize.loc[:, cols[i]].apply(lambda x: x - x.mean() / x.std())

    return data_normalize


def inverse_normalize(test_df, pred_y):
    test_y_mean = test_df.loc[:, "Y"].mean()
    test_y_std = test_df.loc[:, "Y"].std()
    inverse_normalize_pred_y = pred_y * test_y_std + test_y_mean

    return inverse_normalize_pred_y
