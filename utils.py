import numpy as np
import pandas as pd

# # auxiliary function 辅助函数

# array_to_batch表示制作分批量的数据
def array_to_batch(data, batch_size):
    num_batches = np.floor(len(data) / batch_size)  # np.floor(x)对x向下取整

    if len(data) % batch_size == 0:
        # Split an array into multiple sub-arrays, 是按顺序分组的, 没有随机性
        batches = np.array_split(data, num_batches)
    else:
        batches = np.array_split(data[: -(len(data) % batch_size)], num_batches)

    return np.array(batches)


def normalization(data):
    """规范化"""
    data_normalize = data.copy()  # 将data内容复制给data_normlize进行初始化
    cols = data_normalize.columns  # 取列名
    # for i in range(data.shape[-1]):  # 取每一列的索引
    # i = data.shape[-1] - 1
    i = data.shape[-1] - 2  # 获取Y的数据
    data_normalize.loc[:, cols[i]] = (data.loc[:, cols[i]] - data.loc[:, cols[i]].min()) / (
            data.loc[:, cols[i]].max() - data.loc[:, cols[i]].min())
    # data_normalize.loc[:, cols[i]].apply(lambda x: x - x.mean() / x.std())

    return data_normalize


def normalize(data):
    """对DataFrame进行列标准化
    Parameters:
        data (DataFrame): 需要标准化的数据集
    Returns:
        data_normlize (DataFrame): 已标准化的数据集
    Example:
        >>> data = pd.DataFrame({'0':[1,2,3], '1':[2,3,4]})
        >>> data_normalize = normalize(data)
        >>> data_normalize
        Output:
             0    1
        0 -1.0 -1.0
        1  0.0  0.0
        2  1.0  1.0
    Note:
        # https://blog.csdn.net/onlyforbest/article/details/110820814
        # https://blog.csdn.net/Asher117/article/details/86530816
        # https://gitcode.csdn.net/65ed7f4a1a836825ed79b855.html
        # https://blog.csdn.net/ziqingnian/article/details/114936316
        # https://blog.csdn.net/weixin_35757704/article/details/124795226
        # https://cloud.baidu.com/article/2794043
        # https://www.investopedia.com/terms/z/zscore.asp
    """
    data_normalize = data.copy()  # 将data内容复制给data_normlize进行初始化
    cols = data_normalize.columns  # 取列名
    for i in range(data.shape[-1]):  # 取每一列的索引
        data_normalize.loc[:, cols[i]] = (data.loc[:, cols[i]] - data.loc[:, cols[i]].mean()) / data.loc[:, cols[i]].std()
        # data_normalize.loc[:, cols[i]].apply(lambda x: x - x.mean() / x.std())

    return data_normalize


def inverse_normalize(test_df, pred_y):
    """将目标变量Y的数据逆标准化"""
    # cols = test_df.columns  # 取列名
    # y_index = test_df.shape[-1] - 2  # 列名"Y"的索引
    # test_y_mean = test_df.loc[:, cols[y_index]].mean()
    # test_y_std = test_df.loc[:, cols[y_index]].std()

    test_y_mean = test_df.loc[:, "Y"].mean()  # 不可用test_df.loc[:, ["Y"]], 这得出的是dataframe, 需要的是series
    test_y_std = test_df.loc[:, "Y"].std()
    inverse_normalize_pred_y = pred_y * test_y_std + test_y_mean

    return inverse_normalize_pred_y
