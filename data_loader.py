import pandas as pd
import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import utils


class DataLoader:
    def __init__(self,
            data_path="data/",
            n_instances=1000,
            n_vars=5,
            X_noise=0,
            Y_noise=1,
            test_perc=0.2,
            cv_splits=5,
            experiment_params=None,
            rep_i=0,
    ) -> None:

        self.data_path = data_path

        self.datasets = {
            "ihdp": self.load_ihdp_data,
            "cpt": self.load_cpt_data,
            "bd": self.load_bd_data,
            "jobs": self.load_jobs_data,
            "simulation_data": self.load_causalml_mode_2,
        }

        self.X_noise = X_noise
        self.Y_noise = Y_noise
        self.n_instances = n_instances
        self.n_vars = n_vars
        self.test_perc = test_perc
        self.cv_splits = cv_splits
        self.experiment_params = experiment_params

        if experiment_params is not None:
            if "cv_splits" in experiment_params.keys():
                self.cv_splits = experiment_params["cv_splits"]
            if "n_instances" in experiment_params.keys():
                self.n_instances = experiment_params["n_instances"]
            if "n_vars" in experiment_params.keys():
                self.n_vars = experiment_params["n_vars"]
            if "X_noise" in experiment_params.keys():
                self.X_noise = experiment_params["X_noise"]
            if "Y_noise" in experiment_params.keys():
                self.Y_noise = experiment_params["Y_noise"]
            if "test_perc" in experiment_params.keys():
                self.test_perc = experiment_params["test_perc"]

        self.test_weights = None
        self.rep_i = rep_i
        self.train_df_orig = None

    def load_dataset(self, dataset_name):
        self.datasets[dataset_name]()  # self.datasets是字典, 其键是实例的方法所以后面用()
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=0, ) # shuffle=self.random_state is not None,
        self.cv_indexes = list(kf.split(self.train_df))

    def load_ihdp_data(self):
        i = self.rep_i  # repetition index, 指定数据集的重复索引, 类似于第几号数据集, 范围[0, 999]

        data = np.load(self.data_path + "ihdp/ihdp_npci_1-1000.train.npz")
        data_test = np.load(self.data_path + "ihdp/ihdp_npci_1-1000.test.npz")
        # data = np.load(self.data_path + "ihdp/ihdp_npci_1-100.train.npz")
        # data_test = np.load(self.data_path + "ihdp/ihdp_npci_1-100.test.npz")

        # data["x"]是一个三维数组
        train_df = pd.DataFrame(data["x"][:, :, i])
        colnames = [f"x{c}" for c in range(train_df.shape[1])]
        train_df.columns = colnames
        train_df["Y"] = data["yf"][:, i]
        train_df["T"] = data["t"][:, i]

        test_df = pd.DataFrame(data_test["x"][:, :, i])
        test_df.columns = colnames
        test_df["Y"] = data_test["yf"][:, i]
        test_df["T"] = data_test["t"][:, i]

        self.X_features = colnames + ["T"]
        self.Y_feature = ["Y"]

        self.train_df = train_df
        self.test_df = test_df
        self.train_ite = data["mu1"][:, i] - data["mu0"][:, i]
        self.test_ite = data_test["mu1"][:, i] - data_test["mu0"][:, i]
        self.mu0_train = data["mu0"][:, i]
        self.mu1_train = data["mu1"][:, i]
        self.mu0_test = data_test["mu0"][:, i]
        self.mu1_test = data_test["mu1"][:, i]

    def load_cpt_data(self):

        data = pd.read_csv(self.data_path + "cpt/cpt.csv")

        mean_values = data.mean()
        data = data.fillna(mean_values)  # 使用每列的均值填充缺失值

        # # corr >= 0.1
        # filtered_features = ['CPT', 'Mo', 'Ni', 'N', 'PP', 'P', 'Si', 'W', 'Cr', 'NaCl', 'testing_area', 'Nb', 'Cu', 'time', 'Mn', 'temperature', 'S', 'C']

        # # corr >= 0.05
        # filtered_features = ['CPT', 'Mo', 'Ni', 'N', 'PP', 'P', 'Si', 'W', 'Cr', 'NaCl', 'testing_area', 'Nb', 'Cu', 'time', 'Mn', 'temperature', 'S', 'C', 'heating_rate', 'Al', 'V']
        #
        # data = data.loc[:, filtered_features]

        # data.drop(['PP'], axis=1)  # 删除指定列
        # data = data_loc

        # data = data_drop
        data = utils.normalize(data)  # 对整个数据集标准化

        # data.dropna(axis=0, how='any', subset=['PP'], inplace=True)  # 删除缺失值
        # data.drop(['Co', 'Fe'], axis=1, inplace=True)

        # X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.25, random_state=42)

        # self.test_df = data.iloc[::5]
        # self.train_df = data.drop(self.test_df.index)

        # 取 70% 的数据作为训练集
        self.train_df = data.sample(frac=0.7, random_state=0)
        # 剩余 30% 的数据作为测试集
        self.test_df = data.drop(self.train_df.index)
        # self.test_df = self.train_df

        colnames = [f"x{c}" for c in range(self.train_df.shape[1] - 1)]
        self.X_features = colnames
        self.Y_feature = ["Y"]  # CPT
        data.columns = colnames + ["Y"]

        self.train_df = data.iloc[self.train_df.index, :].reset_index()
        self.test_df = data.iloc[self.test_df.index, :].reset_index()

    def load_bd_data(self):
        data = pd.read_csv(self.data_path + "bd/bd.csv", index_col=0)  # brain damage dataset

        data = data.drop(['AIDS', 'Surgically evacuated'], axis=1)

        # # corr >= 0.1
        # filtered_features = ['Expired', 'ICP-24H', 'ICP-POR', 'Light(R)_0.0', 'Light(L)_0.0',
        #                      'Light(R)_1.0', 'Midline shift', 'RDW', 'Surgery', 'Light(B)',
        #                      'Basal cistern_0.0', 'APTT', 'INR(PT)', 'PT', 'Light(L)_1.0',
        #                      'Basal cistern_2.0', 'SAH convexities_2.0', 'RBC', 'SDH Rt_2.0',
        #                      'M_1.0', 'HCT', 'HGB', 'CREA', 'Thrid ventricle_2.0', 'Antiplatelet',
        #                      'PLT', 'V_4.0', 'Cause of trauma_1.0', 'SAH convexities_1.0', 'eGFR(M)',
        #                      'BUN', 'Thrid ventricle_0.0', 'SAH basal cistern_0.0',
        #                      'End stage renal disease', 'Metastatic cancer', 'Temperature',
        #                      'SAH basal cistern_2.0', 'M_3.0', 'Other body trauma', 'GLU', 'BAND',
        #                      'V_5.0']

        # corr >= 0.15
        filtered_features = ['Expired', 'ICP-24H', 'ICP-POR', 'Light(R)_0.0', 'Light(L)_0.0',
                             'Light(R)_1.0', 'Midline shift', 'RDW', 'Surgery', 'Light(B)',
                             'Basal cistern_0.0', 'APTT', 'INR(PT)', 'PT', 'Light(L)_1.0',
                             'Basal cistern_2.0', 'SAH convexities_2.0', 'RBC', 'SDH Rt_2.0',
                             'M_1.0', 'HCT', 'HGB', 'CREA']
        data = data.loc[:, filtered_features]


        # 通过 pop() 提取列并追加到最后
        column_to_move = data.pop('Expired')  # 移除指定列数据
        data['Expired'] = column_to_move  # 将列 'Expired' 添加到最后

        # 取 70% 的数据作为训练集
        self.train_df = data.sample(frac=0.7, random_state=0)
        # 剩余 30% 的数据作为测试集
        self.test_df = data.drop(self.train_df.index)

        colnames = [f"x{c}" for c in range(self.train_df.shape[1] - 1)]
        self.X_features = colnames
        self.Y_feature = ["Y"]  # CPT
        data.columns = colnames + ["Y"]

        self.train_df = data.iloc[self.train_df.index, :].reset_index()
        self.test_df = data.iloc[self.test_df.index, :].reset_index()

    def load_jobs_data(self):

        i = self.rep_i
        data = np.load(self.data_path + "jobs/train.npz")
        data_test = np.load(self.data_path + "jobs/test.npz")

        train_df = pd.DataFrame(data["x"][:, :, i])
        colnames = [f"x{c}" for c in range(train_df.shape[1])]
        train_df.columns = colnames
        train_df["T"] = data["t"][:, i]
        train_df["Y"] = data["yf"][:, i]
        # train_df["e"] = data["e"][:, i]

        test_df = pd.DataFrame(data_test["x"][:, :, i])
        test_df.columns = colnames
        test_df["T"] = data_test["t"][:, i]
        test_df["Y"] = data_test["yf"][:, i]
        # test_df["e"] = data_test["e"][:, i]

        self.X_features = colnames + ["T"]
        self.Y_feature = ["Y"]

        self.train_df = train_df
        self.test_df = test_df

        # self.real_ate = data["ate"]
        # self.test_ite = np.array([99999] * test_df.shape[0])
        # self.train_ite = np.array([99999] * train_df.shape[0])

    def load_causalml_mode_2(self, adj=0.0):

        # import time
        # np.random.seed(int(time.time()))  # 每次运行，数据集都会变动; 不使用的话每次运行数据集是一样的, 因为数据分布生成函数默认数据保持不变
        # np.random.seed(0)  # default seed=0
        # np.random.seed(12)

        self.X_noise = 0

        X_noise = self.X_noise
        Y_noise = self.Y_noise
        n = self.n_instances * 2
        p = self.n_vars

        def generate_lingam_data(n_samples=1000, n_features=3, noise_type='laplace'):

            # 1. 随机生成一个上三角矩阵 B，保证是无环的
            B = np.triu(np.random.uniform(low=-2, high=2, size=(n_features, n_features)), k=1)

            # 2. 生成非高斯噪声 e
            if noise_type == 'laplace':
                e = np.random.laplace(loc=0.0, scale=1.0, size=(n_samples, n_features))
            elif noise_type == 'uniform':
                e = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
            elif noise_type == 'exponential':
                e = np.random.exponential(scale=1.0, size=(n_samples, n_features)) - 1  # 使其均值为0
            else:
                raise ValueError("Unsupported noise type")

            # 3. 解线性方程：X = (I - B)^{-1} e
            I = np.eye(n_features)
            X = e @ np.linalg.inv(I - B).T

            return X, B

        # X, B = generate_lingam_data(n_samples=1000, n_features=10, noise_type='uniform')
        # X = (np.random.uniform(size=n * p) + np.random.normal(0, X_noise, size=n * p)).reshape((n, -1))

        X = (np.random.uniform(size=n * p)).reshape((n, -1))


        X[:, 0] = X[:, 3] + X[:, 4] + np.random.uniform(low=0, high=1, size=n)
        X[:, 1] = X[:, 5] + X[:, 6] + np.random.uniform(low=0, high=1, size=n)
        X[:, 2] = X[:, 1] + np.random.uniform(low=0, high=2, size=n)


        b = (np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1], X[:, 2]) + np.maximum(np.repeat(0.0, n), X[:, 3] + X[:, 4]))

        # b = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1], X[:, 2]) + np.maximum(np.repeat(0.0, n), X[:, 3] + X[:, 4])

        tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))

        # 二元处理变量
        e = np.repeat(0.5, n)
        w = np.random.binomial(1, e, size=n)

        # y = np.power((w - 0.5) * tau, 2) + Y_noise * np.random.uniform(low=-1.0, high=1.0, size=n)
        # y = np.power(b + (w - 0.5) * tau, 2) + Y_noise * np.random.uniform(low=-1.0, high=1.0, size=n)

        # y =  np.power(X[:, 1] + X[:, 2] + X[:, 3], 3) + Y_noise * np.random.uniform(low=-1, high=1.0, size=n)
        y = np.power(1 + (w - 0.5) * tau, 2) + np.exp(b) + Y_noise * np.random.uniform(low=-1, high=1.0, size=n)

        # X[:, 7] = y + np.random.uniform(low=-1, high=1.0, size=n)

        # print("Causal coefficient matrix B:\n", B)

        # X = (np.random.normal(loc=0, scale=1, size=n * p) + np.random.uniform(low=0, high=X_noise, size=n * p)).reshape((n, -1))  # X_noise=0
        #
        # X[:, 0] = X[:, 3] + X[:, 4] + np.random.uniform(low=0, high=1, size=n)
        # X[:, 1] = X[:, 5] + X[:, 6] + np.random.uniform(low=0, high=1, size=n)
        # X[:, 2] = X[:, 1] + np.random.uniform(low=0, high=2, size=n)

        # X[:, 0] = X[:, 2] + X[:, 3] + np.random.uniform(low=-1, high=1, size=n)
        # X[:, 1] = X[:, 5] + X[:, 6] + np.random.uniform(low=-1, high=1, size=n)
        # X[:, 2] = X[:, 8] + X[:, 9] + np.random.uniform(low=-1, high=1, size=n)
        #
        # X[:, 11] = X[:, 0] + np.random.uniform(low=-1, high=1, size=n)
        # X[:, 12] = X[:, 2] + np.random.uniform(low=-1, high=1, size=n)
        #
        # y = X[:, 0] + X[:, 1] + X[:, 2] + Y_noise * np.random.uniform(low=-1, high=1, size=n)



        # X = (np.random.uniform(size=n * p) + np.random.normal(0, X_noise, size=n * p)).reshape((n, -1))  # X_noise=0
        # b = 2 * np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1], X[:, 2]) + np.maximum(
        #     np.repeat(0.0, n), X[:, 3] + X[:, 4]
        # )
        # e = np.repeat(0.5, n)
        # tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))
        #
        # w = np.random.binomial(1, e, size=n)
        # y = b + (w - 0.5) * tau + Y_noise * np.random.normal(size=n)


        train_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        train_df["Y"] = y

        train_df["T"] = w
        self.X_features = [f"col_{i}" for i in range(X.shape[1])] + ["T"]

        # self.X_features = [f"col_{i}" for i in range(X.shape[1])]
        self.Y_feature = ["Y"]

        # 训练集和测试集的比例7:3, 2*self.n_instances为样本总量
        self.train_df = train_df.iloc[: int(2*self.n_instances*0.7-1), :].reset_index()
        self.test_df = train_df.iloc[int(2*self.n_instances*0.7):, :].reset_index()

        return

