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
        self.datasets[dataset_name]() 
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=0, )
        self.cv_indexes = list(kf.split(self.train_df))

    def load_ihdp_data(self):

        self.train_df =pd.read_csv(self.data_path + "ihdp/train_df.csv")
        self.test_df =pd.read_csv(self.data_path + "ihdp/test_df.csv")

        columns = [f"x{c}" for c in range(self.train_df.shape[1]-3)]

        self.X_features = columns + ["T"]
        self.Y_feature = ["Y"]