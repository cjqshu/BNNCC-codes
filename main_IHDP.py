import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn import metrics
import configargparse
from torch.optim import lr_scheduler

import causal_discovery, data_loader, utils

from models.bnncc_regression import BnnccRegressor, bnncc_reg_loss

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.display.float_format = '{:.3f}'.format


def get_parser():
    """get default arguments."""
    parser = configargparse.ArgumentParser(
        description="causal learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument('--dataset_name', default="ihdp", type=str, choices=["ihdp", "cpt", "jobs"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--hidden_size", default=12, type=int)
    parser.add_argument("--hidden_layers", default=10, type=int)
    parser.add_argument("---output_size", default=1, type=int)
    parser.add_argument("--data_dir", default="data/", type=str, help="data directory")
    parser.add_argument("--rep_i", default=16, type=int, help="index number of the repetitive dataset is in [0, 999]")
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--n_epoch", default=50, type=int, help="number of epochs")
    parser.add_argument("--early_stop", default=0, type=int, help="early stopping patience")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float) 
    parser.add_argument("--n_experiments", default=1, type=int, help="number of experiments")
    parser.add_argument("--save_model", default=False, type=bool, help="save model or not")
    parser.add_argument("--save_dir", default="results/", type=str, help="directory of results")
    parser.add_argument("--baselines_status", default=False, type=bool, help="baselines or not")
    parser.add_argument("--ablations", default=False, type=bool, help="ablation or not")
    parser.add_argument("--predefined_dag", default=False, type=bool, help="predefined dag or not")

    return parser


def set_random_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.set_default_dtype(torch.float32)


def load_data(args, n_instances=170, n_vars=6, Y_noise=0.5):
    """data to load"""
    data = data_loader.DataLoader(
        data_path=args.data_dir,  # data_path="./data/"
        n_instances=n_instances,
        n_vars=n_vars,
        X_noise=0,
        Y_noise=Y_noise,
        test_perc=0.2,
        cv_splits=5,
        rep_i=args.rep_i,
    )

    data.load_dataset(args.dataset_name)

    return data


def data_preprocessing(train_df, test_df, X_features, y_feature):
    train_df = utils.normalize(train_df)
    test_df = utils.normalize(test_df)
    train_X = train_df[X_features].to_numpy()
    train_y = train_df[y_feature].to_numpy()
    test_X = test_df[X_features].to_numpy()
    test_y = test_df[y_feature].to_numpy()

    return train_X, train_y, test_X, test_y


def get_causal_groups(args, data):
    """Extract DAG"""
    dag_creator = causal_discovery.DAGCreator(data.train_df, data.X_features, data.Y_feature, method="icalingam", max_samples=1000, )

    if args.predefined_dag:
        dag_edges = dag_creator.return_predefined_dag("ihdp")
    else:
        dag_edges = dag_creator.create_dag_edges()

    nodes_in_edges = [item for sublist in dag_edges for item in sublist]

    nodes_not_in_edges = [ x for x in dag_creator.X_features if x not in nodes_in_edges ]
    dag_edges.append(nodes_not_in_edges)

    results = dag_edges

    return results


def get_causal_groups_data(args, causal_groups, train_X, train_y, test_X, test_y, X_features):
    dag_edges = causal_groups
    dag_edges_idx = [[X_features.index(col) for col in sublist] for sublist in dag_edges]
    print(dag_edges_idx)

    structure_params = {}
    train_Xs = {}

    for i in range(len(dag_edges_idx)):
        structure_params[f"CG{i}"] = {
            "input_size": len(dag_edges_idx[i]),
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "output_size": args.output_size,
        }

        train_Xs[f"CG{i}"] = ( torch.from_numpy(utils.array_to_batch(train_X[:, dag_edges_idx[i]], args.batch_size))
            .to(torch.float32).to(args.device) )  # Creates a Tensor from a numpy.ndarray

    print(f"structure_params: {structure_params}")

    batched_train_y = ( torch.from_numpy(utils.array_to_batch(train_y, args.batch_size)).to(torch.float32).to(args.device) )


    test_Xs = {}
    for i in range(len(dag_edges_idx)):
        test_Xs[f"CG{i}"] = ( torch.from_numpy(utils.array_to_batch(test_X[:, dag_edges_idx[i]], len(test_X)))
            .to(torch.float32).to(args.device) )

        test_CG_shape = test_Xs[f"CG{i}"].shape

    batched_test_y = ( torch.from_numpy(utils.array_to_batch(test_y, len(test_y))).to(torch.float32).to(args.device) )

    return structure_params, train_Xs, batched_train_y, test_Xs, batched_test_y


def get_batch_ids(batched_train_y):
    train_batch_ids = np.random.choice(range(len(batched_train_y)), int(len(batched_train_y) * 1.0))

    return train_batch_ids


def get_model(args, structure_params):
    model = BnnccRegressor(structure_params=structure_params)

    return model


def get_optimizer(args, model):
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    return optimizer


def get_scheduler(optimizer):
    lambda1 = lambda x: 0.1 ** (x // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, ])

    return scheduler


def train(i_epoch, n_epoch, model, train_batch_ids, train_Xs, batched_train_y, optimizer):
    model.train()

    losses = []
    for i in train_batch_ids:
        Xs_batch = {k: v[i] for k, v in train_Xs.items()}
        y_batch = batched_train_y[i]
        optimizer.zero_grad()  
        output = model(Xs_batch)
        loss = bnncc_reg_loss(output, y_batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    train_losses = np.mean(losses)  # sum(losses) / len(losses)
    print(f"epoch: {i_epoch:>4}/{n_epoch} | train_losses: {train_losses:.3f}")

    return train_losses


def val(i_epoch, n_epoch, model, test_Xs, batched_test_y):
    model.eval()

    losses = []
    with torch.no_grad():
        test_Xs = {k: v[0] for k, v in test_Xs.items()}
        output = model(test_Xs)

        y_batch = batched_test_y[0]
        val_losses = bnncc_reg_loss(output, y_batch)

        # # metrics
        mse = metrics.mean_squared_error(y_batch, output)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_batch, output)
        # mape = metrics.mean_absolute_percentage_error(y_batch, output)
        val_r2 = metrics.r2_score(y_batch, output)
        print(f"epoch: {i_epoch:>4}/{n_epoch} |   val_losses: {val_losses:.3f} | rmse: {rmse:.3f} | mae: {mae:.3f} | r2: {val_r2:.3f}")

        y_pred = output

    return y_pred, val_losses, val_r2


def run_experiments(args, structure_params, train_batch_ids, train_Xs, batched_train_y, test_Xs, batched_test_y):
    n_epoch = args.n_epoch
    n_experiments = args.n_experiments
    for experiment in range(n_experiments):
        print(f"\nExperiment: {experiment + 1}")

        model = get_model(args, structure_params)
        optimizer = get_optimizer(args, model)

        for i_epoch in range(1, n_epoch + 1):
            train(i_epoch, n_epoch, model=model, train_batch_ids=train_batch_ids, train_Xs=train_Xs,
                  batched_train_y=batched_train_y, optimizer=optimizer)
            pred_y = val(i_epoch, n_epoch, model=model, test_Xs=test_Xs, batched_test_y=batched_test_y)

            # learning rate scheduler
            scheduler = get_scheduler(optimizer)
            scheduler.step()

    return pred_y


def main():
    parser = get_parser()
    args = parser.parse_args()

    args.dataset_name = "ihdp"
    print(f"starting {args.dataset_name}")

    # # BNN-CC (ours)
    args.hidden_size = 14
    args.hidden_layers = 1 + 5 + 1  # 7
    args.batch_size = 15
    args.n_epoch = 300
    args.n_experiments = 1
    args.predefined_dag = True
    
    print(args)

    data = load_data(args)
    train_df, test_df, X_features, y_feature = data.train_df, data.test_df, data.X_features, data.Y_feature
    train_X, train_y, test_X, test_y = data_preprocessing(train_df, test_df, X_features, y_feature)

    causal_groups = get_causal_groups(args, data)
    structure_params, train_Xs, batched_train_y, test_Xs, batched_test_y = get_causal_groups_data(
        args, causal_groups, train_X, train_y, test_X, test_y, X_features)

    train_batch_ids = get_batch_ids(batched_train_y)

    run_experiments(args, structure_params, train_batch_ids, train_Xs, batched_train_y, test_Xs, batched_test_y)


if __name__ == "__main__":
    main()