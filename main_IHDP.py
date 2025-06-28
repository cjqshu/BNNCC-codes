# import sys
# sys.path.append("..")
import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn import metrics
import configargparse
from torch.optim import lr_scheduler

# 导入自定义模块
import causal_discovery, data_loader, utils

# 导入模型
from models.bnncc_regression import BnnccRegressor, bnncc_reg_loss, BnnnccRegressor, BnnccnrRegressor, BnnccnaRegressor, NncgcRegressor
# from models.baselines import BaselinesRegressor

# visualizations
from visualizations.true_predicted_scatter import true_predicted_scatter_plot

import warnings
warnings.filterwarnings('ignore')

# 运行的结果保留小数点后三位
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format

# # 训练和测试结果展示, 配合run_experiments中的399行
# from torch.utils.tensorboard import SummaryWriter
# folder_name = "IHDP/ours"
# writer = SummaryWriter(log_dir=f"./logs/{folder_name}")  # 图形工具, logs日志文件，这个会自动创建此文件夹

# # cmd input
# tensorboard --logdir=logs --bind_all


def get_parser():
    """get default arguments."""
    parser = configargparse.ArgumentParser(
        description="causal learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument('--dataset_name', default="ihdp", type=str, choices=["ihdp", "cpt", "jobs"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "cuda"])

    # network related
    parser.add_argument("--hidden_size", default=12, type=int)
    parser.add_argument("--hidden_layers", default=10, type=int)
    parser.add_argument("---output_size", default=1, type=int)

    # data loading related
    parser.add_argument("--data_dir", default="data/", type=str, help="data directory")
    parser.add_argument("--rep_i", default=16, type=int, help="index number of the repetitive dataset is in [0, 999]")

    # training related
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--n_epoch", default=50, type=int, help="number of epochs")
    parser.add_argument("--early_stop", default=0, type=int, help="early stopping patience")

    # optimizer related
    parser.add_argument("--lr", default=1e-3, type=float)  # 1e-3 is equal to 1*10^(-3)=0.001
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)  # SGD优化器可选参数

    # experiments related
    parser.add_argument("--n_experiments", default=1, type=int, help="number of experiments")

    # save results
    parser.add_argument("--save_model", default=False, type=bool, help="save model or not")
    parser.add_argument("--save_dir", default="results/", type=str, help="directory of results")

    # baselines
    parser.add_argument("--baselines_status", default=False, type=bool, help="baselines or not")

    # ablations
    parser.add_argument("--ablations", default=False, type=bool, help="ablation or not")

    # predefined dag
    parser.add_argument("--predefined_dag", default=False, type=bool, help="predefined dag or not")

    return parser


def set_random_seed(seed=0):
    """seed setting, 控制python中哈希随机化的种子, 确保实验结果的可复现性"""
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
    # 类的实例
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

    # 类的实例方法
    data.load_dataset(args.dataset_name)

    return data


def data_preprocessing(train_df, test_df, X_features, y_feature):
    # 数据集列标准化
    train_df = utils.normalize(train_df)
    test_df = utils.normalize(test_df)

    # 将Dataframe数据结构类型转成ndarray结构类型
    train_X = train_df[X_features].to_numpy()
    train_y = train_df[y_feature].to_numpy()
    test_X = test_df[X_features].to_numpy()
    test_y = test_df[y_feature].to_numpy()

    return train_X, train_y, test_X, test_y


def get_causal_groups(args, data):
    """Extract DAG"""
    dag_creator = causal_discovery.DAGCreator(data.train_df, data.X_features, data.Y_feature, method="icalingam",max_samples=1000, )

    if args.predefined_dag:
        dag_edges = dag_creator.return_predefined_dag("ihdp") # 自定义或者已确定的因果分组
    else:
        dag_edges = dag_creator.create_dag_edges() # 采用因果发现方法的因果分组

    nodes_in_edges = [item for sublist in dag_edges for item in sublist]

    # 添加不在因果分组的特征
    nodes_not_in_edges = [ x for x in dag_creator.X_features if x not in nodes_in_edges ]
    dag_edges.append(nodes_not_in_edges)

    results = dag_edges

    return results


def get_causal_groups_data(args, causal_groups, train_X, train_y, test_X, test_y, X_features):
    dag_edges = causal_groups
    dag_edges_idx = [[X_features.index(col) for col in sublist] for sublist in dag_edges]
    print(dag_edges_idx)

    structure_params = {}  # 字典
    train_Xs = {}

    # input
    # 自动化指定输入层大小(依赖于因果图约束，可变)，隐藏层大小(12)，和输出层大小(1)
    for i in range(len(dag_edges_idx)):
        structure_params[f"CG{i}"] = {
            "input_size": len(dag_edges_idx[i]),
            "hidden_size": args.hidden_size,  # 12
            "hidden_layers": args.hidden_layers,  # 10
            "output_size": args.output_size,
        }  # 字典中的元素也是字典

        # 按照因果分组确定协变量的数据输入, array_to_batch表示制作分批量的数据, Xs为dict
        train_Xs[f"CG{i}"] = ( torch.from_numpy(utils.array_to_batch(train_X[:, dag_edges_idx[i]], args.batch_size))
            .to(torch.float32).to(args.device) )  # Creates a Tensor from a numpy.ndarray

    print(f"structure_params: {structure_params}")

    # 确定目标变量的真实值
    batched_train_y = ( torch.from_numpy(utils.array_to_batch(train_y, args.batch_size)).to(torch.float32).to(args.device) )

    # print(f"batched_train_y: {batched_train_y.shape}")

    test_Xs = {}
    for i in range(len(dag_edges_idx)):
        # 按照因果分组确定协变量的数据输入, array_to_batch表示制作分批量的数据, Xs为dict, Creates a Tensor from a numpy.ndarray
        test_Xs[f"CG{i}"] = ( torch.from_numpy(utils.array_to_batch(test_X[:, dag_edges_idx[i]], len(test_X)))
            .to(torch.float32).to(args.device) )

        test_CG_shape = test_Xs[f"CG{i}"].shape

    batched_test_y = ( torch.from_numpy(utils.array_to_batch(test_y, len(test_y))).to(torch.float32).to(args.device) )

    return structure_params, train_Xs, batched_train_y, test_Xs, batched_test_y


def get_batch_ids(batched_train_y):
    # 随机化batch的id构成一个数组, 例如array([3, 1, 2]), 元素表示batch的id
    # 从range(0, 1, 2, 3, 4, 5)中随机取int(6*0.8)=4个元素
    # train_batch_ids = np.random.choice(range(len(batched_train_y)), int(len(batched_train_y) * 0.8))
    train_batch_ids = np.random.choice(range(len(batched_train_y)), int(len(batched_train_y) * 1.0))

    return train_batch_ids


def get_model(args, structure_params):

    # model = BnnccRegressor(structure_params=structure_params)  # 模型初始化, 包括参数初始化

    if args.ablations:
        # # 模型不能覆盖，否则无法训练，估计是没更新优化器
        # model = BnnnccRegressor(structure_params=structure_params)  # 需要配合main()中 no_causal_groups 的共同执行
        # model = BnnccnrRegressor(structure_params=structure_params)
        model = BnnccnaRegressor(structure_params=structure_params)
        # model = NncgcRegressor(structure_params=structure_params)  # 最先进的方法nn-cgc变体进行实验

    else:
        model = BnnccRegressor(structure_params=structure_params)  # 模型初始化, 包括参数初始化

    return model


def get_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    return optimizer


def get_scheduler(optimizer):
    # Assuming optimizer has two groups.
    lambda1 = lambda x: 0.1 ** (x // 20)  # lambda1是lr前面乘法因子的函数
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, ])

    return scheduler


def train(i_epoch, n_epoch, model, train_batch_ids, train_Xs, batched_train_y, optimizer):
    model.train()

    losses = []
    for i in train_batch_ids:
        Xs_batch = {k: v[i] for k, v in train_Xs.items()}
        y_batch = batched_train_y[i]
        optimizer.zero_grad()  # 梯度清零/置零
        output = model(Xs_batch)  # 前向传播
        loss = bnncc_reg_loss(output, y_batch)  # 平均损失
        loss.backward()  # 反向传播, 计算模型参数的梯度
        optimizer.step()  # 更新模型参数/权重

        # 记录每个批次的损失
        losses.append(loss.item())
    train_losses = np.mean(losses)  # sum(losses) / len(losses)
    print(f"epoch: {i_epoch:>4}/{n_epoch} | train_losses: {train_losses:.3f}")

    return train_losses


def val(i_epoch, n_epoch, model, test_Xs, batched_test_y):  # validation
    model.eval()

    losses = []
    with torch.no_grad():  # 禁用梯度计算的上下文管理器
        test_Xs = {k: v[0] for k, v in test_Xs.items()}
        output = model(test_Xs)

        y_batch = batched_test_y[0]  # 没有划分批量, 即取整个测试集的数量
        val_losses = bnncc_reg_loss(output, y_batch)

        # # metrics
        mse = metrics.mean_squared_error(y_batch, output)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_batch, output)
        # mape = metrics.mean_absolute_percentage_error(y_batch, output)
        val_r2 = metrics.r2_score(y_batch, output)
        print(f"epoch: {i_epoch:>4}/{n_epoch} |   val_losses: {val_losses:.3f} | rmse: {rmse:.3f} | mae: {mae:.3f} | r2: {val_r2:.3f}")

        y_pred = output

    # writer.add_graph(model, input_to_model=test_Xs)
    # writer.close()

    return y_pred, val_losses, val_r2


def run_experiments(args, structure_params, train_batch_ids, train_Xs, batched_train_y, test_Xs, batched_test_y):
    """模型训练和测试的多次实验"""

    # train_losses = []
    # test_accuracies = []

    n_epoch = args.n_epoch
    n_experiments = args.n_experiments
    for experiment in range(n_experiments):
        print(f"\nExperiment: {experiment + 1}")  # \n表示换行

        # 载入模型和优化器
        model = get_model(args, structure_params)
        optimizer = get_optimizer(args, model)

        for i_epoch in range(1, n_epoch + 1):  # range一个可迭代对象(类型是对象)，而不是列表类型
            # training process
            train_losses = train(i_epoch, n_epoch, model=model, train_batch_ids=train_batch_ids, train_Xs=train_Xs,
                  batched_train_y=batched_train_y, optimizer=optimizer)
            # validating process
            pred_y, val_losses, val_r2 = val(i_epoch, n_epoch, model=model, test_Xs=test_Xs, batched_test_y=batched_test_y)

            # learning rate scheduler
            scheduler = get_scheduler(optimizer)
            scheduler.step()

            # # # 记录模型训练的结果信息
            # # writer.add_scalar("train_loss", train_losses, i_epoch)
            # # 记录模型验证的结果信息
            # writer.add_scalar("r2", val_r2, i_epoch)
            # writer.add_scalar("val_loss", val_losses, i_epoch)


        # train_losses.append(train_loss)
        # test_accuracies.append(test_accuracy)

    # # 计算均值和标准差
    # train_loss_mean = np.mean(train_losses)
    # train_loss_std = np.std(train_losses)
    # test_accuracy_mean = np.mean(test_accuracies)
    # test_accuracy_std = np.std(test_accuracies)

    return pred_y


def main():
    """ 主函数 (程序运行的起点) """

    # # 解析命令行参数
    parser = get_parser()
    args = parser.parse_args()

    # # 指定实验数据集
    args.dataset_name = "ihdp"
    print(f"starting {args.dataset_name}")

    args.rep_i = 16

    # # bnn-cc (ours)
    # self.dropout = nn.Dropout(p=0.1)
    args.hidden_size = 14
    args.hidden_layers = 1 + 5 + 1  # 7
    args.batch_size = 15
    args.n_epoch = 300
    args.n_experiments = 1

    # # # bnnncc
    # args.hidden_size = 14
    # args.hidden_layers = 1 + 5 + 1
    # args.batch_size = 15
    # args.n_epoch = 66
    # args.n_experiments = 1

    # # # bnnccnr
    # args.hidden_size = 14
    # args.hidden_layers = 1 + 5 + 1
    # args.batch_size = 15
    # args.n_epoch = 300
    # args.n_experiments = 1

    # # # bnnccna
    # args.hidden_size = 14
    # args.hidden_layers = 1 + 5 + 1
    # args.batch_size = 15
    # args.n_epoch = 300
    # args.n_experiments = 1

    # # # nncgc (sota=0.949)
    # args.hidden_size = 12
    # args.hidden_layers = 1 + 8 + 1
    # args.batch_size = 20
    # args.n_epoch = 300
    # args.n_experiments = 1

    # # # nncgc
    # args.hidden_size = 200
    # args.hidden_layers = 1
    # args.batch_size = 15
    # args.n_epoch = 300
    # args.n_experiments = 1

    # # # # 网格搜索
    # args.hidden_size = 14       # 10, 12, 14, 16
    # args.hidden_layers = 7      # 5, 6, 7, 8
    # args.batch_size = 15        # 5, 15, 20, 30
    # args.n_epoch = 300
    # args.n_experiments = 1

    print(args)

    # args.baselines_status = True
    # args.ablations = True
    args.predefined_dag = True

    # # 载入数据集
    data = load_data(args)  # object

    # # 数据预处理
    train_df, test_df, X_features, y_feature = data.train_df, data.test_df, data.X_features, data.Y_feature
    train_X, train_y, test_X, test_y = data_preprocessing(train_df, test_df, X_features, y_feature)

    # # 因果分组
    causal_groups = get_causal_groups(args, data)
    # # 获取因果分组的数据
    structure_params, train_Xs, batched_train_y, test_Xs, batched_test_y = get_causal_groups_data(
        args, causal_groups, train_X, train_y, test_X, test_y, X_features)

    # # BNN-NCC: 无因果分组的消融实验 （需要配合get_model中的model = BnnnccRegressor(structure_params=structure_params)进行使用）
    # no_causal_groups = [data.X_features]
    # structure_params, train_Xs, batched_train_y, test_Xs, batched_test_y \
    #     = get_causal_groups_data(args, no_causal_groups, train_X, train_y, test_X, test_y, X_features)
    # print(f"no_causal_groups: \n {no_causal_groups} \n structure params: \n {structure_params} \n")

    # # 获取批次的id序号
    train_batch_ids = get_batch_ids(batched_train_y)

    # # 模型训练和测试的多次实验
    pred_y = run_experiments(args, structure_params, train_batch_ids, train_Xs, batched_train_y, test_Xs, batched_test_y)
    # print(f"\n--- Results after 10 experiments ---")
    # print(f"Train Loss Mean: {train_loss_mean:.4f}, Train Loss Std: {train_loss_std:.4f}")
    # print(f"Test Accuracy Mean: {test_accuracy_mean:.4f}, Test Accuracy Std: {test_accuracy_std:.4f}")

    # # visualizations
    inverse_normalize_test_y = test_df[y_feature].to_numpy()  # inverse normalize
    inverse_normalize_pred_y = utils.inverse_normalize(test_df, pred_y)
    true_predicted_scatter_plot(inverse_normalize_test_y, inverse_normalize_pred_y)

    # # # 保存y_test和y_pred的结果
    # inverse_normalize_pred_y = inverse_normalize_pred_y.numpy()  # tensor转ndarray
    # inverse_normalize_test_y = np.squeeze(inverse_normalize_test_y)  # 将(76,1)压缩为(76,), 去掉含维数为1的维
    # inverse_normalize_pred_y = np.squeeze(inverse_normalize_pred_y)
    # y_test2pred = pd.DataFrame({"y_test": inverse_normalize_test_y, "y_pred": inverse_normalize_pred_y})
    # # y_test2pred.to_csv("./tests/IHDP_test2pred.csv")
    # # y_test2pred.to_csv("./tests/IHDP_test2pred_bnnncc.csv")
    # # y_test2pred.to_csv("./tests/IHDP_test2pred_bnnccnr.csv")
    # y_test2pred.to_csv("./tests/IHDP_test2pred_bnnccna.csv")

    # # # baselines experiments or not
    # if args.baselines_status:
    #     baselines = BaselinesRegressor(train_X, train_y, test_X, test_y)
    #     baselines_results = baselines()
    #     # print(baselines_results)
    #
    #     # 行列索引筛选
    #     sub_baselines_results = baselines_results.loc[
    #         ['LinearRegression', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor',
    #          'AdaBoostRegressor', 'SVR', 'MLPRegressor', ], ['RMSE', 'mean_absolute_error', 'R-Squared', ]].round(3)
    #     sub_baselines_results.index = ['OLS', 'k-NN', 'DT', 'RF', 'ABR', 'SVR', 'MLP', ]  # rename index
    #     sub_baselines_results.columns = ['RMSE', 'MAE', 'R^2',]  # rename colunms
    #     # sub_baselines_results.to_csv(args.save_dir + "sub_baselines_results_0.05_30_200.csv")
    #     print(sub_baselines_results)

    # # # 模型保存与读取 (checkpoint检查点)
    # # torch.save(model, './BnnccRegressor_IHDP.pth')  # 保存整个模型
    # loaded_model = torch.load('BnnccRegressor_IHDP.pth')  # 读取模型
    # val(1, 1, model=loaded_model, test_Xs=test_Xs, batched_test_y=batched_test_y)  # 模型推理


    pass


if __name__ == "__main__":
    main()
