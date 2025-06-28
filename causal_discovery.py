import numpy as np
from castle.algorithms import PC, GES, ICALiNGAM, GOLEM, Notears, DirectLiNGAM, GAE
import networkx as nx


class DAGCreator:
    def __init__(self, train_df, X_features, Y_feature, max_samples=1000, method="notears", ) -> None:
        self.train_df = train_df
        self.X_features = X_features
        self.Y_feature = Y_feature
        self.max_samples = max_samples
        self.method = method
        self.predefined_dags = {
            "cpt": self.load_predefined_cpt_icalingam,
            "jobs": self.load_predefined_jobs_icalingam,
            "ihdp": self.load_predefined_ihdp_icalingam,
            "bd": self.load_predefined_bd_icalingam,
            "simulation_data": self.load_predefined_causalml_mode_2_icalingam,

        }

    def run_gcastle(self, method):
        algo_dict = {
            "pc": PC,
            "ges": GES,
            "icalingam": ICALiNGAM,
            "golem": GOLEM,
            "notears": Notears,
            "directlingam": DirectLiNGAM,
            "gae": GAE
        }
        algo = algo_dict[method]()
        # algo.learn(
        #     self.train_df[self.X_features + self.Y_feature].sample(
        #         np.min([self.train_df.shape[0], self.max_samples])
        #     )
        # )
        algo.learn(self.train_df[self.X_features + self.Y_feature].sample(
                np.min([self.train_df.shape[0], self.max_samples]), random_state=42
            )
        )

        # Relabel the nodes
        learned_graph = nx.DiGraph(algo.causal_matrix)
        MAPPING = {
            k: v for k, v in zip(range(algo.causal_matrix.shape[0]), self.X_features + self.Y_feature)
        }
        learned_graph = nx.relabel_nodes(learned_graph, MAPPING, copy=True)
        self.dag_edges = list(learned_graph.edges())

    def remove_target_from_dag_edges(self):
        new_edges = []
        for l in self.dag_edges:
            new_edge = []

            for i in l:
                if i != self.Y_feature[0]:
                    new_edge.append(i)

            # if len(new_edge) >= 2:
            new_edges.append(new_edge)

        self.dag_edges = new_edges

    def add_nodes_not_in_edges(self):
        nodes_in_edges = [item for sublist in self.dag_edges for item in sublist]
        nodes_not_in_edges = [x for x in self.X_features if x not in nodes_in_edges]
        self.dag_edges.append(nodes_not_in_edges)

    def create_dag_constraints(self):
        """Right now, it changes dag edges to subgroups of variables from the roots"""

        def get_parents(dag, node):
            # Added childs
            parents = []
            for origin, destination in dag:
                if destination == node:
                    parents.append(origin)
                # if origin == node:
                #     parents.append(destination)

            return parents

        # def get_children(dag, node):
        #     # Added childs
        #     children = []
        #     for origin, destination in dag:
        #         if origin == node:
        #             children.append(destination)
        #     return children

        def get_ancestors(dag, node):
            ancestors = []
            for origin, destination in dag:

                # Dont take into account ancestors that go through Y
                if destination == self.Y_feature[0]:
                    continue

                if destination == node:
                    ancestors.append(origin)
                    ancestors.extend(get_ancestors(dag, origin))
            return ancestors

        dag_edges = self.dag_edges
        # obtain target parent nodes
        target_parents = get_parents(dag_edges, self.Y_feature[0])
        # target_childs = get_children(dag_edges, self.Y_feature[0])
        dag_constraints = { parent: get_ancestors(dag_edges, parent) + [parent] for parent in target_parents }
        # Add parents group
        dag_constraints["parents"] = target_parents

        self.dag_constraints = dag_constraints
        # CAUTION, dag_edges is not dag edges anymore
        self.dag_edges = list(dag_constraints.values())

    # 类似于main函数
    def create_dag_edges(self, add_nodes_not_in_edges=False):

        if self.method in ["pc", "ges", "icalingam", "golem", "notears"]:
            self.run_gcastle(self.method)

        else:
            print("Method not supported")

        self.create_dag_constraints()
        self.remove_target_from_dag_edges()
        if add_nodes_not_in_edges:
            self.add_nodes_not_in_edges()

        return [list(set(d)) for d in self.dag_edges if len(d) > 0]

    def return_predefined_dag(self, dataset_name):
        """指定因果分组"""
        return self.predefined_dags[dataset_name]()

    @staticmethod
    def load_predefined_cpt_icalingam():
        dag = [
            ['x13', 'x7', 'x2'],
            ['x20', 'x7', 'x4', 'x12', 'x14', 'x3', 'x11', 'x5', 'x13', 'x15', 'x16', 'x2', 'x9'],
            ['x7', 'x12', 'x14', 'x11', 'x5', 'x13', 'x15', 'x21', 'x2', 'x16'],
            ['x4', 'x2', 'x21'],
            ['x0', 'x1', 'x6', 'x8', 'x10', 'x17', 'x18', 'x19', 'x22', 'x23', 'x24', 'x25']
        ]
        return dag


    @staticmethod
    def load_predefined_jobs_icalingam():
        dag = [
            ["x9", "x11", "x15", "x12", "x8", "x7", "x2", "x6", "x0"],
            ["x2", "x7"],
            ["x8", "x2", "x9", "x0"],
            ["x8", "x2"],
            ["x8", "x2", "x9"],
            ["x9", "x11", "x15", "x1", "x12", "x8", "x7", "x6", "x0"],
            ["x9", "x11", "x15", "x12", "x8", "x7", "x6", "x0"],
            ["x1", "x7"],
            ["x16"],
        ]
        return dag

    @staticmethod
    def load_predefined_ihdp_icalingam():
        dag = [
            ['x0'],
            ['x0', 'x1'],
            ['x2', 'x0'],
            ['x11', 'x10', 'x3', 'x9', 'x13'],
            ['x2', 'x4', 'x0'],
            ['x8', 'x5', 'x11', 'x10', 'x9'],
            ['x8'],
            ['x11', 'x10', 'x9', 'x13'],
            ['x15'],
            ['x18'],
            ['x19'],
            ['x20', 'x21', 'x19', 'x23', 'x18', 'x24', 'x22'],
            ['x21'],
            ['x23'],
            ['T'],
            ['x8', 'x4', 'x20', 'x21', 'x5', 'x19', 'x23', 'x18', 'x0', 'T', 'x2', 'x3', 'x15', 'x13', 'x1'],
            # ['x6', 'x7', 'x12', 'x14', 'x16', 'x17']
        ]

        # dag = [
        #     ['x0'],
        #     ['x0', 'x1'],
        #     ['x0', 'x2'],
        #     ['x3', 'x9', 'x10', 'x11', 'x13'],
        #     ['x0', 'x2', 'x4'],
        #     ['x5', 'x8', 'x9', 'x10', 'x11'],
        #     ['x8'],
        #     ['x9', 'x10', 'x11', 'x13', ],
        #     ['x15'],
        #     ['x18'],
        #     ['x19'],
        #     ['x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24'],
        #     ['x21'],
        #     ['x23'],
        #     ['T'],  # [25]
        #     ['x0', 'x1', 'x3', 'x2', 'x4', 'x8', 'x5', 'x13', 'x15', 'x18', 'x19', 'x20', 'x21', 'x23', 'T', ],
        #     # ['x6', 'x7', 'x12', 'x14', 'x16', 'x17']  # 不在因果分组的特征
        # ]

        return dag

    @staticmethod
    def load_predefined_bd_icalingam():
        dag = [
            ['x0'],
            ['x11'],
            ['x21'],
            ['x0', 'x21', 'x11'],
            # ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20']
        ]

        # dag = [
        #     ['x0'],
        #     ['x11'],
        #     ['x21'],
        #     ['x21', 'x11', 'x0'],
        #     # ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20']
        # ]
        return dag

    @staticmethod
    def load_predefined_causalml_mode_2_icalingam():
        # # n200_p20_sigma0.01
        # dag = [
        #     ['col_1', 'col_2'],
        #     ['col_4', 'col_3', 'col_1', 'col_2', 'col_0'],
        #     ['col_4'],
        #     ['col_3', 'col_4', 'col_2'],
        #     # ['col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_13', 'col_14', 'col_15', 'col_16', 'col_17', 'col_18', 'col_19', 'T']
        # ]

        # # n200_p30_sigma0.01
        # dag = [
        #     ['col_0'],
        #     ['col_0', 'col_2', 'col_1'],
        #     ['col_3'],
        #     ['col_0', 'col_4', 'col_3'],
        #     ['col_4', 'col_0', 'col_2', 'col_3'],
        #     # ['col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_13', 'col_14', 'col_15',
        #     # 'col_16', 'col_17', 'col_18', 'col_19', 'col_20', 'col_21', 'col_22', 'col_23', 'col_24', 'col_25', 'col_26',
        #     # 'col_27', 'col_28', 'col_29', 'T']
        # ]

        # # n200_p40_sigma0.01
        # dag = [
        #     ['col_2', 'col_0'],
        #     ['col_3', 'col_0'],
        #     ['col_3', 'col_0', 'col_4'],
        #     ['col_3', 'col_2', 'col_4'],
        #     # ['col_1', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_13', 'col_14',
        #     # 'col_15', 'col_16', 'col_17', 'col_18', 'col_19', 'col_20', 'col_21', 'col_22', 'col_23', 'col_24', 'col_25',
        #     # 'col_26', 'col_27', 'col_28', 'col_29', 'col_30', 'col_31', 'col_32', 'col_33', 'col_34', 'col_35', 'col_36',
        #     # 'col_37', 'col_38', 'col_39', 'T']
        # ]

        # # n600_p20_sigma0.01
        # dag =[
        #     ['col_2', 'col_1', 'col_0'],
        #     ['col_2'],
        #     ['col_3'],
        #     ['col_4', 'col_2', 'col_1', 'col_0', 'col_3'],
        #     ['col_4', 'col_2', 'col_3', 'col_0'],
        #     # ['col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_13', 'col_14', 'col_15',
        #     # 'col_16', 'col_17', 'col_18', 'col_19', 'T']
        # ]

        # # n600_p30_sigma0.01
        # dag = [
        #     ['col_2'],
        #     ['col_3'],
        #     ['col_4', 'col_0', 'col_3', 'col_2'],
        #     ['col_4', 'col_3', 'col_2'],
        #     # ['col_1', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_13', 'col_14',
        #     # 'col_15', 'col_16', 'col_17', 'col_18', 'col_19', 'col_20', 'col_21', 'col_22', 'col_23', 'col_24', 'col_25',
        #     #  'col_26', 'col_27', 'col_28', 'col_29', 'T']
        # ]

        # n600_p40_sigma0.01
        dag = [
            ['col_0'],
            ['col_2', 'col_1', 'col_0'],
            ['col_0', 'col_3'],
            ['col_4', 'col_0', 'col_3'],
            ['col_4', 'col_2', 'col_0', 'col_3'],
            # ['col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_13', 'col_14', 'col_15',
            # 'col_16', 'col_17', 'col_18', 'col_19', 'col_20', 'col_21', 'col_22', 'col_23', 'col_24', 'col_25', 'col_26',
            # 'col_27', 'col_28', 'col_29', 'col_30', 'col_31', 'col_32', 'col_33', 'col_34', 'col_35', 'col_36', 'col_37',
            #  'col_38', 'col_39', 'T']
        ]

        return dag