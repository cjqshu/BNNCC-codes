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

            "ihdp": self.load_predefined_ihdp_icalingam,

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
        return self.predefined_dags[dataset_name]()


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

        return dag