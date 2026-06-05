from __future__ import annotations

import numpy as np
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


class RotationTree:
    @classmethod
    def from_model(cls, tree, n_features=5, sample_prop=1.0, bootstrap=True):
        rt = RotationTree(n_features, sample_prop, bootstrap)
        rt.model = tree
        return rt

    def __init__(self, n_features=5, sample_prop=1.0, bootstrap=True):
        self.model = DecisionTreeClassifier()
        self.n_features = n_features
        self.sample_prop = sample_prop
        self.bootstrap = bootstrap
        self.partition_nums = []
        self.rotation_matrices = []

    def fit(self, x, y):
        partitions = self.partition_features(x)
        transformed_partitions = []

        for partition in partitions:
            sampled_data = self.get_samples(partition, y)
            rotation_matrix = self.get_rotation_matrix(sampled_data)
            transformed_partitions.append(np.dot(partition, rotation_matrix))

        new_x = np.concatenate(transformed_partitions, axis=1)
        if self.bootstrap:
            xx, yy = self.boot_sample(new_x, y)
            self.model.fit(xx, yy)
        else:
            self.model.fit(new_x, y)

    def partition_features(self, x):
        n_cols = x.shape[1]
        cols = np.arange(n_cols)
        np.random.shuffle(cols)
        partitions = [cols[i::self.n_features] for i in range(self.n_features)]
        self.partition_nums = partitions
        return [x[:, part] for part in partitions]

    def get_samples(self, x_partition, y):
        xy = np.column_stack([x_partition, y])
        sampled_rows = []
        for cls in np.unique(y):
            cls_rows = xy[y == cls, :]
            n_sample = max(1, round(self.sample_prop * cls_rows.shape[0]))
            idx = np.random.choice(cls_rows.shape[0], size=n_sample, replace=True)
            sampled_rows.append(cls_rows[idx, :])
        sampled_data = np.vstack(sampled_rows)
        return sampled_data[:, :-1]

    def get_rotation_matrix(self, samples):
        n_features = samples.shape[1]
        n_samples = samples.shape[0]
        n_components = min(n_features, n_samples)

        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(samples)
        rotation_matrix = pca.components_.T

        if n_components < n_features:
            pad = np.eye(n_features)
            pad[:, :n_components] = rotation_matrix
            rotation_matrix = pad

        self.rotation_matrices.append(rotation_matrix)
        return rotation_matrix

    def boot_sample(self, x, y):
        data = np.column_stack([x, y])
        idx = np.random.choice(data.shape[0], size=data.shape[0], replace=True)
        sample = data[idx, :]
        return sample[:, :-1], sample[:, -1]

    def predict(self, x):
        partitions = [x[:, part] for part in self.partition_nums]
        transformed = [np.dot(part, self.rotation_matrices[i]) for i, part in enumerate(partitions)]
        new_x = np.concatenate(transformed, axis=1)
        return self.model.predict(new_x)


class RotationForest:
    def __init__(self, n_trees=100, n_features=5, sample_prop=1.0, bootstrap=False):
        self.n_trees = n_trees
        self.n_features = n_features
        self.sample_prop = sample_prop
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, x, y, base_tree=None):
        self.trees = []
        for _ in range(self.n_trees):
            if base_tree:
                tree = RotationTree.from_model(
                    base_tree,
                    n_features=self.n_features,
                    sample_prop=self.sample_prop,
                    bootstrap=self.bootstrap,
                )
            else:
                tree = RotationTree(
                    n_features=self.n_features,
                    sample_prop=self.sample_prop,
                    bootstrap=self.bootstrap,
                )
            tree.fit(x, y)
            self.trees.append(tree)

    def predict(self, x):
        all_preds = np.array([tree.predict(x) for tree in self.trees])
        return mode(all_preds, axis=0)[0].flatten()

    def predict_proba(self, x):
        all_preds = np.array([tree.predict(x) for tree in self.trees])
        return all_preds.mean(axis=0)
