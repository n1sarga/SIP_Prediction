from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


SPECIES = "Diabates"
ROOT = Path(__file__).resolve().parents[2]
FUSED_FEATURES_FILE = ROOT / "artifacts" / "fusion" / SPECIES / "fused_feature_vectors.csv"
INTERACTION_FILE = ROOT / "data" / "processed" / SPECIES / f"{SPECIES}_All.csv"
REPORTS_DIR = ROOT / "reports" / "results" / SPECIES


def prepare_pair_features(df_feat: pd.DataFrame, df_pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features_index = df_feat.set_index("Protein Identifier")
    features_list = []
    labels = []
    missing_proteins = []

    for _, row in df_pairs.iterrows():
        prot_a, prot_b = row["Identifier A"], row["Identifier B"]
        if prot_a not in features_index.index or prot_b not in features_index.index:
            missing_proteins.append((prot_a, prot_b))
            continue

        feat_a = features_index.loc[prot_a].to_numpy().flatten()
        feat_b = features_index.loc[prot_b].to_numpy().flatten()
        features_list.append(np.concatenate([feat_a, feat_b]))
        labels.append(row["Interaction"])

    if missing_proteins:
        print(f"Missing proteins in features: {missing_proteins}")

    return np.array(features_list), np.array(labels)


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

    def fit(self, X, y):
        partitions = self.partition_features(X)
        transformed_partitions = []

        for partition in partitions:
            sampled_data = self.get_samples(partition, y)
            rotation_matrix = self.get_rotation_matrix(sampled_data)
            transformed_partitions.append(np.dot(partition, rotation_matrix))

        new_X = np.concatenate(transformed_partitions, axis=1)

        if self.bootstrap:
            xx, yy = self.boot_sample(new_X, y)
            self.model.fit(xx, yy)
        else:
            self.model.fit(new_X, y)

    def partition_features(self, X):
        n_cols = X.shape[1]
        cols = np.arange(n_cols)
        np.random.shuffle(cols)
        partitions = [cols[i::self.n_features] for i in range(self.n_features)]

        partitioned_data = []
        for part in partitions:
            partitioned_data.append(X[:, part])
        self.partition_nums = partitions
        return partitioned_data

    def get_samples(self, X_partition, y):
        Xy = np.column_stack([X_partition, y])
        sampled_rows = []
        for cls in np.unique(y):
            cls_rows = Xy[y == cls, :]
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

    def boot_sample(self, X, y):
        data = np.column_stack([X, y])
        idx = np.random.choice(data.shape[0], size=data.shape[0], replace=True)
        sample = data[idx, :]
        return sample[:, :-1], sample[:, -1]

    def predict(self, X):
        partitions = [X[:, part] for part in self.partition_nums]
        transformed = [np.dot(part, self.rotation_matrices[i]) for i, part in enumerate(partitions)]
        new_X = np.concatenate(transformed, axis=1)
        return self.model.predict(new_X)


class RotationForest:
    def __init__(self, n_trees=100, n_features=5, sample_prop=1.0, bootstrap=False):
        self.n_trees = n_trees
        self.n_features = n_features
        self.sample_prop = sample_prop
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y, base_tree=None):
        for _ in range(self.n_trees):
            if base_tree:
                tree = RotationTree.from_model(base_tree, n_features=self.n_features, sample_prop=self.sample_prop, bootstrap=self.bootstrap)
            else:
                tree = RotationTree(n_features=self.n_features, sample_prop=self.sample_prop, bootstrap=self.bootstrap)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return mode(all_preds, axis=0)[0].flatten()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_features = pd.read_csv(FUSED_FEATURES_FILE)
    df_interactions = pd.read_csv(INTERACTION_FILE)
    X, y = prepare_pair_features(df_features, df_interactions)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RotationForest(n_trees=100, n_features=5, bootstrap=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "metrics.csv", index=False)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Fused Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png")
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    main()
