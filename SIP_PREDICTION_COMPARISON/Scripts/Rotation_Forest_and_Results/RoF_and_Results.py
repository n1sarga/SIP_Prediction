import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Rotation Tree Classifier Class
class RotationTree:
    @classmethod
    def from_model(cls, tree, n_features=3, sample_prop=0.5, bootstrap=True):
        rt = RotationTree(n_features, sample_prop, bootstrap)
        rt.model = tree
        return rt

    def __init__(self, n_features=3, sample_prop=0.5, bootstrap=True):
        self.model = DecisionTreeClassifier()
        self.n_features = n_features
        self.sample_prop = sample_prop
        self.bootstrap = bootstrap
        self.partitions = []
        self.rotation_matrices = []

    def fit(self, X, y):
        feature_partitions = self.partition_features(X)
        transformed_partitions = []
        for partition in feature_partitions:
            sampled_data = self.get_samples(partition, y)
            rotation_matrix = self.get_rotation_matrix(sampled_data)
            transformed_partitions.append(np.dot(partition, rotation_matrix))

        new_X = np.concatenate(transformed_partitions, axis=1)
        if self.bootstrap:
            xx, yy = self.boot_sample(new_X, y)
            self.model.fit(xx, yy)
        else:
            self.model.fit(new_X, y)

    def partition_features(self, x):
        n_cols = x.shape[1]
        column_cases = [i for i in range(n_cols)]
        np.random.shuffle(column_cases)
        case_output = [[m for m in column_cases[i::self.n_features]] for i in range(self.n_features)]
        feature_output = []
        for partition in case_output:
            feature_output.append(np.array([x[:, i] for i in partition]).T)
        self.partition_nums = case_output
        return feature_output

    def get_samples(self, featureset, y):
        featureset = np.column_stack([featureset, y])
        n_rows = featureset.shape[0]
        unique_y = np.unique(y)
        include = []
        for cat in unique_y:
            i = np.random.uniform(size=1)
            if i > 0.5:
                include.append(cat)
        if len(include) == 0:
            include = unique_y
        mask = np.isin(featureset[:, -1], include)
        rotation_seed = featureset[mask, :]
        cases = np.random.choice(int(rotation_seed.shape[0]), size=round(self.sample_prop * rotation_seed.shape[0]))
        rotation_seed = rotation_seed[cases, :]
        sample_y = rotation_seed[:, -1]
        rotation_seed = np.delete(rotation_seed, -1, 1)
        return rotation_seed

    def get_rotation_matrix(self, samples):
        pca = PCA()
        pca.fit(samples)
        rotation_matrix = pca.components_
        self.rotation_matrices.append(rotation_matrix)
        return rotation_matrix

    def boot_sample(self, x, y):
        newdata = np.concatenate((x, y[:, np.newaxis]), axis=1)
        cases = np.random.choice(newdata.shape[0], size=newdata.shape[0], replace=True)
        samples = newdata[cases, ]
        return samples[:, :-1], samples[:, -1]

    def predict(self, X):
        partitions = [np.array([X[:, i] for i in partition]).T for partition in self.partition_nums]
        transformed_partitions = []
        for i, p in enumerate(partitions):
            transformed_partitions.append(np.dot(p, self.rotation_matrices[i]))
        new_X = np.concatenate(transformed_partitions, axis=1)
        return self.model.predict(new_X)

    
# Rotation Forest Class
class RotationForest:
    def __init__(self, n_trees=100, n_features=3, sample_prop=0.5, bootstrap=False):
        self.bootstrap = bootstrap
        self.n_trees = n_trees
        self.n_features = n_features
        self.sample_prop = sample_prop
        self.is_fit = False
        self.trees = []

    def fit(self, X, y, model=None):
        if model:
            for i in range(self.n_trees):
                tree = RotationTree.from_model(model,
                                               n_features=self.n_features,
                                               sample_prop=self.sample_prop,
                                               bootstrap=self.bootstrap)
                tree.fit(X, y)
                self.trees.append(tree)
        else:
            for i in range(self.n_trees):
                tree = RotationTree(n_features=self.n_features,
                                    sample_prop=self.sample_prop,
                                    bootstrap=self.bootstrap)
                tree.fit(X, y)
                self.trees.append(tree)

    def predict(self, X):
        all = []
        for model in self.trees:
            all.append(model.predict(X))
        all = np.asarray(all)
        preds = mode(all)[0].flatten()
        return preds
    
# Training and Evaluation
df_dst = pd.read_csv('C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Maize/Features/SIP_DST.csv')
df_dct = pd.read_csv('C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Maize/Features/SIP_DCT.csv')
df_wt = pd.read_csv('C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Maize/Features/SIP_WT.csv')
df_hb = pd.read_csv('C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Maize/Features/SIP_Hilbert.csv')
df_fr = pd.read_csv('C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Maize/Features/SIP_Fourier.csv')

def prepare_data(df, feature_suffix):
    feature_a_col = f'Features_A_{feature_suffix}'
    feature_b_col = f'Features_B_{feature_suffix}'

    df[feature_a_col] = df[feature_a_col].apply(lambda x: np.array(eval(x)))
    df[feature_b_col] = df[feature_b_col].apply(lambda x: np.array(eval(x)))

    X = np.array([np.concatenate([a, b]) for a, b in zip(df[feature_a_col], df[feature_b_col])])
    y = df['Interaction'].values
    
    return X, y

# Prepare data for each transformation
X_dst, y_dst = prepare_data(df_dst, 'DST')
X_dct, y_dct = prepare_data(df_dct, 'DCT')
X_wt, y_wt = prepare_data(df_wt, 'WT')
X_hb, y_hb = prepare_data(df_hb, 'Hilbert')
X_fr, y_fr = prepare_data(df_fr, 'Fourier')

# Split dataset for train and test
X_train_dst, X_test_dst, y_train_dst, y_test_dst = train_test_split(X_dst, y_dst, test_size=0.3, random_state=42)
X_train_dct, X_test_dct, y_train_dct, y_test_dct = train_test_split(X_dct, y_dct, test_size=0.3, random_state=42)
X_train_wt, X_test_wt, y_train_wt, y_test_wt = train_test_split(X_wt, y_wt, test_size=0.3, random_state=42)
X_train_hb, X_test_hb, y_train_hb, y_test_hb = train_test_split(X_hb, y_hb, test_size=0.3, random_state=42)
X_train_fr, X_test_fr, y_train_fr, y_test_fr = train_test_split(X_fr, y_fr, test_size=0.3, random_state=42)

# Initialize Rotation Forest for each transformation
rf_dst = RotationForest(n_trees=100, n_features=3, bootstrap=True)
rf_dct = RotationForest(n_trees=100, n_features=3, bootstrap=True)
rf_wt = RotationForest(n_trees=100, n_features=3, bootstrap=True)
rf_hb = RotationForest(n_trees=100, n_features=3, bootstrap=True)
rf_fr = RotationForest(n_trees=100, n_features=3, bootstrap=True)

# Fit models
rf_dst.fit(X_train_dst, y_train_dst)
rf_dct.fit(X_train_dct, y_train_dct)
rf_wt.fit(X_train_wt, y_train_wt)
rf_hb.fit(X_train_hb, y_train_hb)
rf_fr.fit(X_train_fr, y_train_fr)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

y_prob_dst = rf_dst.predict(X_test_dst)
y_prob_dct = rf_dct.predict(X_test_dct)
y_prob_wt = rf_wt.predict(X_test_wt)
y_prob_hb = rf_hb.predict(X_test_hb)
y_prob_fr = rf_fr.predict(X_test_fr)

fpr_dst, tpr_dst, _ = roc_curve(y_test_dst, y_prob_dst)
roc_auc_dst = auc(fpr_dst, tpr_dst)

fpr_dct, tpr_dct, _ = roc_curve(y_test_dct, y_prob_dct)
roc_auc_dct = auc(fpr_dct, tpr_dct)

fpr_wt, tpr_wt, _ = roc_curve(y_test_wt, y_prob_wt)
roc_auc_wt = auc(fpr_wt, tpr_wt)

fpr_hb, tpr_hb, _ = roc_curve(y_test_hb, y_prob_hb)
roc_auc_hb = auc(fpr_hb, tpr_hb)

fpr_fr, tpr_fr, _ = roc_curve(y_test_fr, y_prob_fr)
roc_auc_fr = auc(fpr_fr, tpr_fr)

plt.figure(figsize=(10, 8))
plt.plot(fpr_dst, tpr_dst, color='darkorange', lw=2, label=f'DST ROC Curve (area = {roc_auc_dst:.2f})')
plt.plot(fpr_dct, tpr_dct, color='blue', lw=2, label=f'DCT ROC Curve (area = {roc_auc_dct:.2f})')
plt.plot(fpr_wt, tpr_wt, color='green', lw=2, label=f'WT ROC Curve (area = {roc_auc_wt:.2f})')
plt.plot(fpr_hb, tpr_hb, color='red', lw=2, label=f'Hilbert ROC Curve (area = {roc_auc_hb:.2f})')
plt.plot(fpr_fr, tpr_fr, color='purple', lw=2, label=f'Fourier ROC Curve (area = {roc_auc_fr:.2f})')

# Plot a diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('Receiver Operating Characteristic for All Transformation Techniques')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt

# Initialize a list to hold the metrics for each model
metrics_list = []

# Function to calculate metrics and store them in the list
def collect_classification_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
    pearson_corr = pearsonr(y_true, y_pred)[0]  # Pearson correlation coefficient

    # Append metrics to the list as a dictionary
    metrics_list.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'MCC': mcc,
        'RMSE': rmse,
        'Pearson Correlation': pearson_corr
    })

# Predictions
y_pred_dst = rf_dst.predict(X_test_dst)
y_pred_dct = rf_dct.predict(X_test_dct)
y_pred_wt = rf_wt.predict(X_test_wt)
y_pred_hb = rf_hb.predict(X_test_hb)
y_pred_fr = rf_fr.predict(X_test_fr)

# Collect metrics for each transformation
collect_classification_metrics(y_test_dst, y_pred_dst, "DST")
collect_classification_metrics(y_test_dct, y_pred_dct, "DCT")
collect_classification_metrics(y_test_wt, y_pred_wt, "WT")
collect_classification_metrics(y_test_hb, y_pred_hb, "Hilbert")
collect_classification_metrics(y_test_fr, y_pred_fr, "Fourier")

# Create a DataFrame from the metrics list
metrics_df = pd.DataFrame(metrics_list)

# Display the metrics DataFrame
print(metrics_df)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predictions for confusion matrix
y_pred_dst = rf_dst.predict(X_test_dst)
y_pred_dct = rf_dct.predict(X_test_dct)
y_pred_wt = rf_wt.predict(X_test_wt)
y_pred_hb = rf_hb.predict(X_test_hb)
y_pred_fr = rf_fr.predict(X_test_fr)

# List of predictions and names for easier iteration
predictions = [y_pred_dst, y_pred_dct, y_pred_wt, y_pred_hb, y_pred_fr]
models = ['DST', 'DCT', 'WT', 'Hilbert', 'Fourier']
y_tests = [y_test_dst, y_test_dct, y_test_wt, y_test_hb, y_test_fr]

# Plotting confusion matrices for each model
for i, model in enumerate(models):
    cm = confusion_matrix(y_tests[i], predictions[i])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_tests[i]))
    disp.plot(cmap=plt.cm.Blues)  # You can change the colormap
    plt.title(f'Confusion Matrix for {model} Model')
    plt.show()

# Set the index to 'Model' for easier plotting
metrics_df.set_index('Model', inplace=True)

# List of metrics to plot
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'RMSE', 'Pearson Correlation']

import matplotlib.pyplot as plt
import seaborn as sns
# Define colors for each model using seaborn's color palette
colors = sns.color_palette("Set2", len(metrics_df.index))

# Plot bar charts for each metric with different colors
for metric in metrics_to_plot:
    plt.figure(figsize=(8, 6))
    metrics_df[metric].plot(kind='bar', color=colors)
    plt.title(f'{metric} Comparison Across Models')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    if metric != 'RMSE':  # RMSE may have a different scale
        plt.ylim([0, 1])  # Adjust y-axis limits if necessary
    plt.tight_layout()
    plt.show()
