import numpy as np
import pandas as pd
import torch
from logic_explained_networks import lens
from logic_explained_networks.lens.utils.datasets import StructuredDataset
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from logic_explained_networks.lens.utils.metrics import Metric, F1Score

# Set the random seed for repeatibility purposes
lens.utils.base.set_seed(0)


def create_boolean_columns(df: pd.DataFrame, n: int):
    new_df = df.copy()

    for column in df.columns:
        if df[column].dtype in [int, float]:
            min_val = df[column].min()
            max_val = df[column].max()
            step = (max_val - min_val) / n

            for i in range(n):
                lower_bound = np.round(min_val + i * step, 2)
                upper_bound = np.round(min_val + (i + 1) * step, 2)
                bool_col_name = f"{lower_bound}<={column}<{upper_bound}"
                new_df[bool_col_name] = (df[column] >= lower_bound) & (df[column] < upper_bound)
            new_df.drop(column, axis=1, inplace=True)
    return new_df


def print_confusion_matrix(targets: torch.Tensor, predictions_np: np.array):
    # Convert tensors to numpy arrays
    targets_np = targets.cpu().numpy()

    # Calculate confusion matrix
    cm = confusion_matrix(targets_np, predictions_np)

    # Print confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


##############################################################
def train_len(data: StructuredDataset, print_conf_matrix=0, metric: Metric = F1Score(), target_class=0, epoch=500,
              l_r=0.1, hidden_neurons=20):
    # Dataset splitting into train, validation, and testing with stratification
    x_train, x_test, y_train, y_test = train_test_split(data.x, data.y, test_size=0.3, stratify=data.y, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test, random_state=42)

    # Convert data to PyTorch tensors
    train_data = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_data = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    test_data = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    # Model instantiation
    model_mu = lens.models.XMuNN(n_classes=len(torch.unique(data.y)), n_features=data.x.shape[1],
                                 hidden_neurons=[hidden_neurons], loss=torch.nn.CrossEntropyLoss())

    model_psi = lens.models.XPsiNetwork(n_classes=len(torch.unique(data.y)), n_features=data.x.shape[1],
                                        hidden_neurons=[hidden_neurons], loss=torch.nn.CrossEntropyLoss())

    model_relu = lens.models.XReluNN(n_classes=len(torch.unique(data.y)), n_features=data.x.shape[1],
                                     hidden_neurons=[hidden_neurons], loss=torch.nn.CrossEntropyLoss())

    models = [model_mu, model_psi, model_relu]

    # Training
    for model in models:
        model.fit(train_data, val_data, epochs=epoch, l_r=l_r)

        # Get accuracy on test samples
        test_acc = model_mu.evaluate(test_data, metric=metric)
        print("Test accuracy:", test_acc)

        concept_names = data.feature_names

        # Create a DataFrame to store explanations and their metrics
        explanations_dict = {"Explanation": [], "Metric Result": []}

        for target in range(target_class):
            formula = model.get_global_explanation(x_train, y_train, target,
                                                   top_k_explanations=2, metric=metric, concept_names=concept_names)

            # Compute explanation accuracy
            exp_accuracy, predictions = lens.logic.test_explanation(formula, target, x_test, y_test, metric=metric,
                                                                    concept_names=concept_names)

            # Store the explanation and its metric result in the dictionary
            explanations_dict["Explanation"].append(formula)
            explanations_dict["Metric Result"].append(exp_accuracy)

            if print_conf_matrix:
                print_confusion_matrix(targets=y_test, predictions_np=predictions)
        explanations_df = pd.DataFrame(explanations_dict)
        print(explanations_df)


dataset = pd.read_csv('Crop_Recommendation.csv')
y = dataset["Crop"].to_numpy()
label_encoder = LabelEncoder()
dataset['Crop'] = label_encoder.fit_transform(dataset['Crop'])
y_encoded = dataset["Crop"].to_numpy()
dataset.drop("Crop", axis=1, inplace=True)

# Pre-process

new_dataset = create_boolean_columns(dataset, 3)
X_tensor = torch.tensor(new_dataset.to_numpy(), dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.float32)

dataset = StructuredDataset(X_tensor, y_tensor, dataset_name="crops",
                            feature_names=new_dataset.columns.to_numpy(), class_names=np.unique(y).tolist())

train_len(dataset, target_class=22)

# # create a dataset, a tensor of 100 rows and 4 columns with data uniformly between 0 and 1
# x = torch.rand([100, 4])
#
# # basically, the result of the xor applied to the booleanize values of X, using 0,5 as threshold.
# # I think one of the threshold should include the 0.5
# y = (x[:, 0] >= 0.5) & (x[:, 1] >= 0.5)
#
# data = torch.utils.data.TensorDataset(x, y)
#
# # Dataset splitting into train, validation and testing.
# train_data, val_data, test_data = torch.utils.data.random_split(data, [80, 10, 10])
# x_train, y_train = data[train_data.indices]
# x_val, y_val = data[val_data.indices]
# x_test, y_test = data[test_data.indices]
#
# # model instantiation
# model = lens.models.XMuNN(n_classes=2, n_features=4,
#                           hidden_neurons=[3], loss=torch.nn.CrossEntropyLoss())
#
# # training
# model.fit(train_data, val_data, epochs=50, l_r=0.1)
#
# # get accuracy on test samples
# test_acc = model.evaluate(test_data)
# print("Test accuracy:", test_acc)
#
# # get first order logic explanations for a specific target class
# target_class = 1
# concept_names = ['x1', 'x2', 'x3', 'x4']
# formula = model.get_global_explanation(x_train, y_train, target_class,
#                                        top_k_explanations=2, concept_names=concept_names)
# print(f"{formula} <-> f{target_class}")
#
# # compute explanation accuracy
# exp_accuracy, _ = lens.logic.test_explanation(formula, target_class, x_test, y_test, metric= F1Score(),
#                                               concept_names=concept_names)
# print("Logic Test Accuracy:", exp_accuracy)
