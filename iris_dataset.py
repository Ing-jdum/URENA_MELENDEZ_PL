from sklearn.datasets import load_iris
import torch
from sklearn.preprocessing import MinMaxScaler

from logic_explained_networks import lens
from logic_explained_networks.lens.utils.datasets import StructuredDataset
from torch.utils.data import random_split


##############################################################
def train_len(data: StructuredDataset, target_class=0):

    # Dataset splitting into train, validation and testing.
    dataset_length = len(data.x)

    # Calculate the lengths for train, validation, and test
    train_length = dataset_length * 70 // 100
    val_length = dataset_length * 15 // 100
    test_length = dataset_length - train_length - val_length

    tensor_data = torch.utils.data.TensorDataset(data.x, data.y)

    # Use random_split to split the dataset
    train_data, val_data, test_data = random_split(tensor_data, [train_length, val_length, test_length])

    x_train, y_train = tensor_data[train_data.indices]
    x_test, y_test = tensor_data[test_data.indices]

    # model instantiation
    model = lens.models.XMuNN(n_classes=len(torch.unique(data.y)), n_features=x_train.shape[1],
                              hidden_neurons=[20], loss=torch.nn.CrossEntropyLoss())

    # training
    model.fit(train_data, val_data, epochs=500, l_r=0.1)

    # get accuracy on test samples
    test_acc = model.evaluate(test_data)
    print("Test accuracy:", test_acc)

    concept_names = data.feature_names
    formula = model.get_global_explanation(x_train, y_train, target_class,
                                           top_k_explanations=2, concept_names=concept_names)
    print(f"{formula} <-> f{target_class}")

    # compute explanation accuracy
    exp_accuracy, _ = lens.logic.test_explanation(formula, target_class, x_test, y_test,
                                                  concept_names=concept_names)
    print("Logic Test Accuracy:", exp_accuracy)
###############################################################################


# set random seed
lens.utils.base.set_seed(0)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# Normalize X to be between 0 and 1
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Verify shapes
print("X_tensor shape:", X_tensor.shape)  # (150, 4)
print("y_tensor shape:", y_tensor.shape)  # (150, 3)

# Verify values between 0 and 1
print("X_tensor:", X_tensor)

# Create PyTorch TensorDataset
data = torch.utils.data.TensorDataset(X_tensor, y_tensor)
train_len(data=data)
###############################################################################
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# lets define custom ranges for each feature
sepal_length_bins = [0, 5, 7, 10]
sepal_width_bins = [0, 2.5, 3.5, 5]
petal_length_bins = [0, 2, 4, 6]
petal_width_bins = [0, 0.5, 1, 1.5, 2]


# Function to encode data based on ranges
def encode_ranges(data, bins, labels):
    encoded_data = pd.cut(data, bins=bins, labels=labels)
    return pd.get_dummies(encoded_data, prefix='', prefix_sep='')


# Encode the data based on these ranges
sepal_length_encoded = encode_ranges(iris_df['sepal length (cm)'], sepal_length_bins,
                                     ['sepal L 0-5', 'sepal L 5-7', 'sepal L 7-10'])
sepal_width_encoded = encode_ranges(iris_df['sepal width (cm)'], sepal_width_bins,
                                    ['sepal W 0-2.5', 'sepal W 2.5-3.5', 'sepal W 3.5-5'])
petal_length_encoded = encode_ranges(iris_df['petal length (cm)'], petal_length_bins,
                                     ['petal L 0-2', 'petal L 2-4', 'petal L 4-6'])
petal_width_encoded = encode_ranges(iris_df['petal width (cm)'], petal_width_bins,
                                    ['petal W 0-0.5', 'petal W 0.5-1', 'petal W 1-1.5', 'petal W 1.5-2'])

# Concatenate the encoded columns
encoded_df = pd.concat([sepal_length_encoded, sepal_width_encoded, petal_length_encoded, petal_width_encoded], axis=1)
# shuffled_df = encoded_df.sample(frac=1, axis=1, random_state=42)


# Convert to PyTorch tensors
X_tensor = torch.tensor(encoded_df.values, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

data = StructuredDataset(X_tensor, y_tensor, dataset_name="iris",
                         feature_names=encoded_df.columns, class_names=iris.target_names)

train_len(data=data)
######################################################################################
# Diabetes dataset

from ucimlrepo import fetch_ucirepo

# fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# data (as pandas dataframes)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# drop most likely features to be not relevant
X_filtered = X.iloc[:, :-2]

# Separate in ranges continous features like bmi, PhysHlth, MentHlth, GenHlth
# Here we have to use our knowledge of the dataset. PhysHlth and  MentHlth represent
# The number of days you felt you had a bad PhysHlth and MentHlth in the last 30 days.
# we can split it into intervals like less than 10 days, between 10 and 20 and more than 20 days.
# On the other hand, bmi has a sort of stablished range of what makes you lean, healthy or obese based on sex.
# So we expect to see correlation between sex and bmi in the rules.
# GenHealth and age can be normalized. Not that by normalizing GenHealth we are not making any difference
# between a health of 3 and a health of 5. Because now it will be included in the rule if it is >0.5
# or it's negation if it is less than 0.5

# bmi < 18, 18 <= bmi < 25, 25 <= bmi < 30, bmi>=30
X_filtered.loc[:, 'BMI < 18'] = X_filtered['BMI'] < 18
X_filtered.loc[:, '18 <= BMI < 25'] = (X_filtered['BMI'] >= 18) & (X_filtered['BMI'] < 25)
X_filtered.loc[:, '25 <= BMI < 30'] = (X_filtered['BMI'] >= 25) & (X_filtered['BMI'] < 30)
X_filtered.loc[:, 'BMI >= 30'] = X_filtered['BMI'] >= 30

# Create boolean columns based on intervals for 'PhysHlth'
X_filtered.loc[:, 'PhysHlth < 10'] = X_filtered['PhysHlth'] < 10
X_filtered.loc[:, '10 <= PhysHlth <= 20'] = (X_filtered['PhysHlth'] >= 10) & (X_filtered['PhysHlth'] <= 20)
X_filtered.loc[:, 'PhysHlth > 20'] = X_filtered['PhysHlth'] > 20

# Create boolean columns based on intervals for 'MentHlth'
X_filtered.loc[:, 'MentHlth < 10'] = X_filtered['MentHlth'] < 10
X_filtered.loc[:, '10 <= MentHlth <= 20'] = (X_filtered['MentHlth'] >= 10) & (X_filtered['MentHlth'] <= 20)
X_filtered.loc[:, 'MentHlth > 20'] = X_filtered['MentHlth'] > 20

X_filtered = X_filtered.astype('float32')

# Min-Max normalization for 'Age' and 'GenHlth'
X_filtered.loc[:, 'Age'] = (X_filtered['Age'] - X_filtered['Age'].min()) / (X_filtered['Age'].max() - X_filtered['Age'].min())
X_filtered.loc[:, 'GenHlth'] = (X_filtered['GenHlth'] - X_filtered['GenHlth'].min()) / (X_filtered['GenHlth'].max() - X_filtered['GenHlth'].min())


X_filtered = X_filtered.drop(columns=['BMI', 'MentHlth', 'PhysHlth'])

X_tensor = torch.tensor(X_filtered.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

data = StructuredDataset(X_tensor, y_tensor, dataset_name="diabetes",
                         feature_names=X_filtered.columns, class_names=['not diabetes', 'diabetes'])

train_len(data=data)
