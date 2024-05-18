import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('train.csv')
print(df.head())

# Display basic information about the dataset
df.shape
df.info()
df.describe().T

# Display value counts for specific columns
df['ethnicity'].value_counts()
df['relation'].value_counts()

# Replace specific values
df = df.replace({'yes': 1, 'no': 0, '?': 'Others', 'others': 'Others'})

# Plot pie chart of the target variable
plt.pie(df['Class/ASD'].value_counts().values, autopct='%1.1f%%')
plt.show()

# Separate columns by data type
ints, objects, floats = [], [], []
for col in df.columns:
    if df[col].dtype == int:
        ints.append(col)
    elif df[col].dtype == object:
        objects.append(col)
    else:
        floats.append(col)

ints.remove('ID')
ints.remove('Class/ASD')

# Plot count plots for integer columns
plt.subplots(figsize=(15, 15))
for i, col in enumerate(ints):
    plt.subplot(4, 3, i + 1)
    sb.countplot(df[col], hue=df['Class/ASD'])
plt.tight_layout()
plt.show()

# Plot count plots for object columns
plt.subplots(figsize=(15, 30))
for i, col in enumerate(objects):
    plt.subplot(5, 3, i + 1)
    sb.countplot(df[col], hue=df['Class/ASD'])
    plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

# Plot distribution plots for float columns
plt.subplots(figsize=(15, 5))
for i, col in enumerate(floats):
    plt.subplot(1, 2, i + 1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

# Plot box plots for float columns
plt.subplots(figsize=(15, 5))
for i, col in enumerate(floats):
    plt.subplot(1, 2, i + 1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

# Filter out rows based on specific condition
df = df[df['result'] > -5]
df.shape

# Function to convert age to age groups
def convertAge(age):
    if age < 4:
        return 'Toddler'
    elif age < 12:
        return 'Kid'
    elif age < 18:
        return 'Teenager'
    elif age < 40:
        return 'Young'
    else:
        return 'Senior'

df['ageGroup'] = df['age'].apply(convertAge)
sb.countplot(x=df['ageGroup'], hue=df['Class/ASD'])
plt.show()

# Function to add features to the dataset
def add_feature(data):
    data['sum_score'] = 0
    for col in data.loc[:, 'A1_Score':'A10_Score'].columns:
        data['sum_score'] += data[col]
    data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']
    return data

df = add_feature(df)
sb.countplot(x=df['sum_score'], hue=df['Class/ASD'])
plt.show()

# Apply logarithmic transformation to age
df['age'] = df['age'].apply(lambda x: np.log(x))
sb.distplot(df['age'])
plt.show()

# Function to encode labels
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

df = encode_labels(df)

# Define features and target variable
removal = ['ID', 'age_desc', 'used_app_before', 'austim']
features = df.drop(removal + ['Class/ASD'], axis=1)
target = df['Class/ASD']

# Split the dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)

# Handle imbalanced data
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)
print(X.shape, Y.shape)

# Visualize correlation matrix
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

# Define models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

# Train and evaluate models
for model in models:
    model.fit(X, Y)
    print(f'{model} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(Y, model.predict(X)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, model.predict(X_val)))
    print()

# Plot confusion matrix
metrics.plot_confusion_matrix(models[0], X_val, Y_val)
plt.show()
