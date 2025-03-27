import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pickle

def read_data(file_path = "activity_context_tracking_data.csv"):
    df = pd.read_csv(file_path)
    return df
# df = read_data()

def explore_data(df):
    # Exploring a random sample of data
    sample = df.sample(10)
    print(sample)

    # Just a peek into the data statistics
    description = df.describe()
    print("\nDescription")
    print(description)

    # Viewing the number of columns and rows
    shape = df.shape
    print("\nShape")
    print(shape)

    # Checking for null values
    null_counts = df.isnull().sum()
    print("\nNull value count")
    print(null_counts)

    # Information about the data
    print("\nInfo")
    info = df.info()
    print(info)
# explore_data(df)

def drop_columns(df):
    df = df.drop("_id", axis=1)
    return df
# df = drop_columns(df)

# drop_columns(df)

def duplicate_rows(df):
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

# duplicate_rows(df)

def descriptive_statistics(df):
    df = read_data()
    mean = df.mean(numeric_only=True)
    median = df.median(numeric_only=True)
    std = df.std(numeric_only=True)
    var = df.var(numeric_only=True)
    min_val = df.min(numeric_only=True)
    max_val = df.max(numeric_only=True)
    skew = df.skew(numeric_only=True)
    kurt = df.kurt(numeric_only=True)

    # Combine all statistics into a single dataframe
    df_stats = pd.DataFrame({'Mean': mean, 'Median': median, 'Std Dev': std, 'Variance': var,
                             'Minimum': min_val, 'Maximum': max_val, 'Skewness': skew, 'Kurtosis': kurt})

    # Print the results as a table
    print('Descriptive Statistics:\n')
    print(df_stats.to_string())

# descriptive_statistics(df)

def histograms(df):
    df.hist(figsize=(20, 12), bins=20)
    plt.suptitle("Frequency distribution Histograms")
    plt.show()

# histograms(df)

def correlation_heatmap(df):
    df_corr = df.corr()
    plt.figure(figsize=(16, 8))
    plt.title('Correlation Heatmap')
    sns.heatmap(df_corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    plt.show()

# correlation_heatmap(df)

def variable_dependency(df):
    plt.figure(figsize=(10, 6))
    plt.title('Variable Dependency')
    sns.scatterplot(x='orZ', y='accX', hue='activity', data=df)
    plt.show()

# variable_dependency(df)

def activity_frequency_distribution(df):
    activity_count = df.groupby('activity').count()
    plt.figure(figsize=(24, 8))
    plt.title('Frequency Distribution of Activity before Balancing')
    sns.countplot(x=df['activity'])
    plt.show()

# activity_frequency_distribution(df)

def oversampling(df):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    smote = SMOTE()
    x_resampled, y_resampled = smote.fit_resample(x, y)

    print(x_resampled.shape)
    print(y_resampled.shape)

    return x_resampled, y_resampled
# x_resampled, y_resampled = oversampling(df)

def resampled_activity_freq(y_resampled):
    yo_count_pd = pd.Series(y_resampled)
    yo_count = yo_count_pd.value_counts()

    plt.figure(figsize=(24, 8))
    plt.title('Count of Activity after balancing')
    sns.countplot(x=yo_count_pd)
    plt.show()

# resampled_activity_freq(y_resampled)

def split_data(x_resampled, y_resampled):
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test

# x_train, x_test, y_train, y_test = split_data(x_resampled, y_resampled)

def scale_data(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)
    return x_train_scaled, x_test_scaled

# x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

def random_forest_classifier(x_train_scaled, y_train):
    rfc = RandomForestClassifier(random_state=42)
    rfc_model = rfc.fit(x_train_scaled, y_train)
    return rfc_model

# rfc_model = random_forest_classifier(x_train_scaled, y_train)

def svm_classifier(x_train_scaled, y_train):
    svm = SVC()
    svm_model = svm.fit(x_train_scaled, y_train)
    return svm_model

# svm_model = svm_classifier(x_train_scaled, y_train)

def mlp_classifier(x_train_scaled, y_train):
    mlp = MLPClassifier(random_state=42, max_iter=200)
    mlp_model = mlp.fit(x_train_scaled, y_train)
    return mlp_model

# mlp_model = mlp_classifier(x_train_scaled, y_train)

def evaluate_classifier(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
#     cm.plot()
    plt.xticks(rotation=90)
    plt.show()

# evaluate_classifier(y_test, rfc_model.predict(x_test_scaled))

# evaluate_classifier(y_test, svm_model.predict(x_test_scaled))

# evaluate_classifier(y_test, mlp_model.predict(x_test_scaled))

def save_model(model, file_path):
    pickle.dump(model, open(file_path, 'wb'))
    
def load_model(file_path):
    return pickle.load(open(file_path, 'rb'))

# save_model(rfc_model, 'rfc_model.pkl')
# save_model(svm_model, 'svm_model.pkl')
# save_model(mlp_model, 'mlp_model.pkl')

# pickled_rfc_model = load_model('rfc_model.pkl')
# pickled_svm_model = load_model('svm_model.pkl')
# pickled_mlp_model = load_model('mlp_model.pkl')

# pickled_rfc_model.predict(x_test_scaled)

# pickled_svm_model.predict(x_test_scaled)

# pickled_mlp_model.predict(x_test_scaled)