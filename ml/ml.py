import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv("./dataset/preprocessed_data.csv")

# shuffle the rows
data = data.sample(frac=1).reset_index(drop=True)

# Separate features and target variable
X = data.drop('gender', axis=1)  # Features
y = data['gender']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def train_svm(X_train_scaled, y_train, X_test_scaled):
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)

    y_pred_train = svm_classifier.predict(X_train_scaled)
    y_pred_test = svm_classifier.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("SVM")
train_svm(X_train_scaled, y_train, X_test_scaled)

# linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    linear_regression = LinearRegression()
    linear_regression.fit(X_train_scaled, y_train)

    y_pred_train = linear_regression.predict(X_train_scaled)
    y_pred_test = linear_regression.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    print("Training MSE:", train_mse)
    print("Test MSE:", test_mse)

print("\nLinear Regression")
train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test):
    random_forest = RandomForestClassifier(random_state=42)
    random_forest.fit(X_train_scaled, y_train)

    y_pred_train = random_forest.predict(X_train_scaled)
    y_pred_test = random_forest.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nRandom Forest")
train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(X_train_scaled, y_train, X_test_scaled, y_test):
    gradient_boosting = GradientBoostingClassifier(random_state=42)
    gradient_boosting.fit(X_train_scaled, y_train)

    y_pred_train = gradient_boosting.predict(X_train_scaled)
    y_pred_test = gradient_boosting.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nGradient Boosting")
train_gradient_boosting(X_train_scaled, y_train, X_test_scaled, y_test)


# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train_scaled, y_train, X_test_scaled, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)

    y_pred_train = knn.predict(X_train_scaled)
    y_pred_test = knn.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nK-Nearest Neighbors")
train_knn(X_train_scaled, y_train, X_test_scaled, y_test)


# Logistic Regression
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train_scaled, y_train)

    y_pred_train = logistic_regression.predict(X_train_scaled)
    y_pred_test = logistic_regression.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nLogistic Regression")
train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)


