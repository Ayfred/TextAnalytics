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
    # Initialize Support Vector Classifier
    svm_classifier = SVC(kernel='linear', random_state=42)

    # Train the Support Vector Classifier
    svm_classifier.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = svm_classifier.predict(X_train_scaled)
    y_pred_test = svm_classifier.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("SVM")
train_svm(X_train_scaled, y_train, X_test_scaled)

# linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize Linear Regression model
    linear_regression = LinearRegression()

    # Train the model
    linear_regression.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = linear_regression.predict(X_train_scaled)
    y_pred_test = linear_regression.predict(X_test_scaled)

    # Evaluate model performance
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    print("Training MSE:", train_mse)
    print("Test MSE:", test_mse)

print("\nLinear Regression")
train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize Decision Tree Classifier
    decision_tree = DecisionTreeClassifier(random_state=42)

    # Train the model
    decision_tree.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = decision_tree.predict(X_train_scaled)
    y_pred_test = decision_tree.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nDecision Tree")
train_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize Random Forest Classifier
    random_forest = RandomForestClassifier(random_state=42)

    # Train the model
    random_forest.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = random_forest.predict(X_train_scaled)
    y_pred_test = random_forest.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nRandom Forest")
train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize Gradient Boosting Classifier
    gradient_boosting = GradientBoostingClassifier(random_state=42)

    # Train the model
    gradient_boosting.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = gradient_boosting.predict(X_train_scaled)
    y_pred_test = gradient_boosting.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nGradient Boosting")
train_gradient_boosting(X_train_scaled, y_train, X_test_scaled, y_test)

# Neural Network
from sklearn.neural_network import MLPClassifier

def train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize Neural Network Classifier
    neural_network = MLPClassifier(random_state=42)

    # Train the model
    neural_network.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = neural_network.predict(X_train_scaled)
    y_pred_test = neural_network.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nNeural Network")
train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize K-Nearest Neighbors Classifier
    knn = KNeighborsClassifier()

    # Train the model
    knn.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = knn.predict(X_train_scaled)
    y_pred_test = knn.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nK-Nearest Neighbors")
train_knn(X_train_scaled, y_train, X_test_scaled, y_test)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize Naive Bayes Classifier
    naive_bayes = GaussianNB()

    # Train the model
    naive_bayes.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = naive_bayes.predict(X_train_scaled)
    y_pred_test = naive_bayes.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nNaive Bayes")
train_naive_bayes(X_train_scaled, y_train, X_test_scaled, y_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize Logistic Regression model
    logistic_regression = LogisticRegression()

    # Train the model
    logistic_regression.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = logistic_regression.predict(X_train_scaled)
    y_pred_test = logistic_regression.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nLogistic Regression")
train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)

# XGBoost
from xgboost import XGBClassifier

def train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize XGBoost Classifier
    xgboost = XGBClassifier(random_state=42)

    # Train the model
    xgboost.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = xgboost.predict(X_train_scaled)
    y_pred_test = xgboost.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nXGBoost")
train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)

# LightGBM
from lightgbm import LGBMClassifier

def train_lightgbm(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize LightGBM Classifier
    lightgbm = LGBMClassifier(random_state=42)

    # Train the model
    lightgbm.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = lightgbm.predict(X_train_scaled)
    y_pred_test = lightgbm.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nLightGBM")
train_lightgbm(X_train_scaled, y_train, X_test_scaled, y_test)

# CatBoost
from catboost import CatBoostClassifier

def train_catboost(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize CatBoost Classifier
    catboost = CatBoostClassifier(random_state=42, verbose=0)

    # Train the model
    catboost.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = catboost.predict(X_train_scaled)
    y_pred_test = catboost.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nCatBoost")
train_catboost(X_train_scaled, y_train, X_test_scaled, y_test)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

def train_adaboost(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize AdaBoost Classifier
    adaboost = AdaBoostClassifier(random_state=42)

    # Train the model
    adaboost.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = adaboost.predict(X_train_scaled)
    y_pred_test = adaboost.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nAdaBoost")
train_adaboost(X_train_scaled, y_train, X_test_scaled, y_test)

# Bagging
from sklearn.ensemble import BaggingClassifier

def train_bagging(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize Bagging Classifier
    bagging = BaggingClassifier(random_state=42)

    # Train the model
    bagging.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = bagging.predict(X_train_scaled)
    y_pred_test = bagging.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nBagging")
train_bagging(X_train_scaled, y_train, X_test_scaled, y_test)

# Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def train_stacking(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize base estimators
    base_estimators = [
        ('svm', SVC(kernel='linear', random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ]

    # Initialize Stacking Classifier
    stacking = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())

    # Train the model
    stacking.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = stacking.predict(X_train_scaled)
    y_pred_test = stacking.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nStacking")
#train_stacking(X_train_scaled, y_train, X_test_scaled, y_test)

# Voting
from sklearn.ensemble import VotingClassifier

def train_voting(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize Voting Classifier
    voting = VotingClassifier(estimators=[
        ('svm', SVC(kernel='linear', random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ], voting='hard')

    # Train the model
    voting.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = voting.predict(X_train_scaled)
    y_pred_test = voting.predict(X_test_scaled)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

print("\nVoting")
#train_voting(X_train_scaled, y_train, X_test_scaled, y_test)

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(X_train_scaled, y_train):
    # Initialize Support Vector Classifier
    svm_classifier = SVC(random_state=42)

    # Define hyperparameters to tune
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }

    # Initialize Grid Search
    grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5)

    # Train the model
    grid_search.fit(X_train_scaled, y_train)

    # Print best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)

#tune_hyperparameters(X_train_scaled, y_train)
""" 
# Cross-Validation
from sklearn.model_selection import cross_val_score

def cross_validate(X_train_scaled, y_train):
    # Initialize Support Vector Classifier
    svm_classifier = SVC(kernel='linear', C=1, random_state=42)

    # Perform 5-fold cross-validation
    scores = cross_val_score(svm_classifier, X_train_scaled, y_train, cv=5)

    print("Cross-Validation Scores:", scores)

cross_validate(X_train_scaled, y_train)

# Feature Importance
from sklearn.ensemble import RandomForestClassifier

def feature_importance(X_train_scaled, y_train):
    # Initialize Random Forest Classifier
    random_forest = RandomForestClassifier(random_state=42)

    # Train the model
    random_forest.fit(X_train_scaled, y_train)

    # Get feature importances
    feature_importances = random_forest.feature_importances_

    print("Feature Importances:", feature_importances) """

#feature_importance(X_train_scaled, y_train)

# Save Model
import joblib

""" def save_model(X_train_scaled, y_train):

    # Initialize Random Forest Classifier
    random_forest = RandomForestClassifier(random_state=42)

    # Train the model
    random_forest.fit(X_train_scaled, y_train)

    # Save the model
    joblib.dump(random_forest, 'random_forest_model.pkl')


# Load Model

def load_model(X_test_scaled, y_test):
    # Load the model
    random_forest = joblib.load('random_forest_model.pkl')

    # Predictions
    y_pred = random_forest.predict(X_test_scaled)

    # Evaluate model performance
    test_accuracy = accuracy_score(y_test, y_pred)

    print("Test Accuracy:", test_accuracy)

     """