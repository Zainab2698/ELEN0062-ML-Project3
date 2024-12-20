from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils.helpers import write_submission
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def train_evaluate_NeuralNet(X_train, y_train, X_test):
    # Define hyperparameter grid for the Decision Tree
    param_grid = {
            'hidden_layer_sizes':[(100,), (128, 64), (128, 64, 32), (128,), (256, 128), (256, 128, 54), (512, 256)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam', 'sgd']
    }
    # Initialize the Decision Tree classifier
    mlp = MLPClassifier(max_iter = 1000, random_state=42)

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"\nBest Hyperparameters: {best_params}")

    # Manual validation for additional insights
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    best_model.fit(X_train_split, y_train_split)

    confusion_matrix_evaluate(best_model, X_val_split, y_val_split) # Evaluate with confusion matrix
    y_val_pred = best_model.predict(X_val_split)
    val_accuracy = accuracy_score(y_val_split, y_val_pred)

    print(f"Validation Accuracy with Best Model: {val_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_val_split, y_val_pred))

    # Generating the submission file for this model
    print(f"\nGenerating submission file for Neural Network")
    best_model.fit(X_train, y_train)  # Retrain on the full training set
    y_test_pred = best_model.predict(X_test)  # Predict on the test set
    write_submission(y_test_pred, submission_path="submission_neural_network.csv")

    return best_model, best_params

def train_evaluate_dt(X_train, y_train, X_test):
    # Define hyperparameter grid for the Decision Tree
    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # Initialize the Decision Tree classifier
    decision_tree = DecisionTreeClassifier(random_state=42)

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(
        estimator=decision_tree,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"\nBest Hyperparameters: {best_params}")

    # Manual validation for additional insights
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    best_model.fit(X_train_split, y_train_split)

    confusion_matrix_evaluate(best_model, X_val_split, y_val_split) # Evaluate with confusion matrix
    y_val_pred = best_model.predict(X_val_split)
    val_accuracy = accuracy_score(y_val_split, y_val_pred)

    print(f"Validation Accuracy with Best Model: {val_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_val_split, y_val_pred))

    # Generating the submission file for this model
    print(f"\nGenerating submission file for Decision Tree")
    best_model.fit(X_train, y_train)  # Retrain on the full training set
    y_test_pred = best_model.predict(X_test)  # Predict on the test set
    write_submission(y_test_pred, submission_path="submission_decision_tree.csv")

    return best_model, best_params


def confusion_matrix_evaluate(model, X_val, y_val):
    # Predict on validation set
    y_val_pred = model.predict(X_val)

    # Compute the confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print("Confusion Matrix:")
    print(cm)

    # Optionally display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

