from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def decision_tree(X, y):
    """
    Performs hyperparameter tuning on a Decision Tree Classifier using GridSearchCV.
    Args:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target labels.
    Returns:
        best_model (DecisionTreeClassifier): Trained model with the best parameters.
    """
    # Define the hyperparameter grid
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],  # Control tree depth
        'min_samples_split': [2, 5, 10, 20],  # Minimum samples to split
        'min_samples_leaf': [1, 2, 5, 10],    # Minimum samples in a leaf
        'criterion': ['gini', 'entropy']      # Splitting strategy
    }
    
    # Initialize a Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    
    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,                   # 5-fold cross-validation
        scoring='accuracy',     # Evaluation metric
        verbose=1,              # Show progress
        n_jobs=-1               # Use all available cores
    )
    
    # Fit the model
    grid_search.fit(X, y)
    
    # Print the best parameters and score
    print("Best Hyperparameters:")
    print(grid_search.best_params_)
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    
    # Return the best model
    return grid_search.best_estimator_
