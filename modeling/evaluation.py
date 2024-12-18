from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier

def train_evaluate(X_train, y_train):
    # Splitting training daat for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # The classifiers to be tested
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }

    for name, model in classifiers.items():
        print(f"\nTraining and Evaluating: {name}")
        model.fit(X_train_split, y_train_split)
        y_val_pred = model.predict(X_val_split)
        
        # Report accuracy and detailed classification report
        accuracy = accuracy_score(y_val_split, y_val_pred)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_val_split, y_val_pred))

def train_evaluate1(X_train, y_train):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    params = {
        'n_estimators': [50, 75 ,100, 150 ,200],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]
    }

    adaboost = AdaBoostClassifier(random_state=0, algorithm="SAMME")

    grid_search = GridSearchCV(estimator=adaboost, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_split, y_train_split)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"\nBest Hyperparameters: {best_params}")

    y_val_pred = best_model.predict(X_val_split)
    accuracy = accuracy_score(y_val_split, y_val_pred)

    print(f"Validation Accuracy with Best Model: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_val_split, y_val_pred))

    return best_model, best_params
