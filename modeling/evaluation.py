from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils.helpers import write_submission
from modeling.validation import cross_validate_model

def train_evaluate(X_train, y_train, X_test):
    # The classifiers to be tested
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        # Add more classifiers here if needed
    }

    for name, model in classifiers.items():
        print(f"\nTraining and Evaluating: {name}")

        # Perform cross-validation
        cv_accuracy = cross_validate_model(model, X_train, y_train, cv=5)
        print(f"Cross-Validated Accuracy: {cv_accuracy:.4f}")

        # Manual validation for additional insights
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        model.fit(X_train_split, y_train_split)
        y_val_pred = model.predict(X_val_split)
        val_accuracy = accuracy_score(y_val_split, y_val_pred)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_val_split, y_val_pred))

        # Generating the submission file for this model
        print(f"\nGenerating submission file for: {name}")
        model.fit(X_train, y_train)  # Retrain on the full training set
        y_test_pred = model.predict(X_test)  # Predict on the test set
        write_submission(y_test_pred, submission_path=f"submission_{name.replace(' ', '_').lower()}.csv")
