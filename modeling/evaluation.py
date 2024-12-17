from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

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