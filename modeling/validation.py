from sklearn.model_selection import cross_val_score
def cross_validate_model(model, X, y, cv=5):
    """
    Performs cross-validation for a given model.
    """
    print(f"Performing {cv}-Fold Cross-Validation...")
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross-Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()