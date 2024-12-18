from data_processing import loader,preprocessing
from feature_engineering import extractor
from modeling import evaluation

if __name__ == '__main__':
    X_t, y_t, X_te = loader.load_data(data_path='Data/')

    X_train, X_test = preprocessing.normalize_data(X_t, X_te)
    X_train = extractor.extract_basic_features1(X_train)
    X_test = extractor.extract_basic_features1(X_test)
    
    evaluation.train_evaluate1(X_train, y_t)

    X_train, X_test = preprocessing.normalize_data(X_t, X_te)
    X_train = extractor.extract_basic_features(X_train)
    X_test = extractor.extract_basic_features(X_test)

    evaluation.train_evaluate1(X_train, y_t)