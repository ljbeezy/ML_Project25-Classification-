import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


def load_data(train_file, train_label_file, test_file):
    X_train = np.loadtxt(train_file)
    y_train = np.loadtxt(train_label_file)
    X_test = np.loadtxt(test_file)

    X_train[X_train == 1.00000000000000e+99] = np.nan
    X_test[X_test == 1.00000000000000e+99] = np.nan

    return X_train, y_train, X_test


def get_dataset_config(dataset_name):
    configs = {
        'Dataset1': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'poly'],
            'cv_folds': 5
        },
        'Dataset2': {
            'C': [1, 10, 100],
            'gamma': ['scale', 'auto', 0.01],
            'kernel': ['rbf', 'poly'],
            'cv_folds': 5
        },
        'Dataset3': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'poly'],
            'cv_folds': 5
        },
        'Dataset4': {
            'C': [1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'cv_folds': 3
        },
    }
    return configs.get(dataset_name, configs['Dataset1'])


def classify_dataset(train_file, train_label_file, test_file,
                     dataset_name='Dataset1'):

    X_train, y_train, X_test = load_data(train_file, train_label_file, test_file)

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    config = get_dataset_config(dataset_name)
    
    class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(y_train), 
                                         y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

    print(f"Training SVM model for {dataset_name}...")
    print(f"  Hyperparameter tuning with GridSearchCV (cv={config['cv_folds']} folds)...")
    
    param_grid = {
        'svm__C': config['C'],
        'svm__gamma': config['gamma'],
        'svm__kernel': config['kernel'],
    }

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('svm', SVC(class_weight='balanced'))
    ])

    cv = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best cross-val score: {grid_search.best_score_:.4f}")

    predictions = grid_search.predict(X_test)

    print("Predictions:")
    np.set_printoptions(threshold=np.inf)
    print(predictions.astype(int))
    print()


if __name__ == '__main__':
    classify_dataset('TrainData1.txt', 'TrainLabel1.txt', 'TestData1.txt', 'Dataset1')
    classify_dataset('TrainData2.txt', 'TrainLabel2.txt', 'TestData2.txt', 'Dataset2')
    classify_dataset('TrainData3.txt', 'TrainLabel3.txt', 'TestData3.txt', 'Dataset3')
    classify_dataset('TrainData4.txt', 'TrainLabel4.txt', 'TestData4.txt', 'Dataset4')
