import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'cv_folds': 5
        },
        'Dataset2': {
            'n_estimators': [150, 200],
            'max_depth': [15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'cv_folds': 5
        },
        'Dataset3': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'cv_folds': 5
        },
        'Dataset4': {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 8, 10, 12],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
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

    print(f"Training Random Forest model for {dataset_name}...")
    print(f"  Hyperparameter tuning with GridSearchCV (cv={config['cv_folds']} folds)...")

    param_grid = {
        'rf__n_estimators': config['n_estimators'],
        'rf__max_depth': config['max_depth'],
        'rf__min_samples_split': config['min_samples_split'],
        'rf__min_samples_leaf': config['min_samples_leaf'],
    }

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
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
