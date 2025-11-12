import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load training and test data from text files
def load_data(train_file, label_file, test_file):
    X_train = np.loadtxt(train_file)
    y_train = np.loadtxt(label_file)
    X_test = np.loadtxt(test_file)

    # Replace sentinel values (1e99) with NaN for imputation
    sentinel = 1.00000000000000e+99
    X_train[X_train == sentinel] = np.nan
    X_test[X_test == sentinel] = np.nan

    return X_train, y_train, X_test


# Train an SVM pipeline on one dataset and save test predictions
def classify_dataset(train_file, label_file, test_file, output_file, val_frac=0.1):
    X_train, y_train, X_test = load_data(train_file, label_file, test_file)

    # Pipeline: impute missing → scale → SVM
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1, gamma='scale'))  # reduced C to prevent overfitting
    ])

    print("\n" + "=" * 60)
    print(f"Training model for {train_file}")
    print("-" * 60)

    # Validation split (only if dataset large enough)
    if 0.0 < val_frac < 1.0 and len(X_train) >= 2:
        n_classes = len(np.unique(y_train))
        val_size = int(np.floor(val_frac * len(X_train)))
        stratify = y_train if val_size >= n_classes else None

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=val_frac, stratify=stratify, random_state=42
        )

        pipeline.fit(X_tr, y_tr)
        train_acc = accuracy_score(y_tr, pipeline.predict(X_tr))
        val_acc = accuracy_score(y_val, pipeline.predict(X_val))

        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        # Refit on full training set for test predictions
        pipeline.fit(X_train, y_train)
    else:
        # Train directly on all data
        pipeline.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        print(f"Training accuracy: {train_acc:.4f}")

    # Predict test set
    predictions = pipeline.predict(X_test)

    # Save integer predictions
    np.savetxt(output_file, predictions.astype(int), fmt='%d')
    print(f"Saved predictions to {output_file}")
    print("=" * 60)


# Run classifier for all datasets
if __name__ == '__main__':
    datasets = [
        ('TrainData1.txt', 'TrainLabel1.txt', 'TestData1.txt', 'SVMTestResultS1.txt'),
        ('TrainData2.txt', 'TrainLabel2.txt', 'TestData2.txt', 'SVMTestResultS2.txt'),
        ('TrainData3.txt', 'TrainLabel3.txt', 'TestData3.txt', 'SVMTestResultS3.txt'),
        ('TrainData4.txt', 'TrainLabel4.txt', 'TestData4.txt', 'SVMTestResultS4.txt'),
    ]

    for train_file, label_file, test_file, output_file in datasets:
        classify_dataset(train_file, label_file, test_file, output_file)
