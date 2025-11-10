import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Read the training data, labels, and test data from text files
def load_data(train_file, train_label_file, test_file):

    X_train = np.loadtxt(train_file)
    y_train = np.loadtxt(train_label_file)
    X_test = np.loadtxt(test_file)

    # Replace sentinel values with NaN so they can be imputed later
    X_train[X_train == 1.00000000000000e+99] = np.nan
    X_test[X_test == 1.00000000000000e+99] = np.nan

    return X_train, y_train, X_test


# Fit an SVM pipeline on one dataset and save its test predictions
def classify_dataset(train_file, train_label_file, test_file, output_file):

    X_train, y_train, X_test = load_data(train_file, train_label_file, test_file)

    # Build a pipeline that fills missing data, scales features, and trains an SVM
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale'))
    ])

    print(f"Training model for {train_file}...")
    pipeline.fit(X_train, y_train)

    # Use the trained pipeline to make predictions for the test data
    predictions = pipeline.predict(X_test)

    # Write integer predictions to the output file
    np.savetxt(output_file, predictions.astype(int), fmt='%d')
    print(f"Saved predictions to {output_file}\n")


classify_dataset('TrainData1.txt', 'TrainLabel1.txt', 'TestData1.txt', 'ClassificationTestResultS1.txt')
classify_dataset('TrainData2.txt', 'TrainLabel2.txt', 'TestData2.txt', 'ClassificationTestResultS2.txt')
classify_dataset('TrainData3.txt', 'TrainLabel3.txt', 'TestData3.txt', 'ClassificationTestResultS3.txt')
classify_dataset('TrainData4.txt', 'TrainLabel4.txt', 'TestData4.txt', 'ClassificationTestResultS4.txt')
