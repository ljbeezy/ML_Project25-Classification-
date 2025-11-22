import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def load_data(train_file, train_label_file, test_file):
    X_train = np.loadtxt(train_file)
    y_train = np.loadtxt(train_label_file)
    X_test = np.loadtxt(test_file)

    X_train[X_train == 1.00000000000000e+99] = np.nan
    X_test[X_test == 1.00000000000000e+99] = np.nan

    return X_train, y_train, X_test


class RegularizedMLP(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, hidden3=32,
                 num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden3, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_neural_network(X_train, y_train, X_val, y_val, X_test, epochs=50,
                        batch_size=32, lr=5e-4, dropout=0.2, device='cpu'):
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))
    X_test_t = torch.from_numpy(X_test.astype(np.float32))

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    model = RegularizedMLP(input_dim=input_dim, num_classes=num_classes,
                          dropout_rate=dropout).to(device)

    class_weights_arr = compute_class_weight('balanced',
                                             classes=np.unique(y_train),
                                             y=y_train)
    weight_tensor = torch.tensor(class_weights_arr, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        model.train()

    model.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val.astype(np.float32))
        X_val_device = X_val_t.to(device)
        val_outputs = model(X_val_device)
        val_preds_nn = val_outputs.argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_val, val_preds_nn, average='weighted')
        
        X_test_device = X_test_t.to(device)
        test_outputs = model(X_test_device)
        preds = test_outputs.argmax(dim=1).cpu().numpy()

    return preds, val_f1


def train_svm(X_train, y_train, X_val, y_val, X_test):
    svm = SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced')
    svm.fit(X_train, y_train)

    val_preds = svm.predict(X_val)
    val_score = f1_score(y_val, val_preds, average='weighted')

    test_preds = svm.predict(X_test)

    return test_preds, val_score


def train_random_forest(X_train, y_train, X_val, y_val, X_test):
    rf = RandomForestClassifier(n_estimators=150, max_depth=15,
                                min_samples_split=5, min_samples_leaf=2,
                                random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    val_preds = rf.predict(X_val)
    val_score = f1_score(y_val, val_preds, average='weighted')

    test_preds = rf.predict(X_test)

    return test_preds, val_score


def adaptive_classify_dataset(train_file, train_label_file, test_file,
                             output_file_prefix, dataset_name='Dataset1'):

    X_train, y_train, X_test = load_data(train_file, train_label_file, test_file)

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    val_split = 0.2
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=42, stratify=y_train
    )

    unique_labels = np.unique(y_train)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    y_tr_idx = np.vectorize(label_to_idx.get)(y_tr)
    y_val_idx = np.vectorize(label_to_idx.get)(y_val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"ADAPTIVE MODEL SELECTION for {dataset_name}")
    print(f"{'='*60}")

    print(f"Training Neural Network...")
    nn_preds, nn_val_score = train_neural_network(X_tr, y_tr_idx, X_val, y_val_idx,
                                                  X_test, epochs=50, batch_size=16,
                                                  lr=5e-4, dropout=0.2, device=device)
    nn_val_score_float = float(nn_val_score)
    print(f"  NN Validation Score: {nn_val_score_float:.4f}")

    print(f"Training SVM...")
    svm_preds, svm_val_score = train_svm(X_tr, y_tr, X_val, y_val, X_test)
    print(f"  SVM Validation Score: {svm_val_score:.4f}")

    print(f"Training Random Forest...")
    rf_preds, rf_val_score = train_random_forest(X_tr, y_tr, X_val, y_val, X_test)
    print(f"  RF Validation Score: {rf_val_score:.4f}")

    scores = {
        'NN': nn_val_score_float,
        'SVM': svm_val_score,
        'RF': rf_val_score
    }

    best_model = max(scores, key=scores.get)
    print(f"\n  BEST MODEL: {best_model} (score: {scores[best_model]:.4f})")

    if best_model == 'NN':
        final_preds = nn_preds
        final_preds = np.vectorize(idx_to_label.get)(final_preds)
    elif best_model == 'SVM':
        final_preds = svm_preds
    else:
        final_preds = rf_preds

    np.savetxt(f'{output_file_prefix}.txt', final_preds.astype(int), fmt='%d')
    print(f"Saved predictions to {output_file_prefix}.txt\n")


if __name__ == '__main__':
    adaptive_classify_dataset('TrainData1.txt', 'TrainLabel1.txt', 'TestData1.txt',
                             'ClassificationTestResult1', 'Dataset1')
    adaptive_classify_dataset('TrainData2.txt', 'TrainLabel2.txt', 'TestData2.txt',
                             'ClassificationTestResult2', 'Dataset2')
    adaptive_classify_dataset('TrainData3.txt', 'TrainLabel3.txt', 'TestData3.txt',
                             'ClassificationTestResult3', 'Dataset3')
    adaptive_classify_dataset('TrainData4.txt', 'TrainLabel4.txt', 'TestData4.txt',
                             'ClassificationTestResult4', 'Dataset4')
