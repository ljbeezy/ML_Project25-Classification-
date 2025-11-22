import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


def get_class_weights(y):
    unique, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(unique) * counts)
    return {int(u): w for u, w in zip(unique, weights)}


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


def get_dataset_config(dataset_name):
    configs = {
        'Dataset1': {'epochs': 50, 'batch_size': 32, 'lr': 5e-4, 'dropout': 0.2, 'val_split': 0.15},
        'Dataset2': {'epochs': 40, 'batch_size': 32, 'lr': 1e-3, 'dropout': 0.2, 'val_split': 0.15},
        'Dataset3': {'epochs': 50, 'batch_size': 16, 'lr': 5e-4, 'dropout': 0.3, 'val_split': 0.2},
        'Dataset4': {'epochs': 100, 'batch_size': 8, 'lr': 1e-3, 'dropout': 0.4, 'val_split': 0.25},
    }
    return configs.get(dataset_name, configs['Dataset1'])


def classify_dataset(train_file, train_label_file, test_file,
                     dataset_name='Dataset1', cpu_only=False):

    X_train, y_train, X_test = load_data(train_file, train_label_file, test_file)

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    unique_labels = np.unique(y_train)
    if unique_labels.size == 0:
        raise ValueError('No training labels found')
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    y_train_idx = np.vectorize(label_to_idx.get)(y_train)

    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train_idx.astype(np.int64))
    X_test_t = torch.from_numpy(X_test.astype(np.float32))

    config = get_dataset_config(dataset_name)
    val_split = config['val_split']
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_t, y_train_t, test_size=val_split, random_state=42, stratify=y_train_t
    )

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    if cpu_only:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = len(unique_labels)
    input_dim = X_train.shape[1]

    model = RegularizedMLP(input_dim=input_dim, num_classes=num_classes, 
                          dropout_rate=config['dropout']).to(device)
    
    class_weights = get_class_weights(y_train_idx)
    weight_tensor = torch.tensor([class_weights[i] for i in range(num_classes)], 
                                dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5)

    print(f"Training PyTorch model for {dataset_name} on device={device}...")
    print(f"  Config: epochs={config['epochs']}, batch_size={config['batch_size']}, "
          f"lr={config['lr']}, dropout={config['dropout']}")
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    model.train()
    for epoch in range(1, config['epochs'] + 1):
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

        if epoch in (1, config['epochs']) or (config['epochs'] >= 5 and epoch % max(1, config['epochs'] // 5) == 0):
            print(f"Epoch {epoch}/{config['epochs']} - train_loss: {epoch_loss:.4f}, val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        model.train()

    model.eval()
    with torch.no_grad():
        X_test_device = X_test_t.to(device)
        outputs = model(X_test_device)
        preds_idx = outputs.argmax(dim=1).cpu().numpy()

    preds_labels = np.vectorize(idx_to_label.get)(preds_idx)
    print("Predictions:")
    np.set_printoptions(threshold=np.inf)
    print(preds_labels.astype(int))
    print()


if __name__ == '__main__':
    classify_dataset('TrainData1.txt', 'TrainLabel1.txt', 'TestData1.txt', 'Dataset1')
    classify_dataset('TrainData2.txt', 'TrainLabel2.txt', 'TestData2.txt', 'Dataset2')
    classify_dataset('TrainData3.txt', 'TrainLabel3.txt', 'TestData3.txt', 'Dataset3')
    classify_dataset('TrainData4.txt', 'TrainLabel4.txt', 'TestData4.txt', 'Dataset4')
