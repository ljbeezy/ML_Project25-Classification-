import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ...existing code...


# Read feature, label, and test arrays from text files
def load_data(train_file, train_label_file, test_file):

    X_train = np.loadtxt(train_file)
    y_train = np.loadtxt(train_label_file)
    X_test = np.loadtxt(test_file)

    # Swap out the placeholder value with NaN for later cleaning
    X_train[X_train == 1.00000000000000e+99] = np.nan
    X_test[X_test == 1.00000000000000e+99] = np.nan

    return X_train, y_train, X_test


class SimpleMLP(nn.Module):

    def __init__(self, input_dim, hidden1=64, hidden2=32, num_classes=2, dropout=0.2):
        super().__init__()
        # Keep network simple: two hidden layers with small dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        # Xavier init for better starting point
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Push data through the network and return class scores
        return self.net(x)


# Train the neural network on one dataset and store its predictions
def classify_dataset(train_file, train_label_file, test_file, output_file,
                     epochs=20, batch_size=32, lr=1e-3,
                     hidden1=64, hidden2=32, cpu_only=False, dropout=0.2, weight_decay=1e-5):

    X_train, y_train, X_test = load_data(train_file, train_label_file, test_file)

    # Fill missing values and scale each feature to zero mean, unit variance
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Map label values to integer class IDs and remember how to reverse them
    unique_labels = np.unique(y_train)
    if unique_labels.size == 0:
        raise ValueError('No training labels found')
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    y_train_idx = np.vectorize(label_to_idx.get)(y_train)

    num_classes = len(unique_labels)
    input_dim = X_train.shape[1]

    # Convert NumPy arrays to PyTorch tensors for training
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train_idx.astype(np.int64))
    X_test_t = torch.from_numpy(X_test.astype(np.float32))

    if cpu_only:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Wrap tensors in a DataLoader to feed batches to the model
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Set up the model, loss function, and optimizer
    model = SimpleMLP(input_dim=input_dim, hidden1=hidden1,
                      hidden2=hidden2, num_classes=num_classes, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Title / header for clarity between datasets
    print("\n" + "=" * 72)
    print(f"Training for dataset:\n  train: {train_file}\n  labels: {train_label_file}\n  test: {test_file}")
    print("=" * 72)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_correct = 0
        running_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            running_n += xb.size(0)
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / running_n if running_n > 0 else 0.0
        epoch_acc = running_correct / running_n if running_n > 0 else 0.0

        # Print every epoch and add an empty line to separate epochs visually
        print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - train acc: {epoch_acc:.4f}")
        print()  # blank line for separation

    # Footer to separate completion of training from evaluation/saving
    print("-" * 72)
    print(f"Finished training for {train_file}. Evaluating on test set and saving predictions...")
    print()

    model.eval()
    with torch.no_grad():
        # Run the trained model on the test data to get class scores
        X_test_device = X_test_t.to(device)
        outputs = model(X_test_device)
        preds_idx = outputs.argmax(dim=1).cpu().numpy()

    # Turn predicted class IDs back into the original label values
    preds_labels = np.vectorize(idx_to_label.get)(preds_idx)
    np.savetxt(output_file, preds_labels.astype(int), fmt='%d')
    print(f"Saved predictions to {output_file}\n")


if __name__ == '__main__':
    classify_dataset('TrainData1.txt', 'TrainLabel1.txt', 'TestData1.txt', 'NNTestResult1.txt')
    classify_dataset('TrainData2.txt', 'TrainLabel2.txt', 'TestData2.txt', 'NNTestResult2.txt')
    classify_dataset('TrainData3.txt', 'TrainLabel3.txt', 'TestData3.txt', 'NNTestResult3.txt')
    classify_dataset('TrainData4.txt', 'TrainLabel4.txt', 'TestData4.txt', 'NNTestResult4.txt')