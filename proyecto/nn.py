import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import lib  # pybind11 bridge to C++

# ================== CONFIG ==================
torch.manual_seed(42)
input_dim = 11
num_classes = 3
hidden1 = 128
hidden2 = 64
batch_size = 50

# ============== Modelo ======================
class MulticlassClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden1=128, hidden2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.class_logits = nn.Linear(hidden2, num_classes)
        self.class_log_vars = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))  # capa 1
        x2 = F.relu(self.fc2(x1)) # capa 2
        logits = self.class_logits(x2)
        log_vars = self.class_log_vars(x2)
        return logits, log_vars, x1, x2

# ============== Conexión con server ================
lib.connect("127.0.0.1")
X_np, num_epochs = lib.receive_dataset()
X = torch.tensor(X_np, dtype=torch.float32)
y = X[:, -num_classes:]
X = X[:, :input_dim]

# ============== Dataset ======================
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============== Inicialización =====================
model = MulticlassClassifier(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============== Entrenamiento ===============
train_tracker, test_tracker, accuracy_tracker = [], [], []
y_true, y_pred = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        logits, log_vars, x1_out, x2_out = model(batch_x)

        # Capa 1 (entrada -> hidden1)
        layer1_matrix = x1_out.detach().numpy().tolist()
        lib.send_matrix(layer1_matrix)
        x1_avg = torch.tensor(lib.receive_average(), dtype=torch.float32)

        # Capa 2 (hidden1 -> hidden2)
        layer2_matrix = x2_out.detach().numpy().tolist()
        lib.send_matrix(layer2_matrix)
        x2_avg = torch.tensor(lib.receive_average(), dtype=torch.float32)

        # Capa 3 (hidden2 -> output logits)
        logits_out = logits.detach().numpy().tolist()
        lib.send_matrix(logits_out)
        final_avg = lib.receive_average()  # pero no se usa para nada

        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_tracker.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_tracker[-1]:.4f} | ", end="")

    # Evaluación
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits, log_vars, *_ = model(batch_x)
            loss = criterion(logits, batch_y)
            test_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == torch.argmax(batch_y, dim=1)).sum().item()
            total += batch_y.size(0)

            y_true.extend(torch.argmax(batch_y, dim=1))
            y_pred.extend(preds.tolist())

    acc = correct / total
    test_tracker.append(test_loss / len(test_loader))
    accuracy_tracker.append(acc)
    print(f"Test loss: {test_loss / len(test_loader):.4f} | Accuracy: {acc:.4f}")

# ============== Graficación =================
plt.plot(train_tracker, label='Training Loss')
plt.plot(test_tracker, label='Test Loss')
plt.plot(accuracy_tracker, label='Test Accuracy')
plt.legend()
plt.grid(True)
plt.title("Entrenamiento")
plt.show()

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=3))
