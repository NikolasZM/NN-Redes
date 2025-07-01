import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import lib  # Puente a C++ via pybind11

# ================== CONFIGURACIÓN ==================
torch.manual_seed(42)
INPUT_DIM = 11
NUM_CLASSES = 3
HIDDEN1 = 128
HIDDEN2 = 64
BATCH_SIZE = 32

# ================== MODELO NEURONAL ==================
class FederatedSplitNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, NUM_CLASSES)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ================== CONEXIÓN Y DATOS ==================
try:
    print("Conectando al servidor...")
    lib.connect("127.0.0.1", 45015)
    
    print("Recibiendo dataset del servidor...")
    X_np, num_epochs = lib.receive_dataset()
    num_epochs = int(num_epochs)
    print(f"Dataset recibido. Entrenando por {num_epochs} épocas.")

    X = torch.tensor(X_np, dtype=torch.float32)
    y = X[:, -NUM_CLASSES:]
    X = X[:, :INPUT_DIM]

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

except Exception as e:
    print(f"Error durante la configuración: {e}")
    lib.disconnect()
    exit()

# ================== INICIALIZACIÓN ==================
model = FederatedSplitNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================== ENTRENAMIENTO FEDERADO ==================
print("\nIniciando entrenamiento federado...")
try:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()

            # --- Forward Pass ---
            # CAPA 1
            z1 = model.fc1(batch_x)
            print(f"Enviando capa 1 (tamaño: {z1.shape})")
            lib.send_matrix(z1.detach().numpy())
            a1_avg_np = lib.receive_average()
            print(f"Recibido promedio capa 1 (tamaño: {a1_avg_np.shape})")
            
            # CAPA 2
            a1 = torch.tensor(a1_avg_np, dtype=torch.float32)
            a1_activated = F.relu(a1)
            z2 = model.fc2(a1_activated)
            print(f"Enviando capa 2 (tamaño: {z2.shape})")
            lib.send_matrix(z2.detach().numpy())
            a2_avg_np = lib.receive_average()
            print(f"Recibido promedio capa 2 (tamaño: {a2_avg_np.shape})")

            # CAPA 3
            a2 = torch.tensor(a2_avg_np, dtype=torch.float32)
            a2_activated = F.relu(a2)
            z3 = model.fc3(a2_activated)
            print(f"Enviando capa 3 (tamaño: {z3.shape})")
            lib.send_matrix(z3.detach().numpy())
            final_logits_np = lib.receive_average()
            print(f"Recibido promedio capa final (tamaño: {final_logits_np.shape})")


            # --- Backward Pass Corregido y Simplificado ---
            
            # 1. Reconstruir el grafo desde el último punto promediado (a2)
            #    para poder calcular la pérdida.
            final_logits = torch.tensor(final_logits_np, dtype=torch.float32)
            loss = criterion(final_logits, batch_y)

            # 2. Reconstruir el grafo completo para la retropropagación
            #    Esto es necesario porque la conversión a NumPy rompe el grafo de PyTorch.
            a1_graph = torch.tensor(a1_avg_np, requires_grad=True)
            a2_graph = F.relu(model.fc2(F.relu(a1_graph)))
            final_logits_graph = model.fc3(a2_graph)
            
            # 3. Calcular gradientes para fc3 y fc2 con respecto a la pérdida
            surrogate_loss = criterion(final_logits_graph, batch_y)
            surrogate_loss.backward()
            
            # 4. Propagar manualmente el gradiente de a1 hacia atrás a través de fc1
            z1.backward(a1_graph.grad)
            
            # 5. Actualizar todos los pesos
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\nEntrenamiento finalizado o interrumpido.")
    lib.disconnect()
