import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import lib  # Puente a C++ via pybind11

# ================== CONFIGURACIÓN ==================
torch.manual_seed(42)
INPUT_DIM = 11
NUM_CLASSES = 3
HIDDEN1 = 128
HIDDEN2 = 64

# ================== MODELO NEURONAL ==================
class FederatedSplitNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ================== CONEXIÓN Y DATOS ==================
try:
    print("Conectando al servidor...")
    lib.connect("127.0.0.1", 9001) # Cambiado a puerto 9001 como en tu captura
    
    print("Recibiendo dataset del servidor...")
    X_np, num_epochs = lib.receive_dataset()
    num_epochs = int(num_epochs)
    print(f"Dataset recibido ({len(X_np)} muestras). Entrenando por {num_epochs} épocas.")

    y = torch.tensor(X_np[:, -NUM_CLASSES:], dtype=torch.float32)
    X = torch.tensor(X_np[:, :INPUT_DIM], dtype=torch.float32)

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
        
        # --- Forward Pass Manual y Federado ---
        # El forward pass se hace una vez por época, ya que no hay batches
        
        # CAPA 1
        z1 = model.fc1(X)
        lib.send_matrix(z1.detach().numpy())
        z1_avg_np = lib.receive_average()
        
        # CAPA 2
        a1_activated = F.relu(torch.tensor(z1_avg_np))
        z2 = model.fc2(a1_activated)
        lib.send_matrix(z2.detach().numpy())
        z2_avg_np = lib.receive_average()

        # CAPA 3
        a2_activated = F.relu(torch.tensor(z2_avg_np))
        z3 = model.fc3(a2_activated)
        lib.send_matrix(z3.detach().numpy())
        final_logits_np = lib.receive_average()

        # --- Backward Pass Corregido ---
        # Implementación manual de la regla de la cadena para la retropropagación
        optimizer.zero_grad()

        # 1. Crear tensores para los promedios del servidor, habilitando el cálculo de gradientes
        z1_avg = torch.tensor(z1_avg_np, requires_grad=True)
        z2_avg = torch.tensor(z2_avg_np, requires_grad=True)
        final_logits_avg = torch.tensor(final_logits_np, requires_grad=True)
        
        # 2. Calcular la pérdida y su gradiente con respecto a la salida final promediada
        loss = criterion(final_logits_avg, y)
        loss.backward() # Esto calcula final_logits_avg.grad

        # 3. Propagar el gradiente hacia atrás a través de la Capa 3
        #    Esto calcula los gradientes para los pesos de fc3 y para z2_avg
        z3_recomputed = model.fc3(F.relu(z2_avg))
        z3_recomputed.backward(final_logits_avg.grad)

        # 4. Propagar el gradiente hacia atrás a través de la Capa 2
        #    Esto calcula los gradientes para los pesos de fc2 y para z1_avg
        z2_recomputed = model.fc2(F.relu(z1_avg))
        z2_recomputed.backward(z2_avg.grad)

        # 5. Propagar el gradiente hacia atrás a través de la Capa 1
        #    Aquí usamos el 'z1' original que está conectado a los pesos de fc1
        z1.backward(z1_avg.grad)
        
        # 6. Actualizar todos los pesos con los gradientes acumulados
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

except Exception as e:
    print(f"Error durante el entrenamiento: {e}")

finally:
    print("\nEntrenamiento finalizado o interrumpido.")
    lib.disconnect()

