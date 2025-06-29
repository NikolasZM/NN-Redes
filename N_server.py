import socket
import pickle
import numpy as np

# Helper function to create the packet based on protocol
def create_packet(protocol_type, data):
    if protocol_type == 'e':  # Protocol for sending dataset chunk
        total_size = len(data) * 4 + 14  # 4 bytes per float + header size
        is_last = 0  # Initially not the last
        if len(data) <= 100:
            is_last = 1  # Last packet if data is 100 or less
        header = struct.pack("!I B B I H", total_size, 101, is_last, len(data), len(data) * 4)  # 'e' = 101 in ASCII
        print(f"NN Server: Enviando Protocolo 1: {len(data)} datos, Last: {is_last}")
        return header + pickle.dumps(data)

    elif protocol_type == 'm':  # Protocol for sending matrix (weights from hidden layer)
        total_size = len(data) * len(data[0]) * 4 + 14  # 4 bytes per float + header size
        is_last = 1  # Assuming it's the last packet
        header = struct.pack("!I B B I H", total_size, 109, is_last, len(data) * len(data[0]), len(data[0]) * 4)  # 'm' = 109 in ASCII
        print(f"NN Server: Enviando Protocolo 2: {len(data)} pesos, Last: {is_last}")
        return header + pickle.dumps(data)
    
    elif protocol_type == 'v':  # Protocol for small vector with 2 floats (final output)
        total_size = 2 * 4 + 14  # 2 floats, each 4 bytes
        is_last = 1  # Assuming it's the last packet
        header = struct.pack("!I B B I H", total_size, 118, is_last, 2, 8)  # 'v' = 118 in ASCII
        print(f"NN Server: Enviando Protocolo 3: 2 resultados finales, Last: {is_last}")
        return header + pickle.dumps(data)

    return None

def start_nn_server(host='localhost', port=5000, server_id=1):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"NN Server {server_id} connected to Calculation Server")

    # Example data (small vector with 2 floats)
    if server_id == 1:
        data = [0.5, 1.2]  # Example 2D vector
        protocol_type = 'v'  # Protocol 3 for output
    else:
        data = np.random.rand(110).tolist()  # Example larger data for a vector (110 floats)
        protocol_type = 'e'  # Protocol 1 for dataset chunk
    
    # Send packet based on protocol
    packet = create_packet(protocol_type, data)
    client_socket.send(packet)  # Send the packet to calculation server
    print(f"NN Server {server_id} sent data to Calculation Server")

    # Receive the updated weights back from the calculation server (Protocol 2)
    mean_weights = client_socket.recv(1024 * 1024)  # Receive up to 1MB data
    mean_weights = pickle.loads(mean_weights)  # Deserialize the data
    print(f"NN Server {server_id} received updated weights: {mean_weights}")

    # Close the connection
    client_socket.close()

# Start the NN server (example for NN Server 1)
start_nn_server(server_id=1)
