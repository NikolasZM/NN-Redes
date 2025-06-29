import socket
import pickle
import numpy as np
import struct

# Helper function to create the packet based on protocol
def create_packet(protocol_type, data):
    if protocol_type == 'e':  # Protocol for sending dataset chunk
        total_size = len(data) * 4 + 14  # 4 bytes per float + header size
        is_last = 0  # Initially not the last
        if len(data) <= 100:
            is_last = 1  # Last packet if data is 100 or less
        header = struct.pack("!I B B I H", total_size, 101, is_last, len(data), len(data) * 4)  # 'e' = 101 in ASCII
        print(f"Enviando Protocolo 1: {len(data)} datos, Last: {is_last}")
        return header + pickle.dumps(data)

    elif protocol_type == 'm':  # Protocol for sending matrix (weights from hidden layer)
        total_size = len(data) * len(data[0]) * 4 + 14  # 4 bytes per float + header size
        is_last = 1  # Assuming it's the last packet
        header = struct.pack("!I B B I H", total_size, 109, is_last, len(data) * len(data[0]), len(data[0]) * 4)  # 'm' = 109 in ASCII
        print(f"Enviando Protocolo 2: {len(data)} pesos, Last: {is_last}")
        return header + pickle.dumps(data)
    
    elif protocol_type == 'v':  # Protocol for small vector with 2 floats (final output)
        total_size = 2 * 4 + 14  # 2 floats, each 4 bytes
        is_last = 1  # Assuming it's the last packet
        header = struct.pack("!I B B I H", total_size, 118, is_last, 2, 8)  # 'v' = 118 in ASCII
        print(f"Enviando Protocolo 3: 2 resultados finales, Last: {is_last}")
        return header + pickle.dumps(data)

    return None

# Start the calculation server
def start_calculation_server(host='localhost', port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(3)  # Listen for up to 3 connections (one for each NN server)
    
    print("Calculation Server started. Waiting for NN servers to connect...")
    
    # Accept connections from 3 NN servers
    nn_connections = []
    for i in range(3):
        conn, addr = server_socket.accept()
        print(f"NN Server {i+1} connected from {addr}")
        nn_connections.append(conn)
    
    # Step 1: Send dataset chunks to NN servers (Protocol 1)
    dataset = np.random.rand(330).tolist()  # Example dataset with 330 values
    chunk_size = 110  # 110 for 3 NN servers
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i+chunk_size]
        protocol_type = 'e'  # Protocol 1 for chunks
        packet = create_packet(protocol_type, chunk)
        
        # Send to NN servers
        for conn in nn_connections:
            conn.send(packet)  # Send the data chunk

    # Step 2: Receive the updated weights back from NN servers (Protocol 2)
    data_chunks = []
    for conn in nn_connections:
        data_packet = conn.recv(1024 * 1024)  # Receive up to 1MB of data
        header = data_packet[:14]  # The first 14 bytes are the header
        data = pickle.loads(data_packet[14:])  # Deserialize the data
        data_chunks.append(data)
    
    # Calculate the mean weights
    mean_weights = np.mean(data_chunks, axis=0)
    print(f"Calculated mean weights: {mean_weights}")
    
    # Step 3: Send the final output back to NN servers (Protocol 2)
    for conn in nn_connections:
        packet = create_packet('m', mean_weights)  # Protocol 2 for hidden layer
        conn.send(packet)  # Send the mean weights

    # Step 4: Receive the final output (Protocol 3)
    results = []
    for conn in nn_connections:
        result_packet = conn.recv(1024 * 1024)  # Receive final output data
        result = pickle.loads(result_packet[14:])
        results.append(result)
    
    # Print final results from NN servers
    print("Final output from NN servers (Protocol 3):", results)

    # Close the connections
    for conn in nn_connections:
        conn.close()

# Start the server
start_calculation_server()
