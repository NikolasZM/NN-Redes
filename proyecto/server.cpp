#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <map>
#include <arpa/inet.h>
#include <algorithm>
#include <chrono>

constexpr int PORT = 9000;
constexpr int INPUT_DIM = 11;
constexpr int CLASS_DIM = 3;
constexpr int TOTAL_DIM = INPUT_DIM + CLASS_DIM;

int num_epochs = 10;

// ================= CSV LOAD =================

using Sample = std::vector<float>;
using Dataset = std::vector<Sample>;

Dataset read_csv(const std::string& filename) {
    Dataset data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) throw std::runtime_error("No se pudo abrir el CSV");

    // Saltar encabezado
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        Sample row;

        while (std::getline(ss, token, ',')) {
            row.push_back(std::stof(token));
        }

        if (row.size() == TOTAL_DIM) {
            data.push_back(row);
        }
    }

    return data;
}

// ================= BALANCEO =================

std::vector<Dataset> split_balanced(Dataset& data, int num_clients) {
    std::vector<Dataset> splits(num_clients);
    std::vector<Dataset> class_buckets(CLASS_DIM);

    // Separar por clase
    for (auto& row : data) {
        int class_idx = std::distance(row.end() - CLASS_DIM, std::max_element(row.end() - CLASS_DIM, row.end()));
        class_buckets[class_idx].push_back(row);
    }

    // Distribuir equitativamente por clase
    for (int c = 0; c < CLASS_DIM; ++c) {
        int per_client = class_buckets[c].size() / num_clients;
        int index = 0;

        for (int i = 0; i < num_clients; ++i) {
            for (int j = 0; j < per_client && index < class_buckets[c].size(); ++j, ++index) {
                splits[i].push_back(class_buckets[c][index]);
            }
        }

        // Lo que sobra se reparte
        for (; index < class_buckets[c].size(); ++index) {
            splits[index % num_clients].push_back(class_buckets[c][index]);
        }
    }

    return splits;
}

// ================= ENVÍO DE DATOS =================

void send_column_protocol(int client_sock, const std::vector<std::vector<float>>& matrix) {
    size_t num_cols = matrix[0].size();
    size_t num_rows = matrix.size();

    for (size_t col = 0; col < num_cols; ++col) {
        std::vector<float> column;
        for (size_t row = 0; row < num_rows; ++row) {
            column.push_back(matrix[row][col]);
        }

        // Crear paquete
        std::vector<char> buffer;

        uint64_t total_size = 1 + 4 + 1 + 5 + column.size() * (2 + 4); // sin header
        buffer.resize(5 + total_size); // 5 = Size_total

        uint64_t size_total = buffer.size();
        memcpy(&buffer[0], &size_total, 5);

        buffer[5] = 'e';
        memcpy(&buffer[6], &num_epochs, 4);
        buffer[10] = (col == num_cols - 1 ? 1 : 0);

        uint64_t total_numbers = column.size();
        memcpy(&buffer[11], &total_numbers, 5);

        size_t offset = 16;
        for (float val : column) {
            uint16_t size_data = 4;
            memcpy(&buffer[offset], &size_data, 2);
            memcpy(&buffer[offset + 2], &val, 4);
            offset += 6;
        }

        send(client_sock, buffer.data(), buffer.size(), 0);
    }
}

// ================= CLIENT HANDLER =================

void handle_client(int client_sock, const Dataset& client_data) {
    std::cout << "[INFO] Cliente conectado, enviando dataset...\n";

    // Reorganizar datos por columnas
    std::vector<std::vector<float>> matrix(client_data.size(), std::vector<float>(TOTAL_DIM));
    for (size_t i = 0; i < client_data.size(); ++i) {
        matrix[i] = client_data[i];
    }

    send_column_protocol(client_sock, matrix);
    close(client_sock);
}

// ================= MAIN =================

int main() {
    Dataset data = read_csv("Dataset_of_Diabetes.csv");

    int num_clients;
    std::cout << "Ingrese el número de clientes a conectar: ";
    std::cin >> num_clients;

    auto split_data = split_balanced(data, num_clients);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) throw std::runtime_error("Error al crear socket");

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_fd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0)
        throw std::runtime_error("Error al hacer bind");

    listen(server_fd, num_clients);
    std::cout << "[INFO] Servidor esperando conexiones en el puerto " << PORT << "...\n";

    std::vector<std::thread> client_threads;

    for (int i = 0; i < num_clients; ++i) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(server_fd, (sockaddr*)&client_addr, &client_len);

        if (client_sock < 0) {
            std::cerr << "Error al aceptar cliente\n";
            continue;
        }

        client_threads.emplace_back(handle_client, client_sock, split_data[i]);
    }

    for (auto& t : client_threads) t.join();

    std::cout << "[INFO] Todos los datasets enviados. Servidor finalizado (fase 1).\n";
    return 0;
}

