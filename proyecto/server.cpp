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

using Sample = std::vector<float>;
using Dataset = std::vector<Sample>;

std::mutex mtx;
bool timeout_started = false;
std::chrono::steady_clock::time_point timeout_start;
std::map<int, std::vector<std::vector<float>>> client_matrices;
std::vector<int> client_sockets;

// ================= CSV LOAD =================

Dataset read_csv(const std::string& filename) {
    Dataset data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) throw std::runtime_error("No se pudo abrir el CSV");

    std::getline(file, line); // skip header

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

    for (auto& row : data) {
        int class_idx = std::distance(row.end() - CLASS_DIM, std::max_element(row.end() - CLASS_DIM, row.end()));
        class_buckets[class_idx].push_back(row);
    }

    for (int c = 0; c < CLASS_DIM; ++c) {
        int per_client = class_buckets[c].size() / num_clients;
        int index = 0;
        for (int i = 0; i < num_clients; ++i) {
            for (int j = 0; j < per_client && index < class_buckets[c].size(); ++j, ++index) {
                splits[i].push_back(class_buckets[c][index]);
            }
        }
        for (; index < class_buckets[c].size(); ++index) {
            splits[index % num_clients].push_back(class_buckets[c][index]);
        }
    }

    return splits;
}

// ================= ENVÍO DE DATOS [e] =================

void send_column_protocol(int client_sock, const std::vector<std::vector<float>>& matrix) {
    size_t num_cols = matrix[0].size();
    size_t num_rows = matrix.size();

    for (size_t col = 0; col < num_cols; ++col) {
        std::vector<float> column;
        for (size_t row = 0; row < num_rows; ++row) {
            column.push_back(matrix[row][col]);
        }

        std::vector<char> buffer;
        uint64_t total_size = 1 + 4 + 1 + 5 + column.size() * (2 + 4);
        buffer.resize(5 + total_size);

        memcpy(&buffer[0], &total_size, 5);
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

        write(client_sock, buffer.data(), buffer.size());
    }
}

// =================== RECEPCIÓN DE MATRICES [M] ===================

std::vector<float> receive_column(int sock, bool& is_last_col) {
    char header[12];
    int total_read = 0;
    while (total_read < 12) {
        int r = read(sock, header + total_read, 12 - total_read);
        if (r <= 0) throw std::runtime_error("Fallo al leer header");
        total_read += r;
    }

    uint64_t size_total;
    memcpy(&size_total, &header[0], 5);
    char type = header[5];
    if (type != 'M') throw std::runtime_error("Tipo incorrecto, se esperaba 'M'");

    is_last_col = header[6];
    uint64_t total_numbers;
    memcpy(&total_numbers, &header[7], 5);

    std::vector<float> column;
    int payload_size = size_total - 12;
    std::vector<char> payload(payload_size);

    total_read = 0;
    while (total_read < payload_size) {
        int r = read(sock, payload.data() + total_read, payload_size - total_read);
        if (r <= 0) throw std::runtime_error("Fallo al leer payload");
        total_read += r;
    }

    size_t offset = 0;
    for (size_t i = 0; i < total_numbers; ++i) {
        uint16_t size_data;
        memcpy(&size_data, &payload[offset], 2);
        float value;
        memcpy(&value, &payload[offset + 2], 4);
        column.push_back(value);
        offset += 6;
    }

    return column;
}

void send_average_matrix(const std::vector<std::vector<float>>& avg_matrix) {
    for (auto sock : client_sockets) {
        for (size_t col = 0; col < avg_matrix[0].size(); ++col) {
            std::vector<float> column;
            for (size_t row = 0; row < avg_matrix.size(); ++row) {
                column.push_back(avg_matrix[row][col]);
            }

            std::vector<char> buffer;
            uint64_t total_size = 1 + 1 + 5 + column.size() * (2 + 4);
            buffer.resize(5 + total_size);

            memcpy(&buffer[0], &total_size, 5);
            buffer[5] = 'm';
            buffer[6] = (col == avg_matrix[0].size() - 1 ? 1 : 0);
            uint64_t total_numbers = column.size();
            memcpy(&buffer[7], &total_numbers, 5);

            size_t offset = 12;
            for (float val : column) {
                uint16_t size_data = 4;
                memcpy(&buffer[offset], &size_data, 2);
                memcpy(&buffer[offset + 2], &val, 4);
                offset += 6;
            }

            write(sock, buffer.data(), buffer.size());
        }
    }
}

std::vector<std::vector<float>> compute_average_matrix() {
    std::map<size_t, std::vector<float>> col_accumulators;
    int num_clients = client_matrices.size();

    for (auto& [client_id, columns] : client_matrices) {
        for (size_t col = 0; col < columns.size(); ++col) {
            if (col_accumulators.count(col) == 0)
                col_accumulators[col] = columns[col];
            else {
                for (size_t i = 0; i < columns[col].size(); ++i)
                    col_accumulators[col][i] += columns[col][i];
            }
        }
    }

    std::vector<std::vector<float>> avg_matrix;
    for (auto& [col, sum_vec] : col_accumulators) {
        for (float& val : sum_vec) val /= num_clients;
        avg_matrix.push_back(sum_vec);
    }

    return avg_matrix;
}

void receiver_thread(int client_id, int client_sock) {
    client_sockets.push_back(client_sock);
    try {
        while (true) {
            bool is_last = false;
            auto column = receive_column(client_sock, is_last);

            {
                std::lock_guard<std::mutex> lock(mtx);
                client_matrices[client_id].push_back(column);

                if (is_last && !timeout_started) {
                    timeout_started = true;
                    timeout_start = std::chrono::steady_clock::now();
                    std::cout << "[INFO] Última columna recibida de cliente " << client_id << ". Esperando 5s...\n";
                }
            }

            if (timeout_started) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - timeout_start).count() >= 5) break;
            }
        }
    } catch (...) {
        std::cerr << "[ERROR] Cliente " << client_id << " desconectado inesperadamente\n";
    }
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

    std::vector<std::thread> sender_threads;
    for (int i = 0; i < num_clients; ++i) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(server_fd, (sockaddr*)&client_addr, &client_len);

        if (client_sock < 0) {
            std::cerr << "Error al aceptar cliente\n";
            continue;
        }

        sender_threads.emplace_back(send_column_protocol, client_sock, split_data[i]);
    }

    for (auto& t : sender_threads) t.join();

    std::cout << "[INFO] Todos los datasets enviados.\n";

    std::vector<std::thread> receiver_threads;
    for (int i = 0; i < num_clients; ++i) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(server_fd, (sockaddr*)&client_addr, &client_len);

        if (client_sock < 0) {
            std::cerr << "Error al aceptar cliente para recepción\n";
            continue;
        }

        receiver_threads.emplace_back(receiver_thread, i, client_sock);
    }

    for (auto& t : receiver_threads) t.join();

    std::cout << "[INFO] Calculando promedio y reenviando a los clientes...\n";
    auto average = compute_average_matrix();
    send_average_matrix(average);

    std::cout << "[INFO] Finalizado.\n";
    close(server_fd);
    return 0;
}
