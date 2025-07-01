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
#include <condition_variable>
#include <iomanip>

using namespace std;

// ================== CONFIGURACIÃ“N ==================
constexpr int PORT = 45018;
constexpr int INPUT_DIM = 11;
constexpr int CLASS_DIM = 3;
constexpr int TOTAL_DIM = INPUT_DIM + CLASS_DIM;
constexpr int NUM_LAYERS_FEDERATED = 3; // 2 ocultas + 1 salida
int NUM_EPOCHS = 5;
int NUM_CLIENTS = 0;

using Sample = std::vector<float>;
using Dataset = std::vector<Sample>;
using Matrix = std::vector<std::vector<float>>;

// ================== GLOBALES Y SINCRONIZACIÃ“N ==================
std::mutex mtx;
std::condition_variable cv_clients_sent;
std::condition_variable cv_main_sent_avg;
int clients_ready_count = 0;
bool average_is_ready = false;

std::vector<int> all_client_sockets;
std::map<int, Matrix> client_matrices;

// ================== UTILIDADES ==================
std::string encode_float_string(float val) {
    std::ostringstream oss;
    oss << val;
    std::string s = oss.str();
    std::ostringstream result;
    result << std::setw(2) << std::setfill('0') << s.size() << s;
    return result.str();
}

float decode_float_string(const std::vector<char>& buffer, size_t& offset) {
    if (offset + 2 > buffer.size())
        throw std::runtime_error("decode_float_string: fuera de rango al leer tamaÃ±o");

    std::string len_str(buffer.begin() + offset, buffer.begin() + offset + 2);
    offset += 2;

    int len = std::stoi(len_str);
    if (offset + len > buffer.size())
        throw std::runtime_error("decode_float_string: fuera de rango al leer string float");

    std::string float_str(buffer.begin() + offset, buffer.begin() + offset + len);
    offset += len;

    try {
        return std::stof(float_str);
    } catch (...) {
        std::cerr << "[ERROR] ConversiÃ³n a float fallida: [" << float_str << "]\n";
        throw;
    }
}

// ================== LECTURA CSV ==================
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

std::vector<Dataset> split_balanced(Dataset& data, int num_clients) {
    std::vector<Dataset> splits(num_clients);
    std::map<int, Dataset> class_buckets;
    for (const auto& row : data) {
        int class_idx = -1;
        float max_val = -1.0f;
        for (int i = 0; i < CLASS_DIM; ++i) {
            if (row[INPUT_DIM + i] > max_val) {
                max_val = row[INPUT_DIM + i];
                class_idx = i;
            }
        }
        if (class_idx != -1) {
            class_buckets[class_idx].push_back(row);
        }
    }
    for(auto const& pair : class_buckets) {
        for(size_t i = 0; i < pair.second.size(); ++i) {
            splits[i % num_clients].push_back(pair.second[i]);
        }
    }
    return splits;
}

// ================== COMUNICACIÃ“N ==================
void safe_write(int sock, const char* buffer, size_t count) {
    ssize_t sent = 0;
    while (sent < count) {
        ssize_t res = write(sock, buffer + sent, count - sent);
        if (res <= 0) throw std::runtime_error("Fallo al escribir en socket");
        sent += res;
    }
}

void safe_read(int sock, char* buffer, size_t count) {
    ssize_t received = 0;
    while (received < count) {
        ssize_t res = read(sock, buffer + received, count - received);
        if (res <= 0) throw std::runtime_error("Fallo al leer de socket");
        received += res;
    }
}

void send_matrix_by_columns(int sock, char protocol_type, const Matrix& matrix) {
    if (matrix.empty() || matrix[0].empty()) return;

    size_t num_rows = matrix.size();
    size_t num_cols = matrix[0].size();

    for (size_t j = 0; j < num_cols; ++j) {
        std::ostringstream payload;
        payload << protocol_type;
        if (protocol_type == 'e') {
            payload << std::setw(4) << std::setfill('0') << NUM_EPOCHS;
        }
        payload << (j == num_cols - 1 ? '1' : '0');
        payload << std::setw(5) << std::setfill('0') << num_rows;

        for (size_t i = 0; i < num_rows; ++i) {
            payload << encode_float_string(matrix[i][j]);
        }

        std::string msg = payload.str();
        std::ostringstream full;
        full << std::setw(5) << std::setfill('0') << msg.size() << msg;
        safe_write(sock, full.str().c_str(), full.str().size());
    }
}

Matrix receive_any_matrix(int sock, char& out_type) {
    Matrix received_matrix;
    bool all_columns_received = false;

    while (!all_columns_received) {
        char size_buf[5];
        safe_read(sock, size_buf, 5);
        int total_size = std::stoi(std::string(size_buf, 5));

        std::vector<char> buffer(total_size);
        safe_read(sock, buffer.data(), total_size);

        size_t offset = 0;
        char protocol_type = buffer[offset++];
        out_type = protocol_type;

        if (protocol_type == 'V') {
            std::string num_vals_str(buffer.begin() + offset, buffer.begin() + offset + 5);
            int num_vals = std::stoi(num_vals_str);
            offset += 5;

            received_matrix.resize(num_vals);
            for (int i = 0; i < num_vals; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float val = decode_float_string(buffer, offset);
                    received_matrix[i].push_back(val);
                }
            }
            break;
        }

        if (protocol_type == 'e') offset += 4;
        char is_last_col = buffer[offset++];
        all_columns_received = (is_last_col == '1');

        std::string num_vals_str(buffer.begin() + offset, buffer.begin() + offset + 5);
        int num_vals = std::stoi(num_vals_str);
        offset += 5;

        if (received_matrix.empty()) received_matrix.resize(num_vals);
        for (int i = 0; i < num_vals; ++i) {
            float val = decode_float_string(buffer, offset);
            received_matrix[i].push_back(val);
        }
    }

    return received_matrix;
}

// ================== CLIENTE ==================
void handle_client(int client_id, int client_sock, const Dataset& data) {
    try {
        std::cout << "[INFO] Enviando dataset al cliente " << client_id << std::endl;
        send_matrix_by_columns(client_sock, 'e', data);

        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            for (int layer = 0; layer < NUM_LAYERS_FEDERATED; ++layer) {
                std::cout << "[DEBUG] Esperando matriz del cliente " << client_id 
                          << " (epoch " << epoch << ", capa " << layer << ")" << std::endl;

                Matrix m;
                Matrix outputs;

                if (layer == NUM_LAYERS_FEDERATED - 1) {
                    for (int i = 0; i < 2; ++i) {
                        char proto_type;
                        Matrix mat = receive_any_matrix(client_sock, proto_type);

                        if (proto_type == 'M') {
                            m = mat;
                        } else if (proto_type == 'V') {
                            outputs = mat;
                            std::cout << "[CLIENTE " << client_id << "] Resultados de salida (protocolo V):\n";
                            for (size_t i = 0; i < outputs.size(); ++i) {
                                std::cout << "  Muestra " << i << ": ";
                                for (float val : outputs[i]) {
                                    std::cout << val << " ";
                                }
                                std::cout << "\n";
                            }
                        }
                    }
                } else {
                    m = receive_any_matrix(client_sock, *(new char));
                }

                {
                    std::unique_lock<std::mutex> lock(mtx);
                    client_matrices[client_id] = m;
                    clients_ready_count++;
                    cv_clients_sent.notify_one();
                }

                {
                    std::unique_lock<std::mutex> lock(mtx);
                    auto now = std::chrono::system_clock::now();
                    if (!cv_main_sent_avg.wait_until(lock, now + std::chrono::seconds(10),
                        []{ return average_is_ready; })) {
                        std::cerr << "[WARN] Cliente " << client_id << " timeout esperando promedio" << std::endl;
                    }
                }
            }
        }

        std::cout << "[INFO] Cliente " << client_id << " completÃ³ el entrenamiento." << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "[ERROR] Cliente " << client_id << ": " << e.what() << std::endl;
    }

    close(client_sock);
}

// ================== MAIN ==================
int main() {
    std::cout << "Ingrese el nÃºmero de clientes: ";
    std::cin >> NUM_CLIENTS;

    Dataset data = read_csv("Dataset_of_Diabetes.csv");
    auto split_data = split_balanced(data, NUM_CLIENTS);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;
    int opt = 1;
setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));


    if (bind(server_fd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) { perror("bind"); return 1; }
    listen(server_fd, NUM_CLIENTS);
    std::cout << "[INFO] Servidor esperando " << NUM_CLIENTS << " clientes en el puerto " << PORT << "...\n";

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_CLIENTS; ++i) {
        int client_sock = accept(server_fd, nullptr, nullptr);
        if (client_sock < 0) { perror("accept"); continue; }
        std::cout << "[INFO] Cliente " << i << " conectado." << std::endl;
        all_client_sockets.push_back(client_sock);
        threads.emplace_back(handle_client, i, client_sock, split_data[i]);
    }

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        std::cout << "\n--- Epoch " << epoch + 1 << "/" << NUM_EPOCHS << " ---" << std::endl;
        for (int layer = 0; layer < NUM_LAYERS_FEDERATED; ++layer) {
            {
                std::unique_lock<std::mutex> lock(mtx);
                std::cout << "  Capa " << layer + 1 << ": Esperando matrices de clientes..." << std::endl;
                cv_clients_sent.wait(lock, []{ return clients_ready_count == NUM_CLIENTS; });
            }

            Matrix average_matrix;
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (client_matrices.empty()) continue;

                average_matrix = client_matrices.begin()->second;
                auto it = std::next(client_matrices.begin());
                while (it != client_matrices.end()) {
                    for (size_t r = 0; r < it->second.size(); ++r)
                        for (size_t c = 0; c < it->second[r].size(); ++c)
                            average_matrix[r][c] += it->second[r][c];
                    ++it;
                }
                for (auto& row : average_matrix)
                    for (auto& val : row)
                        val /= client_matrices.size();
                client_matrices.clear();
            }

            for (int sock : all_client_sockets) {
                try {
                    send_matrix_by_columns(sock, 'm', average_matrix);
                } catch (...) {}
            }

            {
                std::lock_guard<std::mutex> lock(mtx);
                average_is_ready = true;
                clients_ready_count = 0;
            }
            cv_main_sent_avg.notify_all();
            std::lock_guard<std::mutex> lock(mtx);
            average_is_ready = false;
        }
    }

    for (auto& t : threads) t.join();
    std::cout << "\n[INFO] Entrenamiento Federado Finalizado.\n";
    close(server_fd);
    return 0;
}
