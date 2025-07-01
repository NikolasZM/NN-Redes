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

// ================== CONFIGURACIÓN ==================
constexpr int PORT = 45015;
constexpr int INPUT_DIM = 11;
constexpr int CLASS_DIM = 3;
constexpr int TOTAL_DIM = INPUT_DIM + CLASS_DIM;
constexpr int NUM_LAYERS_FEDERATED = 3; // 2 ocultas + 1 de salida
int NUM_EPOCHS = 5;
int NUM_CLIENTS = 0;

using Sample = std::vector<float>;
using Dataset = std::vector<Sample>;
using Matrix = std::vector<std::vector<float>>;

// ================== GLOBALES Y SINCRONIZACIÓN ==================
std::mutex mtx;
std::condition_variable cv_clients_sent;
std::condition_variable cv_main_sent_avg;
int clients_ready_count = 0;
bool average_is_ready = false;

std::vector<int> all_client_sockets;
std::map<int, Matrix> client_matrices;

std::string encode_float_string(float val) {
    std::ostringstream oss;
    oss << val;
    std::string s = oss.str();
    std::ostringstream result;
    result << std::setw(2) << std::setfill('0') << s.size() << s;
    return result.str();
}


float decode_float_string(const std::vector<char>& buffer, size_t& offset) {
    std::string len_str(buffer.begin() + offset, buffer.begin() + offset + 2);
    offset += 2;
    int len = std::stoi(len_str);
    std::string float_str(buffer.begin() + offset, buffer.begin() + offset + len);
    offset += len;
    return std::stof(float_str);
}



// ================== LECTURA CSV Y BALANCEO ==================
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

// ================== PROTOCOLO DE COMUNICACIÓN ==================
void safe_write(int sock, const char* buffer, size_t count) {
    ssize_t sent = 0;
    while (sent < count) {
        ssize_t res = write(sock, buffer + sent, count - sent);
        if (res <= 0) throw std::runtime_error("Fallo al escribir en socket");
        sent += res;
    }
    cout << "[ENVIANDO] " << buffer << endl;
}

void safe_read(int sock, char* buffer, size_t count) {
    ssize_t received = 0;
    while (received < count) {
        ssize_t res = read(sock, buffer + received, count - received);
        if (res <= 0) throw std::runtime_error("Fallo al leer de socket");
        received += res;
    }
    cout << "[LEYENDO] " << buffer << endl;
}

void send_matrix_by_columns(int sock, char protocol_type, const Matrix& matrix) {
    if (matrix.empty() || matrix[0].empty()) return;

    size_t num_rows = matrix.size();
    size_t num_cols = matrix[0].size();

    for (size_t j = 0; j < num_cols; ++j) {
        std::ostringstream payload;

        // [tipo protocolo]
        payload << protocol_type;

        // [número de épocas] si es 'e'
        if (protocol_type == 'e') {
            payload << std::setw(4) << std::setfill('0') << NUM_EPOCHS;
        }

        // [es última columna]
        payload << (j == num_cols - 1 ? '1' : '0');

        // [cantidad de datos]
        payload << std::setw(5) << std::setfill('0') << num_rows;

        // [datos codificados]
        for (size_t i = 0; i < num_rows; ++i) {
            payload << encode_float_string(matrix[i][j]);
        }

        std::string msg = payload.str();
        std::ostringstream final_msg;
        final_msg << std::setw(5) << std::setfill('0') << msg.size() << msg;

        std::string full = final_msg.str();
        safe_write(sock, full.c_str(), full.size());
    }
}



Matrix receive_matrix_by_columns(int sock, char expected_type, bool with_epochs = false) {
    Matrix matrix;
    bool finished = false;

    while (!finished) {
        char size_buf[5];
        safe_read(sock, size_buf, 5);
        int total_size = std::stoi(std::string(size_buf, 5));

        std::vector<char> buffer(total_size);
        safe_read(sock, buffer.data(), total_size);

        size_t offset = 0;
        char type = buffer[offset++];
        if (type != expected_type) throw std::runtime_error("Tipo de protocolo inválido");

        int num_epochs = 0;
        if (expected_type == 'e' && with_epochs) {
            std::string ep_str(buffer.begin() + offset, buffer.begin() + offset + 4);
            num_epochs = std::stoi(ep_str);
            offset += 4;
        }

        char is_last_col = buffer[offset++];
        finished = (is_last_col == '1');

        std::string total_str(buffer.begin() + offset, buffer.begin() + offset + 5);
        int num_vals = std::stoi(total_str);
        offset += 5;

        if (matrix.empty()) matrix.resize(num_vals);

        for (int i = 0; i < num_vals; ++i) {
            float val = decode_float_string(buffer, offset);
            matrix[i].push_back(val);
        }
    }

    return matrix;
}


// ================== LÓGICA DE MANEJO DE CLIENTE ==================
void handle_client(int client_id, int client_sock, const Dataset& data) {
    try {
        std::cout << "[INFO] Enviando dataset al cliente " << client_id << std::endl;
        send_matrix_by_columns(client_sock, 'e', data);

        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            for (int layer = 0; layer < NUM_LAYERS_FEDERATED; ++layer) {
                std::cout << "[DEBUG] Esperando matriz del cliente " << client_id 
                          << " (epoch " << epoch << ", capa " << layer << ")" << std::endl;
                Matrix m = receive_matrix_by_columns(client_sock, 'M');
                
                std::cout << "[DEBUG] Cliente " << client_id << " envió matriz " 
                          << m.size() << "x" << (m.empty() ? 0 : m[0].size()) << std::endl;
                
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
        std::cout << "[INFO] Cliente " << client_id << " completó el entrenamiento." << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "[ERROR] Cliente " << client_id << ": " << e.what() << std::endl;
    }
    close(client_sock);
}
// ================== MAIN (ORQUESTADOR) ==================
int main() {
    std::cout << "Ingrese el número de clientes: ";
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
                if (client_matrices.empty()) {
                    std::cerr << "  [AVISO] Ningún cliente envió datos. Saltando ronda." << std::endl;
                    continue;
                }
                
                average_matrix = client_matrices.begin()->second;
                auto it = std::next(client_matrices.begin());
                while(it != client_matrices.end()){
                    for(size_t r=0; r<it->second.size(); ++r)
                        for(size_t c=0; c<it->second[r].size(); ++c)
                            average_matrix[r][c] += it->second[r][c];
                    ++it;
                }
                for(auto& row : average_matrix)
                    for(auto& val : row)
                        val /= client_matrices.size();
                client_matrices.clear();
            }
            std::cout << "  Capa " << layer + 1 << ": Promedio calculado. Enviando a todos..." << std::endl;

            for (int sock : all_client_sockets) {
                try {
                    send_matrix_by_columns(sock, 'm', average_matrix);
                } catch(...) { /* Ignorar */ }
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

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\n[INFO] Entrenamiento Federado Finalizado.\n";
    close(server_fd);
    return 0;
}
