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
#include <iomanip> // Para std::setw

// ================== CONFIGURACIÓN ==================
constexpr int PORT = 9001;
constexpr int INPUT_DIM = 11;
constexpr int CLASS_DIM = 3;
constexpr int TOTAL_DIM = INPUT_DIM + CLASS_DIM;
constexpr int NUM_LAYERS_FEDERATED = 3;
constexpr int TIMEOUT_SECONDS = 5;
int NUM_EPOCHS = 10;
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
bool timeout_timer_started = false;
std::chrono::steady_clock::time_point round_start_time;

std::vector<int> all_client_sockets;
std::map<int, Matrix> client_matrices;

// ================== LOGGING DE PROTOCOLO (NUEVO) ==================
void print_protocol_log(const std::string& direction, int client_id, char type, bool is_last, size_t data_size) {
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << std::left << std::setw(10) << direction 
              << "| Cliente " << std::setw(2) << client_id
              << "| Tipo: " << type 
              << " | Last: " << (is_last ? "1" : "0")
              << " | Size: " << std::setw(5) << data_size << " bytes" << std::endl;
}


// ================== LECTURA CSV Y BALANCEO (CORREGIDO) ==================
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

// CORREGIDO: Ahora asegura que todos los clientes tengan el mismo tamaño de dataset
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

    size_t min_size_per_client = -1;

    for(auto const& pair : class_buckets) {
        size_t size_per_client = pair.second.size() / num_clients;
        if (min_size_per_client == -1 || size_per_client < min_size_per_client) {
            min_size_per_client = size_per_client;
        }
    }
    
    for (int i = 0; i < num_clients; ++i) {
        for(auto const& pair : class_buckets) {
            int class_id = pair.first;
            const auto& samples = pair.second;
            size_t start_index = i * min_size_per_client;
            for (size_t j = 0; j < min_size_per_client; ++j) {
                splits[i].push_back(samples[start_index + j]);
            }
        }
    }
    
    std::cout << "[INFO] Data distribuida. Cada cliente recibe " << splits[0].size() << " muestras." << std::endl;
    return splits;
}


// ================== PROTOCOLO DE COMUNICACIÓN (CON LOGS) ==================
void safe_write(int sock, const char* buffer, size_t count) { /* ... sin cambios ... */ 
    ssize_t sent = 0;
    while (sent < count) {
        ssize_t res = write(sock, buffer + sent, count - sent);
        if (res <= 0) throw std::runtime_error("Fallo al escribir en socket");
        sent += res;
    }
}
void safe_read(int sock, char* buffer, size_t count) { /* ... sin cambios ... */ 
    ssize_t received = 0;
    while (received < count) {
        ssize_t res = read(sock, buffer + received, count - received);
        if (res <= 0) throw std::runtime_error("Fallo al leer de socket");
        received += res;
    }
}

void send_matrix_by_columns(int sock, int client_id, char protocol_type, const Matrix& matrix) {
    if (matrix.empty() || matrix[0].empty()) return;

    size_t num_rows = matrix.size();
    size_t num_cols = matrix[0].size();

    for (size_t j = 0; j < num_cols; ++j) {
        std::vector<char> buffer;
        bool is_last = (j == num_cols - 1);
        
        buffer.push_back(protocol_type);
        if (protocol_type == 'e') {
            uint32_t epochs = NUM_EPOCHS;
            buffer.insert(buffer.end(), (char*)&epochs, (char*)&epochs + sizeof(uint32_t));
        }
        buffer.push_back(is_last ? 1 : 0);
        uint32_t num_floats = num_rows;
        buffer.insert(buffer.end(), (char*)&num_floats, (char*)&num_floats + sizeof(uint32_t));

        for (size_t i = 0; i < num_rows; ++i) {
            float val = matrix[i][j];
            buffer.insert(buffer.end(), (char*)&val, (char*)&val + sizeof(float));
        }

        uint32_t total_size = buffer.size();
        safe_write(sock, (char*)&total_size, sizeof(uint32_t));
        safe_write(sock, buffer.data(), buffer.size());
        
        print_protocol_log("[SEND]", client_id, protocol_type, is_last, total_size);
    }
}

Matrix receive_matrix_by_columns(int sock, int client_id, char expected_type) {
    Matrix received_matrix;
    bool all_columns_received = false;

    while (!all_columns_received) {
        uint32_t total_size;
        safe_read(sock, (char*)&total_size, sizeof(uint32_t));
        
        std::vector<char> buffer(total_size);
        safe_read(sock, buffer.data(), total_size);

        size_t offset = 0;
        char protocol_type = buffer[offset++];
        if (protocol_type != expected_type) throw std::runtime_error("Tipo de protocolo inesperado");
        
        all_columns_received = buffer[offset++];
        print_protocol_log("[RECV]", client_id, protocol_type, all_columns_received, total_size);
        
        uint32_t num_floats;
        memcpy(&num_floats, &buffer[offset], sizeof(uint32_t));
        offset += sizeof(uint32_t);

        if (received_matrix.empty()) {
            received_matrix.resize(num_floats);
        }

        for (size_t i = 0; i < num_floats; ++i) {
            float val;
            memcpy(&val, &buffer[offset], sizeof(float));
            offset += sizeof(float);
            received_matrix[i].push_back(val);
        }
    }
    return received_matrix;
}


// ================== LÓGICA DE MANEJO DE CLIENTE ==================
void handle_client(int client_id, int client_sock, const Dataset& data) {
    try {
        send_matrix_by_columns(client_sock, client_id, 'e', data);

        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            for (int layer = 0; layer < NUM_LAYERS_FEDERATED; ++layer) {
                Matrix m = receive_matrix_by_columns(client_sock, client_id, 'M');
                
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    client_matrices[client_id] = m;
                    clients_ready_count++;
                    
                    if (!timeout_timer_started) {
                        timeout_timer_started = true;
                        round_start_time = std::chrono::steady_clock::now();
                        std::lock_guard<std::mutex> print_lock(mtx);
                        std::cout << "  [INFO] Primer cliente (" << client_id << ") terminó. Iniciando timeout de " << TIMEOUT_SECONDS << "s..." << std::endl;
                    }
                    
                    cv_clients_sent.notify_one();
                }
                
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv_main_sent_avg.wait(lock, []{ return average_is_ready; });
                }
            }
        }
    } catch (const std::runtime_error& e) {
        std::lock_guard<std::mutex> lock(mtx);
        std::cerr << "[ERROR] Cliente " << client_id << ": " << e.what() << ". Desconectando." << std::endl;
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
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "[INFO] Cliente " << i << " conectado." << std::endl;
        }

        all_client_sockets.push_back(client_sock);
        threads.emplace_back(handle_client, i, client_sock, split_data[i]);
    }

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        mtx.lock();
        std::cout << "\n--- Epoch " << epoch + 1 << "/" << NUM_EPOCHS << " ---" << std::endl;
        mtx.unlock();
        for (int layer = 0; layer < NUM_LAYERS_FEDERATED; ++layer) {
            {
                std::unique_lock<std::mutex> lock(mtx);
                std::cout << "  Capa " << layer + 1 << ": Esperando matrices de clientes..." << std::endl;

                while (clients_ready_count < NUM_CLIENTS) {
                    if (timeout_timer_started) {
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - round_start_time);
                        if (elapsed.count() >= TIMEOUT_SECONDS) {
                            std::cout << "  [INFO] Timeout! " << clients_ready_count << "/" << NUM_CLIENTS << " clientes respondieron." << std::endl;
                            break; 
                        }
                        cv_clients_sent.wait_for(lock, std::chrono::seconds(1));
                    } else {
                        cv_clients_sent.wait(lock);
                    }
                }
            }

            Matrix average_matrix;
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (client_matrices.empty()) {
                    std::cerr << "  [AVISO] Ningún cliente envió datos a tiempo. Saltando ronda." << std::endl;
                } else {
                    average_matrix = client_matrices.begin()->second;
                    auto it = std::next(client_matrices.begin());
                    while(it != client_matrices.end()){
                        // Este bucle ahora es seguro porque todos los splits tienen el mismo tamaño
                        for(size_t r=0; r<it->second.size(); ++r)
                            for(size_t c=0; c<it->second[r].size(); ++c)
                                average_matrix[r][c] += it->second[r][c];
                        ++it;
                    }
                    for(auto& row : average_matrix)
                        for(auto& val : row)
                            val /= client_matrices.size();
                    
                    std::cout << "  Capa " << layer + 1 << ": Promedio calculado con " << client_matrices.size() << " clientes. Enviando a todos..." << std::endl;
                }
                client_matrices.clear();
            }
            
            for (size_t i = 0; i < all_client_sockets.size(); ++i) {
                try {
                    if (!average_matrix.empty()) {
                        send_matrix_by_columns(all_client_sockets[i], i, 'm', average_matrix);
                    }
                } catch(...) { /* Ignorar si un cliente se cayó */ }
            }

            {
                std::lock_guard<std::mutex> lock(mtx);
                average_is_ready = true;
                clients_ready_count = 0;
                timeout_timer_started = false;
            }
            cv_main_sent_avg.notify_all();
            
            {
                std::lock_guard<std::mutex> lock(mtx);
                average_is_ready = false;
            }
        }
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    std::cout << "\n[INFO] Entrenamiento Federado Finalizado.\n";
    close(server_fd);
    return 0;
}

