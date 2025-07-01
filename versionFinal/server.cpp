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

// ================== CONFIGURACIÓN ==================
constexpr int PORT = 45001;
constexpr int INPUT_DIM = 11;
constexpr int CLASS_DIM = 3;
constexpr int TOTAL_DIM = INPUT_DIM + CLASS_DIM;
constexpr int NUM_LAYERS_FEDERATED = 3;
int NUM_EPOCHS = 5;
int NUM_CLIENTS = 0;

using Sample = vector<float>;
using Dataset = vector<Sample>;
using Matrix = vector<vector<float>>;

// ================== GLOBALES ==================
mutex mtx;
condition_variable cv_clients_sent, cv_main_sent_avg;
int clients_ready_count = 0;
bool average_is_ready = false;

vector<int> all_client_sockets;
map<int, Matrix> client_matrices;
map<int, vector<int>> client_true_labels;

// ================== UTILIDADES ==================
string encode_float_string(float val) {
    ostringstream oss;
    oss << val;
    string s = oss.str();
    ostringstream result;
    result << setw(2) << setfill('0') << s.size() << s;
    return result.str();
}

float decode_float_string(const vector<char>& buffer, size_t& offset) {
    if (offset + 2 > buffer.size()) throw runtime_error("decode_float_string: fuera de rango");
    int len = stoi(string(buffer.begin() + offset, buffer.begin() + offset + 2));
    offset += 2;
    if (offset + len > buffer.size()) throw runtime_error("decode_float_string: fuera de rango");

    string float_str(buffer.begin() + offset, buffer.begin() + offset + len);
    offset += len;
    try {
        return stof(float_str);
    } catch (...) {
        cerr << "[ERROR] Conversión a float fallida: [" << float_str << "]\n";
        throw;
    }
}

// ================== CSV ==================
Dataset read_csv(const string& filename) {
    Dataset data;
    ifstream file(filename);
    if (!file.is_open()) throw runtime_error("No se pudo abrir el CSV");

    string line;
    getline(file, line); // Header
    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        Sample row;
        while (getline(ss, token, ',')) {
            row.push_back(stof(token));
        }
        if (row.size() == TOTAL_DIM) data.push_back(row);
    }
    return data;
}

vector<Dataset> split_balanced(Dataset& data, int num_clients) {
    vector<Dataset> splits(num_clients);
    map<int, Dataset> buckets;

    for (const auto& row : data) {
        int class_idx = max_element(row.begin() + INPUT_DIM, row.end()) - row.begin() - INPUT_DIM;
        buckets[class_idx].push_back(row);
    }
    for (auto& [_, samples] : buckets) {
        for (size_t i = 0; i < samples.size(); ++i) {
            splits[i % num_clients].push_back(samples[i]);
        }
    }
    return splits;
}

// ================== COMUNICACIÓN ==================
void safe_write(int sock, const char* buffer, size_t count) {
    size_t sent = 0;
    while (sent < count) {
        ssize_t res = write(sock, buffer + sent, count - sent);
        if (res <= 0) throw runtime_error("Fallo al escribir en socket");
        sent += res;
    }
    cout << "[ESCRIBIENDO] " << buffer << endl;
}

void safe_read(int sock, char* buffer, size_t count) {
    size_t received = 0;
    while (received < count) {
        ssize_t res = read(sock, buffer + received, count - received);
        if (res <= 0) throw runtime_error("Fallo al leer de socket");
        received += res;
    }
    cout << "[LEYENDO] " << buffer << endl;
}

void send_matrix_by_columns(int sock, char type, const Matrix& matrix) {
    if (matrix.empty() || matrix[0].empty()) return;

    for (size_t col = 0; col < matrix[0].size(); ++col) {
        ostringstream payload;
        payload << type;
        if (type == 'e') payload << setw(4) << setfill('0') << NUM_EPOCHS;
        payload << (col == matrix[0].size() - 1 ? '1' : '0');
        payload << setw(5) << setfill('0') << matrix.size();
        for (auto& row : matrix) {
            payload << encode_float_string(row[col]);
        }

        string msg = payload.str();
        ostringstream final_msg;
        final_msg << setw(5) << setfill('0') << msg.size() << msg;
        safe_write(sock, final_msg.str().c_str(), final_msg.str().size());
    }
}

Matrix receive_any_matrix(int sock, char& out_type) {
    Matrix result;
    bool all_cols_received = false;

    while (!all_cols_received) {
        char size_buf[5];
        safe_read(sock, size_buf, 5);
        int total_size = stoi(string(size_buf, 5));
        vector<char> buffer(total_size);
        safe_read(sock, buffer.data(), total_size);

        size_t offset = 0;
        char type = buffer[offset++];
        out_type = type;

        if (type == 'V') {
            int rows = stoi(string(buffer.begin() + offset, buffer.begin() + offset + 5));
            offset += 5;
            result.resize(rows);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < 3; ++j)
                    result[i].push_back(decode_float_string(buffer, offset));
            break;
        }

        if (type == 'e') offset += 4;
        bool last = buffer[offset++] == '1';
        all_cols_received = last;

        int rows = stoi(string(buffer.begin() + offset, buffer.begin() + offset + 5));
        offset += 5;
        if (result.empty()) result.resize(rows);

        for (int i = 0; i < rows; ++i)
            result[i].push_back(decode_float_string(buffer, offset));
    }
    return result;
}

// ================== CLIENTE ==================
void handle_client(int id, int sock, const Dataset& data) {
    try {
        cout << "[INFO] Enviando dataset al cliente " << id << endl;
        send_matrix_by_columns(sock, 'e', data);

        for (const auto& row : data) {
            int label = max_element(row.begin() + INPUT_DIM, row.end()) - row.begin() - INPUT_DIM;
            client_true_labels[id].push_back(label);
        }

        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            for (int layer = 0; layer < NUM_LAYERS_FEDERATED; ++layer) {
                cout << "[DEBUG] Cliente " << id << " epoch " << epoch << ", capa " << layer << endl;

                Matrix m, v;
                if (layer == NUM_LAYERS_FEDERATED - 1) {
                    for (int i = 0; i < 2; ++i) {
                        char t;
                        Matrix mat = receive_any_matrix(sock, t);
                        if (t == 'M') m = mat;
                        else if (t == 'V') {
                            v = mat;
                            cout << "[CLIENTE " << id << "] Salidas (V):\n";
                            int correct = 0;
                            const vector<int>& true_labels = client_true_labels[id];
                            for (size_t i = 0; i < v.size(); ++i) {
                                int pred = max_element(v[i].begin(), v[i].end()) - v[i].begin();
                                if (pred == true_labels[i]) correct++;
                                cout << "  Muestra " << i << ": ";
                                for (float val : v[i]) cout << val << " ";
                                cout << "\n";
                            }
                            float acc = static_cast<float>(correct) / v.size();
                            cout << fixed << setprecision(2);
                            cout << "[CLIENTE " << id << "] Accuracy local: " << acc * 100 << "% (" << correct << "/" << v.size() << ")\n";
                        }
                    }
                } else {
                    m = receive_any_matrix(sock, *(new char));
                }

                {
                    lock_guard<mutex> lock(mtx);
                    client_matrices[id] = m;
                    clients_ready_count++;
                    cv_clients_sent.notify_one();
                }

                {
                    unique_lock<mutex> lock(mtx);
                    auto now = chrono::system_clock::now();
                    cv_main_sent_avg.wait_until(lock, now + chrono::seconds(10), []{ return average_is_ready; });
                }
            }
        }

        cout << "[INFO] Cliente " << id << " terminó.\n";
    } catch (const exception& e) {
        cerr << "[ERROR] Cliente " << id << ": " << e.what() << endl;
    }
    close(sock);
}

// ================== MAIN ==================
int main() {
    cout << "Ingrese el número de clientes: ";
    cin >> NUM_CLIENTS;

    Dataset data = read_csv("Dataset_of_Diabetes.csv");
    auto split = split_balanced(data, NUM_CLIENTS);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    listen(server_fd, NUM_CLIENTS);
    cout << "[INFO] Esperando " << NUM_CLIENTS << " clientes en el puerto " << PORT << "...\n";

    vector<thread> threads;
    for (int i = 0; i < NUM_CLIENTS; ++i) {
        int client_sock = accept(server_fd, nullptr, nullptr);
        if (client_sock < 0) { perror("accept"); continue; }
        cout << "[INFO] Cliente " << i << " conectado.\n";
        all_client_sockets.push_back(client_sock);
        threads.emplace_back(handle_client, i, client_sock, split[i]);
    }

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        cout << "\n--- Epoch " << epoch + 1 << " ---\n";
        for (int layer = 0; layer < NUM_LAYERS_FEDERATED; ++layer) {
            {
                unique_lock<mutex> lock(mtx);
                cout << "  Capa " << layer + 1 << ": Esperando matrices...\n";
                cv_clients_sent.wait(lock, []{ return clients_ready_count == NUM_CLIENTS; });
            }

            Matrix avg;
            {
                lock_guard<mutex> lock(mtx);
                if (client_matrices.empty()) continue;
                avg = client_matrices.begin()->second;
                for (auto it = ++client_matrices.begin(); it != client_matrices.end(); ++it) {
                    for (size_t i = 0; i < it->second.size(); ++i)
                        for (size_t j = 0; j < it->second[i].size(); ++j)
                            avg[i][j] += it->second[i][j];
                }
                for (auto& row : avg)
                    for (auto& val : row) val /= client_matrices.size();
                client_matrices.clear();
            }

            for (int sock : all_client_sockets) {
                try {
                    send_matrix_by_columns(sock, 'm', avg);
                } catch (...) {}
            }

            {
                lock_guard<mutex> lock(mtx);
                average_is_ready = true;
                clients_ready_count = 0;
            }
            cv_main_sent_avg.notify_all();
            lock_guard<mutex> lock(mtx);
            average_is_ready = false;
        }
    }

    for (auto& t : threads) t.join();
    cout << "\n[INFO] Entrenamiento federado finalizado.\n";
    close(server_fd);
    return 0;
}
