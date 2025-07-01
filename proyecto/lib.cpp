#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

namespace py = pybind11;
using MatrixF = std::vector<std::vector<float>>;

int client_sock = -1;

// ================== FUNCIONES DE COMUNICACIÓN (CLIENTE) ==================
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

void send_matrix_by_columns_client(char protocol_type, const MatrixF& matrix) {
    if (client_sock < 0) throw std::runtime_error("No conectado al servidor");
    if (matrix.empty() || matrix[0].empty()) return;

    size_t num_rows = matrix.size();
    size_t num_cols = matrix[0].size();

    for (size_t j = 0; j < num_cols; ++j) {
        std::vector<char> buffer;
        bool is_last = (j == num_cols - 1);
        
        buffer.push_back(protocol_type);
        buffer.push_back(is_last ? 1 : 0);
        uint32_t num_floats = num_rows;
        buffer.insert(buffer.end(), (char*)&num_floats, (char*)&num_floats + sizeof(uint32_t));

        for (size_t i = 0; i < num_rows; ++i) {
            float val = matrix[i][j];
            buffer.insert(buffer.end(), (char*)&val, (char*)&val + sizeof(float));
        }

        uint32_t total_size = buffer.size();
        safe_write(client_sock, (char*)&total_size, sizeof(uint32_t));
        safe_write(client_sock, buffer.data(), buffer.size());
    }
}

py::object receive_matrix_by_columns_client(char expected_type, bool with_epochs = false) {
    if (client_sock < 0) throw std::runtime_error("No conectado al servidor");

    MatrixF received_matrix;
    bool all_columns_received = false;
    uint32_t num_epochs = 0;

    while (!all_columns_received) {
        uint32_t total_size;
        safe_read(client_sock, (char*)&total_size, sizeof(uint32_t));
        
        std::vector<char> buffer(total_size);
        safe_read(client_sock, buffer.data(), total_size);

        size_t offset = 0;
        char protocol_type = buffer[offset++];
        if (protocol_type != expected_type) throw std::runtime_error("Tipo de protocolo inesperado");

        if (with_epochs) {
            memcpy(&num_epochs, &buffer[offset], sizeof(uint32_t));
            offset += sizeof(uint32_t);
        }

        all_columns_received = buffer[offset++];
        
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
    
    py::array_t<float> result_np({received_matrix.size(), received_matrix.empty() ? 0 : received_matrix[0].size()});
    auto buf = result_np.request();
    float *ptr = static_cast<float *>(buf.ptr);
    for(size_t i=0; i<received_matrix.size(); ++i) {
        for(size_t j=0; j<received_matrix[0].size(); ++j) {
            ptr[i * received_matrix[0].size() + j] = received_matrix[i][j];
        }
    }

    if (with_epochs) {
        return py::make_tuple(result_np, num_epochs);
    }
    return result_np;
}

// ================== FUNCIONES EXPUESTAS A PYTHON ==================
void connect_to_server(const std::string& ip, int port) {
    client_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (client_sock < 0) throw std::runtime_error("No se pudo crear el socket");

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr) <= 0) {
        throw std::runtime_error("Dirección IP inválida");
    }

    if (connect(client_sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        throw std::runtime_error("Conexión fallida con el servidor");
    }
    std::cout << "Conectado al servidor " << ip << ":" << port << std::endl;
}

void disconnect_from_server() {
    if (client_sock >= 0) {
        close(client_sock);
        client_sock = -1;
    }
}

// ================== DEFINICIÓN DEL MÓDULO PYBIND11 ==================
PYBIND11_MODULE(lib, m) {
    m.doc() = "Librería C++ para comunicación de red";

    m.def("connect", &connect_to_server, "Conecta al servidor", py::arg("ip"), py::arg("port") = 9000);
    m.def("disconnect", &disconnect_from_server, "Desconecta del servidor");

    m.def("receive_dataset", []() {
        return receive_matrix_by_columns_client('e', true);
    }, "Recibe el dataset inicial y las épocas del servidor");

    m.def("send_matrix", [](py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 2) throw std::runtime_error("La matriz debe tener 2 dimensiones");

        MatrixF matrix(buf.shape[0], std::vector<float>(buf.shape[1]));
        float* ptr = static_cast<float*>(buf.ptr);
        for(ssize_t i=0; i<buf.shape[0]; ++i) {
            for(ssize_t j=0; j<buf.shape[1]; ++j) {
                matrix[i][j] = ptr[i * buf.shape[1] + j];
            }
        }
        send_matrix_by_columns_client('M', matrix);
    }, "Envía una matriz al servidor");

    m.def("receive_average", []() {
        return receive_matrix_by_columns_client('m', false);
    }, "Recibe la matriz promediada del servidor");
}

