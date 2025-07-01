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
#include <iomanip>

namespace py = pybind11;
using MatrixF = std::vector<std::vector<float>>;

int client_sock = -1;

std::string encode_float_string(float val) {
    std::ostringstream oss;
    oss << val;
    std::string s = oss.str();
    return (std::ostringstream() << std::setw(2) << std::setfill('0') << s.size() << s).str();
}

float decode_float_string(const std::vector<char>& buffer, size_t& offset) {
    if (offset + 2 > buffer.size()) throw std::runtime_error("decode_float_string: fuera de rango");
    int len = std::stoi(std::string(buffer.begin() + offset, buffer.begin() + offset + 2));
    offset += 2;
    if (offset + len > buffer.size()) throw std::runtime_error("decode_float_string: fuera de rango");

    std::string float_str(buffer.begin() + offset, buffer.begin() + offset + len);
    offset += len;
    try {
        return std::stof(float_str);
    } catch (...) {
        std::cerr << "[ERROR] No se pudo convertir a float: [" << float_str << "]\n";
        throw;
    }
}

void safe_write(int sock, const char* buffer, size_t count) {
    size_t sent = 0;
    while (sent < count) {
        ssize_t res = write(sock, buffer + sent, count - sent);
        if (res <= 0) throw std::runtime_error("Fallo al escribir en socket");
        sent += res;
    }
}

void safe_read(int sock, char* buffer, size_t count) {
    size_t received = 0;
    while (received < count) {
        ssize_t res = read(sock, buffer + received, count - received);
        if (res <= 0) throw std::runtime_error("Fallo al leer de socket");
        received += res;
    }
}

void send_matrix_by_columns_client(char type, const MatrixF& matrix) {
    if (client_sock < 0) throw std::runtime_error("No conectado al servidor");
    if (matrix.empty() || matrix[0].empty()) return;

    size_t rows = matrix.size(), cols = matrix[0].size();
    for (size_t j = 0; j < cols; ++j) {
        std::ostringstream payload;
        payload << type;
        if (type == 'e') payload << std::setw(4) << std::setfill('0') << 5;
        payload << (j == cols - 1 ? '1' : '0');
        payload << std::setw(5) << std::setfill('0') << rows;
        for (size_t i = 0; i < rows; ++i) payload << encode_float_string(matrix[i][j]);

        std::string body = payload.str();
        std::ostringstream msg;
        msg << std::setw(5) << std::setfill('0') << body.size() << body;
        safe_write(client_sock, msg.str().c_str(), msg.str().size());
    }
}

py::object receive_matrix_by_columns_client(char expected_type, bool with_epochs = false) {
    if (client_sock < 0) throw std::runtime_error("No conectado al servidor");

    MatrixF matrix;
    uint32_t num_epochs = 0;
    bool last_column = false;

    while (!last_column) {
        char size_buf[5];
        safe_read(client_sock, size_buf, 5);
        int total_size = std::stoi(std::string(size_buf, 5));

        std::vector<char> buffer(total_size);
        safe_read(client_sock, buffer.data(), total_size);

        size_t offset = 0;
        char proto = buffer[offset++];
        if (proto != expected_type) throw std::runtime_error("Tipo de protocolo inesperado");

        if (with_epochs) {
            num_epochs = std::stoi(std::string(buffer.begin() + offset, buffer.begin() + offset + 4));
            offset += 4;
        }

        last_column = (buffer[offset++] == '1');
        int num_vals = std::stoi(std::string(buffer.begin() + offset, buffer.begin() + offset + 5));
        offset += 5;

        if (matrix.empty()) matrix.resize(num_vals);
        for (int i = 0; i < num_vals; ++i)
            matrix[i].push_back(decode_float_string(buffer, offset));
    }

    py::array_t<float> result({matrix.size(), matrix.empty() ? 0 : matrix[0].size()});
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    for (size_t i = 0; i < matrix.size(); ++i)
        for (size_t j = 0; j < matrix[i].size(); ++j)
            ptr[i * matrix[0].size() + j] = matrix[i][j];

    return with_epochs ? py::make_tuple(result, num_epochs) : py::object(result);
}

void connect_to_server(const std::string& ip, int port) {
    client_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (client_sock < 0) throw std::runtime_error("No se pudo crear el socket");

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0)
        throw std::runtime_error("Dirección IP inválida");

    if (connect(client_sock, (sockaddr*)&addr, sizeof(addr)) < 0)
        throw std::runtime_error("Conexión fallida con el servidor");

    std::cout << "Conectado al servidor " << ip << ":" << port << std::endl;
}

void disconnect_from_server() {
    if (client_sock >= 0) {
        close(client_sock);
        client_sock = -1;
    }
}

PYBIND11_MODULE(lib, m) {
    m.doc() = "Librería C++ para comunicación de red";

    m.def("connect", &connect_to_server, "Conecta al servidor", py::arg("ip"), py::arg("port") = 9000);
    m.def("disconnect", &disconnect_from_server, "Desconecta del servidor");

    m.def("receive_dataset", []() {
        return receive_matrix_by_columns_client('e', true);
    });

    m.def("send_matrix", [](py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        auto buf = arr.request();
        if (buf.ndim != 2) throw std::runtime_error("La matriz debe tener 2 dimensiones");
        MatrixF mat(buf.shape[0], std::vector<float>(buf.shape[1]));
        float* ptr = static_cast<float*>(buf.ptr);
        for (ssize_t i = 0; i < buf.shape[0]; ++i)
            for (ssize_t j = 0; j < buf.shape[1]; ++j)
                mat[i][j] = ptr[i * buf.shape[1] + j];
        send_matrix_by_columns_client('M', mat);
    });

    m.def("receive_average", []() {
        return receive_matrix_by_columns_client('m', false);
    });

    m.def("send_output", [](py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        auto buf = arr.request();
        if (buf.ndim != 2) throw std::runtime_error("La matriz debe tener 2 dimensiones");
        if (buf.shape[1] != 3) throw std::runtime_error("La salida debe tener 3 columnas");

        std::ostringstream payload;
        payload << 'V';
        payload << std::setw(5) << std::setfill('0') << buf.shape[0];
        float* ptr = static_cast<float*>(buf.ptr);

        for (ssize_t i = 0; i < buf.shape[0]; ++i)
            for (ssize_t j = 0; j < 3; ++j)
                payload << encode_float_string(ptr[i * 3 + j]);

        std::string body = payload.str();
        std::ostringstream full_msg;
        full_msg << std::setw(5) << std::setfill('0') << body.size() << body;
        safe_write(client_sock, full_msg.str().c_str(), full_msg.str().size());
    });
}
