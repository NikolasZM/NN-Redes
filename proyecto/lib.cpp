
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

constexpr int PORT = 9000;

int sockfd = -1;

// ============== Conexión TCP ==============
void connect_to_server(const std::string& server_ip) {
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) throw std::runtime_error("No se pudo crear el socket");

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, server_ip.c_str(), &serv_addr.sin_addr) <= 0)
        throw std::runtime_error("IP inválida");

    if (connect(sockfd, (sockaddr*)&serv_addr, sizeof(serv_addr)) < 0)
        throw std::runtime_error("Conexión fallida con el servidor");
}

// ============== Recepción de datos tipo 'e' ==============
std::pair<std::vector<std::vector<float>>, int> receive_dataset() {
    std::vector<std::vector<float>> columns;
    int epochs = 0;

    while (true) {
        char header[16];
        int total_read = read(sockfd, header, 16);
        if (total_read <= 0) throw std::runtime_error("Error al leer cabecera");

        uint64_t size_total;
        memcpy(&size_total, &header[0], 5);

        char type = header[5];
        if (type != 'e') throw std::runtime_error("Tipo incorrecto, se esperaba 'e'");

        memcpy(&epochs, &header[6], 4);
        bool is_last = header[10];

        uint64_t total_numbers;
        memcpy(&total_numbers, &header[11], 5);

        int payload_size = size_total - 16;
        std::vector<char> payload(payload_size);
        int read_bytes = read(sockfd, payload.data(), payload_size);
        if (read_bytes <= 0) throw std::runtime_error("Error al leer datos");

        std::vector<float> column;
        size_t offset = 0;
        for (uint64_t i = 0; i < total_numbers; ++i) {
            uint16_t sz;
            memcpy(&sz, &payload[offset], 2);
            float val;
            memcpy(&val, &payload[offset + 2], 4);
            column.push_back(val);
            offset += 6;
        }

        columns.push_back(column);
        if (is_last) break;
    }

    // Reorganizar columnas en filas
    size_t num_rows = columns[0].size();
    size_t num_cols = columns.size();
    std::vector<std::vector<float>> matrix(num_rows, std::vector<float>(num_cols));

    for (size_t c = 0; c < num_cols; ++c)
        for (size_t r = 0; r < num_rows; ++r)
            matrix[r][c] = columns[c][r];

    return {matrix, epochs};
}

// ============== Enviar activaciones con protocolo 'M' ==============
void send_matrix(const std::vector<std::vector<float>>& matrix) {
    size_t num_cols = matrix[0].size();
    size_t num_rows = matrix.size();

    for (size_t col = 0; col < num_cols; ++col) {
        std::vector<char> buffer;
        uint64_t size_total = 1 + 1 + 5 + num_rows * (2 + 4);
        buffer.resize(5 + size_total);

        memcpy(&buffer[0], &size_total, 5);
        buffer[5] = 'M';
        buffer[6] = (col == num_cols - 1 ? 1 : 0);
        uint64_t total_numbers = num_rows;
        memcpy(&buffer[7], &total_numbers, 5);

        size_t offset = 12;
        for (size_t i = 0; i < num_rows; ++i) {
            uint16_t sz = 4;
            memcpy(&buffer[offset], &sz, 2);
            memcpy(&buffer[offset + 2], &matrix[i][col], 4);
            offset += 6;
        }

        write(sockfd, buffer.data(), buffer.size());
    }
}

// ============== Recibir promedio 'm' ==============
std::vector<std::vector<float>> receive_average_matrix() {
    std::vector<std::vector<float>> columns;

    while (true) {
        char header[12];
        int total_read = read(sockfd, header, 12);
        if (total_read <= 0) throw std::runtime_error("Error al leer header de promedio");

        uint64_t size_total;
        memcpy(&size_total, &header[0], 5);
        char type = header[5];
        if (type != 'm') throw std::runtime_error("Esperado 'm'");

        bool is_last = header[6];
        uint64_t total_numbers;
        memcpy(&total_numbers, &header[7], 5);

        int payload_size = size_total - 12;
        std::vector<char> payload(payload_size);
        read(sockfd, payload.data(), payload_size);

        std::vector<float> column;
        size_t offset = 0;
        for (uint64_t i = 0; i < total_numbers; ++i) {
            uint16_t sz;
            memcpy(&sz, &payload[offset], 2);
            float val;
            memcpy(&val, &payload[offset + 2], 4);
            column.push_back(val);
            offset += 6;
        }

        columns.push_back(column);
        if (is_last) break;
    }

    size_t num_rows = columns[0].size();
    size_t num_cols = columns.size();
    std::vector<std::vector<float>> matrix(num_rows, std::vector<float>(num_cols));

    for (size_t c = 0; c < num_cols; ++c)
        for (size_t r = 0; r < num_rows; ++r)
            matrix[r][c] = columns[c][r];

    return matrix;
}

// ============== Export pybind11 ==============

PYBIND11_MODULE(lib, m) {
    m.def("connect", &connect_to_server, "Conectar con el servidor");
    m.def("receive_dataset", &receive_dataset, "Recibir dataset inicial");
    m.def("send_matrix", &send_matrix, "Enviar matriz de activaciones");
    m.def("receive_average", &receive_average_matrix, "Recibir matriz promedio");
}
