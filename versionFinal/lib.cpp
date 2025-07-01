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
    uint16_t len = s.size();
    std::ostringstream result;
    result << std::setw(2) << std::setfill('0') << len << s;
    return result.str();
}

float decode_float_string(const std::vector<char>& buffer, size_t& offset) {
    if (offset + 2 > buffer.size()) throw std::runtime_error("decode_float_string: fuera de rango al leer tamaño");

    std::string len_str(buffer.begin() + offset, buffer.begin() + offset + 2);
    offset += 2;

    int len = std::stoi(len_str);
    if (offset + len > buffer.size()) throw std::runtime_error("decode_float_string: fuera de rango al leer string float");

    std::string float_str(buffer.begin() + offset, buffer.begin() + offset + len);
    offset += len;

    try {
        return std::stof(float_str);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] No se pudo convertir a float: [" << float_str << "] (len: " << len << ")\n";
        throw;
    }
}


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
        std::ostringstream payload;

        // [letra protocolo]
        payload << protocol_type;

        // [número de épocas] solo si es 'e' (este lado no lo usa, pero lo dejamos por claridad)
        if (protocol_type == 'e') {
            payload << std::setw(4) << std::setfill('0') << 5;  // si quieres parametrizar, cámbialo
        }

        // [última columna]
        payload << (j == num_cols - 1 ? '1' : '0');

        // [cantidad de datos]
        payload << std::setw(5) << std::setfill('0') << num_rows;

        // [tamaño dato + dato] por cada elemento
        for (size_t i = 0; i < num_rows; ++i) {
            payload << encode_float_string(matrix[i][j]);
        }

        // [tamaño total como string de 5 bytes] + payload
        std::string body = payload.str();
        std::ostringstream full_msg;
        full_msg << std::setw(5) << std::setfill('0') << body.size() << body;

        std::string final_str = full_msg.str();
        safe_write(client_sock, final_str.c_str(), final_str.size());
    }
}


py::object receive_matrix_by_columns_client(char expected_type, bool with_epochs = false) {
    if (client_sock < 0) throw std::runtime_error("No conectado al servidor");

    MatrixF received_matrix;
    bool all_columns_received = false;
    uint32_t num_epochs = 0;

    while (!all_columns_received) {
        // Leer los primeros 5 bytes que indican el tamaño total
        char size_buf[5];
        safe_read(client_sock, size_buf, 5);
        int total_size = std::stoi(std::string(size_buf, 5));

        // Leer el resto del mensaje
        std::vector<char> buffer(total_size);
        safe_read(client_sock, buffer.data(), total_size);

        size_t offset = 0;

        // Tipo de protocolo
        char protocol_type = buffer[offset++];
        if (protocol_type != expected_type) throw std::runtime_error("Tipo de protocolo inesperado");

        // Épocas (si corresponde)
        if (with_epochs) {
            std::string ep_str(buffer.begin() + offset, buffer.begin() + offset + 4);
            num_epochs = std::stoi(ep_str);
            offset += 4;
        }

        // Última columna
        char is_last_col = buffer[offset++];
        all_columns_received = (is_last_col == '1');

        // Número total de datos
        std::string total_vals_str(buffer.begin() + offset, buffer.begin() + offset + 5);
        int num_vals = std::stoi(total_vals_str);
        offset += 5;

        // Decodificar los datos
        if (received_matrix.empty()) received_matrix.resize(num_vals);

        for (int i = 0; i < num_vals; ++i) {
            float val = decode_float_string(buffer, offset);
            received_matrix[i].push_back(val);
        }
    }

    // Convertir a numpy
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

m.def("send_output", [](py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) throw std::runtime_error("La matriz debe tener 2 dimensiones");

    size_t num_rows = buf.shape[0];
    size_t num_cols = buf.shape[1];
    if (num_cols != 3) throw std::runtime_error("La salida debe tener 3 columnas (clases)");

    float* ptr = static_cast<float*>(buf.ptr);

    // Construir mensaje tipo V (una sola vez, no por columnas)
    std::ostringstream payload;
    payload << 'V';  // tipo
    payload << std::setw(5) << std::setfill('0') << num_rows;

    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float val = ptr[i * 3 + j];
            std::ostringstream val_str;
            val_str << val;
            std::string s = val_str.str();
            payload << std::setw(2) << std::setfill('0') << s.size() << s;
        }
    }

    std::string body = payload.str();
    std::ostringstream final_msg;
    final_msg << std::setw(5) << std::setfill('0') << body.size() << body;

    std::string full = final_msg.str();
    safe_write(client_sock, full.c_str(), full.size());
}, "Envía la salida final del forward pass al servidor");


}

