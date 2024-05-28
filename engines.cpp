#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace py = pybind11;

// Wrapper class for 2D matrix using 1D vector
class Matrix {
public:
    Matrix(const std::vector<double>& data, size_t rows, size_t cols)
        : data_(data), rows_(rows), cols_(cols) {}

    double operator()(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    double& operator()(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

private:
    std::vector<double> data_;
    size_t rows_, cols_;
};

// Function to normalize a vector
Eigen::VectorXd to_normalize(const Eigen::VectorXd& vec) {
    return vec / vec.norm();
}

// Function to normalize a matrix row-wise
Eigen::MatrixXd to_normalize_matrix(const Eigen::MatrixXd& mat) {
    Eigen::MatrixXd normalized(mat.rows(), mat.cols());
    #pragma omp parallel for
    for (int i = 0; i < mat.rows(); ++i) {
        double norm = mat.row(i).norm();
        if (norm > 0) {
            normalized.row(i) = mat.row(i) / norm;
        } else {
            normalized.row(i) = mat.row(i);
        }
    }
    return normalized;
}

// Function to compute cosine distance
double cosine_distance(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2) {
    return vec1.dot(vec2) / (vec1.norm() * vec2.norm());
}

// Function to compute Euclidean distance
double euclidean_distance(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2) {
    return (vec1 - vec2).norm();
}

// Function to compute Euclidean distance for each row in a matrix
Eigen::VectorXd euclidean_distance_matrix(const Eigen::MatrixXd& mat, const Eigen::VectorXd& vec) {
    Eigen::VectorXd distances(mat.rows());
    Eigen::RowVectorXd vec_row = vec.transpose();
    #pragma omp parallel for
    for (int i = 0; i < mat.rows(); ++i) {
        distances(i) = (mat.row(i) - vec_row).norm();
    }
    return distances;
}

// Function to find top k indices
std::vector<int> argsort_topk(const Eigen::VectorXd& vec, int k) {
    std::vector<int> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (k >= vec.size()) {
        std::sort(indices.begin(), indices.end(), [&vec](int a, int b) { return vec(a) < vec(b); });
    } else {
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&vec](int a, int b) { return vec(a) < vec(b); });
        indices.resize(k);
    }

    return indices;
}

// Function for matrix multiplication
Eigen::MatrixXd matmul(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2) {
    return mat1 * mat2;
}

// Function to convert numpy array to Matrix wrapper class
Matrix to_matrix(const py::array_t<double>& array) {
    py::buffer_info info = array.request();
    if (info.ndim != 2) {
        throw std::runtime_error("Input should be a 2D numpy array");
    }
    std::vector<double> data(info.size);
    std::memcpy(data.data(), info.ptr, info.size * sizeof(double));
    return Matrix(data, info.shape[0], info.shape[1]);
}

// Optimized function for dot product (vector and matrix)
py::array_t<double> dot_vector_matrix(const py::array_t<double>& vec, const py::array_t<double>& mat) {
    py::buffer_info vec_info = vec.request();
    py::buffer_info mat_info = mat.request();

    if (vec_info.ndim != 1 || mat_info.ndim != 2) {
        throw std::runtime_error("Invalid dimensions for dot product");
    }

    size_t vec_size = vec_info.shape[0];
    size_t mat_rows = mat_info.shape[0];
    size_t mat_cols = mat_info.shape[1];

    if (vec_size != mat_cols) {
        throw std::invalid_argument("Vector size must match matrix columns for multiplication");
    }

    std::vector<double> vec_data(vec_size);
    std::memcpy(vec_data.data(), vec_info.ptr, vec_size * sizeof(double));

    Matrix matrix = to_matrix(mat);

    std::vector<double> result_data(mat_rows);
    #pragma omp parallel for
    for (size_t i = 0; i < mat_rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < vec_size; ++j) {
            sum += vec_data[j] * matrix(i, j);
        }
        result_data[i] = sum;
    }

    py::array_t<double> result(mat_rows);
    std::memcpy(result.mutable_data(), result_data.data(), mat_rows * sizeof(double));

    return result;
}

// Optimized function for dot product (vector and vector)
double dot_vector_vector(const py::array_t<double>& vec1, const py::array_t<double>& vec2) {
    py::buffer_info vec1_info = vec1.request();
    py::buffer_info vec2_info = vec2.request();

    if (vec1_info.ndim != 1 || vec2_info.ndim != 1) {
        throw std::runtime_error("Both inputs should be 1D numpy arrays");
    }

    size_t size1 = vec1_info.shape[0];
    size_t size2 = vec2_info.shape[0];

    if (size1 != size2) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }

    const double* data1 = static_cast<const double*>(vec1_info.ptr);
    const double* data2 = static_cast<const double*>(vec2_info.ptr);

    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < size1; ++i) {
        result += data1[i] * data2[i];
    }

    return result;
}

// Wrapper function to handle different types of dot product
py::object dot(const py::object& a, const py::object& b) {
    if (py::isinstance<py::array_t<double>>(a) && py::isinstance<py::array_t<double>>(b)) {
        auto vec = a.cast<py::array_t<double>>();
        auto mat = b.cast<py::array_t<double>>();

        if (vec.ndim() == 1 && mat.ndim() == 1) {
            return py::float_(dot_vector_vector(vec, mat));
        } else if (vec.ndim() == 1 && mat.ndim() == 2) {
            return dot_vector_matrix(vec, mat);
        } else {
            throw std::runtime_error("Unsupported dimensions for dot product");
        }
    } else {
        throw std::runtime_error("Unsupported types for dot product");
    }
}

// Function for matrix-matrix dot product
Eigen::MatrixXd dot_matrix(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2) {
    if (mat1.cols() != mat2.rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    return mat1 * mat2;
}

PYBIND11_MODULE(engines, m) {
    m.def("to_normalize", (Eigen::VectorXd (*)(const Eigen::VectorXd&)) &to_normalize, "Normalize a vector");
    m.def("to_normalize_matrix", &to_normalize_matrix, "Normalize a matrix row-wise");
    m.def("cosine_distance", &cosine_distance, "Compute cosine distance");
    m.def("euclidean_distance", (double (*)(const Eigen::VectorXd&, const Eigen::VectorXd&)) &euclidean_distance, "Compute Euclidean distance between two vectors");
    m.def("euclidean_distance_matrix", &euclidean_distance_matrix, "Compute Euclidean distance for each row in a matrix");
    m.def("argsort_topk", &argsort_topk, "Argsort the array and return the top k indices");
    m.def("matmul", &matmul, "Matrix multiplication");
    m.def("dot", &dot, "Optimized dot product between numpy arrays");
    m.def("dot_matrix", &dot_matrix, "Dot product between two matrices");
}
