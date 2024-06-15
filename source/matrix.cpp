#include "matrix.h"

Matrix::Matrix(size_t row, size_t col)
    : row(row), col(col), matrix(row, std::vector<double>(col, 0.0)) {}

void Matrix::set(size_t i, size_t j, double value) {
    if (i >= row || j >= col) {
        throw std::out_of_range("Индекс вне диапазона");
    }
    matrix[i][j] = value;
}

void Matrix::Rand() {
    srand(static_cast<unsigned int>(time(0)));

    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < col; ++j) {
            matrix[i][j] = (rand() % 100) * 0.03;
        }
    }
}

std::vector<double> Matrix::MultVec(const Matrix& m, const std::vector<double>& b) {
    if (b.size() != m.col) {
        throw std::invalid_argument("Vector size must match matrix column size");
    }

    std::vector<double> c(m.row);
    for (size_t i = 0; i < m.row; ++i) {
        double t = 0;
        for (size_t j = 0; j < m.col; ++j) {
            t += m.matrix[i][j] * b[j];
        }
        c[i] = t;
    }
    return c;
}

std::vector<double> Matrix::SumVec(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix transposed(col, row);
    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < col; ++j) {
            transposed.matrix[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    for (size_t i = 0; i < m.row; ++i) {
        for (size_t j = 0; j < m.col; ++j) {
            os << m.matrix[i][j] << " ";
        }
        os << std::endl;
    }
    return os;
}

std::istream& operator >> (std::istream& is, Matrix& m) {
    for (size_t i = 0; i < m.row; ++i) {
        for (size_t j = 0; j < m.col; ++j) {
            is >> m.matrix[i][j];
        }
    }
    return is;
}