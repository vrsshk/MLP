#pragma once
#include "include.h"

class Matrix {
private:
    size_t row;
    size_t col;
    std::vector<std::vector<double>> matrix;

public:
    Matrix() : row(0), col(0), matrix() {}
    Matrix(size_t row, size_t col);
    size_t getRow() const { return row; }
    size_t getCol() const { return col; }
    double get(size_t i, size_t j) const { return matrix[i][j]; }
    void set(size_t i, size_t j, double value);

    void Rand();
    static std::vector<double> MultVec(const Matrix& m, const std::vector<double>& b);
    static std::vector<double> SumVec(const std::vector<double>& a, const std::vector<double>& b);
    Matrix transpose() const;
    friend std::ostream& operator <<(std::ostream& os, const Matrix& m);
    friend std::istream& operator >>(std::istream& is, Matrix& m);
};

