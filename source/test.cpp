#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "matrix.h"
#include "include.h"
#include "func.h"


TEST_CASE("Установка и получение элементов матрицы") {
    Matrix m(2, 2);
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(1, 0, 3.0);
    m.set(1, 1, 4.0);
    CHECK(m.get(0, 0) == 1.0);
    CHECK(m.get(0, 1) == 2.0);
    CHECK(m.get(1, 0) == 3.0);
    CHECK(m.get(1, 1) == 4.0);
}

TEST_CASE("Умножение матрицы на вектор") {
    Matrix m(2, 2);
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(1, 0, 3.0);
    m.set(1, 1, 4.0);
    std::vector<double> b = {5.0, 6.0};
    std::vector<double> result = Matrix::MultVec(m, b);
    CHECK(result.size() == 2);
    CHECK(result[0] == 17.0);
    CHECK(result[1] == 39.0);
}

TEST_CASE("Сложение векторов") {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    std::vector<double> result = Matrix::SumVec(a, b);
    CHECK(result.size() == 3);
    CHECK(result[0] == 5.0);
    CHECK(result[1] == 7.0);
    CHECK(result[2] == 9.0);
}

TEST_CASE("Транспонирование матрицы") {
    Matrix m(2, 2);
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(1, 0, 3.0);
    m.set(1, 1, 4.0);
    Matrix mt = m.transpose();
    CHECK(mt.getRow() == 2);
    CHECK(mt.getCol() == 2);
    CHECK(mt.get(0, 0) == 1.0);
    CHECK(mt.get(0, 1) == 3.0);
    CHECK(mt.get(1, 0) == 2.0);
    CHECK(mt.get(1, 1) == 4.0);
}

