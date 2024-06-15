#pragma once
#include "matrix.h"
#include "func.h"
#include "include.h"

struct data_network {
    int L;
    std::vector<int> size;
};

class Network {
private:
    int L;
    std::vector<int> size;
public:
    Func f;
    std::vector<Matrix> weights;
    std::vector<std::vector<double>> bios;
    std::vector<std::vector<double>> neurons_val, neurons_err;
    std::vector<double> neurons_bios_val;
    Network() : L(0), size() {}
    Network(const data_network& data);

    void PrintConfig();
    void SetInput(const std::vector<double>& values);
    double ForwardFeed();
    int SearchMaxIndex(const std::vector<double>& value);
    void PrintValues(int layers);
    void BackPropogation(double expect);
    void WeightsUpdater(double lr);
    void SaveWeights();
    void ReadWeights();
};

