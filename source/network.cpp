#include "network.h"

Network::Network(const data_network& data) : L(data.L), size(data.size) {
    f.Set();
    srand(time(NULL));
    weights.resize(L - 1);
    bios.resize(L - 1);
    neurons_bios_val.resize(L - 1);
    neurons_val.resize(L);
    neurons_err.resize(L);
    for (int i = 0; i < L - 1; ++i) {
        bios[i].resize(size[i + 1]);
        weights[i] = Matrix(size[i+1], size[i]);
        for (int j = 0; j < size[i + 1]; ++j) {
            bios[i][j] = (rand() % 50) * 0.06;
        }
        weights[i].Rand();
    }
    for (int i = 0; i < L - 1; ++i) {
        neurons_bios_val[i] = 1;
    }
    for (int i = 0; i != L; ++i) {
        neurons_val[i].resize(size[i]);
        neurons_err[i].resize(size[i]);
    }
}

void Network::PrintConfig() {
    std::cout << "\n********************************************\n";
    std::cout << "Network has " << L << " layers.\n";
    std::cout << "Size: \n";
    for (size_t i = 0; i != L; ++i) {
        std::cout << size[i] << " ";    
    }
    std::cout << "\n********************************************\n";
}

void Network::SetInput(const std::vector<double>& values) {
    if (neurons_val[0].size() < values.size()) {
        std::cout << "Error: neurons_val[0] is too small to hold input values\n";
        return;
    }
    for (int i = 0; i != values.size(); ++i) {
        neurons_val[0][i] = values[i];
    }
}
double Network::ForwardFeed() {
    for (int k = 1; k < L; ++k) {
        neurons_val[k] = Matrix::MultVec(weights[k - 1], neurons_val[k - 1]);
        neurons_val[k] = Matrix::SumVec(neurons_val[k], bios[k - 1]);
        f.Use(neurons_val[k]);
    }
    int pred = SearchMaxIndex(neurons_val[L - 1]);
    return pred;
}

int Network::SearchMaxIndex(const std::vector<double>& value) {
    double max = value[0];
    int prediction = 0;
    double tmp;
    for (int j = 1; j < size[L - 1]; j++) {
        tmp = value[j];
        if (tmp > max) {
            prediction = j;
            max = tmp;
        }
    }
    return prediction;
}

void Network::PrintValues(int L) {
    for (int j = 0; j != size[L]; j++) {
        std::cout << j << " " << neurons_val[L][j] << "\n";
    }
}
/*
void Network::BackPropogation(double expect) {
    for (int i = 0; i < size[L - 1]; i++) {
        if (i != int(expect)) {
            neurons_err[L - 1][i] = -1.0 * neurons_val[L - 1][i] * f.UseDerivative(neurons_val[L - 1][i]);
        }
        else {
            neurons_err[L - 1][i] = (1.0 - neurons_val[L - 1][i]) * f.UseDerivative(neurons_val[L - 1][i]);
        }
    }
    for (int k = L - 2; k > 0; k--) {
        neurons_err[k] = Matrix::MultVec(weights[k].transpose(), neurons_err[k + 1]);
        for (int j = 0; j != size[k]; j++) {
            neurons_err[k][j] *= f.UseDerivative(neurons_val[k][j]);
        }
    }
}
*/
void Network::WeightsUpdater(double lr) {
    for (int i = 0; i < L - 1; ++i) {
        for (int j = 0; j < size[i + 1]; ++j) {
            for (int k = 0; k < size[i]; ++k) {
                if (j < weights[i].getCol() && k < weights[i].getRow()) {
                    double temp = weights[i].get(j, k) + (neurons_val[i][k] * neurons_err[i + 1][j] * lr);
                    weights[i].set(j, k, temp);
                }
            }
        }
    }
    for (int i = 0; i < L - 1; ++i) {
        for (int k = 0; k < size[i]; ++k) {
            if (k < bios[i].size()) {
                bios[i][k] += neurons_err[i + 1][k] * lr;
            }
        }
    }
}

void Network::BackPropogation(double expect) {
    // ������������ ������ ��� ���������� ����
    for (int i = 0; i < size[L - 1]; i++) {
        if (i != int(expect)) {
            neurons_err[L - 1][i] = -1.0 * neurons_val[L - 1][i] * f.UseDerivative(neurons_val[L - 1][i]);
        }
        else {
            neurons_err[L - 1][i] = (1.0 - neurons_val[L - 1][i]) * f.UseDerivative(neurons_val[L - 1][i]);
        }
    }

    // ������������ ������ ��� ��������� �����
    for (int k = L - 2; k >= 0; k--) {
        neurons_err[k] = Matrix::MultVec(weights[k].transpose(), neurons_err[k + 1]);
        for (int j = 0; j < size[k]; j++) {
            neurons_err[k][j] *= f.UseDerivative(neurons_val[k][j]);
        }
    }

}
void Network::SaveWeights() {
    std::ofstream fout;
    fout.open(std::string(DATA_DIR) + "Weights.txt");
    if (!fout.is_open()) {
        std::cout << "Error opening file\n";
            std::cout << "Press enter to continue...";
    std::cin.get(); // wait for the user to press enter
    } 
    for (int i = 0; i < L - 1; ++i) {
        for (int j = 0; j < size[i + 1]; ++j) {
            fout << bios[i][j] << " ";
        }
    }
    std::cout << "Weights saved \n";
    fout.close();
}

void Network::ReadWeights() {
    std::ifstream fin;
    fin.open(std::string(DATA_DIR) + "Weights.txt");
    if (!fin.is_open()) {
        std::cout << "Error opening file\n";
            std::cout << "Press enter to continue...";
    std::cin.get(); // wait for the user to press enter
    }
    for (int i = 0; i < L - 1; ++i) {
        fin >> weights[i];
    }
    for (int i = 0; i < L - 1; ++i) {
        for (int j = 0; j < size[i + 1]; ++j) {
            fin >> bios[i][j];
        }
    }
    std::cout << "Weights saved \n";
    fin.close();
}
