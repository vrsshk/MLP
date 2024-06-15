#include "network.h"


struct data_info {
	int digit;
	std::vector<double> pixels;
};

/**
 * Подсчет количества строк в CSV файле
 */
int RowsInCSV(const std::string& path) {
	std::ifstream fin(path);
	int rowCount = 0;
	std::string line;
	while (std::getline(fin, line, '\n')) {
		++rowCount;
	}
	return rowCount;
}

/**
 * Чтение конфигурационного файла
 */
data_network ReadDataNetwork(std::string path) {
	std::ifstream fin(path);
	if (!fin.is_open()) {
		std::cout << "Error opening file\n" << path << "\n";
		    std::cout << "Press enter to continue...";
    std::cin.get(); // wait for the user to press enter
		return {};
	}

	std::cout << path << " loading...\n";

	data_network data{};
	std::string tmp;
	while (fin >> tmp) {
		if (tmp == "Network") {
			fin >> data.L;
			data.size.resize(data.L);
			for (int i = 0; i < data.L; ++i) {
				fin >> data.size[i];
			}
		}
	}
	fin.close();
	return data;
}

/**
 * Чтения данных
 */
std::vector<data_info> ReadData(std::string path, const data_network& data_NW) {
	std::ifstream fin(path);
	if (!fin.is_open()) {
		std::cout << "Error opening file\n" << path << "\n";
		    std::cout << "Press enter to continue...";
    std::cin.get(); // wait for the user to press enter
		return {}; 
	}

	std::cout << path << " loading...\n";

	std::vector<data_info> data;
	std::string line;
	while (std::getline(fin, line)) {
		data_info info;
		std::istringstream iss(line);
		iss >> info.digit;
		info.pixels.resize(784); 
		for (int j = 0; j < 784; ++j) {
			std::string pixel_str;
			iss >> pixel_str;
			info.pixels[j] = std::stod(pixel_str);
		}
		data.push_back(info);
	}
	fin.close();
	return data;
}

int main()
{
	
	Network NW{};
	data_network NW_config;
	std::vector<data_info> data;
	double ra = 0, maxra = 0;
	int right, predict;

	int epoch = 0;
	bool study, repeat = true;
	std::chrono::duration<double> time;

	NW_config = ReadDataNetwork(std::string(DATA_DIR) + "Config.txt");
	NW = Network(NW_config);
	NW.PrintConfig();

	while (repeat) {
		std::cout << "STUDY? (1/0)\n";
		std::cin >> study;
		if (study) {
			//int examples = RowsInCSV(std::string(DATA_DIR) + "train.csv");
			int examples = 100;
			std::cout << "File size: " << examples << "\n";
			data = ReadData(std::string(DATA_DIR) + "train.csv", NW_config);

			auto begin = std::chrono::steady_clock::now();

			while (ra / examples * 100 < 95) {

				ra = 0;
				auto t1 = std::chrono::steady_clock::now();

				for (int i = 0; i < examples; ++i) {
					NW.SetInput(data[i].pixels);;
					right = data[i].digit;
					predict = NW.ForwardFeed();
					if (predict != right) {
						NW.BackPropogation(right);
						NW.WeightsUpdater(0.15 * exp(-epoch / 20.0));
					}
					else {
						ra++;
					}
				}
				auto t2 = std::chrono::steady_clock::now();
				time = t2 - t2;
				if (ra > maxra) {
					maxra = ra;
				}
				std::cout << "ra: " << ra / examples * 100 << "\t" << "maxra: " << maxra / examples * 100 << "\n";
				epoch++;
				if (epoch == 20) {
					break;
				}
			}
			auto end = std::chrono::steady_clock::now();
			time = end - begin;
			std::cout << "TIME: " << time.count() / 60.0 << " min\n";
			NW.SaveWeights();
		}
		else {
			NW.ReadWeights();
		}
		std::cout << "Test? (1/0)\n";
		bool test_flag;
		std::cin >> test_flag;
		if (test_flag) {
			//int ex_tests = RowsInCSV(std::string(DATA_DIR) + "normalized_test.csv");
			int ex_tests = 100;
			data_info* data_test;
			data = ReadData(std::string(DATA_DIR) + "normalized_test.csv", NW_config);
			ra = 0;
			for (int i = 0; i < ex_tests; ++i) {
				NW.SetInput(data[i].pixels);
				predict = NW.ForwardFeed();
				right = data[i].digit;
				if (predict == right) {
					ra++;
				}
			}
			std::cout << "ra: " << ra / ex_tests * 100 << "\n";
		
		}
		std::cout << "Repeat? (1/0)\n";
		std::cin >> repeat;
	}
	    std::cout << "Press enter to continue...";
    std::cin.get(); // wait for the user to press enter
	return 0;
}