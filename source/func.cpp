#include "func.h"

void Func::Set() {
	std::cout << "Set activation function:\n 1 - sigmoid,\n 2 - th(x),\n 3 - ReLU.\n";
	int t;
	std::cin >> t;
	if (t == 1) {
		this->f = sigmoid;
	}
	else if (t == 2) {
		this->f = thx;
	}
	else if (t == 3) {
		this->f = RelU;
	}
	else {
		std::cerr << "Wrong choice. By default, it is selected sigmoid." << std::endl;
		this->f = sigmoid;
	}
}

void Func::Use(std::vector<double>& a) {
	if (f == activation_function::sigmoid) {
		for (size_t i = 0; i != a.size(); ++i) {
			a[i] = 1 / (1 + exp(-a[i]));
		}
	}
	else if (f == activation_function::thx) {
		for (size_t i = 0; i != a.size(); ++i) {
			if (a[i] >= 0) {
				a[i] = (exp(a[i]) - exp(-a[i])) / (exp(a[i]) + exp(-a[i]));
			}
			else {
				a[i] = 0.01 * (exp(a[i]) - exp(-a[i])) / (exp(a[i]) + exp(-a[i]));
			}
		}
	}
	else if (f == activation_function::RelU) {
		for (size_t i = 0; i != a.size(); ++i) {
			if (a[i] < 0) {
				a[i] *= 0.01;
			}
			else if (a[i] > 1) {
				a[i] = 1.0 + 0.01 * (a[i] - 1.0);
			}
		}
	}
}

void Func::UseDerivative(std::vector<double>& a) {
	if (f == activation_function::sigmoid) {
		for (size_t i = 0; i != a.size(); ++i) {
			a[i] = a[i] * (1 - a[i]);
		}
	}
	else if (f == activation_function::thx) {
		for (size_t i = 0; i != a.size(); ++i) {
			if (a[i] < 0) {
				a[i] = 0.01 * (1 - a[i] * a[i]);
			}
			else {
				a[i] = (1 - a[i] * a[i]);
			}
		}
	}
	else if (f == activation_function::RelU) {
		for (size_t i = 0; i != a.size(); ++i) {
			if (a[i] < 0 || a[i] >1) {
				a[i] = 0.01;
			}
			else {
				a[i] = 1.0;
			}
		}
	}
}

double Func::UseDerivative(double value) {
	if (f == activation_function::sigmoid) {
		value = value * (1 - value);
	}
	else if (f == activation_function::thx) {
		if (value < 0) {
			value = 0.01 * (1 - value * value);
		}
		else {
			value = (1 - value * value);
		}
	}
	else if (f == activation_function::RelU) {
		if (value < 0 || value >1) {
			value = 0.01;
		}
		else {
			value = 1.0;
		}
	}
	return(value);
}