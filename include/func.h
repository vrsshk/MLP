#pragma once 
#include "include.h"

enum activation_function { sigmoid, thx, RelU }; 
class Func 
{
private: 
	activation_function f;
public: 
	void Set();
	void Use(std::vector<double>& a);
	void UseDerivative(std::vector<double>& a);
	double UseDerivative(double value);
}; 
