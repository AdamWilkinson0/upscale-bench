#pragma once
#include <string>
#include "tensor.h"

enum CoreMLUnits { UNITS_CPU = 0, UNITS_CPU_GPU = 1, UNITS_CPU_NE = 2, UNITS_ALL = 3 };

struct CoreMLModel;

const char* units_name(int units);
std::string coreml_compile(const std::string& mlpackage, double* ms);
CoreMLModel* coreml_load(const std::string& compiled, int units, double* ms);
bool coreml_run(CoreMLModel* m, const Tensor& in, Tensor& out);
void coreml_destroy(CoreMLModel* m);
