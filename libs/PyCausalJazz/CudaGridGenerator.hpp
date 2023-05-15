#ifndef CUDA_GRID_GEN_PYTHON_HPP
#define CUDA_GRID_GEN_PYTHON_HPP

#include "pyhelper.h"
#include "causaljazz/CudaGrid.cuh"
#include <string>
#include <vector>
#include <map>

class CudaGridGenerator {
public:
    std::string function_file_name;
    std::string function_name;

    CPyObject python_func;

    CudaGridGenerator();
    ~CudaGridGenerator();

    void setPythonFunctionFromStrings(std::string function, std::string functionname);
    void setPythonFunction(PyObject* function);

    CudaGrid generateCudaGridFromFunction(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res, unsigned int output_res);
    CudaGrid generateCudaGridFromFunction(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res, double output_base, double output_size, unsigned int output_res);

};

#endif