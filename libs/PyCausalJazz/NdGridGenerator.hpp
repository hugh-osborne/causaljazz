#ifndef GRID_PYTHON_HPP
#define GRID_PYTHON_HPP

#include "pyhelper.h"
#include "causaljazz/NdGrid.hpp"
#include <string>
#include <vector>
#include <map>

class NdGridGenerator {
public:
    std::string function_file_name;
    std::string function_name;

    CPyObject python_func;

    NdGridGenerator(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res,
        double _threshold_v, double _reset_v, std::vector<double> _reset_jump_relative, double _timestep);
    ~NdGridGenerator();

    void setPythonFunctionFromStrings(std::string function, std::string functionname);
    void setPythonFunction(PyObject* function);

    std::map<std::vector<unsigned int>, std::vector<double>> calculateTransitionMatrix();

    NdGrid* getGrid() { return grid; }
private:

    NdGrid* grid;
};

#endif