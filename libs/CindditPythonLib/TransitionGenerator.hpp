#ifndef GRID_PYTHON_HPP
#define GRID_PYTHON_HPP

#include "pyhelper.h"
#include "graph/Transition.hpp"
#include <string>
#include <vector>
#include <map>

class TransitionGenerator {
public:
    std::string function_file_name;
    std::string function_name;

    CPyObject python_func;

    TransitionGenerator(Discretised1DSpace _out);
    TransitionGenerator(unsigned int _out_res);
    ~TransitionGenerator();

    void setPythonFunctionFromStrings(std::string function, std::string functionname);
    void setPythonFunction(PyObject* function);

    unsigned int addInputDistribution(Discretised1DSpace& _in);
    void setInputDistribution(unsigned int id, Discretised1DSpace& _in);

    void calculateTransitionMatrix();

    Transition* getGrid() { return grid; }
private:

    unsigned int desired_out_resolution;
    Discretised1DSpace out_dist;

    std::vector< Discretised1DSpace> in_dists;

    Transition* grid;
};

#endif