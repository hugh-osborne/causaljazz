#include "NdGridGenerator.hpp"

#include <iostream>

NdGridGenerator::NdGridGenerator(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res,
    double _threshold_v, double _reset_v, std::vector<double> _reset_jump_relative, double _timestep) {
    grid = new TimeVaryingNdGrid(_base, _dims, _res, _threshold_v, _reset_v, _reset_jump_relative, _timestep);
}

NdGridGenerator::~NdGridGenerator() {
    Py_Finalize();
}

void NdGridGenerator::setPythonFunctionFromStrings(std::string function, std::string functionname) {

    function_file_name = function;
    function_name = functionname;

    Py_Initialize();

    CPyObject pName = PyUnicode_FromString(function_file_name.c_str());
    PyErr_Print();
    CPyObject pModule = PyImport_Import(pName);
    PyErr_Print();

    if (pModule)
    {
        python_func = PyObject_GetAttrString(pModule, function_name.c_str());
    }
    else
    {
        std::cout << "ERROR: Python module not imported\n";
    }
}

void NdGridGenerator::setPythonFunction(PyObject* func) {

    Py_Initialize();

    python_func.setObject(func);
}

std::map<std::vector<unsigned int>, std::vector<double>> NdGridGenerator::calculateTransitionMatrix() {
    std::map<std::vector<unsigned int>, std::vector<double>> transitions;

    unsigned int num_cells = 1;
    for (unsigned int d : grid->getRes()) {
        num_cells *= d;
    }

    std::cout << "Generating transition matrix.\n";
    std::cout << "Resolution: " << grid->getRes()[0];
    for (unsigned int i = 1; i < grid->getRes().size(); i++) {
        std::cout << "," << grid->getRes()[i];
    }
    std::cout << "\n";

    std::cout << "Number of cells: " << num_cells << "\n";

    std::cout << "Offsets: " << grid->getResOffsets()[0];
    for (unsigned int i = 1; i < grid->getResOffsets().size(); i++) {
        std::cout << "," << grid->getResOffsets()[i];
    }
    std::cout << "\n";   

    std::cout << "Threshold Cell: " << grid->getThresholdCell() << "\n";
    std::cout << "Reset Cell: " << grid->getResetCell() << "\n";
    std::cout << "Reset Jump Offset: " << grid->getResetJumpOffset() << "\n";
    std::cout << "Timestep: " << grid->getTimestep() << "\n";

    for (int c = 0; c < num_cells; c++) {
        std::vector<unsigned int> coords = grid->getCellCoords(c);

        std::vector<double> start_point(grid->getNumDimensions());
        for (int i = 0; i < grid->getNumDimensions(); i++)
            start_point[i] = grid->getBase()[i] + (coords[i] * grid->getCellWidths()[i]) + (grid->getCellWidths()[i] / 2.0);

        std::vector<double> shift(grid->getNumDimensions());

        PyObject* point_coord_list = PyList_New((Py_ssize_t)grid->getNumDimensions());

        std::vector<PyObject*> list_objects(grid->getNumDimensions());

        for (unsigned int d = 0; d < grid->getNumDimensions(); d++) {
            list_objects[d] = PyFloat_FromDouble(start_point[d]);
            PyList_SetItem(point_coord_list, d, list_objects[d]);
        }

        PyObject* tuple = PyList_AsTuple(point_coord_list);
        if (python_func && PyCallable_Check(python_func))
        {
            PyObject* pass = Py_BuildValue("(O)", tuple);
            PyErr_Print();
            PyObject* pValue = PyObject_CallObject(python_func, pass);
            PyErr_Print();
            for (unsigned int d = 0; d < grid->getNumDimensions(); d++) {
                shift[d] = grid->getTimestep() * PyFloat_AsDouble(PyList_GetItem(pValue, d));
            }
            // Extra threshold logic - if the shift puts the point above the threshold, 
            // set the shift to put the location at the threshold
            if (start_point[0] + shift[0] > grid->getThreshold())
                shift[0] = grid->getThreshold() - start_point[0];
        }
        else
        {
            std::cout << "ERROR: function.\n";
        }

        std::vector<double> transition_data;

        for (int d = 0; d < grid->getNumDimensions(); d++) {
            double off = int(abs(shift[d]) / grid->getCellWidths()[d]);
            double rem = abs(shift[d]) - (off * grid->getCellWidths()[d]);
            double offp = off + 1;

            if (shift[d] < 0) {
                off = -off;
                offp = off - 1;
            }

            transition_data.push_back(off);
            transition_data.push_back(offp);
            transition_data.push_back(rem / grid->getCellWidths()[d]);
        }

        transitions[coords] = transition_data;
    }

    grid->setTransitionMatrix(transitions);

    return transitions;
}