#include "CudaGridGenerator.hpp"

#include <iostream>

CudaGridGenerator::CudaGridGenerator() {
}

CudaGridGenerator::~CudaGridGenerator() {
    Py_Finalize();
}

void CudaGridGenerator::setPythonFunctionFromStrings(std::string function, std::string functionname) {

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

void CudaGridGenerator::setPythonFunction(PyObject* func) {

    Py_Initialize();

    python_func.setObject(func);
}

// Take a 2D or 1D base, dims, and res plus an output res.
// Generate a 3D or 2D conditional distribution output|input 
CudaGrid CudaGridGenerator::generateCudaGridFromFunction(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res, unsigned int output_res) {
    std::vector<double> conditional;

    // On the way, calculate the min and max values of the output dimension
    double max = 0.0;
    double min = 0.0;

    if (_base.size() == 1) { // 1D input
        std::vector<double> results(_res[0]);
        for (unsigned int a = 0; a < _res[0]; a++) {
            double start_point = _base[0] + ((_dims[0] / _res[0]) * (a + 0.5));

            // Build the list of values for the starting point to be sent to the python function
            PyObject* point_coord_list = PyList_New((Py_ssize_t)(1));

            std::vector<PyObject*> list_objects(1);

            list_objects[0] = PyFloat_FromDouble(start_point);
            PyList_SetItem(point_coord_list, 0, list_objects[0]);

            double out_value = 0;

            // Pass the start point to the python function and get an end point value
            PyObject* tuple = PyList_AsTuple(point_coord_list);
            if (python_func && PyCallable_Check(python_func))
            {
                PyObject* pass = Py_BuildValue("(O)", tuple);
                PyErr_Print();
                PyObject* pValue = PyObject_CallObject(python_func, pass);
                PyErr_Print();
                out_value = PyFloat_AsDouble(pValue);
            }
            else
            {
                std::cout << "ERROR: function.\n";
            }

            results[a] = out_value;

            if (a == 0) {
                max = out_value;
                min = out_value;
            }
            else {
                if (max < out_value)
                    max = out_value;
                if (min > out_value)
                    min = out_value;
            }
        }

        double out_base = min;
        double out_dim = max - min;
        double out_cell_width = out_dim / output_res;

        std::vector<double> conditional_flattened(_res[0] * output_res);

        for (unsigned int a = 0; a < _res[0]; a++) {
            // Each point gets flattened to two cells in the out distribution
            double hi_value = results[a] + (out_cell_width / 2.0);
            double lo_value = results[a] - (out_cell_width / 2.0);

            double hi_shifted = (hi_value - out_base) / out_cell_width;
            double lo_shifted = (lo_value - out_base) / out_cell_width;

            unsigned int hi_out_cell = int(hi_shifted);
            unsigned int lo_out_cell = int(lo_shifted);

            double hi_out_prop = hi_shifted - hi_out_cell;
            double lo_out_prop = 1.0 - hi_out_prop;

            if (hi_out_cell >= output_res)
                hi_out_cell = output_res - 1;
            if (lo_out_cell >= output_res)
                lo_out_cell = output_res - 1;
            if (hi_out_cell < 0)
                hi_out_cell = 0;
            if (lo_out_cell < 0)
                lo_out_cell = 0;

            unsigned int hi_cell_id = a + (hi_out_cell * _res[0]);
            unsigned int lo_cell_id = a + (lo_out_cell * _res[0]);

            conditional_flattened[hi_cell_id] += hi_out_prop;
            conditional_flattened[lo_cell_id] += lo_out_prop;
        }

        _base.push_back(out_base);
        _dims.push_back(out_dim);
        _res.push_back(output_res);

        return CudaGrid(_base, _dims, _res, conditional_flattened);
    }
    else if (_base.size() == 2) { // 2D input
        std::vector<std::vector<double>> results(_res[1]);
        for (unsigned int b = 0; b < _res[1]; b++) {
            std::vector<double> resrow(_res[0]);
            for (unsigned int a = 0; a < _res[0]; a++) {
                std::vector<double> start_point(2);
                start_point[0] = start_point[0] = _base[0] + ((_dims[0] / _res[0]) * (a + 0.5));
                start_point[1] = _base[1] + ((_dims[1] / _res[1]) * (b + 0.5));

                // Build the list of values for the starting point to be sent to the python function
                PyObject* point_coord_list = PyList_New((Py_ssize_t)(2));

                std::vector<PyObject*> list_objects(2);

                for (unsigned int i = 0; i < 2; i++) {
                    list_objects[i] = PyFloat_FromDouble(start_point[i]);
                    PyList_SetItem(point_coord_list, i, list_objects[i]);
                }

                double out_value = 0;

                // Pass the start point to the python function and get an end point value
                PyObject* tuple = PyList_AsTuple(point_coord_list);
                if (python_func && PyCallable_Check(python_func))
                {
                    PyObject* pass = Py_BuildValue("(O)", tuple);
                    PyErr_Print();
                    PyObject* pValue = PyObject_CallObject(python_func, pass);
                    PyErr_Print();
                    out_value = PyFloat_AsDouble(pValue);
                }
                else
                {
                    std::cout << "ERROR: function.\n";
                }

                resrow[a] = out_value;

                if (a == 0 && b == 0) {
                    max = out_value;
                    min = out_value;
                }
                else {
                    if (max < out_value)
                        max = out_value;
                    if (min > out_value)
                        min = out_value;
                }
            }
            results[b] = resrow;
        }

        double out_base = min;
        double out_dim = max - min;
        double out_cell_width = out_dim / output_res;

        std::vector<double> conditional_flattened(_res[0]*_res[1]*output_res);

        for (unsigned int b = 0; b < _res[1]; b++) {
            for (unsigned int a = 0; a < _res[0]; a++) {
                // Each point gets flattened to two cells in the out distribution
                double hi_value = results[b][a] + (out_cell_width / 2.0);
                double lo_value = results[b][a] - (out_cell_width / 2.0);

                double hi_shifted = (hi_value - out_base) / out_cell_width;
                double lo_shifted = (lo_value - out_base) / out_cell_width;

                unsigned int hi_out_cell = int(hi_shifted);
                unsigned int lo_out_cell = int(lo_shifted);

                double hi_out_prop = hi_shifted - hi_out_cell;
                double lo_out_prop = 1.0 - hi_out_prop;

                if (hi_out_cell >= output_res)
                    hi_out_cell = output_res - 1;
                if (lo_out_cell >= output_res)
                    lo_out_cell = output_res - 1;
                if (hi_out_cell < 0)
                    hi_out_cell = 0;
                if (lo_out_cell < 0)
                    lo_out_cell = 0;

                unsigned int hi_cell_id = a + (_res[0] * b) + (hi_out_cell * _res[0] * _res[1]);
                unsigned int lo_cell_id = a + (_res[0] * b) + (lo_out_cell * _res[0] * _res[1]);

                conditional_flattened[hi_cell_id] += hi_out_prop;
                conditional_flattened[lo_cell_id] += lo_out_prop;
            }
        }

        _base.push_back(out_base);
        _dims.push_back(out_dim);
        _res.push_back(output_res);

        return CudaGrid(_base, _dims, _res, conditional_flattened);
    }

    
}