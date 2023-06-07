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
            double start_point = _base[0] + ((_dims[0] / _res[0]) * (a));

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
            std::cout << start_point << " -> " << results[a] << "\n";

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

        for (unsigned int a = 0; a < _res[0]-1; a++) {
            // Each point gets flattened to two cells in the out distribution
            double hi_value = results[a + 1];
            double lo_value = results[a];

            if (results[a] > results[a + 1]) {
                hi_value = results[a];
                lo_value = results[a + 1];
            }

            double hi_shifted = (hi_value - out_base) / out_cell_width;
            double lo_shifted = (lo_value - out_base) / out_cell_width;

            int hi_out_cell = int(hi_shifted);
            int lo_out_cell = int(lo_shifted);

            double hi_out_prop = hi_shifted - hi_out_cell;
            double lo_out_prop = 1.0 - hi_out_prop;

            if (hi_out_cell >= (int)output_res)
                hi_out_cell = (int)output_res - 1;
            if (lo_out_cell >= (int)output_res)
                lo_out_cell = (int)output_res - 1;
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
                start_point[0] = _base[0] + ((_dims[0] / _res[0]) * (a));
                start_point[1] = _base[1] + ((_dims[1] / _res[1]) * (b));

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

        srand((unsigned)time(NULL));

        for (unsigned int b = 0; b < _res[1]-1; b++) {
            for (unsigned int a = 0; a < _res[0] - 1; a++) {

                std::vector<unsigned int> counter(output_res);

                unsigned int num_points = 10;
                for (unsigned int n = 0; n < num_points; n++) {
                    std::vector<double> start_point(2);

                    start_point[0] = _base[0] + ((_dims[0] / _res[0]) * (a)) + ((_dims[0] / _res[0])*((double)rand() / (double)RAND_MAX));
                    start_point[1] = _base[1] + ((_dims[1] / _res[1]) * (b)) + ((_dims[1] / _res[1]) * ((double)rand() / (double)RAND_MAX));

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

                    unsigned int out_cell = int((out_value - out_base) / out_cell_width);

                    counter[out_cell]++;
                }

                for (unsigned int n = 0; n < output_res; n++) {
                    if (counter[n] > 0) {
                        unsigned int cell_id = a + (_res[0] * b) + (n * _res[0] * _res[1]);
                        conditional_flattened[cell_id] = (double)counter[n] / (double)num_points;
                    }
                    
                }
            }
        }

        _base.push_back(out_base);
        _dims.push_back(out_dim);
        _res.push_back(output_res);

        return CudaGrid(_base, _dims, _res, conditional_flattened);
    }
}

CudaGrid CudaGridGenerator::generateCudaGridFromFunction(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res, double output_base, double output_size, unsigned int output_res) {
    std::vector<double> conditional;

    if (_base.size() == 1) { // 1D input
        std::vector<double> results(_res[0]);
        for (unsigned int a = 0; a < _res[0]; a++) {
            double start_point = _base[0] + ((_dims[0] / _res[0]) * (a));

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
            std::cout << start_point << " -> " << results[a] << "\n";
        }

        double out_base = output_base;
        double out_dim = output_size;
        double out_cell_width = out_dim / output_res;

        std::vector<double> conditional_flattened(_res[0] * output_res);

        for (unsigned int a = 0; a < _res[0]-1; a++) {
            // Each point gets flattened to two cells in the out distribution
            
            double hi_value = results[a+1];
            double lo_value = results[a];

            if (results[a] > results[a + 1]) {
                hi_value = results[a];
                lo_value = results[a + 1];
            }

            double hi_shifted = (hi_value - out_base) / out_cell_width;
            double lo_shifted = (lo_value - out_base) / out_cell_width;

            int hi_out_cell = int(hi_shifted);
            int lo_out_cell = int(lo_shifted);

            double hi_out_prop = hi_shifted - hi_out_cell;
            double lo_out_prop = 1.0 - hi_out_prop;

            if (hi_out_cell >= (int)output_res)
                hi_out_cell = (int)output_res - 1;
            if (lo_out_cell >= (int)output_res)
                lo_out_cell = (int)output_res - 1;
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
                start_point[0] = _base[0] + ((_dims[0] / _res[0]) * (a));
                start_point[1] = _base[1] + ((_dims[1] / _res[1]) * (b));

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
            }
            results[b] = resrow;
        }

        double out_base = output_base;
        double out_dim = output_size;
        double out_cell_width = out_dim / output_res;

        std::vector<double> conditional_flattened(_res[0] * _res[1] * output_res);

        for (unsigned int b = 0; b < _res[1]-1; b++) {
            for (unsigned int a = 0; a < _res[0]-1; a++) {
                // Each point gets flattened to two cells in the out distribution

                double val_max = results[b][a];
                double val_min = results[b][a];

                if (val_max < results[b + 1][a])
                    val_max = results[b + 1][a];

                if (val_max < results[b][a + 1])
                    val_max = results[b][a + 1];

                if (val_max < results[b + 1][a + 1])
                    val_max = results[b + 1][a + 1];

                if (val_min > results[b + 1][a])
                    val_min = results[b + 1][a];

                if (val_min > results[b][a + 1])
                    val_min = results[b][a + 1];

                if (val_min > results[b + 1][a + 1])
                    val_min = results[b + 1][a + 1];

                double hi_value = val_max;
                double lo_value = val_min;

                double hi_shifted = (hi_value - out_base) / out_cell_width;
                double lo_shifted = (lo_value - out_base) / out_cell_width;

                int hi_out_cell = int(hi_shifted);
                int lo_out_cell = int(lo_shifted);

                if (hi_out_cell == lo_out_cell) {
                    unsigned int hi_cell_id = a + (_res[0] * b) + (hi_out_cell * _res[0] * _res[1]);
                    conditional_flattened[hi_cell_id] = 1.0;
                }
                else {
                    unsigned int desired_out_cell = lo_out_cell;

                    double prop = (((lo_out_cell + 1) * out_cell_width) + out_base - val_min) / (val_max - val_min);
                    if (desired_out_cell >= (int)output_res)
                        desired_out_cell = (int)output_res - 1;
                    if (desired_out_cell < 0)
                        desired_out_cell = 0;
                    unsigned int lo_cell_id = a + (_res[0] * b) + (desired_out_cell * _res[0] * _res[1]);
                    conditional_flattened[lo_cell_id] += prop;

                    for (unsigned int c = lo_out_cell + 1; c < hi_out_cell; c++) {
                        desired_out_cell = c;
                        if (desired_out_cell >= (int)output_res)
                            desired_out_cell = (int)output_res - 1;
                        if (desired_out_cell < 0)
                            desired_out_cell = 0;
                        unsigned int lo_cell_id = a + (_res[0] * b) + (desired_out_cell * _res[0] * _res[1]);
                        conditional_flattened[lo_cell_id] += out_cell_width / (val_max - val_min);
                    }

                    desired_out_cell = hi_out_cell;
                    prop = (val_max - (hi_out_cell * out_cell_width) - out_base) / (val_max - val_min);
                    if (desired_out_cell >= (int)output_res)
                        desired_out_cell = (int)output_res - 1;
                    if (desired_out_cell < 0)
                        desired_out_cell = 0;
                    unsigned int hi_cell_id = a + (_res[0] * b) + (desired_out_cell * _res[0] * _res[1]);
                    conditional_flattened[hi_cell_id] += prop;
                }
            }
        }

        _base.push_back(out_base);
        _dims.push_back(out_dim);
        _res.push_back(output_res);

        return CudaGrid(_base, _dims, _res, conditional_flattened);
    }
}