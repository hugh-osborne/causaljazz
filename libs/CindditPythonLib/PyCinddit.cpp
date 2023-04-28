#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <iostream>
#include "TransitionGenerator.hpp"
#include "graph/MassGraph.cuh"

std::vector<TransitionGenerator*> grid_gens;
MassGraph* graph;

void ParseNewFunctionArguments(PyObject* args) {
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    PyObject* python_function;
    unsigned int out_res;

    // First argument is the python function
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyCallable_Check(temp_p) == 1) {
        python_function = temp_p;
        i++;
    }

    // Second argument is the resolution of the output grid
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Long(temp_p);
        out_res = (int)PyLong_AsLong(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    grid_gens.push_back(new TransitionGenerator(out_res));

    grid_gens[grid_gens.size()-1]->setPythonFunction(python_function);
}


PyObject* fastgraph_addfunctionwithoutput(PyObject* self, PyObject* args)
{
    try {
        ParseNewFunctionArguments(args);

        return Py_BuildValue("i", grid_gens.size() - 1);
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

void ParseNewFunctionWithOutputArguments(PyObject* args) {
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    PyObject* python_function;
    double out_min;
    double out_max;
    unsigned int out_res;

    // First argument is the python function
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyCallable_Check(temp_p) == 1) {
        python_function = temp_p;
        i++;
    }

    // Second argument is the min output value
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        out_min = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Third argument is the range of output values
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        out_max = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Fourth argument is the resolution of the output grid
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Long(temp_p);
        out_res = (int)PyLong_AsLong(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    grid_gens.push_back(new TransitionGenerator(Discretised1DSpace(out_min, out_max, out_res)));

    grid_gens[grid_gens.size() - 1]->setPythonFunction(python_function);
}

PyObject* fastgraph_addfunction(PyObject* self, PyObject* args)
{
    try {
        ParseNewFunctionWithOutputArguments(args);

        return Py_BuildValue("i", grid_gens.size() - 1);
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

unsigned int ParseNewInputArguments(PyObject* args) {
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int function_id;
    double in_min;
    double in_max;
    unsigned int in_res;

    // First argument is the function id
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return NULL; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Long(temp_p);
        function_id = (int)PyLong_AsLong(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Second argument is the min input value
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return NULL; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        in_min = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Third argument is the range of input values
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return NULL; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        in_max = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Fourth argument is the resolution of the input grid
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return NULL; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Long(temp_p);
        in_res = (int)PyLong_AsLong(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    return grid_gens[function_id]->addInputDistribution(Discretised1DSpace(in_min, in_max, in_res));
}


PyObject* fastgraph_addinput(PyObject* self, PyObject* args)
{
    try {
        unsigned int id = ParseNewInputArguments(args);

        return Py_BuildValue("i", id);
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

unsigned int ParseNewInputWithFunctionArguments(PyObject* args) {
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int function_id;
    unsigned int function_out_id;

    // First argument is the function id
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return NULL; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Long(temp_p);
        function_id = (int)PyLong_AsLong(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Second argument is the output function id
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return NULL; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Long(temp_p);
        function_out_id = (int)PyLong_AsLong(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    double min = grid_gens[function_out_id]->getGrid()->getOutDistribution().getMin();
    double max = grid_gens[function_out_id]->getGrid()->getOutDistribution().getMax();
    unsigned int res = grid_gens[function_out_id]->getGrid()->getOutDistribution().getRes();
    Discretised1DSpace s = Discretised1DSpace(min, max, res);
    return grid_gens[function_id]->addInputDistribution(s);
}

PyObject* fastgraph_addinputfromoutput(PyObject* self, PyObject* args)
{
    try {
        unsigned int id = ParseNewInputWithFunctionArguments(args);

        return Py_BuildValue("i", id);
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

unsigned int ParseSetInputWithFunctionArguments(PyObject* args) {
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int function_id;
    unsigned int input_id;
    unsigned int function_out_id;

    // First argument is the function id
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return NULL; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Long(temp_p);
        function_id = (int)PyLong_AsLong(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Second argument is the function input id
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return NULL; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Long(temp_p);
        input_id = (int)PyLong_AsLong(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Second argument is the output function id
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return NULL; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Long(temp_p);
        function_out_id = (int)PyLong_AsLong(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    double min = grid_gens[function_out_id]->getGrid()->getOutDistribution().getMin();
    double max = grid_gens[function_out_id]->getGrid()->getOutDistribution().getMax();
    unsigned int res = grid_gens[function_out_id]->getGrid()->getOutDistribution().getRes();

    Discretised1DSpace s = Discretised1DSpace(min, max, res);

    grid_gens[function_id]->setInputDistribution(input_id, s);

}

PyObject* fastgraph_discretiseinputfromoutput(PyObject* self, PyObject* args)
{
    try {
        ParseSetInputWithFunctionArguments(args);

        Py_RETURN_NONE;
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

PyObject* fastgraph_generate(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int function_id;

    try {

        // First argument is the function id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            function_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        grid_gens[function_id]->calculateTransitionMatrix();
        if (!graph)
            graph = new MassGraph();

        graph->addFunctionGrid(grid_gens[function_id]->getGrid());
        Py_RETURN_NONE;
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

PyObject* fastgraph_getfunctionoutputmax(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int function_id;

    try {

        // First argument is the function id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            function_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        return Py_BuildValue("d", grid_gens[function_id]->getGrid()->getOutDistribution().getMax());
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

PyObject* fastgraph_getfunctionoutputmin(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int function_id;

    try {

        // First argument is the function id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            function_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        return Py_BuildValue("d", grid_gens[function_id]->getGrid()->getOutDistribution().getMin());
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}


PyObject* fastgraph_setinput(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int function_id;
    unsigned int input_id;

    try {
        
        // First argument is the function id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            function_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // First argument is the function id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            input_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Third argument is the discretised distribution
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);

        if (pr_length < 0)
            return NULL;

        std::vector<double> d_distribution(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                d_distribution[index] = 0.0;
            else
                d_distribution[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        std::vector<fptype> dist(pr_length);
        for (unsigned int i = 0; i < pr_length; i++)
            dist[i] = (fptype)d_distribution[i];

        graph->setInputDistribution(function_id, input_id, dist);

        Py_RETURN_NONE;
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

PyObject* fastgraph_applyfunction(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int function_id;

    try {
        // First argument is the function id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            function_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        graph->applyFunction(function_id);

        Py_RETURN_NONE;
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

PyObject* fastgraph_moveoutputtoinput(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int output_function_id;
    unsigned int input_function_id;
    unsigned int input_id;

    try {
        // First argument is the output function id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            output_function_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // First argument is the input function id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            input_function_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // First argument is the input id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            input_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        graph->moveOutputDistributionToInput(output_function_id, input_function_id, input_id);

        Py_RETURN_NONE;
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}


PyObject* fastgraph_readoutput(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int output_function_id;

    try {
        // First argument is the output function id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            output_function_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        graph->readOutputDistribution(output_function_id);

        auto check = graph->getHostedOutput(output_function_id);

        std::vector<double> output;
        for (auto a : check)
            output.push_back((double)a);

        PyObject* tuple = PyTuple_New(output.size());

        for (int index = 0; index < output.size(); index++) {
            PyTuple_SetItem(tuple, index, Py_BuildValue("f", output[index]));
        }

        return tuple;
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

PyObject* fastgraph_cleanup(PyObject* self, PyObject* args)
{
    try {

        graph->cleanupGraph();

        Py_RETURN_NONE;
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

/*
 * List of functions to add to WinMiindPython in exec_WinMiindPython().
 */
static PyMethodDef fastgraph_functions[] = {
    {"function", (PyCFunction)fastgraph_addfunctionwithoutput, METH_VARARGS, "Add a new function without a fully defined output state space."},
    {"definedfunction", (PyCFunction)fastgraph_addfunction, METH_VARARGS, "Add a new function with a discretised output state space."},
    {"max", (PyCFunction)fastgraph_getfunctionoutputmax, METH_VARARGS, "Get the output distribution max from a function."},
    {"min", (PyCFunction)fastgraph_getfunctionoutputmin, METH_VARARGS, "Get the output distribution min from a function."},
    {"input", (PyCFunction)fastgraph_addinput, METH_VARARGS, "Add a discretised input space."},
    {"outputtoinput", (PyCFunction)fastgraph_addinputfromoutput, METH_VARARGS, "Add a discretised input space based on a function output."},
    {"generate", (PyCFunction)fastgraph_generate, METH_VARARGS, "Generate the transition matrix for the given function."},
    {"set", (PyCFunction)fastgraph_setinput, METH_VARARGS, "Set a given input for a given function."},
    {"apply", (PyCFunction)fastgraph_applyfunction, METH_VARARGS, "Perform the function to generate an output distribution."},
    {"transfer", (PyCFunction)fastgraph_moveoutputtoinput, METH_VARARGS, "Transfer the output distribution of one function to an input of another."},
    {"read", (PyCFunction)fastgraph_readoutput, METH_VARARGS, "Read the output distribution of a function."},
    {"cleanup", (PyCFunction)fastgraph_cleanup, METH_VARARGS, "Clean up."},
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Documentation for miindsim.
 */
PyDoc_STRVAR(fastgraph_doc, "The FastGraph module");

static PyModuleDef fastgraph_def = {
    PyModuleDef_HEAD_INIT,
    "fastgraph",
    fastgraph_doc,
    -1,
    fastgraph_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_fastgraph() {
    return PyModule_Create(&fastgraph_def);
}