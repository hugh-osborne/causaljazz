#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <iostream>
#include "NdGridGenerator.hpp"
#include "CudaGridGenerator.hpp"
#include "causaljazz/MassSimulation.cuh"
#include "causaljazz/display.hpp"
#include "causaljazz/CausalJazz.cuh"

std::vector<NdGridGenerator*> grid_gens;
std::vector<MassPopulation*> populations;
MassSimulation* sim;
unsigned int iteration_count = 0;
CausalJazz* jazz;
CudaGridGenerator cuda_grid_gen;


void ParseArguments(PyObject* args) {
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    PyObject* python_function;
    std::vector<double> base;
    std::vector<double> size;
    std::vector<unsigned int> resolution;
    double threshold = 0.0;
    double reset = 0.0;
    std::vector<double> relative_jump;
    double timestep = 0.0;
    double timescale = 0.0;

    // First argument is the python function
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyCallable_Check(temp_p) == 1) {
        python_function = temp_p;
        i++;
    }

    // Second argument is the base list
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    int pr_length = PyObject_Length(temp_p);
    if (pr_length < 0)
        return;

    base = std::vector<double>(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject* item;
        item = PyList_GetItem(temp_p, index);
        if (!PyFloat_Check(item))
            base[index] = 0.0;
        base[index] = PyFloat_AsDouble(item);
    }

    Py_XDECREF(temp_p);
    i++;

    // Third argument is the grid size list
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    pr_length = PyObject_Length(temp_p);
    if (pr_length < 0)
        return;

    size = std::vector<double>(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject* item;
        item = PyList_GetItem(temp_p, index);
        if (!PyFloat_Check(item))
            size[index] = 0.0;
        size[index] = PyFloat_AsDouble(item);
    }

    Py_XDECREF(temp_p);
    i++;

    // Fourth argument is the grid resolution list
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    pr_length = PyObject_Length(temp_p);
    if (pr_length < 0)
        return;

    resolution = std::vector<unsigned int>(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject* item;
        item = PyList_GetItem(temp_p, index);
        if (!PyFloat_Check(item))
            resolution[index] = 0.0;
        resolution[index] = PyLong_AsLong(item);
    }

    Py_XDECREF(temp_p);
    i++;

    // Fifth argument is the threshold
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        threshold = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Sixth argument is the reset
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        reset = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Seventh argument is the relative jump list
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    pr_length = PyObject_Length(temp_p);
    if (pr_length < 0)
        return;

    relative_jump = std::vector<double>(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject* item;
        item = PyList_GetItem(temp_p, index);
        if (!PyFloat_Check(item))
            relative_jump[index] = 0.0;
        relative_jump[index] = PyFloat_AsDouble(item);
    }

    Py_XDECREF(temp_p);
    i++;

    // Eighth argument is the time step
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        timestep = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    grid_gens.push_back(new NdGridGenerator(base, size, resolution, threshold, reset, relative_jump, timestep));

    grid_gens[grid_gens.size()-1]->setPythonFunction(python_function);
    grid_gens[grid_gens.size()-1]->calculateTransitionMatrix();
}


PyObject* fastmass_generate(PyObject* self, PyObject* args)
{
    try {
        ParseArguments(args);

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

PyObject* fastmass_init(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    double timestep;
    bool write_frames;

    try {
        // First parameter is the timestep of the simulation
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Float(temp_p);
            timestep = (double)PyFloat_AsDouble(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Second parameter is boolean write frames
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }

        write_frames = PyObject_IsTrue(temp_p);

        sim = new MassSimulation(timestep, write_frames);

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

PyObject* fastmass_add_population(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int grid_id;
    std::vector<double> start_point;
    double refractory_period;
    bool display;

    try {
        // First argument is the grid id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            grid_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Second argument is the start point as a list of doubles
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        start_point = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                start_point[index] = 0.0;
            start_point[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Float(temp_p);
            refractory_period = (double)PyFloat_AsDouble(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }


        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }

        display = PyObject_IsTrue(temp_p);


        std::vector<unsigned int> coords(grid_gens[grid_id]->getGrid()->getNumDimensions());
        for (int d = 0; d < grid_gens[grid_id]->getGrid()->getNumDimensions(); d++) {
            coords[d] = int((start_point[d] - grid_gens[grid_id]->getGrid()->getBase()[d]) / grid_gens[grid_id]->getGrid()->getCellWidths()[d]);
        }
        unsigned int start_cell = grid_gens[grid_id]->getGrid()->getCellNum(coords);

        std::cout << "Starting point " << start_point[0];
        for (int d = 1; d < grid_gens[grid_id]->getGrid()->getNumDimensions(); d++)
            std::cout << "," << start_point[d];
        std::cout << " translates to cell " << coords[0];
        for (int d = 1; d < grid_gens[grid_id]->getGrid()->getNumDimensions(); d++)
            std::cout << "," << coords[d];
        std::cout << " with cell ID " << start_cell << ".\n";


        populations.push_back(new MassPopulation(grid_gens[grid_id]->getGrid(), start_cell, refractory_period));

        // Warn the user if the timesteps don't match
        if (grid_gens[grid_id]->getGrid()->getTimestep() - sim->getTimestep() < -0.00000001 || grid_gens[grid_id]->getGrid()->getTimestep() - sim->getTimestep() > 0.00000001)
            std::cout << "Warning: Grid timestep for population " << populations.size() - 1 << " (" << grid_gens[grid_id]->getGrid()->getTimestep() << ") doesn't match the simulation timestep (" << sim->getTimestep() << "). Continuing.\n";

        sim->addPopulation(populations[populations.size() - 1], display);
        std::cout << "Population created.\n";

        return Py_BuildValue("i", populations.size() - 1);
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

PyObject* fastmass_start(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    try {

        sim->initSimulation();
        std::cout << "Init complete. Starting simulation.\n";

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


PyObject* fastmass_step(PyObject* self, PyObject* args)
{
    try {

        sim->updateSimulation();

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

PyObject* fastmass_shutdown(PyObject* self, PyObject* args)
{
    try {

        sim->cleanupSimulation();

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

PyObject* fastmass_poisson(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int pop_id;
    std::vector<double> jump;

    try {
        // First argument is the population id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            pop_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }
        
        // Second argument is the jump vector
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        jump = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                jump[index] = 0.0;
            jump[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        std::cout << "Adding external input to population " << pop_id << "\n";

        unsigned int id = sim->addInputToPopulation(pop_id, jump);

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

PyObject* fastmass_connect(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int source_id;
    unsigned int target_id;
    double weight;
    std::vector<double> jump;
    double delay;

    try {
        // First argument is the source population id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            source_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Second argument is the target population id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            target_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Third argument is the weight
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Float(temp_p);
            weight = (double)PyFloat_AsDouble(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Fourth argument is the jump vector
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        jump = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                jump[index] = 0.0;
            jump[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // Fifth argument is the delay

        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Float(temp_p);
            delay = (double)PyFloat_AsDouble(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        sim->connectPopulations(source_id, target_id, weight, jump, delay);

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

PyObject* fastmass_updateconn(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int population_id;
    unsigned int connection_id;
    double weight;
    std::vector<double> jump;

    try {
        // First argument is the source population id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            population_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Second argument is the target population id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            connection_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Third argument is the weight
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Float(temp_p);
            weight = (double)PyFloat_AsDouble(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Fourth argument is the jump vector
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        jump = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                jump[index] = 0.0;
            jump[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        sim->manualUpdateConnection(population_id, connection_id, weight, jump);

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

PyObject* fastmass_post(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int pop_id;
    unsigned int conn_id;
    double rate;

    try {
        // First argument is the source population id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            pop_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Second argument is the target population id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            conn_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Third argument is the weight
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Float(temp_p);
            rate = (double)PyFloat_AsDouble(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        sim->postFiringRateToConnection(pop_id, conn_id, rate);

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

PyObject* fastmass_readrates(PyObject* self, PyObject* args)
{
    try {

        std::vector<fptype> rates = sim->readFiringRates();

        PyObject* tuple = PyTuple_New(rates.size());

        for (int index = 0; index < rates.size(); index++) {
            PyTuple_SetItem(tuple, index, Py_BuildValue("f", rates[index]));
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

PyObject* fastmass_readmass(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int pop_id;

    try {

        // First argument is the source population id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            pop_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        std::vector<fptype> mass = sim->getPopulation(pop_id)->getMass();

        PyObject* tuple = PyTuple_New(mass.size());

        for (int index = 0; index < mass.size(); index++) {
            PyTuple_SetItem(tuple, index, Py_BuildValue("f", mass[index]));
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


PyObject* fastmass_newdist(PyObject* self, PyObject* args)
{
    try {
        /* Get arbitrary number of strings from Py_Tuple */
        Py_ssize_t i = 0;
        PyObject* temp_p, * temp_p2;

        std::vector<double> base;
        std::vector<double> size;
        std::vector<unsigned int> resolution;
        std::vector<double> mass;

        // base list
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        base = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                base[index] = 0.0;
            base[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // grid size list
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        size = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                size[index] = 0.0;
            size[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // grid resolution list
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        resolution = std::vector<unsigned int>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                resolution[index] = 0.0;
            resolution[index] = PyLong_AsLong(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // probability mass
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        mass = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                mass[index] = 0.0;
            mass[index] = PyFloat_AsDouble(item);
        }


        if (!jazz) {
            jazz = new CausalJazz();
        }

        unsigned int id = jazz->addDistribution(base, size, resolution, mass);

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

PyObject* fastmass_newdist2(PyObject* self, PyObject* args)
{
    try {
        /* Get arbitrary number of strings from Py_Tuple */
        Py_ssize_t i = 0;
        PyObject* temp_p, * temp_p2;

        unsigned int id_A;
        unsigned int id_B;

        // Dist A
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id_A = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Dist B
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id_B = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* gridA = jazz->getGrid(id_A);
        CudaGrid* gridB = jazz->getGrid(id_B);

        unsigned int id = jazz->addDistribution(gridA, gridB);

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

PyObject* fastmass_newdist3(PyObject* self, PyObject* args)
{
    try {
        /* Get arbitrary number of strings from Py_Tuple */
        Py_ssize_t i = 0;
        PyObject* temp_p, * temp_p2;

        unsigned int id_A;
        unsigned int id_B;
        unsigned int id_C;

        // Dist A
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id_A = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Dist B
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id_B = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // Dist C
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id_C = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* gridA = jazz->getGrid(id_A);
        CudaGrid* gridB = jazz->getGrid(id_B);
        CudaGrid* gridC = jazz->getGrid(id_C);

        unsigned int id = jazz->addDistribution(gridA, gridB, gridC);

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


PyObject* fastmass_readdist(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int id;

    try {

        // grid id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* grid = jazz->getGrid(id);

        std::vector<fptype> mass = grid->readProbabilityMass();

        PyObject* tuple = PyTuple_New(mass.size());

        for (int index = 0; index < mass.size(); index++) {
            PyTuple_SetItem(tuple, index, Py_BuildValue("f", mass[index]));
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

PyObject* fastmass_marginal(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int joint_id;
    unsigned int dimension;
    unsigned int marginal_id;

    try {

        // joint dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            joint_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // dimension number
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            dimension = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // marginal dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            marginal_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* joint_grid = jazz->getGrid(joint_id);

        jazz->buildMarginalDistribution(joint_grid, dimension, marginal_id);

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

PyObject* fastmass_conditional(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int joint_id;
    std::vector<unsigned int> given_dimensions;
    unsigned int given_id;
    unsigned int out_id;

    try {

        // joint dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            joint_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // given dimensions
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        given_dimensions = std::vector<unsigned int>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                given_dimensions[index] = 0.0;
            given_dimensions[index] = PyLong_AsLong(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // given dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            given_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // out dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            out_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* joint_grid = jazz->getGrid(joint_id);
        CudaGrid* given_grid = jazz->getGrid(given_id);

        jazz->reduceJointDistributionToConditional(joint_grid, given_dimensions, given_grid, out_id);

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

PyObject* fastmass_joint2(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int marginal_id;
    unsigned int givendim;
    unsigned int conditional_id;
    unsigned int joint_id;

    try {
        // marginal dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            marginal_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // The dimension which is the "given" marginal. Eg. in P(A)P(B|A): dimension = 0, P(B)P(A|B): dimension = 1
        // Might be better to split this function into two to avoid the user having to work it out.
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            givendim = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // conditional dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            conditional_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // joint dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            joint_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* marginal_grid = jazz->getGrid(marginal_id);
        CudaGrid* conditional_grid = jazz->getGrid(conditional_id);

        jazz->buildJointDistributionFromChain(marginal_grid, givendim, conditional_grid, joint_id);

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

PyObject* fastmass_joint3(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int marginal_id;
    unsigned int givendim;
    unsigned int conditional_id;
    unsigned int joint_id;

    try {
        // marginal dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            marginal_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // The dimension which is the "given" marginal. Eg. in P(A)P(B|A): dimension = 0, P(B)P(A|B): dimension = 1
        // Might be better to split this function into two to avoid the user having to work it out.
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            givendim = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // conditional dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            conditional_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // joint dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            joint_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* marginal_grid = jazz->getGrid(marginal_id);
        CudaGrid* conditional_grid = jazz->getGrid(conditional_id);

        jazz->buildJointDistributionFromChain(marginal_grid, givendim, conditional_grid, joint_id);

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

PyObject* fastmass_chain(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int A_id;
    unsigned int givendimAB;
    unsigned int B_given_A_id;
    unsigned int givendimBC;
    unsigned int C_given_B_id;
    unsigned int joint_id;

    try {
        // marginal dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            A_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // The dimension which is the "given" marginal. Eg. in P(A)P(B|A): dimension = 0, P(B)P(A|B): dimension = 1
        // Might be better to split this function into two to avoid the user having to work it out.
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            givendimAB = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // conditional dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            B_given_A_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // The dimension which is the "given" marginal. Eg. in P(A)P(B|A): dimension = 0, P(B)P(A|B): dimension = 1
        // Might be better to split this function into two to avoid the user having to work it out.
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            givendimBC = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // conditional dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            C_given_B_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // joint dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            joint_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* A_grid = jazz->getGrid(A_id);
        CudaGrid* B_given_A_grid = jazz->getGrid(B_given_A_id);
        CudaGrid* C_given_B_grid = jazz->getGrid(C_given_B_id);

        jazz->buildJointDistributionFromChain(A_grid, givendimAB, B_given_A_grid, givendimBC, C_given_B_grid, joint_id);

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

PyObject* fastmass_fork(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int marginal_id;
    unsigned int givendim;
    unsigned int conditional_id;
    unsigned int givendim2;
    unsigned int conditional_id2;
    unsigned int joint_id;

    try {
        // marginal dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            marginal_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // The dimension which is the "given" marginal. Eg. in P(A)P(B|A): dimension = 0, P(B)P(A|B): dimension = 1
        // Might be better to split this function into two to avoid the user having to work it out.
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            givendim = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // conditional dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            conditional_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // The dimension which is the "given" marginal. Eg. in P(A)P(B|A): dimension = 0, P(B)P(A|B): dimension = 1
        // Might be better to split this function into two to avoid the user having to work it out.
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            givendim2 = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // conditional dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            conditional_id2 = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // joint dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            joint_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* marginal_grid = jazz->getGrid(marginal_id);
        CudaGrid* conditional_grid = jazz->getGrid(conditional_id);
        CudaGrid* conditional_grid2 = jazz->getGrid(conditional_id2);

        jazz->buildJointDistributionFromFork(marginal_grid, givendim, conditional_grid, givendim2, conditional_grid2, joint_id);

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

PyObject* fastmass_collider(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int marginal_id;
    std::vector<unsigned int> givendims;
    unsigned int conditional_id;
    unsigned int joint_id;

    try {
        // marginal dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            marginal_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // given dimensions
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        givendims = std::vector<unsigned int>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                givendims[index] = 0.0;
            givendims[index] = PyLong_AsLong(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // conditional dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            conditional_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // joint dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            joint_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* marginal_grid = jazz->getGrid(marginal_id);
        CudaGrid* conditional_grid = jazz->getGrid(conditional_id);

        jazz->buildJointDistributionFromCollider(marginal_grid, givendims, conditional_grid, joint_id);

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

PyObject* fastmass_function(PyObject* self, PyObject* args)
{
    try {
        /* Get arbitrary number of strings from Py_Tuple */
        Py_ssize_t i = 0;
        PyObject* temp_p, * temp_p2;

        std::vector<double> base;
        std::vector<double> size;
        std::vector<unsigned int> resolution;
        PyObject* function;
        unsigned int output_res;

        // base list
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        base = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                base[index] = 0.0;
            base[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // grid size list
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        size = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                size[index] = 0.0;
            size[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // grid resolution list
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        resolution = std::vector<unsigned int>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                resolution[index] = 0.0;
            resolution[index] = PyLong_AsLong(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // The python function
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyCallable_Check(temp_p) == 1) {
            function = temp_p;
            i++;
        }

        // output resolution
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            output_res = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }


        if (!jazz) {
            jazz = new CausalJazz();
        }

        cuda_grid_gen.setPythonFunction(function);

        unsigned int id = jazz->addGrid(cuda_grid_gen.generateCudaGridFromFunction(base, size, resolution, output_res));

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

PyObject* fastmass_bounded_function(PyObject* self, PyObject* args)
{
    try {
        /* Get arbitrary number of strings from Py_Tuple */
        Py_ssize_t i = 0;
        PyObject* temp_p, * temp_p2;

        std::vector<double> base;
        std::vector<double> size;
        std::vector<unsigned int> resolution;
        PyObject* function;
        double output_base;
        double output_size;
        unsigned int output_res;

        // base list
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        base = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                base[index] = 0.0;
            base[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // grid size list
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        size = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                size[index] = 0.0;
            size[index] = PyFloat_AsDouble(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // grid resolution list
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        resolution = std::vector<unsigned int>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                resolution[index] = 0.0;
            resolution[index] = PyLong_AsLong(item);
        }

        Py_XDECREF(temp_p);
        i++;

        // The python function
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyCallable_Check(temp_p) == 1) {
            function = temp_p;
            i++;
        }

        // output base
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Float(temp_p);
            output_base = (double)PyFloat_AsDouble(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // output size
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Float(temp_p);
            output_size = (double)PyFloat_AsDouble(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // output resolution
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            output_res = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }


        if (!jazz) {
            jazz = new CausalJazz();
        }

        cuda_grid_gen.setPythonFunction(function);

        unsigned int id = jazz->addGrid(cuda_grid_gen.generateCudaGridFromFunction(base, size, resolution, output_base, output_size, output_res));

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

PyObject* fastmass_base(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int id;

    try {

        // grid id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* grid = jazz->getGrid(id);

        std::vector<double> base = grid->getBase();

        PyObject* tuple = PyTuple_New(base.size());

        for (int index = 0; index < base.size(); index++) {
            PyTuple_SetItem(tuple, index, Py_BuildValue("f", base[index]));
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

PyObject* fastmass_size(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int id;

    try {

        // grid id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* grid = jazz->getGrid(id);

        std::vector<double> dims = grid->getDims();

        PyObject* tuple = PyTuple_New(dims.size());

        for (int index = 0; index < dims.size(); index++) {
            PyTuple_SetItem(tuple, index, Py_BuildValue("f", dims[index]));
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

PyObject* fastmass_res(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int id;

    try {

        // grid id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        CudaGrid* grid = jazz->getGrid(id);

        std::vector<unsigned int> res = grid->getRes();

        PyObject* tuple = PyTuple_New(res.size());

        for (int index = 0; index < res.size(); index++) {
            PyTuple_SetItem(tuple, index, Py_BuildValue("i", res[index]));
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

PyObject* fastmass_transfer(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int in_id;
    unsigned int out_id;

    try {
        // in dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            in_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // out dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            out_id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }


        jazz->transferMass(in_id, out_id);

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

PyObject* fastmass_rescale(PyObject* self, PyObject* args)
{
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject* temp_p, * temp_p2;

    unsigned int id;

    try {
        // in dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

 

        jazz->rescale(id);

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

PyObject* fastmass_update(PyObject* self, PyObject* args)
{
    try {
        /* Get arbitrary number of strings from Py_Tuple */
        Py_ssize_t i = 0;
        PyObject* temp_p, * temp_p2;

        unsigned int id;
        std::vector<double> mass;

        // in dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        // probability mass
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        int pr_length = PyObject_Length(temp_p);
        if (pr_length < 0)
            return NULL;

        mass = std::vector<double>(pr_length);

        for (int index = 0; index < pr_length; index++) {
            PyObject* item;
            item = PyList_GetItem(temp_p, index);
            if (!PyFloat_Check(item))
                mass[index] = 0.0;
            mass[index] = PyFloat_AsDouble(item);
        }

        jazz->update(id, mass);

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

PyObject* fastmass_total(PyObject* self, PyObject* args)
{
    try {
        /* Get arbitrary number of strings from Py_Tuple */
        Py_ssize_t i = 0;
        PyObject* temp_p, * temp_p2;

        unsigned int id;

        // in dist id
        temp_p = PyTuple_GetItem(args, i);
        if (temp_p == NULL) { return NULL; }
        if (PyNumber_Check(temp_p) == 1) {
            /* Convert number to python float then C double*/
            temp_p2 = PyNumber_Long(temp_p);
            id = (int)PyLong_AsLong(temp_p2);
            Py_DECREF(temp_p2);
            i++;
        }

        double m = jazz->totalMass(id);

        return Py_BuildValue("f", m);
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
static PyMethodDef pycausaljazz_functions[] = {
    {"generate", (PyCFunction)fastmass_generate, METH_VARARGS, "Generate a simulation."},
    {"addPopulation", (PyCFunction)fastmass_add_population, METH_VARARGS, "Add a population."},
    {"init", (PyCFunction)fastmass_init, METH_VARARGS, "Init a simulation."},
    {"start", (PyCFunction)fastmass_start, METH_VARARGS, "Start a simulation."},
    {"step", (PyCFunction)fastmass_step, METH_VARARGS, "Perform one step of a simulation."},
    {"shutdown", (PyCFunction)fastmass_shutdown, METH_VARARGS, "Shutdown a simulation."},
    {"poisson", (PyCFunction)fastmass_poisson, METH_VARARGS, "Set up a poisson spike train input to the given neurons with given jump."},
    {"connect", (PyCFunction)fastmass_connect, METH_VARARGS, "Connect Two populations."},
    {"updateConnection", (PyCFunction)fastmass_updateconn, METH_VARARGS, "Update connection between two populations."},
    {"postRate", (PyCFunction)fastmass_post, METH_VARARGS, "Post firing rate to poisson input."},
    {"readRates", (PyCFunction)fastmass_readrates, METH_VARARGS, "Read firing rates of each populations."},
    {"readMass", (PyCFunction)fastmass_readmass, METH_VARARGS, "Read mass of a given population."},
    {"newDist", (PyCFunction)fastmass_newdist, METH_VARARGS, "Jazz : Create a new distribution."},
    {"newDistFrom2", (PyCFunction)fastmass_newdist2, METH_VARARGS, "Jazz : Create a new distribution from two independent 1D distributions."},
    {"newDistFrom3", (PyCFunction)fastmass_newdist3, METH_VARARGS, "Jazz : Create a new distribution from three independent 1D distributions."},
    {"readDist", (PyCFunction)fastmass_readdist, METH_VARARGS, "Jazz : Read a distribution."},
    {"marginal", (PyCFunction)fastmass_marginal, METH_VARARGS, "Jazz : Calculate a marginal."},
    {"conditional", (PyCFunction)fastmass_conditional, METH_VARARGS, "Jazz : Calculate a conditional."},
    {"joint2D", (PyCFunction)fastmass_joint2, METH_VARARGS, "Jazz : Calculate a 2D joint distribution from p(A)*p(B|A)."},
    {"joint3D", (PyCFunction)fastmass_joint3, METH_VARARGS, "Jazz : Calculate a 3D joint distribution from p(AB)*p(C|A)."},
    {"chain", (PyCFunction)fastmass_chain, METH_VARARGS, "Jazz : Calculate a joint distribution from a chain structure p(A)*p(B|A)*p(C|B)."},
    {"fork", (PyCFunction)fastmass_fork, METH_VARARGS, "Jazz : Calculate a joint distribution from a fork structure p(A)*p(B|A)*p(C|A)."},
    {"collider", (PyCFunction)fastmass_collider, METH_VARARGS, "Jazz : Calculate a joint distribution from a collider structure p(AB)*p(C|AB)."},
    {"function", (PyCFunction)fastmass_function, METH_VARARGS, "Jazz : Calculate a conditional based on a python function."},
    {"boundedFunction", (PyCFunction)fastmass_bounded_function, METH_VARARGS, "Jazz : Calculate a conditional based on a python function. The output is explicitly bounded. "},
    {"base", (PyCFunction)fastmass_base, METH_VARARGS, "Jazz : Return the base values for a given grid id."},
    {"size", (PyCFunction)fastmass_size, METH_VARARGS, "Jazz : Return the size values for a given grid id."},
    {"res", (PyCFunction)fastmass_res, METH_VARARGS, "Jazz : Return the res values for a given grid id."},
    {"transfer", (PyCFunction)fastmass_transfer, METH_VARARGS, "Jazz : Transfer mass from one grid to another."},
    {"rescale", (PyCFunction)fastmass_rescale, METH_VARARGS, "Jazz : Rescale mass in a grid to sum to 1.0."},
    {"update", (PyCFunction)fastmass_update, METH_VARARGS, "Jazz : Update the mass in a grid."},
    {"total", (PyCFunction)fastmass_total, METH_VARARGS, "Jazz : Check the mass total in this grid."},
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Documentation for miindsim.
 */
PyDoc_STRVAR(pycausaljazz_doc, "The CausalJazz module");

static PyModuleDef pycausaljazz_def = {
    PyModuleDef_HEAD_INIT,
    "pycausaljazz",
    pycausaljazz_doc,
    -1,
    pycausaljazz_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_pycausaljazz() {
    return PyModule_Create(&pycausaljazz_def);
}