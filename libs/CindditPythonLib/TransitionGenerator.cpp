#include "TransitionGenerator.hpp"

#include <iostream>

TransitionGenerator::TransitionGenerator(Discretised1DSpace _out) :
    out_dist(_out),
    desired_out_resolution(0) {}

TransitionGenerator::TransitionGenerator(unsigned int _out_res):
    desired_out_resolution(_out_res) {}

TransitionGenerator::~TransitionGenerator() {
    Py_Finalize();
}

void TransitionGenerator::setPythonFunctionFromStrings(std::string function, std::string functionname) {

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

void TransitionGenerator::setPythonFunction(PyObject* func) {

    Py_Initialize();

    python_func.setObject(func);
}

unsigned int TransitionGenerator::addInputDistribution(Discretised1DSpace& in) {
    in_dists.push_back(in);
    return in_dists.size() - 1;
}

void TransitionGenerator::setInputDistribution(unsigned int id, Discretised1DSpace& _in) {
    in_dists[id] = _in;
    grid->getInDistribution(id) = _in;
}

// Each cell in the out distribution is summed from a number of joint probabilities from the in distributions multiplied by a proportion
// The proportion is calculated based on the python function (as in the MIIND grid method)
void TransitionGenerator::calculateTransitionMatrix() {

    grid = new Transition(in_dists);

    // First, calculate all the values for the joint input distribution. 
    // Then we can work out the range required for the output. This should be taken away from the user.

    std::vector<double> results(grid->getTotalJointInDistributionCells());
    double max = 0.0;
    double min = 0.0;
    // For each joint cell in the in distributions
    for (int c = 0; c < grid->getTotalJointInDistributionCells(); c++) {
        // Build the start point
        std::vector<double> start_point;
        for (unsigned int d = 0; d < grid->getNumInDistributions(); d++) {
            start_point.push_back(grid->getInDistribution(d).cellCentroid(grid->getJointInDistributionCoords(c)[d]));
        }

        // Build the list of values for the starting point to be sent to the python function
        PyObject* point_coord_list = PyList_New((Py_ssize_t)(grid->getNumInDistributions()));

        std::vector<PyObject*> list_objects(grid->getNumInDistributions());

        for (unsigned int i = 0; i < grid->getNumInDistributions(); i++) {
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

        results[c] = out_value;

        if (c == 0) {
            max = results[c];
            min = results[c];
        }
        else {
            if (max < results[c])
                max = results[c];
            if (min > results[c])
                min = results[c];
        }
    }

    if (desired_out_resolution > 0)
        grid->setOutDistribution(min, max, desired_out_resolution);
    else
        grid->setOutDistribution(out_dist);

    std::vector<std::vector<std::pair<std::vector<unsigned int>, double>>> transitions(grid->getOutDistribution().getRes());

    for (int c = 0; c < grid->getTotalJointInDistributionCells(); c++) {
        // Each point gets flattened to two cells in the out distribution
        double hi_value = results[c] + (grid->getOutDistribution().getCellWidth() / 2.0);
        double lo_value = results[c] - (grid->getOutDistribution().getCellWidth() / 2.0);

        double hi_shifted = (hi_value - grid->getOutDistribution().getMin()) / grid->getOutDistribution().getCellWidth();
        double lo_shifted = (lo_value - grid->getOutDistribution().getMin()) / grid->getOutDistribution().getCellWidth();

        unsigned int hi_out_cell = int(hi_shifted);
        unsigned int lo_out_cell = int(lo_shifted);

        double hi_out_prop = hi_shifted - hi_out_cell;
        double lo_out_prop = 1.0 - hi_out_prop;

        std::pair<std::vector<unsigned int>, double> hi_pair(grid->getJointInDistributionCoords(c), hi_out_prop);
        std::pair<std::vector<unsigned int>, double> lo_pair(grid->getJointInDistributionCoords(c), lo_out_prop);

        if (hi_out_cell >= grid->getOutDistribution().getRes())
            hi_out_cell = grid->getOutDistribution().getRes() - 1;
        if (lo_out_cell >= grid->getOutDistribution().getRes())
            lo_out_cell = grid->getOutDistribution().getRes() - 1;
        if (hi_out_cell < 0)
            hi_out_cell = 0;
        if (lo_out_cell < 0)
            lo_out_cell = 0;
        if (hi_pair.second > 0)
            transitions[hi_out_cell].push_back(hi_pair);
        if (lo_pair.second > 0)
            transitions[lo_out_cell].push_back(lo_pair);
    }

    grid->setBackwardTransitions(transitions);
}