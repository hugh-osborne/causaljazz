#ifndef TIMEVARYINGGRID_HPP
#define TIMEVARYINGGRID_HPP

#include <string>
#include <vector>
#include <map>
#include "NdGrid.hpp"

class TimeVaryingNdGrid : public NdGrid {
public:

    TimeVaryingNdGrid(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res,
        double _threshold_v, double _reset_v, std::vector<double> _reset_jump_relative, double _timestep);
    ~TimeVaryingNdGrid();

    unsigned int getThresholdDimOffset() { return threshold_dim_offset; }
    unsigned int getThresholdCell() { return threshold_cell; }
    unsigned int getResetCell() { return reset_cell; }
    unsigned int getResetJumpOffset() { return reset_jump_offset; }

    double getThreshold() { return threshold_v; }
    double getReset() { return reset_v; }
    std::vector<double> getResetJumpRelative() { return reset_jump_relative; }
    double getTimestep() { return timestep; }

    std::map<std::vector<unsigned int>, std::vector<double>>& getTransitionMatrix() { return transition_matrix; }
    void setTransitionMatrix(std::map<std::vector<unsigned int>, std::vector<double>> tm) { transition_matrix = tm; return; }

    std::vector<std::tuple<unsigned int, unsigned int, double>> calculateJumpOffset(std::vector<double> jump);

private:

    unsigned int threshold_dim_offset;
    double threshold_v;
    unsigned int threshold_cell;
    double reset_v;
    unsigned int reset_cell;
    std::vector<double> reset_jump_relative;
    unsigned int reset_jump_offset;
    double timestep;

    std::map<std::vector<unsigned int>, std::vector<double>> transition_matrix;
};

#endif