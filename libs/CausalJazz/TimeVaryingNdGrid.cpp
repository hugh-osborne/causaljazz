#include "TimeVaryingNdGrid.hpp"

#include <iostream>

TimeVaryingNdGrid::TimeVaryingNdGrid(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res,
    double _threshold_v, double _reset_v, std::vector<double> _reset_jump_relative, double _timestep) :
    NdGrid(_base, _dims, _res){

    threshold_dim_offset = res[0];
    threshold_v = _threshold_v;
    reset_v = _reset_v;
    reset_jump_relative = _reset_jump_relative;

    threshold_cell = int((threshold_v - base[0]) / cell_widths[0]);
    reset_cell = int((reset_v - base[0]) / cell_widths[0]);

    reset_jump_offset = 0;
    for (int i = 0; i < num_dimensions; i++) {
        reset_jump_offset += res_offsets[i] * int(reset_jump_relative[i] / cell_widths[i]);
    }

    timestep = _timestep;

}

TimeVaryingNdGrid::~TimeVaryingNdGrid() {
}

std::vector<std::tuple<unsigned int, unsigned int, double>> TimeVaryingNdGrid::calculateJumpOffset(std::vector<double> jump) {

    std::vector<std::tuple<unsigned int, unsigned int, double>> out;

    for (unsigned int d = 0; d < getNumDimensions(); d++) {
        double off = int(abs(jump[d]) / getCellWidths()[d]);
        double rem = abs(jump[d]) - (off * getCellWidths()[d]);
        double offp = off + 1;

        if (jump[d] < 0) {
            off = -off;
            offp = off - 1;
        }
        std::tuple<unsigned int, unsigned int, double> t(off, offp, rem / getCellWidths()[d]);
        out.push_back(t);
    }

    return out;
}