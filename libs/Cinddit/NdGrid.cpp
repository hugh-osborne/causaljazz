#include "NdGrid.hpp"

#include <iostream>

NdGrid::NdGrid(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res,
    double _threshold_v, double _reset_v, std::vector<double> _reset_jump_relative, double _timestep) {

        num_dimensions = _dims.size();
        base = _base;
        dims = _dims;
        res = _res;

        std::vector<unsigned int> temp_res_offsets = { 1 };
        getResOffset(1, temp_res_offsets, res);
        
        res_offsets = std::vector<unsigned int>(num_dimensions);
        for (int i = 0; i < num_dimensions; i++) {
            res_offsets[i] = temp_res_offsets[i];
        }

        cell_widths = std::vector<double>(num_dimensions);

        for (int i = 0; i < num_dimensions; i++) {
            cell_widths[i] = dims[i] / res[i];
        }

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

NdGrid::~NdGrid() {
}

void NdGrid::getResOffset(unsigned int _count, std::vector<unsigned int> &_offsets, std::vector<unsigned int> _res) {
    _count *= _res[0];
    _offsets.push_back(_count);

    std::vector<unsigned int> new_res;
    for (int i = 1; i < _res.size(); i++) {
        new_res.push_back(_res[i]);
    }

    if (new_res.size() == 1)
        return;

    getResOffset(_count, _offsets, new_res);
    return;
}

std::vector<unsigned int> NdGrid::getCellCoords(unsigned int c) {
    std::vector<unsigned int> coords(num_dimensions);

    for (int i = num_dimensions-1; i >= 0; i--) {
        coords[i] = int(c / res_offsets[i]);
        c = c - (coords[i] * res_offsets[i]);
    }

    return coords;
}

unsigned int NdGrid::getCellNum(std::vector<unsigned int> coords) {
    unsigned int c = 1;
    for (int i = 0; i < num_dimensions; i++) {
        c += coords[i] * res_offsets[i];
    }

    return c;
}

std::vector<std::tuple<unsigned int, unsigned int, double>> NdGrid::calculateJumpOffset(std::vector<double> jump) {

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