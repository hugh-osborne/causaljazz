#include "NdGrid.hpp"

#include <iostream>

NdGrid::NdGrid(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res) {

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

    total_cells = 1;
    for (unsigned int d : res) {
        total_cells *= d;
    }
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