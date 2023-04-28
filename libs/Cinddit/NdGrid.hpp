#ifndef GRID_HPP
#define GRID_HPP

#include <string>
#include <vector>
#include <map>

class NdGrid {
public:
    
    NdGrid(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res,
        double _threshold_v, double _reset_v, std::vector<double> _reset_jump_relative, double _timestep);
    ~NdGrid();

    unsigned int getNumDimensions() { return num_dimensions; }
    std::vector<double>& getBase() { return base; }
    std::vector<double>& getDims() { return dims; }
    std::vector<unsigned int>& getRes() { return res; }
    std::vector<unsigned int>& getResOffsets() { return res_offsets; }
    std::vector<double>& getCellWidths() { return cell_widths; }

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

    void getResOffset(unsigned int _count, std::vector<unsigned int>& _offsets, std::vector<unsigned int> _res);
    std::vector<unsigned int> getCellCoords(unsigned int c);
    unsigned int getCellNum(std::vector<unsigned int> coords);
    std::vector<std::tuple<unsigned int, unsigned int, double>> calculateJumpOffset(std::vector<double> jump);

private:

    unsigned int num_dimensions;
    std::vector<double> base;
    std::vector<double> dims;
    std::vector<unsigned int> res;
    std::vector<unsigned int> res_offsets;
    std::vector<double> cell_widths;
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