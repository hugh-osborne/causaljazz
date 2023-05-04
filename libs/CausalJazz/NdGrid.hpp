#ifndef GRID_HPP
#define GRID_HPP

#include <string>
#include <vector>
#include <map>

class NdGrid {
public:
    
    NdGrid(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res);
    ~NdGrid();

    unsigned int getNumDimensions() { return num_dimensions; }
    std::vector<double>& getBase() { return base; }
    std::vector<double>& getDims() { return dims; }
    std::vector<unsigned int>& getRes() { return res; }
    std::vector<unsigned int>& getResOffsets() { return res_offsets; }
    std::vector<double>& getCellWidths() { return cell_widths; }

    unsigned int getTotalNumCells() { return total_cells; }

    void getResOffset(unsigned int _count, std::vector<unsigned int>& _offsets, std::vector<unsigned int> _res);
    std::vector<unsigned int> getCellCoords(unsigned int c);
    unsigned int getCellNum(std::vector<unsigned int> coords);

protected:

    unsigned int total_cells;

    unsigned int num_dimensions;
    std::vector<double> base;
    std::vector<double> dims;
    std::vector<unsigned int> res;
    std::vector<unsigned int> res_offsets;
    std::vector<double> cell_widths;
};

#endif