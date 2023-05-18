#ifndef CUDA_GRID_POPULATION
#define CUDA_GRID_POPULATION

#include "CudaEuler.cuh"
#include <vector>
#include <map>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "NdGrid.hpp"

class CudaGrid : public NdGrid{
public:
	CudaGrid(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res, std::vector<double> A);
	CudaGrid(const CudaGrid& other);
	~CudaGrid();

	fptype* getProbabilityMass() { return probability_mass; }
	std::vector<fptype> readProbabilityMass();
	void updateMass(std::vector<double> A);

private:
	
	fptype* probability_mass;
};

#endif