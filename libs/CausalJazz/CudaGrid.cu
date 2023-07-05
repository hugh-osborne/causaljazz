#include "CudaGrid.cuh"
#include "CudaEuler.cuh"

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <tuple>

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// Build an initial distribution
CudaGrid::CudaGrid(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res, std::vector<double> A)
: NdGrid(_base, _dims, _res) {

	// TODO: Sanity check that A.size == num_cells
	// Convert A from doubles to fptypes
	std::vector<fptype> copydata;
	for (auto a : A)
		copydata.push_back((fptype)a);

	checkCudaErrors(cudaMalloc((fptype**)&probability_mass, getTotalNumCells() * sizeof(fptype)));
	checkCudaErrors(cudaMemcpy(probability_mass, &copydata[0], getTotalNumCells() * sizeof(fptype), cudaMemcpyHostToDevice));
}

CudaGrid::CudaGrid(const CudaGrid& other)
	: NdGrid(other.base, other.dims, other.res),
	probability_mass(other.probability_mass)
{}

CudaGrid::~CudaGrid() {

}

std::vector<fptype> CudaGrid::readProbabilityMass() {
	if (hosted_probability_mass.size() == 0)
		hosted_probability_mass = std::vector<fptype>(getTotalNumCells());

	checkCudaErrors(cudaMemcpy(&hosted_probability_mass[0], probability_mass, getTotalNumCells() * sizeof(fptype), cudaMemcpyDeviceToHost));

	return hosted_probability_mass;
}

void CudaGrid::updateMass(std::vector<double> A) {
	// TODO: Sanity check that A.size == num_cells
	// Convert A from doubles to fptypes
	std::vector<fptype> copydata;
	for (auto a : A)
		copydata.push_back((fptype)a);

	checkCudaErrors(cudaMemcpy(probability_mass, &copydata[0], getTotalNumCells() * sizeof(fptype), cudaMemcpyHostToDevice));
}