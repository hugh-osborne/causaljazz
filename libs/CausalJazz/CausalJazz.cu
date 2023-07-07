#include "CausalJazz.cuh"
#include "CudaEuler.cuh"
#include <iostream>

#define BLOCK_SIZE 128

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

CausalJazz::CausalJazz()
: block_size(BLOCK_SIZE),
  mass_sum_value(0){

}

CausalJazz::~CausalJazz() {

}

// Build a joint distribution from 2 *independent* grids
unsigned int CausalJazz::addDistribution(CudaGrid* A, CudaGrid* B) {
	std::vector<double> newbase = A->getBase();
	std::vector<double> newdims = A->getDims();
	std::vector<unsigned int> newres = A->getRes();

	newbase.push_back(B->getBase()[0]);
	newdims.push_back(B->getDims()[0]);
	newres.push_back(B->getRes()[0]);

	std::vector<double> init(A->getTotalNumCells() * B->getTotalNumCells());
	grids.push_back(CudaGrid(newbase, newdims, newres, init));

	unsigned int id = grids.size() - 1;

	unsigned int numBlocks = (init.size() + block_size - 1) / block_size;

	GenerateJointDistributionFrom2Independents << <numBlocks, block_size >> > (
		grids[id].getTotalNumCells(),
		grids[id].getProbabilityMass(),
		A->getTotalNumCells(),
		A->getProbabilityMass(),
		B->getProbabilityMass());

	return id;
}

// Build a joint distribution from 3 *independent* grids
unsigned int CausalJazz::addDistribution(CudaGrid* A, CudaGrid* B, CudaGrid* C) {
	std::vector<double> newbase = A->getBase();
	std::vector<double> newdims = A->getDims();
	std::vector<unsigned int> newres = A->getRes();

	newbase.push_back(B->getBase()[0]);
	newdims.push_back(B->getDims()[0]);
	newres.push_back(B->getRes()[0]);

	newbase.push_back(C->getBase()[0]);
	newdims.push_back(C->getDims()[0]);
	newres.push_back(C->getRes()[0]);

	std::vector<double> init(A->getTotalNumCells() * B->getTotalNumCells() * C->getTotalNumCells());
	grids.push_back(CudaGrid(newbase, newdims, newres, init));

	unsigned int id = grids.size() - 1;

	unsigned int numBlocks = (init.size() + block_size - 1) / block_size;

	GenerateJointDistributionFrom3Independents << <numBlocks, block_size >> > (
		grids[id].getTotalNumCells(),
		grids[id].getProbabilityMass(),
		A->getTotalNumCells(),
		A->getProbabilityMass(),
		B->getTotalNumCells(),
		B->getProbabilityMass(),
		C->getProbabilityMass());

	return id;
}

// Build a distribution from the values in A
unsigned int CausalJazz::addDistribution(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res, std::vector<double> A) {

	grids.push_back(CudaGrid(_base, _dims, _res, A));

	return grids.size() - 1;
}

void CausalJazz::buildJointDistributionFrom2Independents(CudaGrid* A, CudaGrid* B, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

	GenerateJointDistributionFrom2Independents << <numBlocks, block_size >> > (
		grids[out].getTotalNumCells(),
		grids[out].getProbabilityMass(),
		A->getTotalNumCells(),
		A->getProbabilityMass(),
		B->getProbabilityMass());
}

void CausalJazz::buildJointDistributionFrom3Independents(CudaGrid* A, CudaGrid* B, CudaGrid* C, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

	GenerateJointDistributionFrom3Independents << <numBlocks, block_size >> > (
		grids[out].getTotalNumCells(),
		grids[out].getProbabilityMass(),
		A->getTotalNumCells(),
		A->getProbabilityMass(),
		B->getTotalNumCells(),
		B->getProbabilityMass(),
		C->getProbabilityMass());
}

void CausalJazz::buildJointDistributionFromChain(CudaGrid* A, unsigned int givendim, CudaGrid* BgivenA, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

	if (A->getNumDimensions() == 1) {
		if (givendim == 0)
			GenerateJointDistributionGivenA << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getTotalNumCells(),
				A->getProbabilityMass(),
				BgivenA->getProbabilityMass());
		else if (givendim == 1)
			GenerateJointDistributionGivenB << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getTotalNumCells(),
				A->getProbabilityMass(),
				BgivenA->getProbabilityMass());
	}
	else if (A->getNumDimensions() == 2) {
		if (givendim == 0)
			GenerateJointDistributionFromABCGivenA << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getRes()[0],
				A->getRes()[1],
				A->getProbabilityMass(),
				BgivenA->getProbabilityMass());
		else if (givendim == 1)
			GenerateJointDistributionFromABCGivenB << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getRes()[0],
				A->getRes()[1],
				A->getProbabilityMass(),
				BgivenA->getProbabilityMass());
	}
	
}

void CausalJazz::buildJointDistributionFromChain(CudaGrid* A, unsigned int givendimBA, CudaGrid* BgivenA, unsigned int givendimCB, CudaGrid* CgivenB, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

	if (givendimBA == 0 && givendimCB == 0) {
		GenerateJointDistributionFromABGivenACGivenB << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			grids[out].getRes()[0],
			grids[out].getRes()[1],
			A->getProbabilityMass(),
			BgivenA->getProbabilityMass(),
			CgivenB->getProbabilityMass());
	} else if (givendimBA == 1 && givendimCB == 0) {
		GenerateJointDistributionFromArBGivenACGivenB << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			grids[out].getRes()[0],
			grids[out].getRes()[1],
			A->getProbabilityMass(),
			BgivenA->getProbabilityMass(),
			CgivenB->getProbabilityMass());
	} else if (givendimBA == 0 && givendimCB == 1) {
		GenerateJointDistributionFromABGivenArCGivenB << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			grids[out].getRes()[0],
			grids[out].getRes()[1],
			grids[out].getRes()[2],
			A->getProbabilityMass(),
			BgivenA->getProbabilityMass(),
			CgivenB->getProbabilityMass());
	} else if (givendimBA == 1 && givendimCB == 1) {
		GenerateJointDistributionFromArBGivenArCGivenB << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			grids[out].getRes()[0],
			grids[out].getRes()[1],
			grids[out].getRes()[2],
			A->getProbabilityMass(),
			BgivenA->getProbabilityMass(),
			CgivenB->getProbabilityMass());
	}

}

void CausalJazz::buildJointDistributionFromFork(CudaGrid* A, unsigned int givendimBA, CudaGrid* BgivenA, unsigned int givendimCA, CudaGrid* CgivenA, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

	if (givendimBA == 0 && givendimCA == 0) {
		GenerateJointDistributionFromFork << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			A->getTotalNumCells(),
			A->getProbabilityMass(),
			BgivenA->getRes()[1],
			BgivenA->getProbabilityMass(),
			CgivenA->getRes()[1],
			CgivenA->getProbabilityMass());
	}
	/*else if (givendimBA == 1 && givendimCA == 0) {
		GenerateJointDistributionFromForkBgivenA << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			A->getTotalNumCells(),
			A->getProbabilityMass(),
			BgivenA->getRes()[1],
			BgivenA->getProbabilityMass(),
			CgivenA->getRes()[1],
			CgivenA->getProbabilityMass());
	}
	else if (givendimBA == 0 && givendimCA == 1) {
		GenerateJointDistributionFromForkCgivenA << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			A->getTotalNumCells(),
			A->getProbabilityMass(),
			BgivenA->getRes()[1],
			BgivenA->getProbabilityMass(),
			CgivenA->getRes()[1],
			CgivenA->getProbabilityMass());
	}
	else if (givendimBA == 1 && givendimCA == 1) {
		GenerateJointDistributionFromForkBgivenACgivenA << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			A->getTotalNumCells(),
			A->getProbabilityMass(),
			BgivenA->getRes()[1],
			BgivenA->getProbabilityMass(),
			CgivenA->getRes()[1],
			CgivenA->getProbabilityMass());
	}*/
	
}

void CausalJazz::buildJointDistributionFromCollider(CudaGrid* AB, std::vector<unsigned int> givendims, CudaGrid* CgivenAB, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

	if (givendims[0] == 0 && givendims[1] == 1) {
		GenerateJointDistributionFromColliderGivenAB << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			AB->getRes()[0],
			AB->getRes()[1],
			AB->getProbabilityMass(),
			CgivenAB->getProbabilityMass());
	}
	/*else if (givendims[0] == 0 && givendims[1] == 2) {
		GenerateJointDistributionFromColliderGivenAC << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			AB->getRes()[0],
			AB->getRes()[1],
			AB->getProbabilityMass(),
			CgivenAB->getProbabilityMass());
	}
	else if (givendims[0] == 1 && givendims[1] == 2) {
		GenerateJointDistributionFromColliderGivenBC << <numBlocks, block_size >> > (
			grids[out].getTotalNumCells(),
			grids[out].getProbabilityMass(),
			AB->getRes()[0],
			AB->getRes()[1],
			AB->getProbabilityMass(),
			CgivenAB->getProbabilityMass());
	}*/
}

void CausalJazz::buildMarginalDistribution(CudaGrid* A, unsigned int droppedDim, unsigned int out) {

	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

	if (A->getNumDimensions() == 3) {
		if (droppedDim == 0) { // Drop A
			GenerateMarginalBC << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getRes()[0],
				A->getRes()[1],
				A->getRes()[2],
				A->getProbabilityMass());
		}
		else if (droppedDim == 1) { // Drop B
			GenerateMarginalAC << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getRes()[0],
				A->getRes()[1],
				A->getRes()[2],
				A->getProbabilityMass());
		}
		else if (droppedDim == 2) { // Drop C
			GenerateMarginalAB << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getRes()[0],
				A->getRes()[1],
				A->getRes()[2],
				A->getProbabilityMass());
		}
	}
	else if (A->getNumDimensions() == 2) {
		if (droppedDim == 0) { // Drop A
			GenerateMarginalB << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getRes()[0],
				A->getRes()[1],
				A->getProbabilityMass());
		}
		else if (droppedDim == 1) { // Drop B
			GenerateMarginalA << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getRes()[0],
				A->getRes()[1],
				A->getProbabilityMass());
		}
	}
}

void CausalJazz::reduceJointDistributionToConditional(CudaGrid* A, std::vector<unsigned int> given, CudaGrid* givenDist, unsigned int out) {

	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

	if (A->getNumDimensions() == 3) {
		if (given.size() == 2) {
			if ((given[0] == 0 && given[1] == 1) || (given[0] == 1 && given[1] == 0)) { // C|AB
				GenerateCGivenAB << <numBlocks, block_size >> > (
					grids[out].getTotalNumCells(),
					grids[out].getProbabilityMass(),
					A->getRes()[0],
					A->getRes()[1],
					givenDist->getProbabilityMass(),
					A->getProbabilityMass());
			} else if ((given[0] == 2 && given[1] == 1) || (given[0] == 1 && given[1] == 2)) { // A|BC
				GenerateAGivenBC << <numBlocks, block_size >> > (
					grids[out].getTotalNumCells(),
					grids[out].getProbabilityMass(),
					A->getRes()[0],
					A->getRes()[1],
					givenDist->getProbabilityMass(),
					A->getProbabilityMass());
			} else if ((given[0] == 0 && given[1] == 2) || (given[0] == 2 && given[1] == 0)) { // B|AC
				GenerateBGivenAC << <numBlocks, block_size >> > (
					grids[out].getTotalNumCells(),
					grids[out].getProbabilityMass(),
					A->getRes()[0],
					A->getRes()[1],
					givenDist->getProbabilityMass(),
					A->getProbabilityMass());
			}
		}
		else if (given.size() == 1) {
			if (given[0] == 0) { // BC|A
				GenerateBCGivenA << <numBlocks, block_size >> > (
					grids[out].getTotalNumCells(),
					grids[out].getProbabilityMass(),
					A->getRes()[0],
					A->getRes()[1],
					givenDist->getProbabilityMass(),
					A->getProbabilityMass());
			} else if (given[0] == 1) { // AC|B
				GenerateACGivenB << <numBlocks, block_size >> > (
					grids[out].getTotalNumCells(),
					grids[out].getProbabilityMass(),
					A->getRes()[0],
					A->getRes()[1],
					givenDist->getProbabilityMass(),
					A->getProbabilityMass());
			} else if (given[0] == 2) { // AB|C
				GenerateABGivenC << <numBlocks, block_size >> > (
					grids[out].getTotalNumCells(),
					grids[out].getProbabilityMass(),
					A->getRes()[0],
					A->getRes()[1],
					givenDist->getProbabilityMass(),
					A->getProbabilityMass());
			}
		}
	}
	else if (A->getNumDimensions() == 2) {
		if (given[0] == 0) { // B|A
			GenerateBGivenA << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getRes()[0],
				A->getRes()[1],
				givenDist->getProbabilityMass(),
				A->getProbabilityMass());
		} else if (given[0] == 1) { // A|B
			GenerateAGivenB << <numBlocks, block_size >> > (
				grids[out].getTotalNumCells(),
				grids[out].getProbabilityMass(),
				A->getRes()[0],
				A->getRes()[1],
				givenDist->getProbabilityMass(),
				A->getProbabilityMass());
		}
	}
}

void CausalJazz::buildJointDistributionFromABCDDiamond(CudaGrid* A, CudaGrid* BCgivenA, CudaGrid* DgivenBC, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;
	GenerateJointADFromABCDDiamond << <numBlocks, block_size >> > (
		grids[out].getTotalNumCells(),
		grids[out].getProbabilityMass(),
		BCgivenA->getRes()[0],
		BCgivenA->getRes()[1],
		BCgivenA->getRes()[2],
		A->getProbabilityMass(),
		BCgivenA->getProbabilityMass(),
		DgivenBC->getProbabilityMass());
}

void CausalJazz::transferMass(unsigned int in, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;
	transferMassBetweenGrids << <numBlocks, block_size >> > (
		grids[out].getTotalNumCells(),
		grids[in].getProbabilityMass(),
		grids[out].getProbabilityMass());
}

void CausalJazz::rescale(unsigned int grid) {
	if (!mass_sum_value)
		checkCudaErrors(cudaMalloc((fptype**)&mass_sum_value, sizeof(fptype)));

	sumMass << <1, 1 >> > (
		grids[grid].getTotalNumCells(),
		grids[grid].getProbabilityMass(),
		mass_sum_value);

	unsigned int numBlocks = (grids[grid].getTotalNumCells() + block_size - 1) / block_size;

	rescaleMass << <numBlocks, block_size >> > (
		grids[grid].getTotalNumCells(),
		mass_sum_value,
		grids[grid].getProbabilityMass());
}

void CausalJazz::update(unsigned int grid_id, std::vector<double> A) {
	grids[grid_id].updateMass(A);
}

double CausalJazz::totalMass(unsigned int grid_id) {
	if (!mass_sum_value)
		checkCudaErrors(cudaMalloc((fptype**)&mass_sum_value, sizeof(fptype)));

	sumMass << <1, 1 >> > (
		grids[grid_id].getTotalNumCells(),
		grids[grid_id].getProbabilityMass(),
		mass_sum_value);

	fptype mass = 0.0;

	checkCudaErrors(cudaMemcpy(&mass, mass_sum_value, sizeof(fptype), cudaMemcpyDeviceToHost));

	return (double)mass;
}

void CausalJazz::transpose2D(unsigned int in, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

	transpose << <numBlocks, block_size >> > (
		grids[out].getTotalNumCells(),
		grids[in].getProbabilityMass(),
		grids[in].getRes()[0],
		grids[in].getRes()[1],
		grids[out].getProbabilityMass());
}

double CausalJazz::mult(std::vector<CudaGrid*> grids, std::vector<std::vector<unsigned int>> dimension_ids, std::vector<unsigned int> dim_vals) {
	
	double val = 1.0;

	for (unsigned int g = 0; g < grids.size(); g++) {
		std::vector<unsigned int> coords(grids[g]->getNumDimensions());
		for (unsigned int d = 0; d < dimension_ids[g].size(); d++) {
			coords[d] = dim_vals[dimension_ids[g][d]];
		}
		
		val *= grids[g]->getHostedProbabilityMass()[grids[g]->getCellNum(coords)];
		
	}

	return val;
}

void CausalJazz::multSumDims(std::vector<CudaGrid*> grids, std::vector<std::vector<unsigned int>> dimension_ids, std::vector<unsigned int> dim_sizes, double& val, std::vector<unsigned int> dim_vals, std::vector<unsigned int> sum_dims) {
	
	if (sum_dims.size() == 0) {
		val += mult(grids, dimension_ids, dim_vals);
	}
	else {
		unsigned int dim = sum_dims.back();
		unsigned int size = dim_sizes.back();
		sum_dims.pop_back();
		dim_sizes.pop_back();
		for (unsigned int i = 0; i < size; i++) {
			multSumDims(grids, dimension_ids, dim_sizes, val, dim_vals, sum_dims);
			dim_vals[dim]++;
		}
	}

}

void CausalJazz::multOutDims(std::vector<CudaGrid*> grids, std::vector<std::vector<unsigned int>> dimension_ids, std::vector<unsigned int> dim_sizes, CudaGrid* out, std::vector<unsigned int> dim_vals, std::vector<unsigned int> full_out_dims, std::vector<unsigned int> out_dims, std::vector<unsigned int> sum_dims) {
	
	if (out_dims.size() == 0) {
		std::vector<unsigned int> coords(out->getNumDimensions());
		for (unsigned int d = 0; d < full_out_dims.size(); d++) {
			coords[d] = dim_vals[full_out_dims[d]];
		}
		double total = 0.0;

		multSumDims(grids, dimension_ids, dim_sizes, total, dim_vals, sum_dims);

		out->getHostedProbabilityMass()[out->getCellNum(coords)] = total;
	}
	else {
		unsigned int dim = out_dims.back();
		out_dims.pop_back();
		for (unsigned int i = 0; i < out->getRes()[out_dims.size()]; i++) {
			multOutDims(grids, dimension_ids, dim_sizes, out, dim_vals, full_out_dims, out_dims, sum_dims);
			dim_vals[dim]++;
		}
	}
}

void CausalJazz::multGrids(std::vector<CudaGrid*> grids, std::vector<std::vector<unsigned int>> dimension_ids, CudaGrid* out, std::vector<unsigned int> out_dims) {

	// First calculate which dimensions need to be summed and which we are storing in the out grid.
	// Could use better data structures for this obviously
	std::vector<unsigned int> sum_dims;
	std::vector<unsigned int> sum_sizes;
	for (unsigned int s = 0; s < dimension_ids.size(); s++) {
		for (unsigned int d = 0; d < dimension_ids[s].size(); d++) {
			bool inc = false;
			for (unsigned int o : out_dims) {
				if (o == dimension_ids[s][d]) {
					inc = true;
					break;
				}
			}
			for (unsigned int o : sum_dims) {
				if (o == dimension_ids[s][d]) {
					inc = true;
					break;
				}
			}
			if (!inc) {
				sum_sizes.push_back(grids[s]->getRes()[d]);
				sum_dims.push_back(dimension_ids[s][d]);
			}
				
		}
	}

	std::vector<unsigned int> dim_vals(out_dims.size() + sum_dims.size());

	for (auto grid : grids) {
		grid->readProbabilityMass();
	}

	out->readProbabilityMass();

	multOutDims(grids, dimension_ids, sum_sizes, out, dim_vals, out_dims, out_dims, sum_dims);

	std::vector<double> mass(out->getHostedProbabilityMass().size());
	for (unsigned int i = 0; i < mass.size(); i++)
		mass[i] = (double)out->getHostedProbabilityMass()[i];

	out->updateMass(mass);

}

void CausalJazz::multGrids4D(std::vector<CudaGrid*> grids, std::vector<std::vector<unsigned int>> dimension_ids, CudaGrid* out, std::vector<unsigned int> out_dims) {
	// For the card version, we have a set order of dimensions:
	// the three out dimensions followed by the summed dimension

	// Build the array of grid pointers, array of dimension_id start points, dimension_numbers, res_offsets

	std::vector<fptype*> grid_addresses(grids.size());
	std::vector<inttype> grid_dim_start_points(grids.size());
	std::vector<inttype> grid_dim_counts(grids.size());
	std::vector<inttype> grid_res_offsets;
	std::vector<inttype> grid_flattened_dimension_ids;
	unsigned int grid_start_counter = 0;
	unsigned int d_res = 0;

	for (unsigned int g = 0; g < grids.size(); g++) {
		grid_addresses[g] = grids[g]->getProbabilityMass();
		grid_dim_start_points[g] = grid_start_counter;
		grid_dim_counts[g] = dimension_ids[g].size();
		grid_start_counter += grid_dim_counts[g];
		for (unsigned int d = 0; d < dimension_ids[g].size(); d++) {
			grid_flattened_dimension_ids.push_back(dimension_ids[g][d]);
			grid_res_offsets.push_back(grids[g]->getResOffsets()[d]);
			// Check if this is the fourth dimension we want to add
			if (dimension_ids[g][d] == 3) {
				d_res = grids[g]->getResOffsets()[d];
			}
		}
	}

	if (d_res == 0) {
		std::cout << "ERROR: FOURTH (SUMMED) DIMENSION NOT FOUND.\n";
		return;
	}

	fptype** addresses;
	inttype* dim_start_points;
	inttype* dim_counts;
	inttype* res_offsets;
	inttype* flattened_dimension_ids;

	checkCudaErrors(cudaMalloc((fptype**)&addresses, grids.size() * sizeof(fptype*)));
	checkCudaErrors(cudaMemcpy(addresses, &grid_addresses[0], grids.size() * sizeof(fptype*), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((inttype**)&dim_start_points, grids.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(dim_start_points, &grid_dim_start_points[0], grids.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((inttype**)&dim_counts, grids.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(dim_counts, &grid_dim_counts[0], grids.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((inttype**)&res_offsets, grid_res_offsets.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(res_offsets, &grid_res_offsets[0], grid_res_offsets.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((inttype**)&flattened_dimension_ids, grid_flattened_dimension_ids.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(flattened_dimension_ids, &grid_flattened_dimension_ids[0], grid_flattened_dimension_ids.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	/*
	__global__ void multiplyGrids4D(
		inttype num_ABC_cells,
		fptype * out_ABC,
		inttype A_res,
		inttype B_res,
		inttype C_res,
		inttype D_res,
		inttype num_grids,
		fptype * *grids,
		inttype * grid_dim_start_points,
		inttype * grid_dim_counts,
		inttype * dimension_ids,
		inttype * grid_res_offsets)*/

	unsigned int numBlocks = (out->getTotalNumCells() + block_size - 1) / block_size;

	multiplyGrids4D << <numBlocks, block_size >> > (
		out->getTotalNumCells(),
		out->getProbabilityMass(),
		out->getRes()[0],
		out->getRes()[1],
		out->getRes()[2],
		d_res,
		grids.size(),
		addresses,
		dim_start_points,
		dim_counts,
		flattened_dimension_ids,
		res_offsets);
}