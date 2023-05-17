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
	else if (givendimBA == 1 && givendimCA == 0) {
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
	}
	
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
	else if (givendims[0] == 0 && givendims[1] == 2) {
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
	}
	
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