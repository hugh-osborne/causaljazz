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
: block_size(BLOCK_SIZE) {

}

CausalJazz::~CausalJazz() {

}

unsigned int CausalJazz::addDistribution(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res, std::vector<double> A) {

	grids.push_back(CudaGrid(_base, _dims, _res, A));

	return grids.size() - 1;
}

void CausalJazz::buildJointDistributionFromChain(CudaGrid* A, unsigned int givendim, CudaGrid* BgivenA, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

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

void CausalJazz::buildJointDistributionFromCollider(CudaGrid* A, CudaGrid* B, CudaGrid* CgivenAB, unsigned int out) {
	unsigned int numBlocks = (grids[out].getTotalNumCells() + block_size - 1) / block_size;

	GenerateJointDistributionFromCollider << <numBlocks, block_size >> > (
		grids[out].getTotalNumCells(),
		grids[out].getProbabilityMass(),
		A->getTotalNumCells(),
		A->getProbabilityMass(),
		B->getTotalNumCells(),
		B->getProbabilityMass(),
		CgivenAB->getProbabilityMass());
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