#ifndef CAUSALJAZZ_SIMULATION
#define CAUSALJAZZ_SIMULATION

#include "CudaGrid.cuh"
#include "CausalJazz.cuh"

class CausalJazz {
public:
	CausalJazz();
	~CausalJazz();

	unsigned int addDistribution(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res, std::vector<double> A);
	void buildJointDistributionFromChain(CudaGrid* A, CudaGrid* BgivenA, unsigned int out);
	void buildJointDistributionFromFork(CudaGrid* A, CudaGrid* BgivenA, CudaGrid* CgivenA, unsigned int out);
	void buildJointDistributionFromCollider(CudaGrid* A, CudaGrid* B, CudaGrid* CgivenAB, unsigned int out);
	void buildMarginalDistribution(CudaGrid* A, unsigned int droppedDim, unsigned int out);
	void reduceJointDistributionToConditional(CudaGrid* A, std::vector<unsigned int> given, CudaGrid* givenDist, unsigned int out);
	CudaGrid* getGrid(unsigned int grid) { return grids[grid]; }
	
private:
	std::vector<CudaGrid*> grids;

	// Other CUDA helper values
	int block_size;
};

#endif
