#ifndef CAUSALJAZZ_SIMULATION
#define CAUSALJAZZ_SIMULATION

#include "CudaGrid.cuh"
#include "CausalJazz.cuh"

class CausalJazz {
public:
	CausalJazz();
	~CausalJazz();

	unsigned int addGrid(CudaGrid grid) {
		grids.push_back(grid); return grids.size() - 1;
	}
	unsigned int addDistribution(CudaGrid* A, CudaGrid* B);
	unsigned int addDistribution(CudaGrid* A, CudaGrid* B, CudaGrid* C);
	unsigned int addDistribution(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res, std::vector<double> A);
	void buildJointDistributionFrom2Independents(CudaGrid* A, CudaGrid* B, unsigned int out);
	void buildJointDistributionFrom3Independents(CudaGrid* A, CudaGrid* B, CudaGrid* C, unsigned int out);
	void buildJointDistributionFromChain(CudaGrid* A, unsigned int givendim, CudaGrid* BgivenA, unsigned int out);
	void buildJointDistributionFromChain(CudaGrid* A, unsigned int givendimBA, CudaGrid* BgivenA, unsigned int givendimCB, CudaGrid* CgivenB, unsigned int out);
	void buildJointDistributionFromFork(CudaGrid* A, unsigned int givendimBA, CudaGrid* BgivenA, unsigned int givendimCA, CudaGrid* CgivenA, unsigned int out);
	void buildJointDistributionFromCollider(CudaGrid* AB, std::vector<unsigned int> givendims, CudaGrid* CgivenAB, unsigned int out);
	void buildMarginalDistribution(CudaGrid* A, unsigned int droppedDim, unsigned int out);
	void reduceJointDistributionToConditional(CudaGrid* A, std::vector<unsigned int> given, CudaGrid* givenDist, unsigned int out);
	void buildJointDistributionFromABCDDiamond(CudaGrid* A, CudaGrid* BCgivenA, CudaGrid* DgivenBC, unsigned int out);
	void transferMass(unsigned int in, unsigned int out);
	void rescale(unsigned int grid);
	void transpose2D(unsigned int in, unsigned int out);
	void update(unsigned int grid_id, std::vector<double> A);
	double totalMass(unsigned int grid_id);
	CudaGrid* getGrid(unsigned int grid) { return &grids[grid]; }

	double mult(std::vector<CudaGrid*> grids, std::vector<std::vector<unsigned int>> dimension_ids, std::vector<unsigned int> dim_vals);
	void multGrids(std::vector<CudaGrid*> grids, std::vector<std::vector<unsigned int>> dimension_ids, CudaGrid* out, std::vector<unsigned int> out_dims);
	
private:
	std::vector<CudaGrid> grids;

	// Other CUDA helper values
	int block_size;
	fptype* mass_sum_value;
};

#endif
