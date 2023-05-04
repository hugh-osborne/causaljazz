#ifndef GRID_MASS_POPULATION
#define GRID_MASS_POPULATION

#include "CudaEuler.cuh"
#include <vector>
#include <map>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "TimeVaryingNdGrid.hpp"

class MassSimulation;

class MassPopulationConnection {
public:
	MassPopulationConnection(fptype* _delay_buffer, inttype _delay_buffer_length, fptype _weight, std::vector<fptype> _jump) :
		delay_buffer(_delay_buffer),
		delay_buffer_length(_delay_buffer_length),
		weight(_weight),
		jump(_jump) {}

	MassPopulationConnection(MassPopulationConnection &other) :
		delay_buffer(other.delay_buffer),
		delay_buffer_length(other.delay_buffer_length),
		weight(other.weight),
		jump(other.jump) {}

	fptype* delay_buffer;
	inttype delay_buffer_length;
	fptype weight;
	std::vector<fptype> jump;
};

class MassPopulation {
public:
	MassPopulation(TimeVaryingNdGrid* _grid, unsigned int _start_cell = 0, double refractory_period = 0.0);
	~MassPopulation();

	void addToSimulation(inttype population_id, MassSimulation* sim);

	void InitMass();
	void UpdateMass(unsigned int iteration_count);
	void CleanupMass();

	TimeVaryingNdGrid* getGrid() { return grid; }

	unsigned int getNumCells() { return num_cells; }

	std::vector<fptype>& getMass();

	void setDisplayEnabled() { display = true; return; }

	void setPoissonInput(unsigned int id, double weight, std::vector<double>& jump);
	unsigned int addPoissonInput(double weight, std::vector<double>& jump, fptype* delay_buffer, inttype delay_buffer_length);
	void postPoissonInput(unsigned int id, double rate);

	std::vector<fptype> getAverageMass();
	fptype getResetMass();
	fptype getRefractingMass();

private:

	MassSimulation* sim;

	int fact(int n);
	std::vector<fptype> poissonDistribution(fptype lambda, fptype cutoff);
	void RecurseCalculateTransition(std::vector<int>& off, std::vector<fptype>& prop, unsigned int d, unsigned int c);

	unsigned int current_simulation_iteration_count;
	inttype population_id;

	unsigned int kernel_width;
	unsigned int num_populations;
	unsigned int num_transition_cells;

	bool display;

	unsigned int start_cell;
	unsigned int num_cells;
	TimeVaryingNdGrid* grid;

	fptype sim_time;
	bool loaded_mass_for_this_iteration;
	bool loaded_reset_mass_for_this_iteration;
	bool loaded_refracting_mass_for_this_iteration;

	// vector < pair< rate, jump> >
	// At the moment, the jump vector must only have a positive or negative value in one dimension
	std::vector<MassPopulationConnection> hosted_poisson_connections;

	// For each connection, for each iteration, a kernel is generated based on the input rate and jump size of the connection
	// Array size = number of connections
	std::vector<fptype*> connection_kernels;

	// Now we're calculating the jump offsets/proportion etc, we need to track the cell widths
	fptype* grid_cell_widths;

	// Array to store the mass in each cell
	// Array size = number of cells
	std::vector<fptype> hosted_mass;
	fptype* mass;
	fptype* dydt;
	inttype* transition_counts;
	inttype* transition_offsets;

	// Arrays for transition matrix=
	// Array size = number of cells * 2^Dimensions
	int* offsets;
	fptype* proportions;

	std::vector<fptype> hosted_reset_mass;
	fptype* reset_mass;

	// Threshold cell ids and their mappings to reset cells
	std::map<inttype, inttype> reset_mapping;
	inttype* threshold_cells;
	inttype* reset_cells;
	std::vector<fptype> hosted_reset_refractive_queues;
	fptype* reset_refractive_queues;
	inttype refractive_queue_length;
	inttype current_refractive_queue_position;

	// Other CUDA helper values
	int block_size;
	int num_blocks;
};

#endif
