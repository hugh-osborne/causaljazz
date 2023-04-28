#ifndef GRID_SIMULATION
#define GRID_SIMULATION

#include "CudaEuler.cuh"
#include <vector>
#include <map>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "NdGrid.hpp"

class GridOnCard {
public:
	GridOnCard(NdGrid* grid);
	~GridOnCard();

	NdGrid* getGrid() { return grid; }

	std::vector<int*>& getOffsets() { return offsets; }
	std::vector<int*>& getOffsetsPlusOne() { return offsets_plus_one; }
	std::vector<fptype*>& getProportions() { return proportions; }

	fptype* getCellWidths() { return grid_cell_widths; }
	unsigned int getNumCells() { return num_cells; }
private:

	NdGrid* grid;
	unsigned int num_cells;

	// Arrays for transition matrix
	// Vector size = number of dimensions
	// Inner array size = number of cells
	std::vector<int*> offsets;
	std::vector<int*> offsets_plus_one;
	std::vector<fptype*> proportions;

	// Now we're calculating the jump offsets/proportion etc, we need to track the cell widths
	fptype* grid_cell_widths;

};

class MCSimulation;

class PopulationHelper {
public:
	PopulationHelper(GridOnCard* _grid, fptype refractory, unsigned int _num_neurons = 100, unsigned int _start_cell = 0, bool _display = false);
	~PopulationHelper();

	GridOnCard* getGridOnCard() { return grid_on_card; }
	unsigned int getNumNeurons() { return num_neurons; }
	unsigned int getNeuronOffset() { return neuron_offset; }
	unsigned int getJumpBufferOffset() { return jump_buffer_offset; }
	unsigned int getStartCell() { return start_cell; }

	void setNeuronOffset(unsigned int off) { neuron_offset = off; return; }
	void setJumpBufferOffset(unsigned int off) { jump_buffer_offset = off; return; }
	void setSimulation(MCSimulation* s) { sim = s; return; }

	MCSimulation* getSimulation() { return sim; }
	bool getDisplay() { return display; }
	fptype getRefractoryPeriod() { return refractory_period; }

private:
	GridOnCard* grid_on_card;
	bool display;
	unsigned int num_neurons;
	unsigned int neuron_offset;
	unsigned int jump_buffer_offset;
	unsigned int start_cell;
	fptype refractory_period;

	MCSimulation* sim;
};

class MCSimulation {
public:
	MCSimulation(fptype _timestep, bool _write_frames);
	~MCSimulation();

	void InitNeurons();
	void UpdateNeurons();
	void CleanupNeurons();

	unsigned int addPopulation(PopulationHelper* pop);

	unsigned int getNumNeurons() { return num_neurons; }
	std::vector<unsigned int>& getNeuronCellLocations() { return hosted_neuron_cell_locations; }
	std::vector<unsigned int> getSpikes();
	std::vector<unsigned int> getSpikes(unsigned int pop_id);

	void addPoissonInput(double input_rate, std::vector<double>& jump, std::vector<unsigned int>& target_neurons);
	void addConnection(std::vector<double>& jump, double delay, unsigned int source, unsigned int target);
	void addHebbianConnection(std::vector<double>& jump, double delay, unsigned int source, unsigned int target);
	unsigned int addExternalConnections(std::vector<std::pair<std::vector<double>, unsigned int>> jump_target);
	void sendSpikes(unsigned int input_stream, std::vector<unsigned int> connection_spikes);

	fptype getTimestep() { return sim_timestep; }
private:
	std::vector<PopulationHelper*> populations;
	std::vector<unsigned int> display_pops;
	bool write_display_frames;

	unsigned int num_neurons;
	unsigned int bit_storage_num;

	fptype sim_time;
	fptype sim_timestep;
	inttype iteration_count;

	std::map<inttype, std::vector<std::tuple<std::vector<fptype>, inttype, inttype>>> hosted_connections;

	// For each neuron to neuron connection, store the jump vector, delay, target neuron
	// Array size = number of connections * dimensions
	fptype* jumps;
	// Array size = number of connections
	inttype* delays;
	// Array size = number of connections
	inttype* target_neurons;
	
	// For each pre-synaptic neuron, we need to know which connections we pass spikes to
	// It's a sparse connection matrix, so store the number of connections and an offset
	// to access the above arrrays
	// Array size = number of neurons
	inttype* connection_counts;
	// Array size = number of neurons
	inttype* connection_offsets;

	// poisson inputs < pair< rate, connections <jump vector, target neuron>> 
	std::vector < std::pair<fptype, std::vector<std::pair<std::vector<fptype>, inttype>>>> hosted_poisson_connections;
	// For any poisson inputs, we also want to keep track of connections, just add them on the end
	// Array size = number of poisson inputs
	inttype* poisson_connection_counts;
	// Array size = number of poisson inputs
	inttype* poisson_connection_offsets;

	// External inputs <pair< jump vector, target neuron>> 
	std::vector<std::vector <std::pair<std::vector<fptype>, inttype>>> hosted_externl_connections;
	// To send spikes to neurons in the population, we use the bit set format to identify each connection
	std::vector<std::vector<inttype>> hosted_input_spike_streams;
	std::vector<inttype*> input_spike_streams;
	// Array size = number of input streams
	inttype* external_connection_counts;
	// Array size = number of input streams
	inttype* external_connection_offsets;

	// Each post-synaptic neuron has a ring-buffer of a maximum length to handle 
	// delays and for accumulating jump amounts
	// Array size = number of neurons * num_dimensions * max_delay_timesteps
	fptype* jump_buffer;
	inttype max_delay_timesteps;
	// Remember to keep track of the current jump_buffer pointer_position.
	// This should be the same for all neurons
	inttype current_buffer_position;

	// We need to hold an index to the jump buffer for each neuron and the number of dimensions
	inttype* neuron_jump_indexes;
	inttype* neuron_dim_nums;

	// We also need to hold an index to the jump buffers and queues for each connection and the number of dimensions
	inttype* connection_jump_indexes;
	inttype* connection_dim_nums;

	// Array to store the cell location of each neuron
	// Array size = number of neurons
	std::vector<inttype> hosted_neuron_cell_locations;
	inttype* neuron_cell_locations;

	// refractory periods and refractory timers, one per neuron
	fptype* refractory_periods;
	fptype* refractory_timers;

	// Storage for random state and numbers for picking transitions
	curandState* random_state;

	// For speed (but not accuracy) we can store a bunch of random numbers and just use them
	inttype num_rand_nums;
	fptype* random_numbers;

	// Storage for outgoing spikes
	// Array size = num_neurons
	// Look at T. Nowotney's work to see how to do this quickly and with minimal memory usage
	std::vector<inttype> hosted_outgoing_spikes;
	inttype* outgoing_spikes;

	// Other CUDA helper values
	int block_size;
	int num_blocks;

	// Implement Hebbian STDP :
	// For each connection, store a value which describes the time since the last incoming spike
	// For each neuron, a value which describes the time since the last time the neuron spiked.
	// These values can be used to update the jump value for that connection

	bool useHebbianSTDP;
	// We want to store the difference between pre and post spike times for all connections
	// then for at timestep, we update the jump values accordingly.

	std::map<inttype, std::vector<std::tuple<std::vector<fptype>, inttype, inttype>>> hosted_hebbian_connections;
	
	// Array size = number of neurons
	inttype* hebbian_connection_counts;
	inttype* hebbian_connection_offsets;

	// We, therefore, need an array of differences for each connection.
	// Array size = number of hebbian connections
	fptype* spike_time_post;
	fptype* spike_time_pre;
	// This is updated whenever a pre-synaptic neuron spikes, which is easily handled.
	// However, we also need to update when each post-synaptic neuron spikes.
	// Post-synaptic neurons only look at the jump buffer, not individual connections.
	// We, therefore, need to store the connection ids
	// Array size = number of hebbian connections
	inttype* target_neuron_connection_ids;
	// Array size  = number of neurons
	inttype* target_neuron_connection_counts;
	inttype* target_neuron_connection_offsets;


};

#endif
