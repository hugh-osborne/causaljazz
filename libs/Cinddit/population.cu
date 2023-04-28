#include "population.cuh"
#include "CudaEuler.cuh"
#include "display.hpp"

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <tuple>

#define BLOCK_SIZE 128

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

GridOnCard::GridOnCard(NdGrid* _grid) :
	grid(_grid) {

	// Copy over the transition matrix

	num_cells = 1;
	for (unsigned int d : grid->getRes()) {
		num_cells *= d;
	}

	for (int d = 0; d < grid->getNumDimensions(); d++) {
		std::vector<int> offs(num_cells);
		std::vector<int> offsplus(num_cells);
		std::vector<fptype> props(num_cells);

		for (int c = 0; c < num_cells; c++) {
			std::vector<double> vals = grid->getTransitionMatrix()[grid->getCellCoords(c)];
			offs[c] = int(vals[d * 3]) * grid->getResOffsets()[d];
			offsplus[c] = int(vals[d * 3 + 1]) * grid->getResOffsets()[d];
			props[c] = (fptype)vals[d * 3 + 2];
		}

		offsets.push_back(0);
		offsets_plus_one.push_back(0);
		proportions.push_back(0);

		checkCudaErrors(cudaMalloc((int**)&offsets[d], num_cells * sizeof(int)));
		checkCudaErrors(cudaMemcpy(offsets[d], &offs[0], num_cells * sizeof(int), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((int**)&offsets_plus_one[d], num_cells * sizeof(int)));
		checkCudaErrors(cudaMemcpy(offsets_plus_one[d], &offsplus[0], num_cells * sizeof(int), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((fptype**)&proportions[d], num_cells * sizeof(fptype)));
		checkCudaErrors(cudaMemcpy(proportions[d], &props[0], num_cells * sizeof(fptype), cudaMemcpyHostToDevice));
	}

	// Write the grid cell widths
	std::vector<fptype> widths(grid->getNumDimensions());
	for (int d = 0; d < grid->getNumDimensions(); d++)
		widths[d] = grid->getCellWidths()[d];
	checkCudaErrors(cudaMalloc((fptype**)&grid_cell_widths, grid->getNumDimensions() * sizeof(fptype)));
	checkCudaErrors(cudaMemcpy(grid_cell_widths, &widths[0], grid->getNumDimensions() * sizeof(fptype), cudaMemcpyHostToDevice));
}

GridOnCard::~GridOnCard() {
	for (int d = 0; d < grid->getNumDimensions(); d++) {
		cudaFree(offsets[d]);
		cudaFree(offsets_plus_one[d]);
		cudaFree(proportions[d]);
	}
}

PopulationHelper::PopulationHelper(GridOnCard* _grid, fptype refractory, unsigned int _num_neurons, unsigned int _start_cell, bool _display):
	grid_on_card(_grid),
	refractory_period(refractory),
	num_neurons(_num_neurons),
	start_cell(_start_cell),
	display(_display){
}

PopulationHelper::~PopulationHelper() {
}

MCSimulation::MCSimulation(fptype _timestep, bool _write_frames) :
	num_neurons(0),
	iteration_count(0),
	sim_timestep(_timestep),
	block_size(BLOCK_SIZE),
	sim_time(0.0),
	bit_storage_num(sizeof(inttype) * 8),
	num_rand_nums(10000),
	max_delay_timesteps(32),
	useHebbianSTDP(true),
	display_pops(0),
	write_display_frames(_write_frames)
{
}

MCSimulation::~MCSimulation() {
	CleanupNeurons();
}

unsigned int MCSimulation::addPopulation(PopulationHelper* pop) {
	populations.push_back(pop);
	pop->setSimulation(this);
	pop->setNeuronOffset(num_neurons);
	num_neurons += pop->getNumNeurons();

	if (pop->getDisplay()) {
		Display::getInstance()->addPopulation(populations.size() - 1, pop);

		display_pops.push_back(populations.size() - 1);

		Display::getInstance()->setDisplayNodes(display_pops);
	}

	return populations.size() - 1;
}

void MCSimulation::InitNeurons() {

	Display::getInstance()->animate(write_display_frames, (double)sim_timestep);

	// Initialise neurons - put them all in the start cell to begin with

	std::vector<unsigned int> neurons(num_neurons);
	std::vector<fptype> refractories(num_neurons);
	unsigned int idx = 0;
	for (auto p : populations) {
		for (unsigned int i = 0; i < p->getNumNeurons(); i++) {
			neurons[idx] = p->getStartCell();
			refractories[idx] = p->getRefractoryPeriod();
			idx++;
		}
	}

	hosted_neuron_cell_locations = std::vector<unsigned int>(neurons.size());
	checkCudaErrors(cudaMalloc((inttype**)&neuron_cell_locations, neurons.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(neuron_cell_locations, &neurons[0], neurons.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((fptype**)&refractory_periods, refractories.size() * sizeof(fptype)));
	checkCudaErrors(cudaMemcpy(refractory_periods, &refractories[0], refractories.size() * sizeof(fptype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((fptype**)&refractory_timers, refractories.size() * sizeof(fptype)));
	
	// Initialise random numbers
	inttype numBlocks = (num_neurons + block_size - 1) / block_size;
	checkCudaErrors(cudaMalloc((void**)&random_state, block_size * numBlocks * sizeof(curandState)));
	const auto p1 = std::chrono::system_clock::now();
	initCurand << <numBlocks, block_size >> > (random_state, std::chrono::duration_cast<std::chrono::seconds>(
		p1.time_since_epoch()).count());

	// Initialise a chunk of random numbers which we will use instead of generating on the fly
	numBlocks = (num_rand_nums + block_size - 1) / block_size;
	checkCudaErrors(cudaMalloc((fptype**)&random_numbers, num_rand_nums * sizeof(fptype)));
	initRandomNumbers << <numBlocks, block_size >> > (num_rand_nums, random_numbers, random_state);

	// Store which neurons spiked in a bit format
	hosted_outgoing_spikes = std::vector<inttype>((int(num_neurons / bit_storage_num) + 1));
	checkCudaErrors(cudaMalloc((inttype**)&outgoing_spikes, (int(num_neurons / bit_storage_num)+1) * sizeof(inttype)));

	// Move all connection information to the card

	unsigned int connection_offset = 0;

	std::vector<fptype> _jumps;
	std::vector<inttype> _delays;
	std::vector<inttype> _targets;
	std::vector<fptype> _spike_diffs;
	std::vector<inttype> _connection_counts;
	std::vector<inttype> _connection_offsets;

	unsigned int jump_count = 0;
	std::vector<inttype> jump_indexes;
	std::vector<inttype> jump_dims;

	for (unsigned int n = 0; n < num_neurons; n++) {
		_connection_offsets.push_back(connection_offset);
		unsigned int connection_count = 0;
		for (unsigned int c = 0; c < hosted_connections[n].size(); c++) {
			for (unsigned int d = 0; d < hosted_connections[n][c]._Myfirst._Val.size(); d++) {
				_jumps.push_back(hosted_connections[n][c]._Myfirst._Val[d]);
			}
			_delays.push_back(hosted_connections[n][c]._Get_rest()._Myfirst._Val);
			_targets.push_back(hosted_connections[n][c]._Get_rest()._Get_rest()._Myfirst._Val);
			_spike_diffs.push_back(0);
			jump_indexes.push_back(jump_count);
			jump_count += hosted_connections[n][c]._Myfirst._Val.size();
			jump_dims.push_back(hosted_connections[n][c]._Myfirst._Val.size());
			connection_count++;
			connection_offset++;
		}
		_connection_counts.push_back(connection_count);
		
	}
	// Keep the counts going for any poisson inputs
	std::vector<inttype> _poisson_connection_counts;
	std::vector<inttype> _poisson_connection_offsets;

	for (unsigned int p = 0; p < hosted_poisson_connections.size(); p++) {
		_poisson_connection_offsets.push_back(connection_offset);
		unsigned int connection_count = 0;
		for (unsigned int c = 0; c < hosted_poisson_connections[p].second.size(); c++) {
			for (unsigned int d = 0; d < hosted_poisson_connections[p].second[c].first.size(); d++) {
				_jumps.push_back(hosted_poisson_connections[p].second[c].first[d]);
			}
			_delays.push_back(0);
			_spike_diffs.push_back(0);
			_targets.push_back(hosted_poisson_connections[p].second[c].second);

			jump_indexes.push_back(jump_count);
			jump_count += hosted_poisson_connections[p].second[c].first.size();
			jump_dims.push_back(hosted_poisson_connections[p].second[c].first.size());

			connection_count++;
			connection_offset++;
		}
		_poisson_connection_counts.push_back(connection_count);
		
	}
	// Keep the counts going for any external inputs 
	std::vector<inttype> _external_connection_counts;
	std::vector<inttype> _external_connection_offsets;

	for (unsigned int p = 0; p < hosted_externl_connections.size(); p++) {
		_external_connection_offsets.push_back(connection_offset);
		unsigned int connection_count = 0;
		for (unsigned int c = 0; c < hosted_externl_connections[p].size(); c++) {
			for (unsigned int d = 0; d < hosted_externl_connections[p][c].first.size(); d++) {
				_jumps.push_back(hosted_externl_connections[p][c].first[d]);
			}
			_delays.push_back(0);
			_spike_diffs.push_back(0);
			_targets.push_back(hosted_externl_connections[p][c].second);
			jump_indexes.push_back(jump_count);
			jump_count += hosted_externl_connections[p][c].first.size();
			jump_dims.push_back(hosted_externl_connections[p][c].first.size());
			connection_count++;
			connection_offset++;
		}
		_external_connection_counts.push_back(connection_count);

	}

	// Keep the counts going for any hebbian connections
	// We also need to keep track of the connection ids for each associated target neuron
	std::vector<inttype> _hebbian_connection_counts;
	std::vector<inttype> _hebbian_connection_offsets;

	std::map<inttype, std::vector<inttype>> target_neuron_connections;

	for (unsigned int n = 0; n < num_neurons; n++) {
		_hebbian_connection_offsets.push_back(connection_offset);
		unsigned int connection_count = 0;
		if (hosted_hebbian_connections.count(n) > 0) {
			for (unsigned int c = 0; c < hosted_hebbian_connections[n].size(); c++) {
				for (unsigned int d = 0; d < hosted_hebbian_connections[n][c]._Myfirst._Val.size(); d++) {
					_jumps.push_back(hosted_hebbian_connections[n][c]._Myfirst._Val[d]);
				}
				_delays.push_back(hosted_hebbian_connections[n][c]._Get_rest()._Myfirst._Val);
				_targets.push_back(hosted_hebbian_connections[n][c]._Get_rest()._Get_rest()._Myfirst._Val);
				_spike_diffs.push_back(0);

				if (!target_neuron_connections.count(hosted_hebbian_connections[n][c]._Get_rest()._Get_rest()._Myfirst._Val))
					target_neuron_connections[hosted_hebbian_connections[n][c]._Get_rest()._Get_rest()._Myfirst._Val] = std::vector<inttype>();

				target_neuron_connections[hosted_hebbian_connections[n][c]._Get_rest()._Get_rest()._Myfirst._Val].push_back(connection_offset + connection_count);

				jump_indexes.push_back(jump_count);
				jump_count += hosted_hebbian_connections[n][c]._Myfirst._Val.size();
				jump_dims.push_back(hosted_hebbian_connections[n][c]._Myfirst._Val.size());

				connection_count++;
				connection_offset++;
			}
		}
		_hebbian_connection_counts.push_back(connection_count);
	}

	std::vector<inttype> _target_neuron_connection_ids;
	std::vector<inttype> _hebbian_backward_connection_counts;
	std::vector<inttype> _hebbian_backward_connection_offsets;
	inttype backward_connection_offset = 0;

	for (unsigned int n = 0; n < num_neurons; n++) {
		_hebbian_backward_connection_offsets.push_back(backward_connection_offset);
		inttype backward_connection_count = 0;
		for (unsigned int s = 0; s < target_neuron_connections[n].size(); s++) {
			_target_neuron_connection_ids.push_back(target_neuron_connections[n][s]);
			backward_connection_count++;
			backward_connection_offset++;
		}
		_hebbian_backward_connection_counts.push_back(backward_connection_count);
	}

	checkCudaErrors(cudaMalloc((fptype**)&jumps, _jumps.size() * sizeof(fptype)));
	checkCudaErrors(cudaMemcpy(jumps, &_jumps[0], _jumps.size() * sizeof(fptype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((inttype**)&delays, _delays.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(delays, &_delays[0], _delays.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((inttype**)&target_neurons, _targets.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(target_neurons, &_targets[0], _targets.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((inttype**)&connection_counts, _connection_counts.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(connection_counts, &_connection_counts[0], _connection_counts.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((inttype**)&connection_offsets, _connection_offsets.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(connection_offsets, &_connection_offsets[0], _connection_offsets.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((inttype**)&poisson_connection_counts, _poisson_connection_counts.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(poisson_connection_counts, &_poisson_connection_counts[0], _poisson_connection_counts.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((inttype**)&poisson_connection_offsets, _poisson_connection_offsets.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(poisson_connection_offsets, &_poisson_connection_offsets[0], _poisson_connection_offsets.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((inttype**)&external_connection_counts, _external_connection_counts.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(external_connection_counts, &_external_connection_counts[0], _external_connection_counts.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((inttype**)&external_connection_offsets, _external_connection_offsets.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(external_connection_offsets, &_external_connection_offsets[0], _external_connection_offsets.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((inttype**)&hebbian_connection_counts, _hebbian_connection_counts.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(hebbian_connection_counts, &_hebbian_connection_counts[0], _hebbian_connection_counts.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((inttype**)&hebbian_connection_offsets, _hebbian_connection_offsets.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(hebbian_connection_offsets, &_hebbian_connection_offsets[0], _hebbian_connection_offsets.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((fptype**)&spike_time_post, _spike_diffs.size() * sizeof(fptype)));
	checkCudaErrors(cudaMalloc((fptype**)&spike_time_pre, _spike_diffs.size() * sizeof(fptype)));
	checkCudaErrors(cudaMalloc((inttype**)&target_neuron_connection_ids, _target_neuron_connection_ids.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(target_neuron_connection_ids, &_target_neuron_connection_ids[0], _target_neuron_connection_ids.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((inttype**)&target_neuron_connection_counts, _hebbian_backward_connection_counts.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(target_neuron_connection_counts, &_hebbian_backward_connection_counts[0], _hebbian_backward_connection_counts.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((inttype**)&target_neuron_connection_offsets, _hebbian_backward_connection_offsets.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(target_neuron_connection_offsets, &_hebbian_backward_connection_offsets[0], _hebbian_backward_connection_offsets.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((inttype**)&connection_jump_indexes, jump_indexes.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(connection_jump_indexes, &jump_indexes[0], jump_indexes.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((inttype**)&connection_dim_nums, jump_dims.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(connection_dim_nums, &jump_dims[0], jump_dims.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	// Build the jump buffers
	std::vector<unsigned int> indexes;
	std::vector<unsigned int> dims;
	unsigned int num_buffers = 0;
	unsigned int index_count = 0;
	for (auto p : populations) {
		p->setJumpBufferOffset(num_buffers);
		num_buffers += p->getNumNeurons() * p->getGridOnCard()->getGrid()->getNumDimensions();
		for (unsigned int i = 0; i < p->getNumNeurons(); i++) {
			indexes.push_back(index_count);
			dims.push_back(p->getGridOnCard()->getGrid()->getNumDimensions());
			index_count += p->getGridOnCard()->getGrid()->getNumDimensions() * max_delay_timesteps;
		}
	}
	
	checkCudaErrors(cudaMalloc((inttype**)&neuron_jump_indexes, num_neurons * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(neuron_jump_indexes, &indexes[0], num_neurons * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((inttype**)&neuron_dim_nums, num_neurons * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(neuron_dim_nums, &dims[0], num_neurons * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((fptype**)&jump_buffer, num_buffers * max_delay_timesteps * sizeof(fptype)));
	current_buffer_position = 0;

}

void MCSimulation::addPoissonInput(double input_rate, std::vector<double>& jump, std::vector<unsigned int>& _target_neurons) {
	
	std::vector<std::pair<std::vector<fptype>, inttype>> connections;
	for (unsigned int n = 0; n < _target_neurons.size(); n++) {
		std::vector<fptype> j(jump.size());
		for (unsigned int ji = 0; ji < j.size(); ji++)
			j[ji] = (fptype)jump[ji];
		std::pair<std::vector<fptype>, inttype> p(j, _target_neurons[n]);
		connections.push_back(p);
	}

	std::pair<fptype, std::vector<std::pair<std::vector<fptype>, inttype>>> pp((fptype)input_rate, connections);

	hosted_poisson_connections.push_back(pp);
}

void MCSimulation::addConnection(std::vector<double>& jump, double delay, unsigned int source, unsigned int target) {
	std::vector<fptype> j(jump.size());
	for (unsigned int ji = 0; ji < j.size(); ji++)
		j[ji] = (fptype)jump[ji];

	unsigned int d = floor(delay / sim_timestep + 0.5);

	if (max_delay_timesteps < d)
		max_delay_timesteps = d+1;

	std::tuple<std::vector<fptype>, inttype, inttype> t(j, d, target);

	if (!hosted_connections.count(source))
		hosted_connections[source] = std::vector< std::tuple<std::vector<fptype>, inttype, inttype>>();
	hosted_connections[source].push_back(t);
}

void MCSimulation::addHebbianConnection(std::vector<double>& jump, double delay, unsigned int source, unsigned int target) {
	std::vector<fptype> j(jump.size());
	for (unsigned int ji = 0; ji < j.size(); ji++)
		j[ji] = (fptype)jump[ji];

	unsigned int d = floor(delay / sim_timestep + 0.5);

	if (max_delay_timesteps < d)
		max_delay_timesteps = d + 1;

	std::tuple<std::vector<fptype>, inttype, inttype> t(j, d, target);

	if (!hosted_hebbian_connections.count(source))
		hosted_hebbian_connections[source] = std::vector< std::tuple<std::vector<fptype>, inttype, inttype>>();
	hosted_hebbian_connections[source].push_back(t);
}

unsigned int MCSimulation::addExternalConnections(std::vector<std::pair<std::vector<double>, unsigned int>> jump_target) {
	std::vector< std::pair<std::vector<fptype>, inttype>> conns;

	for (int i = 0; i < jump_target.size(); i++) {
		std::vector<fptype> j(jump_target[i].first.size());
		for (unsigned int ji = 0; ji < j.size(); ji++)
			j[ji] = (fptype)jump_target[i].first[ji];

		std::pair<std::vector<fptype>, inttype> p(j, jump_target[i].second);
		conns.push_back(p);
	}

	unsigned int bit_size = int(jump_target.size() / bit_storage_num) + 1;
	hosted_input_spike_streams.push_back(std::vector<inttype>(bit_size));
	input_spike_streams.push_back(0);
	checkCudaErrors(cudaMalloc((inttype**)&input_spike_streams[input_spike_streams.size()-1], bit_size * sizeof(inttype)));

	hosted_externl_connections.push_back(conns);
	
	return hosted_externl_connections.size()-1;
}
void MCSimulation::sendSpikes(unsigned int input_stream, std::vector<unsigned int> connection_spikes) {
	
	for (int i = 0; i < hosted_input_spike_streams[input_stream].size(); i++)
		hosted_input_spike_streams[input_stream][i] = 0;

	for (unsigned int i = 0; i < connection_spikes.size(); i++) {
		if (connection_spikes[i] == 1)
			hosted_input_spike_streams[input_stream][int(i / bit_storage_num)] |= 1 << (i % bit_storage_num);
	}

	checkCudaErrors(cudaMemcpy(input_spike_streams[input_stream], &hosted_input_spike_streams[input_stream][0], hosted_input_spike_streams[input_stream].size() * sizeof(inttype), cudaMemcpyHostToDevice));
}

void MCSimulation::UpdateNeurons() {

	inttype numBlocks = (num_rand_nums + block_size - 1) / block_size;
	initRandomNumbers << <numBlocks, block_size >> > (num_rand_nums, random_numbers, random_state);

	numBlocks = ((int(num_neurons / bit_storage_num) + 1) + block_size - 1) / block_size;
	ResetSpikes << <numBlocks, block_size >> > (
		(int(num_neurons / bit_storage_num) + 1),
		outgoing_spikes);

	for (int p = 0; p < hosted_poisson_connections.size(); p++) {
		numBlocks = (hosted_poisson_connections[p].second.size() + block_size - 1) / block_size;

		PostPoissonSpikes << <numBlocks, block_size >> > (
			jump_buffer,
			neuron_jump_indexes,
			current_buffer_position,
			max_delay_timesteps,
			poisson_connection_counts,
			poisson_connection_offsets,
			p,
			hosted_poisson_connections[p].first, //rate
			jumps,
			connection_jump_indexes,
			neuron_dim_nums,
			target_neurons,
			sim_timestep,
			random_state);
	}

	for (auto p : populations) {
		numBlocks = (p->getNumNeurons() + block_size - 1) / block_size;

		ReduceRefractoryTimers << <numBlocks, block_size >> > (
			p->getNumNeurons(),
			p->getNeuronOffset(),
			refractory_timers,
			sim_timestep);

		for (int d = 0; d < p->getGridOnCard()->getGrid()->getNumDimensions(); d++) {
			Deterministic << <numBlocks, block_size >> > (
				p->getNumNeurons(),
				p->getNeuronOffset(),
				neuron_cell_locations,
				p->getGridOnCard()->getNumCells(),
				p->getGridOnCard()->getOffsets()[d],
				p->getGridOnCard()->getOffsetsPlusOne()[d],
				p->getGridOnCard()->getProportions()[d],
				random_numbers,
				num_rand_nums,
				refractory_timers);
		}

		HandleSpikes << <numBlocks, block_size >> > (
			p->getNumNeurons(),
			p->getNeuronOffset(),
			neuron_cell_locations,
			p->getGridOnCard()->getNumCells(),
			jump_buffer,
			neuron_jump_indexes,
			current_buffer_position,
			max_delay_timesteps,
			p->getGridOnCard()->getGrid()->getNumDimensions(),
			p->getGridOnCard()->getCellWidths(),
			p->getGridOnCard()->getGrid()->getThresholdDimOffset(),
			random_numbers,
			num_rand_nums,
			refractory_timers);

		ThresholdReset << <numBlocks, block_size >> > (
			p->getNumNeurons(),
			p->getNeuronOffset(),
			neuron_cell_locations,
			p->getGridOnCard()->getNumCells(),
			p->getGridOnCard()->getGrid()->getThresholdDimOffset(),
			1,
			p->getGridOnCard()->getGrid()->getThresholdCell(),
			p->getGridOnCard()->getGrid()->getResetCell(),
			p->getGridOnCard()->getGrid()->getResetJumpOffset(),
			bit_storage_num,
			outgoing_spikes,
			jumps,
			connection_jump_indexes,
			delays,
			target_neurons,
			connection_counts,
			connection_offsets,
			hebbian_connection_counts,
			hebbian_connection_offsets,
			jump_buffer,
			neuron_jump_indexes,
			p->getGridOnCard()->getGrid()->getNumDimensions(),
			current_buffer_position,
			max_delay_timesteps,
			refractory_timers,
			refractory_periods);

		if (p->getDisplay())
			checkCudaErrors(cudaMemcpy(&hosted_neuron_cell_locations[p->getNeuronOffset()], &neuron_cell_locations[p->getNeuronOffset()], p->getNumNeurons() * sizeof(inttype), cudaMemcpyDeviceToHost));

	}

	current_buffer_position = (current_buffer_position + 1) % max_delay_timesteps;
	
	numBlocks = (num_neurons + block_size - 1) / block_size;
	// STDP

	UpdateHebbianSpikeTimes << <numBlocks, block_size >> > (
		num_neurons,
		bit_storage_num,
		outgoing_spikes,
		spike_time_pre,
		spike_time_post,
		hebbian_connection_counts,
		hebbian_connection_offsets,
		target_neuron_connection_ids,
		target_neuron_connection_counts,
		target_neuron_connection_offsets,
		sim_time);

	numBlocks = (hosted_hebbian_connections.size() + block_size - 1) / block_size;
	UpdateHebbianJumps << <numBlocks, block_size >> > (
		hosted_hebbian_connections.size(),
		hebbian_connection_counts,
		hebbian_connection_offsets,
		spike_time_pre,
		spike_time_post,
		jumps,
		connection_jump_indexes,
		connection_dim_nums);

	

	checkCudaErrors(cudaMemcpy(&hosted_outgoing_spikes[0], outgoing_spikes, (int(num_neurons / bit_storage_num) + 1) * sizeof(inttype), cudaMemcpyDeviceToHost));

	for (int p = 0; p < hosted_externl_connections.size(); p++) {
		inttype numBlocks = (hosted_externl_connections[p].size() + block_size - 1) / block_size;
		PostSpikes << <numBlocks, block_size >> > (
			jump_buffer,
			neuron_jump_indexes,
			current_buffer_position,
			max_delay_timesteps,
			external_connection_counts,
			external_connection_offsets,
			p,
			bit_storage_num,
			input_spike_streams[p],
			jumps,
			connection_jump_indexes,
			connection_dim_nums,
			target_neurons);

		numBlocks = ((int(hosted_externl_connections[p].size() / bit_storage_num) + 1) + block_size - 1) / block_size;
		ResetSpikes << <numBlocks, block_size >> > (
			(int(hosted_externl_connections[p].size() / bit_storage_num) + 1),
			input_spike_streams[p]
			);
	}

	sim_time += sim_timestep;
	Display::getInstance()->updateDisplay((long)iteration_count);
	iteration_count++;
}

std::vector<unsigned int> MCSimulation::getSpikes() {
	std::vector<unsigned int> spikes;

	for (int i = 0; i < num_neurons; i++) {
		if ((hosted_outgoing_spikes[int(i / bit_storage_num)] >> (i % bit_storage_num)) & 1 == 1)
			spikes.push_back(i);
	}

	return spikes;
}

std::vector<unsigned int> MCSimulation::getSpikes(unsigned int pop_id) {
	std::vector<unsigned int> spikes;

	for (int i = 0; i < populations[pop_id]->getNumNeurons(); i++) {
		if ((hosted_outgoing_spikes[int((populations[pop_id]->getNeuronOffset() + i) / bit_storage_num)] >> ((populations[pop_id]->getNeuronOffset() + i) % bit_storage_num)) & 1 == 1)
			spikes.push_back(i);
	}

	return spikes;
}

void MCSimulation::CleanupNeurons() {
	cudaFree(neuron_cell_locations);
	
	cudaFree(random_state);
	cudaFree(outgoing_spikes);
}