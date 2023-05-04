#include "MassPopulation.cuh"
#include "MassSimulation.cuh"
#include "CudaEuler.cuh"

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

MassPopulation::MassPopulation(TimeVaryingNdGrid* _grid, unsigned int _start_cell, double refractory_period) :
	grid(_grid),
	start_cell(_start_cell),
	block_size(BLOCK_SIZE),
	sim_time(0.0),
	display(false),
	kernel_width(100),
	loaded_mass_for_this_iteration(false),
	loaded_reset_mass_for_this_iteration(false),
	loaded_refracting_mass_for_this_iteration(false)
{
	num_cells = 1;
	for (unsigned int d : grid->getRes()) {
		num_cells *= d;
	}

	num_transition_cells = ((int)pow(2, grid->getNumDimensions()));

	refractive_queue_length = int(refractory_period / grid->getTimestep()) + 1;
	current_refractive_queue_position = 0;
}

MassPopulation::~MassPopulation() {
	CleanupMass();
}

void MassPopulation::addToSimulation(inttype _population_id, MassSimulation* _sim) {
	sim = _sim;
	population_id = _population_id;
}

// Pass prop = {p[d], 1.0-p[d]}, off = {o[d], op[d]}, d = 1, c = cell count
void MassPopulation::RecurseCalculateTransition(std::vector<int>& off, std::vector<fptype>& prop, unsigned int d, unsigned int c) {

	std::vector<double> vals = grid->getTransitionMatrix()[grid->getCellCoords(c)];
	int o = int(vals[d * 3]) * grid->getResOffsets()[d];
	int op = int(vals[d * 3 + 1]) * grid->getResOffsets()[d];
	fptype pr = (fptype)vals[d * 3 + 2];

	std::vector<int> noff;
	std::vector<fptype> nprop;
	unsigned int size = off.size();
	for (unsigned int p = 0; p < size; p++) {
		noff.push_back(off[p] + o);
		noff.push_back(off[p] + op);
		nprop.push_back(prop[p] * (1.0 - pr));
		nprop.push_back(prop[p] * pr);
	}

	off = noff;
	prop = nprop;

	if (d == grid->getNumDimensions() - 1)
		return;
	else
		RecurseCalculateTransition(off, prop, d + 1, c);
}

void MassPopulation::InitMass() {
	
	// Initialise mass - put all mass in the start cell to begin with
	hosted_mass = std::vector<fptype>(num_cells);
	hosted_mass[start_cell] = 1.0;
	checkCudaErrors(cudaMalloc((fptype**)&mass, num_cells * sizeof(fptype)));
	checkCudaErrors(cudaMemcpy(mass, &hosted_mass[0], num_cells * sizeof(fptype), cudaMemcpyHostToDevice));
	
	// Initialise the dydt vector 
	checkCudaErrors(cudaMalloc((fptype**)&dydt, num_cells * sizeof(fptype)));

	// Move the transitions to a map so we can flip

	std::map<unsigned int, std::pair<std::vector<int>, std::vector<fptype>>> ts;

	// initialise the arrays

	for (int c = 0; c < num_cells; c++) {
		ts[c] = std::pair<std::vector<int>, std::vector<fptype>>(std::vector<int>(), std::vector<fptype>());
	}

	// Calculate the forward transitions and generate a backwards map 
	std::vector<std::vector<int>> forward_offs(num_cells);
	std::vector<std::vector<fptype>> forward_props(num_cells);

	for (int c = 0; c < num_cells; c++) {
		std::vector<double> vals = grid->getTransitionMatrix()[grid->getCellCoords(c)];
		int o = int(vals[0]) * grid->getResOffsets()[0];
		int op = int(vals[1]) * grid->getResOffsets()[0];
		fptype pr = (fptype)vals[2];

		forward_offs[c] = { o, op };
		forward_props[c] = { (fptype)(1.0 - pr), pr };

		RecurseCalculateTransition(forward_offs[c], forward_props[c], 1, c);

		for (unsigned int i = 0; i < num_transition_cells; i++) {
			ts[(c + forward_offs[c][i]) % num_cells].first.push_back(-forward_offs[c][i]);
			ts[(c + forward_offs[c][i]) % num_cells].second.push_back(forward_props[c][i]);
		}
	}

	// Flatten the backwards map
	std::vector<int> offs;
	std::vector<fptype> props;

	std::vector<inttype> trans_counts;
	std::vector<inttype> trans_offsets;
	unsigned int trans_offset = 0;
	for (int c = 0; c < num_cells; c++) {
		trans_offsets.push_back(trans_offset);
		unsigned int trans_count = 0;
		for (unsigned int i = 0; i < ts[c].first.size(); i++) {
			offs.push_back(ts[c].first[i]);
			props.push_back(ts[c].second[i]);
			trans_count++;
		}
		trans_offset += trans_count;
		trans_counts.push_back(trans_count);
	}

	// We need to map the transitions in a sparse matrix format

	checkCudaErrors(cudaMalloc((inttype**)&transition_counts, num_cells * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(transition_counts, &trans_counts[0], trans_counts.size() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((inttype**)&transition_offsets, num_cells * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(transition_offsets, &trans_offsets[0], trans_offsets.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	// Copy over the transition matrix

	checkCudaErrors(cudaMalloc((int**)&offsets, offs.size() * sizeof(int)));
	checkCudaErrors(cudaMalloc((fptype**)&proportions, props.size() * sizeof(fptype)));

	checkCudaErrors(cudaMemcpy(offsets, &offs[0], offs.size() * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(proportions, &props[0], props.size() * sizeof(fptype), cudaMemcpyHostToDevice));

	// Calculate the number of threshold cells: assuming this is somewhere along the 0th dimension it's easy.
	unsigned int num_threshold_cells = 1;
	for (unsigned int d = 1; d < grid->getNumDimensions(); d++) {
		num_threshold_cells *= grid->getRes()[d];
	}

	std::vector<unsigned int> threshold_sources;
	std::vector<unsigned int> reset_targets;
	for (unsigned int c = 0; c < num_cells; c++) {
		std::vector<unsigned int> coords = grid->getCellCoords(c);
		if (coords[0] == grid->getThresholdCell()) {
			coords[0] = grid->getResetCell();
			threshold_sources.push_back(c);
			reset_targets.push_back((grid->getCellNum(coords) + grid->getResetJumpOffset()) % num_cells);
			reset_mapping[c] = (grid->getCellNum(coords) + grid->getResetJumpOffset()) % num_cells;
		}
	}

	checkCudaErrors(cudaMalloc((inttype**)&threshold_cells, threshold_sources.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(threshold_cells, &threshold_sources[0], threshold_sources.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((inttype**)&reset_cells, reset_targets.size() * sizeof(inttype)));
	checkCudaErrors(cudaMemcpy(reset_cells, &reset_targets[0], reset_targets.size() * sizeof(inttype), cudaMemcpyHostToDevice));

	// refractory queues
	hosted_reset_refractive_queues = std::vector<fptype>(reset_targets.size() * refractive_queue_length);
	checkCudaErrors(cudaMalloc((fptype**)&reset_refractive_queues, reset_targets.size() * refractive_queue_length * sizeof(fptype)));

	// holds th reset mass each iteration for calculating firing rate
	checkCudaErrors(cudaMalloc((fptype**)&reset_mass, reset_targets.size() * sizeof(fptype)));
	hosted_reset_mass = std::vector<fptype>(reset_targets.size());

	// Write the grid cell widths
	std::vector<fptype> widths(grid->getNumDimensions());
	for (int d = 0; d < grid->getNumDimensions(); d++)
		widths[d] = grid->getCellWidths()[d];
	checkCudaErrors(cudaMalloc((fptype**)&grid_cell_widths, grid->getNumDimensions() * sizeof(fptype)));
	checkCudaErrors(cudaMemcpy(grid_cell_widths, &widths[0], grid->getNumDimensions() * sizeof(fptype), cudaMemcpyHostToDevice));
}

int MassPopulation::fact(int n) {
	if ((n == 0) || (n == 1))
		return 1;
	else
		return n * fact(n - 1);
}

std::vector<fptype> MassPopulation::poissonDistribution(fptype lambda, fptype cutoff) {
	std::vector<fptype> discrete_distribution;

	unsigned int events = 0;
	// Once events is greater than lambda (we've gone past the peak of the distribution), 
	// keep going until the probability is below the cutoff
	while (true) {
		discrete_distribution.push_back((pow(lambda, events) * exp(-lambda)) / fact(events));
		if (events > lambda && discrete_distribution.back() < cutoff)
			break;
		events++;
	}

	return discrete_distribution;
}

unsigned int MassPopulation::addPoissonInput(double weight, std::vector<double>& jump, fptype* delay_buffer, inttype delay_buffer_length) {
	std::vector<fptype> j(jump.size());
	for (unsigned int ji = 0; ji < j.size(); ji++)
		j[ji] = (fptype)jump[ji];

	hosted_poisson_connections.push_back(MassPopulationConnection(delay_buffer, delay_buffer_length, (fptype)weight, j));

	// Copy over the kernel
	connection_kernels.push_back(0);
	checkCudaErrors(cudaMalloc((fptype**)&connection_kernels[hosted_poisson_connections.size() - 1], kernel_width * sizeof(fptype)));

	return hosted_poisson_connections.size() - 1;
}

void MassPopulation::setPoissonInput(unsigned int id, double weight, std::vector<double>& jump) {
	std::vector<fptype> j(jump.size());
	for (unsigned int ji = 0; ji < j.size(); ji++)
		j[ji] = (fptype)jump[ji];

	hosted_poisson_connections[id].weight = weight;
	hosted_poisson_connections[id].jump = j;
}

void MassPopulation::postPoissonInput(unsigned int id, double rate) {
	PostValueToRingBuffer << <1, 1 >> > (hosted_poisson_connections[id].delay_buffer, 
		hosted_poisson_connections[id].delay_buffer_length, 
		current_simulation_iteration_count, 
		(fptype)rate);
}


void MassPopulation::UpdateMass(unsigned int iteration_count) {
	current_simulation_iteration_count = iteration_count;

	loaded_mass_for_this_iteration = false;
	loaded_reset_mass_for_this_iteration = false;
	loaded_refracting_mass_for_this_iteration = false;

	unsigned int numBlocks = (num_cells + block_size - 1) / block_size;

	for (unsigned int c = 0; c < hosted_poisson_connections.size(); c++) {
		/* Currently, we can only do noise in a single direction (dimension). For example, in 2D, we can do left/right or up/down.
		Ideally, we would like to be able to have any direction across all dimensions.
		For now, identify the first dimension in jump with a non-zero value and just that.*/
		unsigned int jump_dimension = 0;
		for (unsigned int d = 0; d < grid->getNumDimensions(); d++) {
			if (std::abs(hosted_poisson_connections[c].jump[d]) > 0.000001) {
				jump_dimension = d;
				break;
			}

		}

		CalculateKernel << < 1, 1 >> > (
			kernel_width,
			connection_kernels[c],
			hosted_poisson_connections[c].delay_buffer,
			hosted_poisson_connections[c].delay_buffer_length,
			current_simulation_iteration_count,
			hosted_poisson_connections[c].weight,
			grid->getTimestep(),
			hosted_poisson_connections[c].jump[jump_dimension],
			grid->getCellWidths()[jump_dimension]
			);

		NaiveConvolveKernel << <numBlocks, block_size >> > (
			num_cells,
			mass,
			dydt,
			connection_kernels[c],
			kernel_width,
			grid->getResOffsets()[jump_dimension]);

		ApplyDydt << <numBlocks, block_size >> > (
			num_cells,
			mass,
			dydt);
	}

	DeterministicMass << <numBlocks, block_size >> > (
		num_cells,
		mass,
		dydt,
		offsets,
		proportions,
		transition_counts,
		transition_offsets);

	ApplyDydt << <numBlocks, block_size >> > (
		num_cells,
		mass,
		dydt);

	numBlocks = (reset_mapping.size() + block_size - 1) / block_size;

	HandleResetMass << <numBlocks, block_size >> > (
		reset_mapping.size(),
		reset_cells,
		reset_refractive_queues,
		refractive_queue_length,
		current_refractive_queue_position,
		mass);

	ThresholdResetMass << <numBlocks, block_size >> > (
		reset_mapping.size(),
		threshold_cells,
		reset_cells,
		reset_mass,
		reset_refractive_queues,
		refractive_queue_length,
		current_refractive_queue_position,
		mass);

	if (sim) {
		SumResetMass << <1, 1 >> > (
			hosted_reset_mass.size(),
			reset_mass,
			sim->getRatesArray(),
			population_id,
			grid->getTimestep());
	}

	if (display) {
		checkCudaErrors(cudaMemcpy(&hosted_mass[0], mass, num_cells * sizeof(fptype), cudaMemcpyDeviceToHost));
		loaded_mass_for_this_iteration = true;
	}

	current_refractive_queue_position = (current_refractive_queue_position + 1) % refractive_queue_length;

	sim_time += grid->getTimestep();
}

void MassPopulation::CleanupMass() {
	cudaFree(mass);
	cudaFree(offsets);
	cudaFree(proportions);
	for (unsigned int c = 0; c < hosted_poisson_connections.size(); c++) {
		cudaFree(connection_kernels[c]);
	}
	cudaFree(reset_refractive_queues);
	cudaFree(grid_cell_widths);
}

std::vector<fptype> MassPopulation::getAverageMass() {
	if (!loaded_mass_for_this_iteration) {
		checkCudaErrors(cudaMemcpy(&hosted_mass[0], mass, num_cells * sizeof(fptype), cudaMemcpyDeviceToHost));
		loaded_mass_for_this_iteration = true;
	}
	std::vector<fptype> avgs(grid->getNumDimensions());
	for (unsigned int a = 0; a < grid->getNumDimensions(); a++) {
		avgs[a] = 0.0;
		for (unsigned int c = 0; c < num_cells; c++) {
			avgs[a] += hosted_mass[c] * (((grid->getCellCoords(c)[a] + 0.5) * grid->getCellWidths()[a]) + grid->getBase()[a]);
		}
		avgs[a] /= grid->getRes()[a];
	}
	return avgs;
}

fptype MassPopulation::getResetMass() {
	if (!loaded_reset_mass_for_this_iteration) {
		checkCudaErrors(cudaMemcpy(&hosted_reset_mass[0], reset_mass, hosted_reset_mass.size() * sizeof(fptype), cudaMemcpyDeviceToHost));
		loaded_reset_mass_for_this_iteration = true;
	}

	fptype reset = 0.0;
	for (fptype m : hosted_reset_mass)
		reset += m;

	return reset;
}

fptype MassPopulation::getRefractingMass() {
	if (!loaded_refracting_mass_for_this_iteration) {
		checkCudaErrors(cudaMemcpy(&hosted_reset_refractive_queues[0], reset_refractive_queues, hosted_reset_refractive_queues.size() * sizeof(fptype), cudaMemcpyDeviceToHost));
		loaded_refracting_mass_for_this_iteration = true;
	}

	fptype refracting = 0.0;
	for (fptype m : hosted_reset_refractive_queues)
		refracting += m;

	return refracting;
}

std::vector<fptype>& MassPopulation::getMass() {
	if (!loaded_mass_for_this_iteration) {
		checkCudaErrors(cudaMemcpy(&hosted_mass[0], mass, num_cells * sizeof(fptype), cudaMemcpyDeviceToHost));
		loaded_mass_for_this_iteration = true;
	}
	return hosted_mass;
}