#include "MassSimulation.cuh"
#include "display.hpp"
#include <iostream>

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

MassSimulation::MassSimulation(fptype _timestep, bool _write_display_frames) :
	timestep(_timestep),
	write_display_frames(_write_display_frames),
	read_rates_this_iteration(false),
	iteration_count(0){

}

MassSimulation::~MassSimulation() {

}

unsigned int MassSimulation::addPopulation(MassPopulation* pop, bool display) {
	if(display)
		pop->setDisplayEnabled();

	populations.push_back(pop);

	hosted_rates.push_back(0);

	if (rates)
		cudaFree(rates);

	checkCudaErrors(cudaMalloc((fptype**)&rates, populations.size() * sizeof(fptype)));

	pop->addToSimulation(populations.size() - 1, this);

	if (display) {
		Display::getInstance()->addMassPopulation(populations.size() - 1, pop);

		display_pops.push_back(populations.size() - 1);

		Display::getInstance()->setDisplayNodes(display_pops);
	}
	
	return populations.size() - 1;
}

void MassSimulation::connectPopulations(unsigned int source, unsigned int target, fptype weight, std::vector<double> jump, fptype delay) {
	// Build the ring buffer
	unsigned int buffer_length = int(delay / timestep) + 1;
	delay_buffers.push_back(0);
	checkCudaErrors(cudaMalloc((fptype**)&delay_buffers[delay_buffers.size()-1], buffer_length * sizeof(fptype)));

	unsigned int connection_id = populations[target]->addPoissonInput(weight, jump, delay_buffers[delay_buffers.size() - 1], buffer_length);
	std::tuple<unsigned int, unsigned int, unsigned int, fptype, fptype> t(source, target, delay_buffers.size() - 1, weight, delay);
	connections.push_back(t);
	buffer_lengths.push_back(buffer_length);
}

unsigned int MassSimulation::addInputToPopulation(unsigned int target, std::vector<double> jump) {
	// Build the ring buffer
	unsigned int buffer_length = 1;
	delay_buffers.push_back(0);
	checkCudaErrors(cudaMalloc((fptype**)&delay_buffers[delay_buffers.size() - 1], buffer_length * sizeof(fptype)));
	buffer_lengths.push_back(buffer_length);

	return populations[target]->addPoissonInput(1.0, jump, delay_buffers[delay_buffers.size() - 1], 1);
	
}

void MassSimulation::manualUpdateConnection(unsigned int population, unsigned int connection, fptype weight, std::vector<double> jump) {
	populations[population]->setPoissonInput(connection, (double)weight, jump);
}

void MassSimulation::postFiringRateToConnection(unsigned int population, unsigned int connection, fptype rate) {
	populations[population]->postPoissonInput(connection, (double)rate);
}

void MassSimulation::initSimulation() {
	Display::getInstance()->animate(write_display_frames, (double)timestep);

	for (auto p : populations)
		p->InitMass();
}

void MassSimulation::updateSimulation() {
	read_rates_this_iteration = false;

	// Pass firing rates to relevant queues
	for (unsigned int c = 0; c < connections.size(); c++) {
		PassRatesToQueues << <1, 1 >> > (
			rates,
			connections[c]._Myfirst._Val,
			delay_buffers[connections[c]._Get_rest()._Get_rest()._Myfirst._Val],
			buffer_lengths[connections[c]._Get_rest()._Get_rest()._Myfirst._Val],
			iteration_count-1);
	}

	for (auto p : populations) {
		// Currently, the simulation time step must match all population time steps
		p->UpdateMass(iteration_count);
	}

	Display::getInstance()->updateDisplay((long)iteration_count);

	iteration_count++;
}

void MassSimulation::cleanupSimulation() {
	for (auto p : populations)
		p->CleanupMass();
}

std::vector<fptype> MassSimulation::readFiringRates() {
	// Even only reading this once per iteration slows things down significantly.
	// A faster but less interactive method would be to cache the rates for each iteration on the card
	// then read them less frequently.

	if (!read_rates_this_iteration) {
		checkCudaErrors(cudaMemcpy(&hosted_rates[0], rates, populations.size() * sizeof(fptype), cudaMemcpyDeviceToHost));
		read_rates_this_iteration = true;
	}
	return hosted_rates;
}