#ifndef GRID_MASS_SIMULATION
#define GRID_MASS_SIMULATION

#include "MassPopulation.cuh"

class MassSimulation {
public:
	MassSimulation(fptype _timestep, bool _write_display_frames);
	~MassSimulation();

	unsigned int addPopulation(MassPopulation* pop, bool display);
	void connectPopulations(unsigned int source, unsigned int target, fptype weight, std::vector<double> jump, fptype delay);
	unsigned int addInputToPopulation(unsigned int target, std::vector<double> jump);

	void manualUpdateConnection(unsigned int population, unsigned int connection, fptype weight, std::vector<double> jump);

	void postFiringRateToConnection(unsigned int population, unsigned int connection, fptype rate);

	void initSimulation();
	void updateSimulation();
	void cleanupSimulation();

	std::vector<fptype> readFiringRates();

	MassPopulation* getPopulation(unsigned int p) { return populations[p]; }

	double getTimestep() { return timestep; }

	fptype* getRatesArray() { return rates; }

private:
	fptype timestep;
	bool write_display_frames;

	std::vector<unsigned int> display_pops;

	std::vector<MassPopulation*> populations;

	// Connections: <source pop, target pop, delay buffer id, weight, delay>
	std::vector<std::tuple<unsigned int, unsigned int, unsigned int, fptype, fptype>> connections;

	// Store the firing rates: only read from the card once per iteration
	bool read_rates_this_iteration;
	std::vector<fptype> hosted_rates;
	fptype* rates;

	// We keep a ring buffer for each connection based on the length of the delay
	// This needs to be passed to the populations so they can update the firing rate
	std::vector<fptype*> delay_buffers;
	unsigned int iteration_count;
	std::vector<unsigned int> buffer_lengths;
};

#endif
