#ifndef CUEMTESTCUH
#define CUEMTESTCUH

#include <curand.h>
#include <curand_kernel.h>

typedef unsigned int inttype;
typedef float fptype;
__device__ int modulo(int a, int b);
__device__ int fact(int n);

__global__ void initCurand(curandState* state, unsigned long seed);

__global__ void initRandomNumbers(inttype num_rands, fptype* random_nums, curandState* state);

__global__ void Deterministic(
    inttype num_neurons,
    inttype neuron_offset,
    inttype* neuron_cell_locations,
    inttype num_cells,
    int* offsets,
    int* offsetsplus,
    fptype* proportions,
    fptype* random_numbers,
    inttype num_rands,
    fptype* refractory_timers);

__global__ void ReduceRefractoryTimers(
    inttype num_neurons,
    inttype neuron_offset,
    fptype* refractory_timers,
    fptype timestep);

__global__ void ResetSpikes(
    inttype num_spiked_values,
    inttype* spiked);

__global__ void ThresholdReset(
    inttype num_neurons,
    inttype neuron_offset,
    inttype* neuron_cell_locations,
    inttype num_cells,
    inttype threshold_dim_offset,
    inttype threshold_dim_stride, // not using this currently but it's in case we have threshold/reset in a different dimension
    inttype threshold_cell,
    inttype reset_cell,
    inttype reset_jump_cell,
    inttype bit_count,
    inttype* spiked,
    fptype* jumps,
    inttype* connections_to_jumps,
    inttype* delays,
    inttype* targets,
    inttype* connection_count,
    inttype* connection_offset,
    inttype* hebbian_connection_count,
    inttype* hebbian_connection_offset,
    fptype* jump_queues,
    inttype* neuron_to_queues,
    inttype num_dimensions,
    inttype current_jump_position,
    inttype max_jump_position,
    fptype* refractory_timers,
    fptype* refractory_periods);

__global__ void HandleSpikes(
    inttype num_neurons,
    inttype neuron_offset,
    inttype* neuron_cell_locations,
    inttype num_cells,
    fptype* jump_queues,
    inttype* neuron_to_queues,
    inttype current_jump_position,
    inttype max_jump_position,
    inttype num_dimensions,
    fptype* grid_cell_width,
    inttype threshold_dim_offset,
    fptype* random_numbers,
    inttype num_rands,
    fptype* refractory_timers);

__global__ void PostPoissonSpikes(
    fptype* jump_queues,
    inttype* neurons_to_queues,
    inttype current_jump_position,
    inttype max_jump_position,
    inttype* connection_counts,
    inttype* connection_offsets,
    inttype connection_id,
    fptype rate,
    fptype* jumps,
    inttype* connection_to_jumps,
    inttype* neurons_to_dims,
    inttype* targets,
    fptype timestep,
    curandState* state);

__global__ void PostSpikes(
    fptype* jump_queues,
    inttype* neurons_to_queues,
    inttype current_jump_position,
    inttype max_jump_position,
    inttype* connection_counts,
    inttype* connection_offsets,
    inttype connection_id,
    inttype bit_count,
    inttype* spikes,
    fptype* jumps,
    inttype* connections_to_jumps,
    inttype* connections_to_dims,
    inttype* targets);

__global__ void SpikeExternalInput(
    inttype num_connections,
    inttype* connection_ids,
    inttype* neuron_incoming_spike_value_queues,
    inttype* neuron_connection_delays,
    curandState* state);

__global__ void UpdateHebbianSpikeTimes(
    inttype num_neurons,
    inttype bit_count,
    inttype* spiked,
    fptype* spike_time_pre,
    fptype* spike_time_post,
    inttype* hebbian_connection_counts,
    inttype* hebbian_connection_offsets,
    inttype* target_neuron_connection_ids,
    inttype* target_neuron_connection_counts,
    inttype* target_neuron_connection_offsets,
    fptype time);

__global__ void UpdateHebbianJumps(
    inttype num_connections,
    inttype* hebbian_connection_counts,
    inttype* hebbian_connection_offsets,
    fptype* spike_time_pre,
    fptype* spike_time_post,
    fptype* jumps,
    inttype* connections_to_jumps,
    inttype* connections_to_dims);

// Mass Simulation

__global__ void DeterministicMass(
    inttype num_cells,
    fptype* mass,
    fptype* dydt,
    int* offsets,
    fptype* proportions,
    inttype* transition_counts,
    inttype* transition_offsets);

__global__ void ApplyDydt(
    inttype num_cells,
    fptype* mass,
    fptype* dydt);

__global__ void NaiveConvolveKernel(
    inttype num_cells,
    fptype* mass,
    fptype* dydt,
    fptype* kernel,
    inttype kernel_width,
    inttype dim_stride = 1);

__global__ void PostValueToRingBuffer(
    fptype* queue,
    inttype queue_length,
    inttype current_position,
    fptype value);

__global__ void CalculateKernel(
    inttype kernel_width,
    fptype* kernel,
    fptype* delay_buffer,
    inttype delay_buffer_length,
    inttype iteration_count,
    fptype weight,
    fptype timestep,
    fptype jump,
    fptype cell_width);

__global__ void ThresholdResetMass(
    inttype num_threshold_cells,
    inttype* threshold_cells,
    inttype* reset_cells,
    fptype* reset_mass,
    fptype* reset_queues,
    inttype queue_length,
    inttype current_queue_pointer,
    fptype* mass);

__global__ void HandleResetMass(
    inttype num_reset_cells,
    inttype* reset_cells,
    fptype* reset_queues,
    inttype queue_length,
    inttype current_queue_pointer,
    fptype* mass);

__global__ void SumResetMass(
    inttype num_reset_cells,
    fptype* reset_mass,
    fptype* stored_rates,
    inttype population_id,
    fptype timestep);

__global__ void PassRatesToQueues(
    fptype* stored_rates,
    inttype population_id,
    fptype* queue,
    inttype queue_length,
    inttype current_position);

// New Causal Jazz

__global__ void GenerateJointDistributionFrom2Independents(
    inttype num_joint_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    fptype* B);

__global__ void GenerateJointDistributionFrom3Independents(
    inttype num_joint_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    inttype num_B_cells,
    fptype* B,
    fptype* C);

__global__ void GenerateJointDistributionGivenA(
    inttype num_AB_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    fptype* BgivenA);

__global__ void GenerateJointDistributionGivenB(
    inttype num_AB_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* B,
    fptype* AgivenB);

__global__ void GenerateJointDistributionFromABCGivenA(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    fptype* AB,
    fptype* CgivenA);

__global__ void GenerateJointDistributionFromABCGivenB(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    fptype* AB,
    fptype* CgivenB);

__global__ void GenerateJointDistributionFromABGivenACGivenB(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    fptype* A,
    fptype* BgivenA,
    fptype* CgivenB);

__global__ void GenerateJointDistributionFromArBGivenACGivenB(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    fptype* A,
    fptype* rBgivenA,
    fptype* CgivenB);

__global__ void GenerateJointDistributionFromABGivenArCGivenB(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    inttype num_C_cells,
    fptype* A,
    fptype* BgivenA,
    fptype* rCgivenB);

__global__ void GenerateJointDistributionFromArBGivenArCGivenB(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    inttype num_C_cells,
    fptype* A,
    fptype* rBgivenA,
    fptype* rCgivenB);

__global__ void GenerateJointDistributionFromFork(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    inttype num_BgivenA_cells,
    fptype* BgivenA,
    inttype num_CgivenA_cells,
    fptype* CgivenA);

__global__ void GenerateJointDistributionFromForkBgivenA(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    inttype num_BgivenA_cells,
    fptype* BgivenA,
    inttype num_CgivenA_cells,
    fptype* CgivenA);

__global__ void GenerateJointDistributionFromForkCgivenA(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    inttype num_BgivenA_cells,
    fptype* BgivenA,
    inttype num_CgivenA_cells,
    fptype* CgivenA);

__global__ void GenerateJointDistributionFromForkBgivenACgivenA(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    inttype num_BgivenA_cells,
    fptype* BgivenA,
    inttype num_CgivenA_cells,
    fptype* CgivenA);

__global__ void GenerateJointDistributionFromColliderGivenAB(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    fptype* AB,
    fptype* CgivenAB);

__global__ void GenerateJointDistributionFromColliderGivenBC(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    fptype* BC,
    fptype* AgivenBC);

__global__ void GenerateJointDistributionFromColliderGivenAC(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    fptype* AC,
    fptype* BgivenAC);

__global__ void GenerateJointDistributionFromColliderGivenBA(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    fptype* AB,
    fptype* CgivenAB);

__global__ void GenerateJointDistributionFromColliderGivenCB(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    inttype num_C_cells,
    fptype* BC,
    fptype* AgivenBC);

__global__ void GenerateJointDistributionFromColliderGivenCA(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    inttype num_B_cells,
    inttype num_C_cells,
    fptype* AC,
    fptype* BgivenAC);

__global__ void GenerateMarginalAB(
    inttype num_marginal_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    inttype C_res,
    fptype* ABC);

__global__ void GenerateMarginalAC(
    inttype num_marginal_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    inttype C_res,
    fptype* ABC);

__global__ void GenerateMarginalBC(
    inttype num_marginal_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    inttype C_res,
    fptype* ABC);

__global__ void GenerateMarginalA(
    inttype num_marginal_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* AB);

__global__ void GenerateMarginalB(
    inttype num_marginal_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* AB);

__global__ void GenerateAGivenBC(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* BC,
    fptype* ABC);

__global__ void GenerateBGivenAC(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* AC,
    fptype* ABC);

__global__ void GenerateCGivenAB(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* AB,
    fptype* ABC);

__global__ void GenerateABGivenC(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* C,
    fptype* ABC);

__global__ void GenerateACGivenB(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* B,
    fptype* ABC);

__global__ void GenerateBCGivenA(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* A,
    fptype* ABC);

__global__ void GenerateAGivenB(
    inttype num_AB_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* B,
    fptype* AB);

__global__ void GenerateBGivenA(
    inttype num_AB_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* A,
    fptype* AB);

__global__ void GenerateJointADFromABCDDiamond(
    inttype num_AD_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    inttype C_res,
    fptype* A,
    fptype* BC_given_A,
    fptype* D_given_BC);

__global__ void transferMassBetweenGrids(
    inttype num_cells,
    fptype* in,
    fptype* out);

__global__ void sumMass(
    inttype num_cells,
    fptype* mass,
    fptype* sum);

__global__ void rescaleMass(
    inttype num_cells,
    fptype* sum,
    fptype* out);

__global__ void transpose(
    inttype num_out_cells,
    fptype* in,
    inttype in_A_cells,
    inttype in_B_cells,
    fptype* out);

#endif