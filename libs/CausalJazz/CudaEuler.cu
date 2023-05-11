#include "CudaEuler.cuh"
#include <stdio.h>

__device__ int modulo(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

__device__ int fact(int n) {
    if ((n == 0) || (n == 1))
        return 1;
    else
        return n * fact(n - 1);
}

__global__ void initCurand(curandState* state, unsigned long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, index, 0, &state[index]);
}

__global__ void initRandomNumbers(inttype num_rands, 
    fptype* random_nums, curandState* state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_rands; i += stride) {
        random_nums[i] = curand_uniform(&state[index]);
    }
}

__global__ void ReduceRefractoryTimers(
    inttype num_neurons,
    inttype neuron_offset,
    fptype* refractory_timers,
    fptype timestep)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_neurons; i += stride) {
        inttype idx = i + neuron_offset;
        
        if (refractory_timers[idx] >= 0) {
            refractory_timers[idx] -= timestep;
        }
            
    }
}

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
    fptype* refractory_timers)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_neurons; i += stride) {
        inttype idx = i + neuron_offset;

        if (refractory_timers[idx] > 0)
            continue;

        fptype r = random_numbers[modulo(idx, num_rands)];
        if (r < proportions[neuron_cell_locations[idx]]) {
            neuron_cell_locations[idx] = modulo(neuron_cell_locations[idx] + offsetsplus[neuron_cell_locations[idx]], num_cells);
        }
        else {
            neuron_cell_locations[idx] = modulo(neuron_cell_locations[idx] + offsets[neuron_cell_locations[idx]], num_cells);
        }
    }
}

__global__ void ResetSpikes(
    inttype num_spiked_values,
    inttype* spiked) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_spiked_values; i += stride) {
        spiked[i] = 0;
    }
}

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
    inttype* connection_counts,
    inttype* connection_offsets,
    inttype* hebbian_connection_counts,
    inttype* hebbian_connection_offsets,
    fptype* jump_queues,
    inttype* neuron_to_queues,
    inttype num_dimensions,
    inttype current_jump_position,
    inttype max_jump_position,
    fptype* refractory_timers,
    fptype* refractory_periods)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_neurons; i += stride) {
        unsigned int io = i + neuron_offset;
        unsigned int idx = io * bit_count;
        inttype v_cell = modulo(neuron_cell_locations[io], threshold_dim_offset);
        if (v_cell > threshold_cell) {
            refractory_timers[io] = refractory_periods[io]; // Note we hold the neurons at their reset potential during refractory period.

            neuron_cell_locations[io] = modulo((neuron_cell_locations[io] - v_cell) + reset_cell + reset_jump_cell, num_cells);
            spiked[int(io /bit_count)] |= 1 << modulo(io,bit_count);
                
            for (int c = 0; c < connection_counts[io]; c++) {
                inttype conn_index = connection_offsets[io] + c;
                inttype neuron_index = targets[conn_index];

                for (unsigned int d = 0; d < num_dimensions; d++) {
                    atomicAdd(&jump_queues[neuron_to_queues[neuron_index] + (d * max_jump_position) + modulo(current_jump_position + delays[conn_index], max_jump_position)], jumps[connections_to_jumps[conn_index] + d]);
                }
            }

            for (int c = 0; c < hebbian_connection_counts[io]; c++) {
                inttype conn_index = hebbian_connection_offsets[io] + c;
                inttype neuron_index = targets[conn_index];

                for (unsigned int d = 0; d < num_dimensions; d++) {
                    atomicAdd(&jump_queues[neuron_to_queues[neuron_index] + (d * max_jump_position) + modulo(current_jump_position + delays[conn_index], max_jump_position)], jumps[connections_to_jumps[conn_index] + d]);
                }
            }
        }
    }
}

__global__ void SpikeExternalInput(
    inttype num_connections,
    inttype* connection_ids,
    inttype* neuron_incoming_spike_value_queues,
    inttype* neuron_connection_delays,
    curandState* state) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_connections; i += stride) {
        inttype bit_setter = 1;
        bit_setter = bit_setter << (neuron_connection_delays[connection_ids[i]]);
        neuron_incoming_spike_value_queues[connection_ids[i]] |= bit_setter;
    }
}

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
    fptype* refractory_timers)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_neurons; i += stride) {
        inttype idx = i + neuron_offset;

        if (refractory_timers[idx] > 0) { // ignore all incoming spikes when refracting
            for (unsigned int d = 0; d < num_dimensions; d++) {
                jump_queues[neuron_to_queues[idx]
                    + (d * max_jump_position) + current_jump_position] = 0.0;
            }
            continue;
        }

        inttype v_cell = modulo(neuron_cell_locations[idx], threshold_dim_offset);

        for (unsigned int d = 0; d < num_dimensions; d++) {
            
            fptype jump = jump_queues[neuron_to_queues[idx]
                + (d * max_jump_position) + current_jump_position];

            if (jump == 0.0)
                continue;

            jump_queues[neuron_to_queues[idx]
                + (d * max_jump_position) + current_jump_position] = 0.0;

            inttype jump_cell_offs = floor(fabs(jump) / grid_cell_width[d]);
            inttype jump_cell_offsp = jump_cell_offs + 1;
            if (jump < 0) {
                jump_cell_offs = -jump_cell_offs;
                jump_cell_offsp = jump_cell_offs - 1;
            }
            fptype rem = fabs(jump) - (fptype)(jump_cell_offs * grid_cell_width[d]);
            fptype jump_prop = rem / grid_cell_width[d];
            
            fptype r = random_numbers[modulo(i + d, num_rands)];

            if (r < jump_prop) {
                neuron_cell_locations[idx] = modulo(neuron_cell_locations[idx] + jump_cell_offsp, num_cells);
            }
            else {
                neuron_cell_locations[idx] = modulo(neuron_cell_locations[idx] + jump_cell_offs, num_cells);
            }
        }
    }
}


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
    curandState* state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < connection_counts[connection_id]; i += stride) {
        inttype conn_index = connection_offsets[connection_id] + i;
        inttype neuron_index = targets[conn_index];

        inttype count = curand_poisson(&state[index], rate * timestep);
        
        for (unsigned int d = 0; d < neurons_to_dims[neuron_index]; d++) {
            atomicAdd(&jump_queues[neurons_to_queues[neuron_index] + (d * max_jump_position) + current_jump_position /* (0 delay) */], count * jumps[connection_to_jumps[conn_index] + d]);
        }
    }
}

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
    inttype* targets)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < connection_counts[connection_id]; i += stride) {
        inttype conn_index = connection_offsets[connection_id] + i;
        inttype neuron_index = targets[conn_index];

        int count = spikes[int(i / bit_count)] >> modulo(i, bit_count) & 1;

        for (unsigned int d = 0; d < connections_to_dims[conn_index]; d++) {
            atomicAdd(&jump_queues[neurons_to_queues[neuron_index] + (d * max_jump_position) + current_jump_position/* (0 delay) */], count * jumps[connections_to_jumps[conn_index] + d]);
        }
    }
}

// STDP rule taken from Song 2000, Nature Neuroscience
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
    fptype time)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_neurons; i += stride) {
        if ((spiked[int(i / bit_count)] >> modulo(i, bit_count)) & 1 == 1) { // this neuron spiked this iteration
            // update the pre spike times for all connections where this is the source neuron
            for (unsigned int c = hebbian_connection_offsets[i]; c < hebbian_connection_offsets[i] + hebbian_connection_counts[i]; c++) {
                spike_time_pre[c] = time;
            }
            // update the post spike times for all connections where this is the target neuron
            for (unsigned int c = target_neuron_connection_offsets[i]; c < target_neuron_connection_offsets[i] + target_neuron_connection_counts[i]; c++) {
                spike_time_post[target_neuron_connection_ids[c]] = time;
            }
        }
    }
}

__global__ void UpdateHebbianJumps(
    inttype num_connections,
    inttype* hebbian_connection_counts,
    inttype* hebbian_connection_offsets,
    fptype* spike_time_pre,
    fptype* spike_time_post,
    fptype* jumps,
    inttype* connections_to_jumps,
    inttype* connections_to_dims)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_connections; i += stride) {
        for (int c = hebbian_connection_offsets[i]; c < hebbian_connection_offsets[i] + hebbian_connection_counts[i]; c++) {
            for (int d = 0; d < connections_to_dims[c]; d++) {
                if (spike_time_pre[c] > 0 && spike_time_post[c] > 0 && (spike_time_pre[c] - spike_time_post[c]) < 0) {
                    jumps[connections_to_jumps[c] + d] = fminf(1.5, jumps[connections_to_jumps[c] + d] + 1 * expf((spike_time_pre[c] - spike_time_post[c]) / 10)); // A_+ = 0.1, tau_+ = 1.0
                }
                if (spike_time_pre[c] > 0 && spike_time_post[c] > 0 && (spike_time_pre[c] - spike_time_post[c]) > 0) {
                    jumps[connections_to_jumps[c] + d] = fmaxf(0.0, jumps[connections_to_jumps[c] + d] - 0.05 * expf(-(spike_time_pre[c] - spike_time_post[c]) / 0.1)); // A_- = 0.1, tau_- = 1.0
                }
            }  
        }
    }
}

__global__ void DeterministicMass(
    inttype num_cells,
    fptype* mass,
    fptype* dydt,
    int* offsets,
    fptype* proportions,
    inttype* transition_counts,
    inttype* transition_offsets)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_cells; i += stride) {
        dydt[i] = 0.0;
        for (int t = transition_offsets[i]; t < transition_offsets[i] + transition_counts[i]; t++) {
            dydt[i] += mass[modulo(i+offsets[t],num_cells)] * proportions[t];
        }
        dydt[i] -= mass[i];
    }
}

__global__ void ApplyDydt(
    inttype num_cells,
    fptype* mass,
    fptype* dydt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_cells; i += stride) {
        mass[i] += dydt[i];
    }
}

__global__ void PostValueToRingBuffer(
    fptype* queue,
    inttype queue_length,
    inttype current_position,
    fptype value)
{
    queue[modulo(current_position, queue_length)] = value;
}

__global__ void CalculateKernel(
    inttype kernel_width,
    fptype* kernel,
    fptype* delay_buffer,
    inttype delay_buffer_length,
    inttype iteration_count,
    fptype weight,
    fptype timestep,
    fptype jump,
    fptype cell_width)
{
    for (inttype i = 0; i < kernel_width; i++)
        kernel[i] = 0.0;

    inttype events = 0;
    fptype jump_cumulative = 0.0;
    fptype lambda = weight * delay_buffer[modulo(iteration_count, delay_buffer_length)] * timestep;
    fptype mass_check = 0.0;
    while (true) {
        if (jump > 0) {
            inttype index = int((kernel_width / 2.0) + (jump_cumulative / cell_width));
            if (index >= kernel_width-1 || index < 0)
                break;

            fptype prob = (pow(lambda, events) * exp(-lambda)) / fact(events);
            
            if (events > lambda && prob < 0.00001)
                break;

            fptype rest = (((kernel_width / 2.0) + (jump_cumulative / cell_width)) - index);

            kernel[index] += prob * (1.0 - rest);
            kernel[index+1] += prob * rest;
            mass_check += prob;
            
        }
        else {
            inttype index = int((kernel_width / 2.0) + (jump_cumulative / cell_width));
            if (index >= kernel_width || index < 1)
                break;

            fptype prob = (pow(lambda, events) * exp(-lambda)) / fact(events);

            if (events > lambda && prob < 0.00001)
                break;

            fptype rest = (((kernel_width / 2.0) + (jump_cumulative / cell_width)) - index);

            kernel[index] += prob * (1.0 - rest);
            kernel[index - 1] += prob * rest;
            mass_check += prob;
        }

        jump_cumulative += jump;
        events++;
        
    }

    // Make sure the kernel probability sums to 1
    for (inttype i = 0; i < kernel_width; i++)
        kernel[i] *= 1.0 / mass_check;
        

}

__global__ void NaiveConvolveKernel(
    inttype num_cells,
    fptype* mass,
    fptype* dydt,
    fptype* kernel,
    inttype kernel_width,
    inttype dim_stride)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_cells; i += stride) {
        dydt[i] = 0.0;
        for (int j = 0; j < kernel_width; j++) {
            dydt[i] += mass[modulo(i - ((j - int(kernel_width/2.0)) * dim_stride), num_cells)] * kernel[j];
        }
        dydt[i] -= mass[i];
    }
}

__global__ void ThresholdResetMass(
    inttype num_threshold_cells,
    inttype* threshold_cells,
    inttype* reset_cells,
    fptype* reset_mass,
    fptype* reset_queues,
    inttype queue_length,
    inttype current_queue_pointer,
    fptype* mass)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_threshold_cells; i += stride) {
        reset_queues[(i * queue_length) + modulo(current_queue_pointer-1,queue_length)] += mass[threshold_cells[i]];
        reset_mass[i] = mass[threshold_cells[i]];
        mass[threshold_cells[i]] = 0.0;
    }
}

__global__ void HandleResetMass(
    inttype num_reset_cells,
    inttype* reset_cells,
    fptype* reset_queues,
    inttype queue_length,
    inttype current_queue_pointer,
    fptype* mass)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_reset_cells; i += stride) {
        mass[reset_cells[i]] += reset_queues[(i * queue_length) + current_queue_pointer];
        reset_queues[(i * queue_length) + current_queue_pointer] = 0.0;
    }
}

__global__ void SumResetMass(
    inttype num_reset_cells,
    fptype* reset_mass,
    fptype* stored_rates,
    inttype population_id,
    fptype timestep)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    stored_rates[population_id] = 0.0;

    for (int i = index; i < num_reset_cells; i += stride) {
        stored_rates[population_id] += reset_mass[i] / timestep;
    }
}

__global__ void PassRatesToQueues(
    fptype* stored_rates,
    inttype population_id,
    fptype* queue,
    inttype queue_length,
    inttype current_position)
{
    queue[modulo(current_position-1, queue_length)] = stored_rates[population_id];
}

// Causal Jazz

// Result is A = dim0, B = dim1
__global__ void GenerateJointDistributionGivenA(
    inttype num_AB_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    fptype* BgivenA)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_AB_cells; i += stride) {
        inttype A_index = modulo(i, num_A_cells);
        out[i] = A[A_index] * BgivenA[i];
    }
}

// Result is A = dim0, B = dim1
__global__ void GenerateJointDistributionGivenB(
    inttype num_AB_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* B,
    fptype* AgivenB)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_AB_cells; i += stride) {
        inttype B_index = int(i / num_A_cells);
        out[i] = B[B_index] * AgivenB[i];
    }
}

// Result is A = dim0, B = dim1, C = dim2
__global__ void GenerateJointDistributionFromFork(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    inttype num_B_cells,
    fptype* BgivenA,
    inttype num_C_cells,
    fptype* CgivenA)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {
        inttype C_joint = int(i / (num_A_cells * num_B_cells));
        inttype B_joint = int(modulo(i, num_A_cells * num_B_cells) / num_A_cells);
        inttype A_joint = modulo(i, num_A_cells);
        out[i] = A[A_joint] * BgivenA[(B_joint*num_A_cells)+A_joint] * CgivenA[(C_joint * num_A_cells) + A_joint];
    }
}

__global__ void GenerateJointDistributionFromForkBgivenA(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    inttype num_B_cells,
    fptype* BgivenA,
    inttype num_C_cells,
    fptype* CgivenA)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {
        inttype C_joint = int(i / (num_A_cells * num_B_cells));
        inttype B_joint = int(modulo(i, num_A_cells * num_B_cells) / num_A_cells);
        inttype A_joint = modulo(i, num_A_cells);
        out[i] = A[A_joint] * BgivenA[(A_joint * num_B_cells) + B_joint] * CgivenA[(C_joint * num_A_cells) + A_joint];
    }
}

__global__ void GenerateJointDistributionFromForkCgivenA(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    inttype num_B_cells,
    fptype* BgivenA,
    inttype num_C_cells,
    fptype* CgivenA)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {
        inttype C_joint = int(i / (num_A_cells * num_B_cells));
        inttype B_joint = int(modulo(i, num_A_cells * num_B_cells) / num_A_cells);
        inttype A_joint = modulo(i, num_A_cells);
        out[i] = A[A_joint] * BgivenA[(B_joint * num_A_cells) + A_joint] * CgivenA[(A_joint * num_C_cells) + C_joint];
    }
}

__global__ void GenerateJointDistributionFromForkBgivenACgivenA(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    inttype num_B_cells,
    fptype* BgivenA,
    inttype num_C_cells,
    fptype* CgivenA)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {
        inttype C_joint = int(i / (num_A_cells * num_B_cells));
        inttype B_joint = int(modulo(i, num_A_cells * num_B_cells) / num_A_cells);
        inttype A_joint = modulo(i, num_A_cells);
        out[i] = A[A_joint] * BgivenA[(A_joint * num_B_cells) + B_joint] * CgivenA[(A_joint * num_C_cells) + C_joint];
    }
}

// Result is A = dim0, B = dim1, C = dim2
__global__ void GenerateJointDistributionFromCollider(
    inttype num_ABC_cells,
    fptype* out,
    inttype num_A_cells,
    fptype* A,
    inttype num_B_cells,
    fptype* B,
    fptype* CgivenAB)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {
        inttype C_joint = int(i / (num_A_cells * num_B_cells));
        inttype B_joint = int(modulo(i, num_A_cells * num_B_cells) / num_A_cells);
        inttype A_joint = modulo(i, num_A_cells);
        out[i] = A[A_joint] * B[B_joint] * CgivenAB[i];
    }
}

// Result is AB from ABC
__global__ void GenerateMarginalAB(
    inttype num_marginal_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    inttype C_res,
    fptype* A)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_marginal_cells; i += stride) {

        inttype B_joint = int(i / A_res);
        inttype A_joint = modulo(i, A_res);

        fptype total_mass = 0.0;
        for (unsigned int j = 0; j < C_res; j++) {
            total_mass += A[(j * A_res * B_res) + (B_joint * A_res) + A_joint];
        }
        out[i] = total_mass;
    }
}

// Result is BC from ABC
__global__ void GenerateMarginalBC(
    inttype num_marginal_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    inttype C_res,
    fptype* A)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_marginal_cells; i += stride) {

        inttype C_joint = int(i / B_res);
        inttype B_joint = modulo(i, B_res);

        fptype total_mass = 0.0;
        for (unsigned int j = 0; j < A_res; j++) {
            total_mass += A[(C_joint * A_res * B_res) + (B_joint * A_res) + j];
        }
        out[i] = total_mass;
    }
}

// Result is AC from ABC
__global__ void GenerateMarginalAC(
    inttype num_marginal_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    inttype C_res,
    fptype* A)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_marginal_cells; i += stride) {

        inttype C_joint = int(i / A_res);
        inttype A_joint = modulo(i, A_res);

        fptype total_mass = 0.0;
        for (unsigned int j = 0; j < B_res; j++) {
            total_mass += A[(C_joint * A_res * B_res) + (j * A_res) + A_joint];
        }
        out[i] = total_mass;
    }
}

// Result is A from AB
__global__ void GenerateMarginalA(
    inttype num_marginal_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* A)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_marginal_cells; i += stride) {
        fptype total_mass = 0.0;
        for (unsigned int j = 0; j < B_res; j++) {
            total_mass += A[(j * A_res) + i];
        }
        out[i] = total_mass;
    }
}

// Result is B from AB
__global__ void GenerateMarginalB(
    inttype num_marginal_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* A)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_marginal_cells; i += stride) {
        fptype total_mass = 0.0;
        for (unsigned int j = 0; j < A_res; j++) {
            total_mass += A[(i * A_res) + j];
        }
        
        out[i] = total_mass;
    }
}

// Result is A|BC
__global__ void GenerateAGivenBC(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* BC,
    fptype* ABC)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {

        inttype C_joint = int(i / (A_res * B_res));
        inttype B_joint = int(modulo(i, A_res * B_res) / A_res);

        out[i] = ABC[i] / BC[(C_joint*B_res) + B_joint];
    }
}

// Result is B|AC
__global__ void GenerateBGivenAC(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* AC,
    fptype* ABC)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {

        inttype C_joint = int(i / (A_res * B_res));
        inttype A_joint = modulo(i, A_res);

        out[i] = ABC[i] / AC[(C_joint * B_res) + A_joint];
    }
}

// Result is C|AB
__global__ void GenerateCGivenAB(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* AB,
    fptype* ABC)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {

        inttype B_joint = int(modulo(i, A_res * B_res) / A_res);
        inttype A_joint = modulo(i, A_res);

        out[i] = ABC[i] / AB[(B_joint * A_res) + A_joint];
    }
}

// Result is AB|C
__global__ void GenerateABGivenC(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* C,
    fptype* ABC)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {

        inttype C_joint = int(i / (A_res * B_res));

        out[i] = ABC[i] / C[C_joint];
    }
}

// Result is AC|B
__global__ void GenerateACGivenB(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* B,
    fptype* ABC)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {

        inttype B_joint = int(modulo(i, A_res * B_res) / A_res);

        out[i] = ABC[i] / B[B_joint];
    }
}

// Result is BC|A
__global__ void GenerateBCGivenA(
    inttype num_ABC_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* A,
    fptype* ABC)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_ABC_cells; i += stride) {

        inttype A_joint = modulo(i, A_res);

        out[i] = ABC[i] / A[A_joint];
    }
}

// Result is A|B
__global__ void GenerateAGivenB(
    inttype num_AB_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* B,
    fptype* AB)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_AB_cells; i += stride) {

        inttype B_joint = int(i / A_res);

        out[i] = AB[i] / B[B_joint];
    }
}

// Result is B|A
__global__ void GenerateBGivenA(
    inttype num_AB_cells,
    fptype* out,
    inttype A_res,
    inttype B_res,
    fptype* A,
    fptype* AB)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_AB_cells; i += stride) {

        inttype A_joint = modulo(i, A_res);

        out[i] = AB[i] / A[A_joint];
    }
}