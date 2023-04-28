from fastmass import fastmass
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time

import miind.grid_generate as grid_generate
import miind.miindgen as miindgen
import miind.miindsimv as miind

from fastmc import fastmc

def adex(y):
    C = 281
    g_l = 30
    E_l = -70.6
    v_t = -50.4
    tau = 2.0
    alpha = 4.0
    tau_w = 144.0
    I = 0.0 #1000.0 

    v = y[0]
    w = y[1]

    v_prime = (-g_l*(v - E_l) + g_l*tau*np.exp((v - v_t)/tau) - w + I) / C
    w_prime = (alpha*(v - E_l) - w) / tau_w

    return [v_prime, w_prime]

def adex_rev_for_miind(y):
    C = 281
    g_l = 30
    E_l = -70.6
    v_t = -50.4
    tau = 2.0
    alpha = 4.0
    tau_w = 144.0
    I = 0.0 #1000.0 

    v = y[1]
    w = y[0]

    v_prime = (-g_l*(v - E_l) + g_l*tau*np.exp((v - v_t)/tau) - w + I) / C
    w_prime = (alpha*(v - E_l) - w) / tau_w

    return [w_prime, v_prime]

def lif(y):
    C = 281
    g_l = 30
    E_l = -70.6
    I = 0.0 #1000.0 

    v = y[0]
    w = y[1]

    v_prime = (-g_l*(v - E_l) + I) / C
    w_prime = 0.0

    return [v_prime, w_prime]

def lif_rev_for_miind(y):
    C = 281
    g_l = 30
    E_l = -70.6
    I = 0.0 #1000.0 

    v = y[1]
    w = y[0]

    v_prime = (-g_l*(v - E_l) + I) / C
    w_prime = 0.0

    return [w_prime, v_prime]

num_iterations = 1000
poisson_input_rate = 3.0
dt = 0.1

adex_grid = fastmass.generate(adex, [-90.0,-5.0], [75.0,445.0], [200,200], -40, -70.6, [0,100], dt)
lif_grid = fastmass.generate(lif,  [-140.0,0.0], [140.0,0.005], [600,1], -40, -70.6, [0,0], dt)

fastmass.init(dt, False)

pop1 = fastmass.addPopulation(adex_grid, [-70.6, 0.1], 5.0, False)
pop2 = fastmass.addPopulation(lif_grid, [-70.6, 0.001], 5.0, False)

conn1 = fastmass.poisson(pop1, [1.0, 0.0])

fastmass.connect(pop1,pop2,50.0,[1.0,0.0],5.0)

fastmass.start()

start_time = time.perf_counter()
rates_mass_adex = []
rates_mass_lif = []
times = []
for i in range(num_iterations):
    fastmass.postRate(pop1,conn1,poisson_input_rate)
    fastmass.step()
    rates_mass_adex = rates_mass_adex + [fastmass.readRates()[pop1]*1000]
    rates_mass_lif = rates_mass_lif + [fastmass.readRates()[pop2]*1000]
    times = times + [i*dt/1000]
    
print()
print()
print("FastMass Completed in : ", time.perf_counter() - start_time, "seconds.")
print()
print()

fastmass.shutdown()

### MIIND simulation

#miindgen.generateNdGrid(adex_rev_for_miind, 'adex', [-5.0,-90], [445.0,75.0], [200,200], -40, -70.6, [100.0,0.0], dt, 0.001)
#miindgen.generateNdGrid(lif_rev_for_miind, 'lif', [0.0,-140], [0.005,140.0], [1,600], -40, -70.6, [0.0,0.0], dt, 0.001)

miind.init(1, "adex.xml")
timestep = miind.getTimeStep()
simulation_length = miind.getSimulationLength()

print('Timestep from XML : {}'.format(timestep))
print('Sim time from XML : {}'.format(simulation_length))

miind.startSimulation()

start_time = time.perf_counter()

constant_input = [poisson_input_rate*1000,0.0]
rates_miind_adex = []
rates_miind_lif = []
for i in range(int(simulation_length/timestep)):
    rates = miind.evolveSingleStep(constant_input)
    #print(rates)
    rates_miind_adex.append(rates[0])
    rates_miind_lif.append(rates[1])

print()
print()
print("Miind Completed in : ", time.perf_counter() - start_time, "seconds.")
print()
print()

#miind.endSimulation()

### Finite Size

num_neurons = 5000

adex_grid = fastmc.generate(adex, [-90.0,-5.0], [75.0,445.0], [200,200], -40, -70.6, [0,100], dt)
lif_grid = fastmc.generate(lif,  [-140.0,0.0], [140.0,0.005], [600,1], -40, -70.6, [0,0], dt)

fastmc.init(dt, False)

pop1 = fastmc.addPopulation(adex_grid, [-70.6, 0.1], 5.0, num_neurons, False)
pop2 = fastmc.addPopulation(lif_grid, [-70.6, 0.001], 5.0, num_neurons, False)

fastmc.poissonPop([1.0 ,0.0], poisson_input_rate, pop1,  [a for a in range(num_neurons)])

fastmc.connectPop([1.0,0.0],5.0,pop1,pop2,50)

fastmc.start()

start_time = time.perf_counter()

rates_spike_adex = []
rates_spike_lif = []
for i in range(num_iterations):
    fastmc.step()
    rates_spike_adex = rates_spike_adex + [len(fastmc.spikesPop(pop1))]
    rates_spike_lif = rates_spike_lif + [len(fastmc.spikesPop(pop2))]

print()
print()
print("Finite Size (", num_neurons,") Completed in : ", time.perf_counter() - start_time, "seconds.")
print()
print()
fastmc.shutdown()

fig, ax = plt.subplots(1, 1)

ax.set_title('Firing Rate')
ax.plot(times, rates_mass_adex)
ax.plot(times, rates_miind_adex)
ax.plot(times, [i/num_neurons/dt*1000 for i in rates_spike_adex])
fig.tight_layout()

fig, ax = plt.subplots(1, 1)

ax.set_title('Firing Rate')
ax.plot(times, rates_mass_lif)
ax.plot(times, rates_miind_lif)
ax.plot(times, [i/num_neurons/dt*1000 for i in rates_spike_lif])
fig.tight_layout()

plt.show()