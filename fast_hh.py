from fastmass import fastmass
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time

from fastmc import fastmc

def hodgkin_huxley(y):
    V_k = -90
    V_na = 50
    V_l = -65
    g_k = 30
    g_na = 100
    g_l = 0.5
    C = 1.0
    V_t = -63

    v = y[0];
    m = y[1];
    n = y[2];
    h = y[3];

    alpha_m = (0.32 * (13 - v + V_t)) / (np.exp((13 - v + V_t) / 4) - 1)
    alpha_n = (0.032 * (15 - v + V_t)) / (np.exp((15 - v + V_t) / 5) - 1)
    alpha_h = 0.128 * np.exp((17 - v + V_t) / 18)

    beta_m = (0.28 * (v - V_t - 40)) / (np.exp((v - V_t - 40) / 5) - 1)
    beta_n = 0.5 * np.exp((10 - v + V_t) / 40)
    beta_h = 4 / (1 + np.exp((40 - v + V_t) / 5))

    v_prime = (-(g_k * n**4 * (v - V_k)) - (g_na * m**3 * h * (v - V_na)) - (g_l * (v - V_l)) + 5) / C
    m_prime = (alpha_m * (1 - m)) - (beta_m * m)
    n_prime = (alpha_n * (1 - n)) - (beta_n * n)
    h_prime = (alpha_h * (1 - h)) - (beta_h * h)

    return [h_prime, n_prime, m_prime, v_prime]

num_iterations = 1000
poisson_input_rate = 3.0
dt = 0.001

hh_grid = fastmass.generate(hodgkin_huxley, [-100.0,-1.0,-1.0,-1.0], [160.0,2.0,2.0,2.0], [50,50,50,50], -99.9, 59.9, [0,0,0,0], dt)

fastmass.init(dt, False)

pop1 = fastmass.addPopulation(hh_grid, [-70, 0.05, 0.3, 0.6], 0.0, False)

fastmass.start()

start_time = time.perf_counter()
times = []
avg_vs = []
for i in range(num_iterations):
    fastmass.step()
    times = times + [i*dt/1000]
    mass = fastmass.readMass(pop1)
    summ = []
    for vi in range(50):
        current_v = -100.0 + ((160.0/50.0)*vi) + ((160.0/2.0)/50.0)
        total = np.sum(mass[(vi*50*50*50):((vi+1)*50*50*50)]) * current_v
        summ = summ + [total]
    avg_vs = avg_vs + [np.sum(summ)]

print()
print()
print("FastMass Completed in : ", time.perf_counter() - start_time, "seconds.")
print()
print()

fastmass.shutdown()

fig, ax = plt.subplots(1, 1)

ax.set_title('Average V')
ax.plot(times, avg_vs)
fig.tight_layout()

### Finite Size

#num_neurons = 5000

#hh_grid = fastmc.generate(hodgkin_huxley, [-100.0,-1.0,-1.0,-1.0], [160.0,2.0,2.0,2.0], [50,50,50,50], -99.9, 59.9, [0,0,0,0], dt)

#fastmc.init(dt, False)

#pop1 = fastmc.addPopulation(hh_grid,[-70, 0.05, 0.3, 0.6], 0.0, num_neurons, False)

#fastmc.start()

#start_time = time.perf_counter()

#for i in range(num_iterations):
#    fastmc.step()

#print()
#print()
#print("Finite Size (", num_neurons,") Completed in : ", time.perf_counter() - start_time, "seconds.")
#print()
#print()
#fastmc.shutdown()

#fig, ax = plt.subplots(1, 1)

#ax.set_title('Firing Rate')
#ax.plot(times, rates_mass_lif)
#ax.plot(times, rates_miind_lif)
#ax.plot(times, [i/num_neurons/dt*1000 for i in rates_spike_lif])
#fig.tight_layout()

plt.show()