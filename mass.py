from pycausaljazz import pycausaljazz
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time

def cond(y):
    E_l = -70.6
    V_thres = -50.4 
    E_e = 0.0
    E_i = -75
    C = 28.1
    g_l = 0.03
    tau_e = 2.728
    tau_i = 10.49

    v = y[0]
    w = y[1]
    u = y[2]

    v_prime = (-g_l*(v - E_l) - w * (v - E_e) - u * (v - E_i)) / C
    w_prime = -(w) / tau_e
    u_prime = -(u) / tau_i

    return [v_prime, w_prime, u_prime]

num_iterations = 1000
poisson_input_rate = 3.0
dt = 0.1

cond_grid = pycausaljazz.generate(cond,  [-80.0,-1.0,-5.0], [40.0,26.0,55.0], [150,100,100], -40.4, -79.6, [0,0,0], dt)

pycausaljazz.init(dt, False)

pop3 = pycausaljazz.addPopulation(cond_grid, [-60.6, 0.001, 0.001], 0.0, True)

pycausaljazz.start()

start_time = time.perf_counter()
rates1 = []
times = []
for i in range(num_iterations):
    pycausaljazz.step()
    rates1 = rates1 + [pycausaljazz.readRates()[0]*1000]
    times = times + [i*dt/1000]
    
print("Completed in : ", time.perf_counter() - start_time, "seconds.")

pycausaljazz.shutdown()

fig, ax = plt.subplots(1, 1)

ax.set_title('Firing Rate')
ax.plot(times, rates1)
fig.tight_layout()

plt.show()