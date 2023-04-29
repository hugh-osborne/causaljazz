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
    C = 281
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

def lif(y):
    C = 281
    g_l = 30
    E_l = -70.6
    v_t = -50.4
    tau = 2.0
    alpha = 4.0
    tau_w = 144.0
    I = 0.0 #1000.0 

    v = y[0]

    v_prime = (-g_l*(v - E_l) + I) / C

    return [v_prime, 0.0]

num_iterations = 1000
poisson_input_rate = 3.0
dt = 0.1

adex_grid = pycausaljazz.generate(adex, [-90.0,-5.0], [75.0,445.0], [300,300], -40, -70.6, [0,100], dt) #600 x 300
lif_grid = pycausaljazz.generate(lif,  [-140.0,0.0], [140.0,0.005], [600,1], -40, -70.6, [0,0], dt)
cond_grid = pycausaljazz.generate(cond,  [-80.0,-1.0,-5.0], [40.0,26.0,55.0], [150,100,100], -50.4, -70.6, [0,0,0], dt)

pycausaljazz.init(dt, False)

pop1 = pycausaljazz.addPopulation(adex_grid, [-70.6, 0.1], 0.0, True)
pop2 = pycausaljazz.addPopulation(lif_grid, [-70.6, 0.001], 0.0, False)

conn1 = pycausaljazz.poisson(pop1, [1.0, 0.0])

pycausaljazz.connect(pop1,pop2,500.0,[1.0,0.0],0.0)

pycausaljazz.start()

start_time = time.perf_counter()
rates1 = []
rates2 = []
times = []
for i in range(num_iterations):
    pycausaljazz.postRate(pop1,conn1,poisson_input_rate)
    pycausaljazz.step()
    rates1 = rates1 + [pycausaljazz.readRates()[0]*1000]
    rates2 = rates2 + [pycausaljazz.readRates()[1]*1000]
    times = times + [i*dt/1000]
    
print("Completed in : ", time.perf_counter() - start_time, "seconds.")

pycausaljazz.shutdown()

fig, ax = plt.subplots(1, 1)

ax.set_title('Firing Rate')
ax.plot(times, rates1)
ax.plot(times, rates2)
fig.tight_layout()

plt.show()