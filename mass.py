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
    w_prime = 0.0
    u_prime = -(u) / tau_i

    return [v_prime, w_prime, u_prime]

num_iterations = 1000
poisson_input_rate = 3.0
dt = 0.1

cond_grid = pycausaljazz.generate(cond,  [-80.0,-1.0,-5.0], [40.0,26.0,55.0], [100,100,100], -50, -70.6, [0,0,0], dt)

pycausaljazz.init(dt, False)

pop3 = pycausaljazz.addPopulation(cond_grid, [-70.6, 13.001, 0.001], 0.0, False)

mass_grid = pycausaljazz.newDist([-80.0,-1.0,-5.0], [40.0,26.0,55.0], [100,100,100], [a for a in np.zeros(100*100*100)])
marginal_vw = pycausaljazz.newDist([-80.0,-1.0], [40.0,26.0], [100,100], [a for a in np.zeros(100*100)])
marginal_vu = pycausaljazz.newDist([-80.0,-5.0], [40.0,55.0], [100,100], [a for a in np.zeros(100*100)])
marginal_v = pycausaljazz.newDist([-80.0], [40.0], [100], [a for a in np.zeros(100)])
marginal_w = pycausaljazz.newDist([-1.0], [26.0], [100], [a for a in np.zeros(100)])
marginal_u = pycausaljazz.newDist([-5.0], [55.0], [100], [a for a in np.zeros(100)])

pycausaljazz.start()

start_time = time.perf_counter()
rates1 = []
times = []
for i in range(num_iterations):
    pycausaljazz.step()
    rates1 = rates1 + [pycausaljazz.readRates()[0]*1000]
    mass = pycausaljazz.readMass(pop3)
    
    pycausaljazz.update(mass_grid, [a for a in mass])

    pycausaljazz.marginal(mass_grid, 2, marginal_vw)
    pycausaljazz.marginal(mass_grid, 1, marginal_vu)
    pycausaljazz.marginal(marginal_vw, 1, marginal_v)
    pycausaljazz.marginal(marginal_vw, 0, marginal_w)
    pycausaljazz.marginal(marginal_vu, 0, marginal_u)

    dist_v = pycausaljazz.readDist(marginal_v)
    dist_w = pycausaljazz.readDist(marginal_w)
    dist_u = pycausaljazz.readDist(marginal_u)

    dist_v = [a / (40.0/100) for a in dist_v]
    dist_w = [a / (26.0/100) for a in dist_w]
    dist_u = [a / (55.0/100) for a in dist_u]

    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(np.linspace(-80.0,-40.0,100), dist_v)
    ax[0,1].plot(np.linspace(-1.0,25.0,100), dist_w)
    ax[1,0].plot(np.linspace(-5.0,55.0,100), dist_u)
    fig.tight_layout()
    plt.show()

    times = times + [i*dt/1000]
    
print("Completed in : ", time.perf_counter() - start_time, "seconds.")

pycausaljazz.shutdown()

fig, ax = plt.subplots(1, 1)

ax.set_title('Firing Rate')
ax.plot(times, rates1)
fig.tight_layout()

plt.show()