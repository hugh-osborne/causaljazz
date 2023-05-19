#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm

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

    return [v + 0.1*v_prime, w + 0.1*w_prime, u + 0.1*u_prime]

def monte_carlo(I=0.0, num_neurons = 5000, sim_time = 10.0):
    timestep = 0.1 # ms

    tolerance = 1e-4
    tspan = np.linspace(0, sim_time, int(sim_time/timestep))
    current = np.array([[norm.rvs(-70.6, 0.4, 1)[0],norm.rvs(23.0, 0.1, 1)[0],norm.rvs(0.0, 0.1, 1)[0]] for a in range(num_neurons)])
    t_1 = np.empty((0,num_neurons,3))
    n = 0
    for t in tspan:
        n += 1
        
        for nn in range(num_neurons):
            current[nn] = cond(current[nn])
            if (current[nn][0] > -50.0):
                current[nn][0] = -70.6

            current[nn][1] = norm.rvs(23.0, 0.1, 1)[0] # override w

        # plot the marginal distributions
        fig, ax = plt.subplots(2,2)
        ax[0,0].hist(current[:,0], density=True, bins=100, range=[-80.0,-40.0], histtype='step')
        ax[0,1].hist(current[:,1], density=True, bins=100, range=[-1.0,25.0], histtype='step')
        ax[1,0].hist(current[:,2], density=True, bins=100, range=[-5.0,50.0], histtype='step')
        fig.tight_layout()
        plt.show()

        t_1 = np.concatenate((t_1,np.reshape(current, (1,num_neurons,3))))
    
    return tspan, np.array(t_1)

times, data = monte_carlo(I=0.0)
