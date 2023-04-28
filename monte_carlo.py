#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import random
import csv
from scipy.stats import poisson

def hh(current, dt, I=0):
    V_k = -90
    V_na = 50
    V_l = -65
    g_k = 30
    g_na = 100
    g_l = 0.5
    C = 1.0
    V_t = -63
    
    v = current[0]
    m = current[1] 
    n = current[2]
    h = current[3]

    alpha_m = (0.32 * (13 - v + V_t)) / (np.exp((13 - v + V_t)/4) - 1) 
    alpha_n = (0.032 * (15 - v + V_t)) / (np.exp((15 - v + V_t)/5) - 1)
    alpha_h = 0.128 * np.exp((17 - v + V_t)/18)

    beta_m = (0.28 * (v - V_t - 40)) / (np.exp((v - V_t - 40)/5) - 1)
    beta_n = 0.5 * np.exp((10 - v + V_t)/40)
    beta_h = 4 / (1 + np.exp((40 - v + V_t)/5))
    

    v_prime = (-(g_k*(n**4)*(v - V_k)) - (g_na*(m**3)*h*(v - V_na)) - (g_l*(v - V_l)) + I) / C
    m_prime = (alpha_m * (1 - m)) - (beta_m*m)
    n_prime = (alpha_n * (1 - n)) - (beta_n*n)
    h_prime = (alpha_h * (1 - h)) - (beta_h*h)

    return [v + dt*v_prime, m + dt*m_prime, n + dt*n_prime, h + dt*h_prime]

def simulate_hh(I=0.0):
    sim_time = 50 # ms
    timestep = 0.001 # ms
    v0 = -70 # mV
    m0 = 0.05
    n0 = 0.3
    h0 = 0.6
    tolerance = 1e-4
    tspan = np.linspace(0, sim_time, int(sim_time/timestep))
    current = [v0,m0,n0,h0]
    t_1 = []
    n = 0
    for t in tspan:
        n += 1
        print(n)
        current = hh(current, timestep, I)
        t_1 = t_1 + [current]

    return tspan, np.array(t_1)
    
def simulate_hh_pop(I=0.0, num_neurons = 500, sim_time = 0.5):
    timestep = 0.001 # ms
    v0 = -70 # mV
    m0 = 0.05
    n0 = 0.3
    h0 = 0.6
    tolerance = 1e-4
    tspan = np.linspace(0, sim_time, int(sim_time/timestep))
    current = np.array([[v0,m0,n0,h0] for a in range(num_neurons)])
    t_1 = np.empty((0,num_neurons,4))
    n = 0
    for t in tspan:
        n += 1
        print(n)
        
        for nn in range(num_neurons):
            current[nn] = hh(current[nn], timestep, I)
            current[nn][0] = current[nn][0] + (poisson.rvs(85000*0.00001)*0.5)
            current[nn][0] = current[nn][0] + (poisson.rvs(225000*0.00001)*0.5)
            current[nn][0] = current[nn][0] - (poisson.rvs(225000*0.00001)*0.5)
            
            current[nn][1] = current[nn][1] + (poisson.rvs(5000*0.00001)*0.01)
            current[nn][1] = current[nn][1] - (poisson.rvs(5000*0.00001)*0.01)
            
            current[nn][2] = current[nn][2] + (poisson.rvs(5000*0.00001)*0.01)
            current[nn][2] = current[nn][2] - (poisson.rvs(5000*0.00001)*0.01)
            
            current[nn][3] = current[nn][3] + (poisson.rvs(5000*0.00001)*0.01)
            current[nn][3] = current[nn][3] - (poisson.rvs(5000*0.00001)*0.01)
        
        t_1 = np.concatenate((t_1,np.reshape(current, (1,num_neurons,4))))
    
    return tspan, np.array(t_1)

times, data = simulate_hh(I=5.0)

#plot v over time
#Bursting lasts for just over 0.2 seconds
f = plt.figure()
plt.plot(times, data[:,0])
plt.show()

# with 3 second quiescence (I = 0.05)
plt.xlabel('Membrane Potential (v), mV')
plt.ylabel('(n)')
plt.plot(data[0:,0],data[0:,2])
plt.show()

plt.xlabel('Membrane Potential (v), mV')
plt.ylabel('(m)')
plt.plot(data[0:,0],data[0:,1])
plt.show()

plt.xlabel('Membrane Potential (v), mV')
plt.ylabel('(h)')
plt.plot(data[0:,0],data[0:,3])
plt.show()
