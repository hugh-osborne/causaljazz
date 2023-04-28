from fastgraph import fastgraph
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time
import math

dt = 0.001

def m_prime(y):
    m = y[0]
    v = y[1]

    V_t = -63

    if (13 - v + V_t) == 0:
        alpha_m = 0.0
    else:
        alpha_m = (0.32 * (13 - v + V_t)) / (np.exp((13 - v + V_t) / 4) - 1)
    beta_m = (0.28 * (v - V_t - 40)) / (np.exp((v - V_t - 40) / 5) - 1)
    m_prime = (alpha_m * (1 - m)) - (beta_m * m)

    return m + (dt*m_prime)

def n_prime(y):
    n = y[0]
    v = y[1]

    V_t = -63
    
    if (15 - v + V_t) == 0:
        alpha_n = 0.0
    else:
        alpha_n = (0.032 * (15 - v + V_t)) / (np.exp((15 - v + V_t) / 5) - 1)
    beta_n = 0.5 * np.exp((10 - v + V_t) / 40)
    n_prime = (alpha_n * (1 - n)) - (beta_n * n)

    return n + (dt*n_prime)

def h_prime(y):
    h = y[0]
    v = y[1]

    V_t = -63

    alpha_h = 0.128 * np.exp((17 - v + V_t) / 18)
    beta_h = 4 / (1 + np.exp((40 - v + V_t) / 5))
    h_prime = (alpha_h * (1 - h)) - (beta_h * h)

    return h + (dt*h_prime)

def pot(y):
    n = y[0]
    v = y[1]

    V_k = -90

    return (n**4 * (v - V_k))

def sod_gate(y):
    m = y[0]
    h = y[1]

    return ( m**3 * h)

def sod(y):
    sod_gate = y[0]
    v = y[1]

    V_na = 50

    return sod_gate * (v - V_na)

def potsod(y):
    pot = y[0]
    sod = y[1]
    
    g_na = 100
    g_k = 30

    return (dt*(-(g_k * pot) - (g_na * sod)))

def v_prime(y):
    potsod = y[0]
    v = y[1]

    g_l = 0.5
    V_l = -65
    C = 1.0

    v_prime = ((g_l*(v - V_l)) + 5) / C
    
    return v + potsod - (dt*v_prime)

v_res = 100
gate_res = 100
pot_res = 100
sod_gate_res = 100
sod_res = 100
potsod_res = 100

gate_min = -1.0
gate_max = 1.0

v_min = -100.0
v_max = 60.0

print("Generating transition matrices...")

gate_space = np.linspace(gate_min,gate_max,gate_res)
v_space = np.linspace(v_min,v_max,v_res)

func_m_prime = fastgraph.definedfunction(m_prime, gate_min, gate_max, gate_res)
m_prime_input_m = fastgraph.input(func_m_prime,gate_min,gate_max,gate_res)
m_prime_input_v = fastgraph.input(func_m_prime,v_min,v_max,v_res)
fastgraph.generate(func_m_prime)

func_n_prime = fastgraph.definedfunction(n_prime, gate_min, gate_max, gate_res)
n_prime_input_n = fastgraph.input(func_n_prime,gate_min,gate_max,gate_res)
n_prime_input_v = fastgraph.input(func_n_prime,v_min,v_max,v_res)
fastgraph.generate(func_n_prime)

func_h_prime = fastgraph.definedfunction(h_prime, gate_min, gate_max, gate_res)
h_prime_input_h = fastgraph.input(func_h_prime,gate_min,gate_max,gate_res)
h_prime_input_v = fastgraph.input(func_h_prime,v_min,v_max,v_res)
fastgraph.generate(func_h_prime)

func_pot = fastgraph.function(pot, pot_res)
pot_input_n = fastgraph.outputtoinput(func_pot,func_n_prime)
pot_input_v = fastgraph.input(func_pot,v_min,v_max,v_res)
fastgraph.generate(func_pot)
print("Potassium Range:", fastgraph.min(func_pot), "->", fastgraph.max(func_pot))

func_sod_gate = fastgraph.function(sod_gate, sod_gate_res)
sod_gate_input_m = fastgraph.outputtoinput(func_sod_gate,func_m_prime)
sod_gate_input_h = fastgraph.outputtoinput(func_sod_gate,func_h_prime)
fastgraph.generate(func_sod_gate)
print("Sodium Gating Range:", fastgraph.min(func_sod_gate), "->", fastgraph.max(func_sod_gate))

func_sod = fastgraph.function(sod, sod_res)
sod_input_sod_gate = fastgraph.outputtoinput(func_sod,func_sod_gate)
sod_input_v = fastgraph.input(func_sod,v_min,v_max,v_res)
fastgraph.generate(func_sod)
print("Sodium Range:", fastgraph.min(func_sod), "->", fastgraph.max(func_sod))

func_potsod = fastgraph.function(potsod, potsod_res)
potsod_input_pot = fastgraph.outputtoinput(func_potsod,func_pot)
potsod_input_sod = fastgraph.outputtoinput(func_potsod,func_sod)
fastgraph.generate(func_potsod)
print("Potassium and Sodium Range:", fastgraph.min(func_potsod), "->", fastgraph.max(func_potsod))

func_v_prime = fastgraph.definedfunction(v_prime, v_min,v_max, v_res)
v_prime_input_potsod = fastgraph.outputtoinput(func_v_prime,func_potsod)
v_prime_input_v = fastgraph.input(func_v_prime,v_min,v_max,v_res)
fastgraph.generate(func_v_prime)

m_start = np.zeros(gate_res)
m_start[int((0.05 - gate_min) / gate_res)] = 1.0

n_start = np.zeros(gate_res)
n_start[int((0.3 - gate_min) / gate_res)] = 1.0

h_start = np.zeros(gate_res)
h_start[int((0.6 - gate_min) / gate_res)] = 1.0

v_start = np.zeros(v_res)
v_start[int((-70 - v_min) / v_res)] = 1.0


num_iterations = int(5 / dt);

fastgraph.set(func_m_prime, m_prime_input_m, m_start.tolist())
fastgraph.set(func_m_prime, m_prime_input_v, v_start.tolist())

fastgraph.set(func_n_prime, n_prime_input_n, n_start.tolist())
fastgraph.set(func_n_prime, n_prime_input_v, v_start.tolist())

fastgraph.set(func_h_prime, h_prime_input_h, h_start.tolist())
fastgraph.set(func_h_prime, h_prime_input_v, v_start.tolist())

fastgraph.set(func_pot, pot_input_n, n_start.tolist())
fastgraph.set(func_pot, pot_input_v, v_start.tolist())

fastgraph.set(func_sod_gate, sod_gate_input_m, m_start.tolist())
fastgraph.set(func_sod_gate, sod_gate_input_h, h_start.tolist())

fastgraph.set(func_sod, sod_input_v, v_start.tolist())

fastgraph.set(func_v_prime, v_prime_input_v, v_start.tolist())

avg_vs = []
times = []

print("Done. Begin simulation...")

for i in range(num_iterations):
    fastgraph.apply(func_m_prime)
    fastgraph.apply(func_n_prime)
    fastgraph.apply(func_h_prime)

    fastgraph.apply(func_pot)
    fastgraph.apply(func_sod_gate)
    fastgraph.transfer(func_sod_gate, func_sod, sod_input_sod_gate)
    fastgraph.apply(func_sod)
    fastgraph.transfer(func_sod, func_potsod, potsod_input_sod)
    fastgraph.transfer(func_pot, func_potsod, potsod_input_pot)
    fastgraph.apply(func_potsod)
    fastgraph.transfer(func_potsod, func_v_prime, v_prime_input_potsod)
    fastgraph.apply(func_v_prime)

    fastgraph.transfer(func_m_prime, func_m_prime, m_prime_input_m)
    fastgraph.transfer(func_v_prime, func_m_prime, m_prime_input_v)

    fastgraph.transfer(func_n_prime, func_n_prime, n_prime_input_n)
    fastgraph.transfer(func_v_prime, func_n_prime, n_prime_input_v)

    fastgraph.transfer(func_h_prime, func_h_prime, h_prime_input_h)
    fastgraph.transfer(func_v_prime, func_h_prime, h_prime_input_v)

    fastgraph.transfer(func_n_prime, func_pot, pot_input_n)
    fastgraph.transfer(func_v_prime, func_pot, pot_input_v)

    fastgraph.transfer(func_m_prime, func_sod_gate, sod_gate_input_m)
    fastgraph.transfer(func_h_prime, func_sod_gate, sod_gate_input_h)

    fastgraph.transfer(func_v_prime, func_sod, sod_input_v)

    fastgraph.transfer(func_v_prime, func_v_prime, v_prime_input_v)
    
    if i % 1 == 0:
        #print(i*dt)
        dist_v_out = fastgraph.read(func_v_prime)

        avg = np.sum([v_space[a]*dist_v_out[a] for a in range(v_res)])
        #print(avg)
        avg_vs = avg_vs + [avg]
        times = times + [i*dt]

        #fig, ax = plt.subplots(1, 1)

        #ax.plot(v_space,dist_v_out, 'r')
        #fig.tight_layout()

        #plt.show()

print("Done.")
fig, ax = plt.subplots(1, 1)

ax.plot(times,avg_vs, 'r')
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Membrane Potential (mV)")
fig.tight_layout()

plt.show()