from fastgraph import fastgraph
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time

def add(y):
    a = y[0]
    b = y[1]

    return a+b

def mult15(y):
    a = y[0]

    return 1.5*a

def reduce(y):
    a = y[0]

    return a - 0.1*a;

# B <- A*1.5
# OUT <- C + B

res = 150

space_in_a = np.linspace(-50,50,res)
space_in_b = np.linspace(-50,50,res)
space_in_c = np.linspace(-50,50,res)
space_out = np.linspace(-100,100,res)

func_id_mult = fastgraph.function(mult15,res)
input_a = fastgraph.input(func_id_mult,-50,50,res)
fastgraph.discretise(func_id_mult)

func_id_add = fastgraph.function(add, res)
input_b = fastgraph.outputtoinput(func_id_add,func_id_mult)
input_c = fastgraph.input(func_id_add,-50,50,res)
fastgraph.discretise(func_id_add)

fastgraph.descretiseinput(func_id_add, input_b, func_id_mult)

fastgraph.generate(func_id_mult)
fastgraph.generate(func_id_add)


test_a = np.zeros(res)
test_c = np.zeros(res)
test_a[int(((10+50)/(50+50))*res)] = 1.0
test_c[int(((-20+50)/(50+50))*res)] = 1.0

fastgraph.set(func_id_mult, input_a, test_a.tolist())
fastgraph.set(func_id_add,  input_c, test_c.tolist())

fastgraph.apply(func_id_mult)
fastgraph.transfer(func_id_mult, func_id_add, input_b)
fastgraph.apply(func_id_add)

dist_mult_out = fastgraph.read(func_id_mult)
dist_add_out = fastgraph.read(func_id_add)

# When we want to read the output of add and mult, the discretisations matter
mult_discretisation = fastgraph.discretisation(func_id_mult)
add_discretisation = fastgraph.discretisation(func_id_add)

print(len(mult_discretisation))

rectified_mult = [mult_discretisation[i] + ((mult_discretisation[i+1]-mult_discretisation[i])/2.0) for i in range(len(mult_discretisation)-1)]
rectified_add = [add_discretisation[i] + ((add_discretisation[i+1]-add_discretisation[i])/2.0) for i in range(len(add_discretisation)-1)]

fig, ax = plt.subplots(1, 1)

ax.set_title('Discretisation')
diffs = [mult_discretisation[i+1] - mult_discretisation[i] for i in range(len(mult_discretisation)-1)]
ax.plot(diffs, 'r')
fig.tight_layout()

fig, ax = plt.subplots(1, 1)

ax.set_title('Addition')
ax.plot(space_in_a, test_a, 'r')
ax.plot(rectified_mult, dist_mult_out, 'g')
ax.plot(space_in_c, test_c, 'b')
ax.plot(rectified_add, dist_add_out, 'k')
fig.tight_layout()

plt.show()

func_id_reduce = fastgraph.definedfunction(reduce, -100, 100, res)
input_d = fastgraph.input(func_id_reduce,-100,100,res)
fastgraph.discretise(func_id_reduce)

fastgraph.descretiseinput(func_id_reduce, input_d, func_id_reduce)

fastgraph.generate(func_id_reduce)

red_discretisation = fastgraph.discretisation(func_id_reduce)
rectified_red = [red_discretisation[i] + ((red_discretisation[i+1]-red_discretisation[i])/2.0) for i in range(len(red_discretisation)-1)]

fig, ax = plt.subplots(1, 1)

ax.set_title('Discretisation')
diffs = [red_discretisation[i+1] - red_discretisation[i] for i in range(len(red_discretisation)-1)]
ax.plot(rectified_red, diffs, 'rx-')
fig.tight_layout()

test_r = np.zeros(res)
for i in range(len(red_discretisation)):
    if 50 > red_discretisation[i] and 50 <= red_discretisation[i+1]:
        test_r[i] = 1.0
        break
    
fastgraph.set(func_id_reduce,  input_d, test_r.tolist())

avg_vs = []
times = []

for i in range(100):
    print("*")
    fastgraph.apply(func_id_reduce)
    fastgraph.transfer(func_id_reduce, func_id_reduce, input_d)
    dist_red_out = fastgraph.read(func_id_reduce)
    avg = np.sum([rectified_red[a]*dist_red_out[a] for a in range(res)])
    avg_vs = avg_vs + [avg]
    times = times + [i]
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(rectified_red, dist_red_out, '')
    fig.tight_layout()

    plt.show()

fig, ax = plt.subplots(1, 1)

ax.plot(times,avg_vs, 'r')
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Membrane Potential (mV)")
fig.tight_layout()

plt.show()

print(avg_vs)