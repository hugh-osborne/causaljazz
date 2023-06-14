from distutils.ccompiler import show_compilers
from pycausaljazz import pycausaljazz as cj
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from scipy.stats import poisson
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def plot3D(points, data):
    flattened_points_x = []
    flattened_points_y = []
    flattened_points_z = []
    flattened_joint_col = []

    for x in range(100):
        for y in range(100):
            for z in range(100):
                if (data[x][y][z] > 0.000001):
                    flattened_points_x = flattened_points_x + [points[x][y][z][0]]
                    flattened_points_y = flattened_points_y + [points[x][y][z][1]]
                    flattened_points_z = flattened_points_z + [points[x][y][z][2]]
                    flattened_col = data[x][y][z]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(flattened_points_x, flattened_points_y, flattened_points_z, marker='s',
                     s=20, alpha=1, color='green')

    ax.set_title("3D Heatmap")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

def plotDist1D(dist, range=range(100)):
    dist_read = cj.readDist(dist)
    fig, ax = plt.subplots(1, 1)
    ax.plot(range, dist_read)
    fig.tight_layout()

def plotDist2D(dist, res=(100,100), _extent=[0,100,0,100]):
    dist = cj.readDist(dist)
    dist = np.array(dist)
    dist = dist.reshape(res)

    fig, ax = plt.subplots(1, 1)
    ax.set(xticks=[0, 99], xticklabels=[_extent[0], _extent[1]], yticks=[0, 99], yticklabels=[int(_extent[2]), int(_extent[3])]);
    plt.imshow(dist, cmap='hot', interpolation='nearest')

def plotDist3D(dist, res=(100,100,100)):

    points = []
    for z in np.linspace(cj.base(dist)[2], cj.base(dist)[2] + cj.size(dist)[2], 100):
        points_col = []
        for y in np.linspace(cj.base(dist)[1], cj.base(dist)[1] + cj.size(dist)[1], 100):
            points_dep = []
            for x in np.linspace(cj.base(dist)[0], cj.base(dist)[0] + cj.size(dist)[0], 100):
                points_dep = points_dep + [(x,y,z)]
            points_col = points_col + [points_dep]
        points = points + [points_col]

    dist = cj.readDist(dist)
    dist = np.array(dist)
    dist = dist.reshape(res)

    plot3D(points, dist)

timestep = 0.1

def cond(y):
    E_l = -70.6
    E_e = 0.0
    E_i = 0.0 #-75.0
    C = 281
    g_l = 0.03
    tau_e =10.49
    tau_i = 10.49

    v = y[0]
    w = y[1]

    v_prime = (-g_l*(v - E_l) - w * (v - E_e)) / C
    w_prime = -(w) / tau_e

    return [v + timestep*v_prime, w + (timestep*w_prime) + 0.5]

def w_prime(y):
    w = y[0]
    jump = y[1]
    tau_e =10.49
    dt = timestep

    w_prime = -(w) / tau_e
    return w + (dt*w_prime) + 0.5


def vw(y):
    v = y[0]
    w = y[1]
    C = 281

    E_e = 0.0

    return (-w * (v - E_e)) / C


def v_prime(y):
    v = y[0]
    vw = y[1] 

    E_l = -70.6
    C = 281
    g_l = 0.03
    dt = timestep

    v_prime = ((-g_l*(v - E_l)) / C) + vw

    return v + dt*v_prime

def v_prime_simple(y):
    v = y[0]
    w = y[1] 

    E_l = -70.6
    E_e = 0.0
    C = 281
    g_l = 0.03
    dt = timestep

    v_prime = ((-g_l*(v - E_l)) - (w * (v - E_e))) / C

    return v + dt*v_prime

res = 100
v_res = 100
w_res = 100
I_res = 100

v_max = -40.0
v_min = -100.0
w_max = 50.0 #25.0
w_min = -20.0 #-1.0

# Set up the starting distribution
v = np.linspace(v_min, v_max, v_res)
w = np.linspace(w_min, w_max, w_res)

points = []
for x in range(res):
    points_col = []
    for y in range(w_res):
        points_dep = []
        for z in range(v_res):
            points_dep = points_dep + [(x,y,z)]
        points_col = points_col + [points_dep]
    points = points + [points_col]
    
#[-70.6, 0.001, 0.001]
# Unfortunately stats.norm doesn't provide a nice pmf approximation of the pdf. 
# So let's just do that ourselves without breaking our backs by multiplying across by the discretisation width and normalising
vpdf = [a * ((v_max-v_min)/v_res) for a in norm.pdf(v, -70.6, 2.4)]
wpdf = [a * ((w_max-w_min)/w_res) for a in norm.pdf(w, 15.0, 2.4)]

vpdf2 = [a * ((v_max-v_min)/v_res) for a in norm.pdf(v, -55.6, 0.4)]
wpdf2 = [a * ((w_max-w_min)/w_res) for a in norm.pdf(w, 25.0, 0.4)]

vpdf = [vpdf[a] + vpdf2[a] for a in range(v_res)]
wpdf = [wpdf[a] + wpdf2[a] for a in range(w_res)]

vpdf = [a / sum(vpdf) for a in vpdf]
wpdf = [a / sum(wpdf) for a in wpdf]

v0 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in vpdf])
w0 = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in wpdf])

w_rate = 0 #4
epsp = 0.5
wI_max_events = 10
wI_min_events = -2
wI_max = wI_max_events*epsp
wI_min = wI_min_events*epsp
epsps = np.linspace(wI_min, wI_max, I_res)
wI_events = np.linspace(wI_min_events, wI_max_events, I_res)
wIpdf_final = [0 for a in wI_events]
for i in range(len(wI_events)-1):
    if (int(wI_events[i]) < int(wI_events[i+1])) or (wI_events[i] < 0 and wI_events[i+1] >= 0): # we have just passed a new event
        e = int(wI_events[i+1])
        if e <= 0:
            e = int(wI_events[i])
        diff = wI_events[i+1] - wI_events[i]
        lower_prop = (int(wI_events[i+1]) - wI_events[i]) / diff 
        upper_prop = 1.0 - lower_prop
        wIpdf_final[i] += poisson.pmf(e, w_rate*timestep) * lower_prop
        wIpdf_final[i+1] += poisson.pmf(e, w_rate*timestep) * upper_prop
wIpdf = wIpdf_final
wI = cj.newDist([wI_min], [wI_max], [I_res], [a for a in wIpdf])

c_w_prime = cj.boundedFunction([w_min,wI_min],[(w_max-w_min),(wI_max-wI_min)],[w_res,I_res], w_prime, w_min, (w_max-w_min), w_res)
c_vw = cj.boundedFunction([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], vw, 0, 10, res)
c_v_prime = cj.boundedFunction([v_min,cj.base(c_vw)[2]],[(v_max-v_min),cj.size(c_vw)[2]],[v_res,res],v_prime, v_min, (v_max-v_min), v_res)
c_v_prime_simple = cj.boundedFunction([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res],v_prime_simple, v_min, (v_max-v_min), v_res)

joint_w0_wI_w1 = cj.newDist([w_min,wI_min,w_min],[(w_max-w_min),(wI_max-wI_min),(w_max-w_min)],[w_res,I_res,w_res],[a for a in np.zeros(w_res*I_res*w_res)])
joint_w0_wI = cj.newDist([w_min,wI_min],[(w_max-w_min),(wI_max-wI_min)],[w_res,I_res],[a for a in np.zeros(w_res*I_res)])

cj.joint2Di(w0, wI, joint_w0_wI)
num_neurons = 10000

w_inputs = np.array([[poisson.rvs(w_rate*timestep)*epsp] for a in range(num_neurons)])
mc_neurons = np.array([[norm.rvs(-70.6, 2.4, 1)[0],norm.rvs(15.0, 2.4, 1)[0]] for a in range(num_neurons)])
for m in mc_neurons:
    r = random.uniform(0,1)
    t = 0.0
    i = 0
    while t < 1.0:
        t += vpdf[i]
        if t >= r:
            m[0] = v_min + ((i+0.5) * ((v_max-v_min)/v_res))
            t = 2.0
        i += 1
    r = random.uniform(0,1)
    t = 0.0
    i = 0
    while t < 1.0:
        t += wpdf[i]
        if t >= r:
            m[1] = w_min + ((i+0.5) * ((w_max-w_min)/w_res))
            t = 2.0
        i += 1

mc_neurons_joint_w_wI_w1 = np.array([[mc_neurons[a][1],w_inputs[a][0],w_prime([mc_neurons[a][1],w_inputs[a][0]])] for a in range(num_neurons)])

cj.collider(joint_w0_wI, [0,1], c_w_prime, joint_w0_wI_w1)

w1 = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in np.zeros(w_res)])
marginal_w0_w1 = cj.newDist([w_min,w_min],[(w_max-w_min),(w_max-w_min)],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])
w1_given_w0 = cj.newDist([w_min,w_min],[(w_max-w_min),(w_max-w_min)],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])

cj.marginal(joint_w0_wI_w1, 1, marginal_w0_w1)
cj.marginal(marginal_w0_w1, 0, w1)
cj.conditional(marginal_w0_w1, [0], w0, w1_given_w0)

joint_v_w = cj.newDist([v_min,w_min], [(v_max-v_min),(w_max-w_min)], [v_res,w_res], [a for a in np.zeros(v_res*w_res)])

cj.joint2Di(v0, w0, joint_v_w)

joint_v_w_vw = cj.newDist([v_min,w_min,cj.base(c_vw)[2]],[(v_max-v_min),(w_max-w_min),cj.size(c_vw)[2]],[v_res,w_res,res], [a for a in np.zeros(v_res*w_res*res)])

cj.collider(joint_v_w, [0,1], c_vw, joint_v_w_vw)

joint_v_vw = cj.newDist([v_min, cj.base(c_vw)[2]], [(v_max-v_min), cj.size(c_vw)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])

cj.marginal(joint_v_w_vw, 1, joint_v_vw)

vw_t = cj.newDist([cj.base(c_vw)[2]], [cj.size(c_vw)[2]], [res], [a for a in np.zeros(res)])

cj.marginal(joint_v_vw, 0, vw_t)

mc_neurons_joint_v_w_vw = np.array([[mc_neurons[a][0],mc_neurons[a][1],vw([mc_neurons[a][0],mc_neurons[a][1]])] for a in range(num_neurons)])

plotDist1D(vw_t, range=range(res))

fig, ax = plt.subplots(1,1)
ax.hist(mc_neurons_joint_v_w_vw[:,2], density=True, bins=res, range=[cj.base(c_vw)[2],cj.base(c_vw)[2]+cj.size(c_vw)[2]], histtype='step')
fig.tight_layout()
plt.show()

plotDist1D(w0, range=range(w_res))

fig, ax = plt.subplots(1,1)
ax.hist(mc_neurons[:,1], density=True, bins=w_res, range=[w_min,w_max], histtype='step')
fig.tight_layout()
plt.show()

marginal_vw_v1 = cj.newDist([cj.base(c_vw)[2],v_min],[cj.size(c_vw)[2],(v_max-v_min)],[res,v_res], [a for a in np.zeros(res*v_res)])
v1 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in np.zeros(v_res)])
joint_v0_vw_v1 = cj.newDist([v_min,cj.base(c_vw)[2],v_min],[(v_max-v_min),cj.size(c_vw)[2],(v_max-v_min)],[v_res,res,v_res], [a for a in np.zeros(v_res*res*v_res)])

cj.collider(joint_v_vw, [0,1], c_v_prime, joint_v0_vw_v1)
cj.marginal(joint_v0_vw_v1, 0, marginal_vw_v1)
cj.marginal(marginal_vw_v1, 0, v1)

simple_joint_v_w_v1 = cj.newDist([v_min,w_min,v_min],[(v_max-v_min),(w_max-w_min),(v_max-v_min)],[v_res,w_res,v_res], [a for a in np.zeros(v_res*w_res*v_res)])
simple_joint_w_v1 = cj.newDist([w_min,v_min],[(w_max-w_min),(v_max-v_min)],[w_res,v_res], [a for a in np.zeros(w_res*v_res)])
simple_v1 = cj.newDist([v_min],[(v_max-v_min)],[v_res], [a for a in np.zeros(v_res)])
cj.collider(joint_v_w, [0,1], c_v_prime_simple, simple_joint_v_w_v1)
cj.marginal(simple_joint_v_w_v1, 0, simple_joint_w_v1)
cj.marginal(simple_joint_w_v1, 0, simple_v1)

simple_v1_given_w = cj.newDist([w_min,v_min],[(w_max-w_min),(v_max-v_min)],[w_res,v_res], [a for a in np.zeros(w_res*v_res)])
cj.conditional(simple_joint_w_v1, [0], w0, simple_v1_given_w)

mc_neurons_joint_v0_vw_v1 = np.array([[mc_neurons[a][0],mc_neurons_joint_v_w_vw[a][2],v_prime([mc_neurons[a][0],mc_neurons_joint_v_w_vw[a][2]])] for a in range(num_neurons)])

for nn in range(len(mc_neurons_joint_v0_vw_v1)):
    if (mc_neurons_joint_v0_vw_v1[nn][2] > -50.0):
        mc_neurons_joint_v0_vw_v1[nn][2] = -70.6

v1_given_w0 = cj.newDist([w_min, v_min], [(w_max-w_min), (v_max-v_min)], [w_res,v_res], [a for a in np.zeros(w_res*v_res)])
marginal_w0_vw = cj.newDist([w_min,cj.base(c_vw)[2]],[(w_max-w_min),cj.size(c_vw)[2]],[w_res,res], [a for a in np.zeros(w_res*res)])

cj.marginal(marginal_vw_v1, 1, vw_t)
plotDist1D(vw_t, range=range(res))

fig, ax = plt.subplots(1,1)
ax.hist(mc_neurons_joint_v_w_vw[:,2], density=True, bins=res, range=[cj.base(c_vw)[2],cj.base(c_vw)[2]+cj.size(c_vw)[2]], histtype='step')
fig.tight_layout()

plotDist2D(marginal_vw_v1, res=(v_res,res))

fig, ax = plt.subplots(1,1)
ax.scatter(mc_neurons_joint_v0_vw_v1[:,1], mc_neurons_joint_v0_vw_v1[:,2], s=0.5)
ax.scatter(mc_neurons_joint_v0_vw_v1[:,1], mc_neurons_joint_v0_vw_v1[:,0], s=0.5)
ax.set_xlim((cj.base(c_vw)[2],cj.base(c_vw)[2]+cj.size(c_vw)[2]))
ax.set_ylim((v_max,v_min))
fig.tight_layout()

v1_given_vw = cj.newDist([cj.base(c_vw)[2],v_min],[cj.size(c_vw)[2],(v_max-v_min)],[res,v_res], [a for a in np.zeros(res*v_res)])

joint_w0_vw_v1 = cj.newDist([w_min,cj.base(c_vw)[2],v_min],[(w_max-w_min),cj.size(c_vw)[2],(v_max-v_min)],[w_res,res,v_res], [a for a in np.zeros(w_res*res*v_res)])
marginal_w0_v1 = cj.newDist([w_min, v_min], [(w_max-w_min), (v_max-v_min)], [w_res,v_res], [a for a in np.zeros(w_res*v_res)])

joint_w0_v1_w1 = cj.newDist([w_min,v_min,w_min],[(w_max-w_min),(v_max-v_min),(w_max-w_min)],[w_res,v_res,w_res], [a for a in np.zeros(w_res*v_res*w_res)])
joint_v1_w1 = cj.newDist([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])
joint_w0_w1_v1 = cj.newDist([w_min,w_min,v_min],[(w_max-w_min),(w_max-w_min),(v_max-v_min)],[w_res,w_res,v_res], [a for a in np.zeros(w_res*w_res*v_res)])
joint_w1_v1 = cj.newDist([w_min,v_min],[(w_max-w_min),(v_max-v_min)],[w_res,v_res], [a for a in np.zeros(w_res*v_res)])

cj.marginal(joint_v_w_vw, 0, marginal_w0_vw)

cj.marginal(marginal_w0_vw, 0, vw_t)
plotDist1D(vw_t, range=range(res))

fig, ax = plt.subplots(1,1)
ax.hist(mc_neurons_joint_v_w_vw[:,2], density=True, bins=res, range=[cj.base(c_vw)[2],cj.base(c_vw)[2]+cj.size(c_vw)[2]], histtype='step')
fig.tight_layout()

plotDist2D(marginal_w0_vw, res=(res,w_res))

fig, ax = plt.subplots(1,1)
ax.scatter(mc_neurons_joint_v_w_vw[:,1], mc_neurons_joint_v_w_vw[:,2], s=0.5)
ax.set_xlim((w_min,w_max))
ax.set_ylim((cj.base(c_vw)[2]+cj.size(c_vw)[2],cj.base(c_vw)[2]))
fig.tight_layout()
plt.show()

cj.conditional(marginal_vw_v1, [0], vw_t, v1_given_vw)
cj.joint3D(marginal_w0_vw, 1, v1_given_vw, joint_w0_vw_v1)
cj.marginal(joint_w0_vw_v1, 1, marginal_w0_v1)

cj.conditional(marginal_w0_v1, [0], w0, v1_given_w0)

cj.fork(w0, 0, v1_given_w0, 0, w1_given_w0, joint_w0_v1_w1)

simple_joint_w0_v1_w1 = cj.newDist([w_min,v_min,w_min],[(w_max-w_min),(v_max-v_min),(w_max-w_min)],[w_res,v_res,w_res], [a for a in np.zeros(w_res*v_res*w_res)])
simple_joint_v1_w1 = cj.newDist([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])
cj.fork(w0, 0, simple_v1_given_w, 0, w1_given_w0, simple_joint_w0_v1_w1)
cj.marginal(simple_joint_w0_v1_w1, 0, simple_joint_v1_w1)

cj.marginal(joint_w0_v1_w1, 0, joint_v1_w1)

mc_neurons_prime = np.array([[mc_neurons_joint_v0_vw_v1[a][2],mc_neurons_joint_w_wI_w1[a][2]] for a in range(num_neurons)])

for nn in range(len(mc_neurons_prime)):
    if (mc_neurons_prime[nn][0] > -50.0):
        mc_neurons_prime[nn][0] = -70.6

plotDist2D(marginal_w0_v1, res=(v_res,w_res))

fig, ax = plt.subplots(1,1)
ax.scatter(mc_neurons_joint_v_w_vw[:,1], mc_neurons_prime[:,0], s=0.5)
ax.set_xlim((w_min,w_max))
ax.set_ylim((v_max,v_min))
fig.tight_layout()
plt.show()

plotDist2D(joint_v1_w1, res=(w_res,v_res))
plotDist2D(simple_joint_v1_w1, res=(w_res,v_res))

fig, ax = plt.subplots(1,1)
ax.scatter(mc_neurons_prime[:,0], mc_neurons_prime[:,1], s=0.3)
ax.scatter(mc_neurons_joint_v0_vw_v1[:,0], mc_neurons_joint_w_wI_w1[:,0], s=0.3)
ax.set_xlim((v_min,v_max))
ax.set_ylim((w_max,w_min))
fig.tight_layout()
plt.show()

# Joint Monte Carlo

mc_joint_neurons = np.array([[norm.rvs(-70.6, 2.4, 1)[0],norm.rvs(15.0, 2.4, 1)[0]] for a in range(num_neurons)])

mc_joint_neurons_prime = np.array([cond(mc_joint_neurons[a]) for a in range(num_neurons)])

for nn in range(len(mc_joint_neurons_prime)):
        if (mc_joint_neurons_prime[nn][0] > -50.0):
            mc_joint_neurons_prime[nn][0] = -70.6

        mc_joint_neurons_prime[nn][1] += epsp*poisson.rvs(w_rate*timestep) # override w

        
# Timestep 1 seems fine so far. Now to start the loop.

vnp1 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in np.zeros(v_res)])
wnp1 = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in np.zeros(w_res)])
joint_vnp1_wnp1 = cj.newDist([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])
joint_vnp1_wnp1_t = cj.newDist([w_min,v_min],[(w_max-w_min),(v_max-v_min)],[w_res,v_res], [a for a in np.zeros(w_res*v_res)])
simple_vnp1 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in np.zeros(v_res)])
simple_wnp1 = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in np.zeros(w_res)])
simple_joint_vnp1_wnp1 = cj.newDist([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])

# We also need to keep track of the latest v and w given the initial w0
vn = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in np.zeros(v_res)])
cj.transfer(v1, vn)
wn = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in np.zeros(w_res)])
cj.transfer(w1, wn)
simple_vn = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in np.zeros(v_res)])
cj.transfer(simple_v1, simple_vn)
simple_wn = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in np.zeros(w_res)])
cj.transfer(w1, simple_wn)
joint_wn_wI = cj.newDist([w_min,wI_min],[(w_max-w_min),(wI_max-wI_min)],[w_res,I_res],[a for a in np.zeros(w_res*I_res)])
joint_wn_wI_wnp1 = cj.newDist([w_min,wI_min,w_min],[(w_max-w_min),(wI_max-wI_min),(w_max-w_min)],[w_res,I_res,w_res],[a for a in np.zeros(w_res*I_res*w_res)])
wnp1_given_wn = cj.newDist([w_min,w_min],[(w_max-w_min),(w_max-w_min)],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])
cj.transfer(w1_given_w0, wnp1_given_wn)

marginal_w0_wn = cj.newDist([w_min,w_min],[(w_max-w_min),(w_max-w_min)],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])
cj.transfer(marginal_w0_w1, marginal_w0_wn)

wn_given_w0 = cj.newDist([w_min,w_min],[(w_max-w_min),(w_max-w_min)],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])
marginal_wn_wnp1 = cj.newDist([w_min,w_min],[(w_max-w_min),(w_max-w_min)],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])
marginal_w0_wnp1 = cj.newDist([w_min,w_min],[(w_max-w_min),(w_max-w_min)],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])
wnp1_given_w0 = cj.newDist([w_min,w_min],[(w_max-w_min),(w_max-w_min)],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])

joint_vn_wn = cj.newDist([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])
cj.transfer(joint_v1_w1, joint_vn_wn)

simple_joint_vn_wn = cj.newDist([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])
cj.transfer(simple_joint_v1_w1, simple_joint_vn_wn)

joint_vn_vw_vnp1 = cj.newDist([v_min,cj.base(c_vw)[2],v_min],[(v_max-v_min),cj.size(c_vw)[2],(v_max-v_min)],[v_res,res,v_res], [a for a in np.zeros(v_res*res*v_res)])
joint_wn_vw_vnp1 = cj.newDist([w_min,cj.base(c_vw)[2],v_min],[(w_max-w_min),cj.size(c_vw)[2],(v_max-v_min)],[w_res,res,v_res], [a for a in np.zeros(w_res*res*v_res)])

marginal_vw_vnp1 = cj.newDist([cj.base(c_vw)[2],v_min],[cj.size(c_vw)[2],(v_max-v_min)],[res,v_res], [a for a in np.zeros(res*v_res)])
marginal_vw_vnp1_t = cj.newDist([v_min,cj.base(c_vw)[2]],[(v_max-v_min),cj.size(c_vw)[2]],[v_res,res], [a for a in np.zeros(v_res*res)])

marginal_vn_vw = cj.newDist([v_min,cj.base(c_vw)[2]],[(v_max-v_min),cj.size(c_vw)[2]],[v_res,res], [a for a in np.zeros(v_res*res)])
marginal_wn_vw = cj.newDist([w_min,cj.base(c_vw)[2]],[(w_max-w_min),cj.size(c_vw)[2]],[w_res,res], [a for a in np.zeros(w_res*res)])
cj.transfer(marginal_w0_vw, marginal_wn_vw)
vnp1_given_vw = cj.newDist([cj.base(c_vw)[2],v_min],[cj.size(c_vw)[2],(v_max-v_min)],[res,v_res], [a for a in np.zeros(res*v_res)])

joint_vn_vnp1 = cj.newDist([v_min,v_min],[(v_max-v_min),(v_max-v_min)],[v_res,v_res], [a for a in np.zeros(v_res*v_res)])
vnp1_given_vn = cj.newDist([v_min,v_min],[(v_max-v_min),(v_max-v_min)],[v_res,v_res], [a for a in np.zeros(v_res*v_res)])

cj.marginal(joint_v0_vw_v1, 1, joint_vn_vnp1)
cj.conditional(joint_vn_vnp1, [0], v0, vnp1_given_vn)

wn_given_vn = cj.newDist([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])

cj.conditional(joint_v_w, [0], v0, wn_given_vn)

joint_vn_wn_vnp1 = cj.newDist([v_min,w_min,v_min],[(v_max-v_min),(w_max-w_min),(v_max-v_min)],[v_res,w_res,v_res], [a for a in np.zeros(v_res*w_res*v_res)])

cj.fork(v0, 0, wn_given_vn, 0, vnp1_given_vn, joint_vn_wn_vnp1)

joint_wn_vnp1 = cj.newDist([w_min,v_min],[(w_max-w_min),(v_max-v_min)],[w_res,v_res], [a for a in np.zeros(w_res*v_res)])
vnp1_given_wn = cj.newDist([w_min,v_min],[(w_max-w_min),(v_max-v_min)],[w_res,v_res], [a for a in np.zeros(w_res*v_res)])

cj.marginal(joint_vn_wn_vnp1, 0, joint_wn_vnp1)
cj.conditional(joint_wn_vnp1, [0], w0, vnp1_given_wn)

cj.fork(w0, 0, vnp1_given_wn, 0, w1_given_w0, joint_w0_v1_w1)
cj.marginal(joint_w0_v1_w1, 0, joint_v1_w1)

joint_w0_vn = cj.newDist([w_min,v_min],[(w_max-w_min),(v_max-v_min)],[w_res,v_res], [a for a in np.zeros(w_res*v_res)])
cj.transfer(marginal_w0_v1, joint_w0_vn)

joint_w0_wn_wnp1 = cj.newDist([w_min,w_min,w_min],[(w_max-w_min),(w_max-w_min),(w_max-w_min)],[w_res,w_res,w_res], [a for a in np.zeros(w_res*w_res*w_res)])
joint_w0_vn_vnp1 = cj.newDist([w_min,v_min,v_min],[(w_max-w_min),(v_max-v_min),(v_max-v_min)],[w_res,v_res,v_res], [a for a in np.zeros(w_res*v_res*v_res)])

joint_w0_vnp1 = cj.newDist([w_min,v_min],[(w_max-w_min),(v_max-v_min)],[w_res,v_res], [a for a in np.zeros(w_res*v_res)])
vnp1_given_w0 = cj.newDist([w_min,v_min],[(w_max-w_min),(v_max-v_min)],[w_res,v_res], [a for a in np.zeros(w_res*v_res)])

joint_w0_vnp1_wnp1 = cj.newDist([w_min,v_min,w_min],[(w_max-w_min),(v_max-v_min),(w_max-w_min)],[w_res,v_res,w_res], [a for a in np.zeros(w_res*v_res*w_res)])
joint_wn_vnp1_wnp1 = cj.newDist([w_min,v_min,w_min],[(w_max-w_min),(v_max-v_min),(w_max-w_min)],[w_res,v_res,w_res], [a for a in np.zeros(w_res*v_res*w_res)])

# alternative alternative

joint_wn_vw_wnp1 = cj.newDist([w_min,cj.base(c_vw)[2],w_min],[(w_max-w_min),cj.size(c_vw)[2],(w_max-w_min)],[w_res,res,w_res], [a for a in np.zeros(w_res*res*w_res)])
vw_given_wn = cj.newDist([w_min,cj.base(c_vw)[2]],[(w_max-w_min),cj.size(c_vw)[2]],[w_res,res], [a for a in np.zeros(w_res*res)])
cj.conditional(marginal_wn_vw, [0], w0, vw_given_wn)
cj.fork(w0, 0, vw_given_wn, 0, wnp1_given_wn, joint_wn_vw_wnp1)

marginal_vw_wnp1 = cj.newDist([cj.base(c_vw)[2],w_min],[cj.size(c_vw)[2],(w_max-w_min)],[res,w_res], [a for a in np.zeros(res*w_res)])
cj.marginal(joint_wn_vw_wnp1, 0, marginal_vw_wnp1)
wnp1_given_vw = cj.newDist([cj.base(c_vw)[2],w_min],[cj.size(c_vw)[2],(w_max-w_min)],[res,w_res], [a for a in np.zeros(res*w_res)])
cj.conditional(marginal_vw_wnp1, [0], vw_t, wnp1_given_vw)

joint_vw_vnp1_wnp1 = cj.newDist([cj.base(c_vw)[2], v_min, w_min],[cj.size(c_vw)[2],(v_max-v_min),(w_max-w_min)],[res,v_res,w_res], [a for a in np.zeros(res*v_res*w_res)])
cj.fork(vw_t, 0, v1_given_vw, 0, wnp1_given_vw, joint_vw_vnp1_wnp1) 




for iteration in range(1000):

    cj.joint2Di(wn, wI, joint_wn_wI)

    w_inputs = np.array([[poisson.rvs(w_rate*timestep)*epsp] for a in range(num_neurons)])
    mc_neurons_joint_w_wI_w1 = np.array([[mc_neurons_prime[a][1],w_inputs[a][0],w_prime([mc_neurons_prime[a][1],w_inputs[a][0]])] for a in range(num_neurons)])


    plotDist1D(wI, range=range(I_res))

    fig, ax = plt.subplots(1,1)
    ax.hist(w_inputs, density=True, bins=I_res, range=[wI_min,wI_max], histtype='step')
    fig.tight_layout()
    plt.show()

    cj.collider(joint_wn_wI, [0,1], c_w_prime, joint_wn_wI_wnp1)

    plotDist2D(joint_wn_wI, res=(I_res,w_res))

    fig, ax = plt.subplots(1,1)
    ax.scatter(mc_neurons_joint_w_wI_w1[:,0], mc_neurons_joint_w_wI_w1[:,1], s=0.3)
    ax.set_xlim((w_min,w_max))
    ax.set_ylim((wI_max,wI_min))
    fig.tight_layout()
    plt.show()

    cj.marginal(joint_wn_wI_wnp1, 1, marginal_wn_wnp1)
    cj.marginal(marginal_wn_wnp1, 0, wnp1)
    cj.conditional(marginal_wn_wnp1, [0], wn, wnp1_given_wn)

    mc_neurons_marginal_w_w1 = np.array([[mc_neurons_prime[a][1],w_prime([mc_neurons_prime[a][1],w_inputs[a][0]])] for a in range(num_neurons)])

    plotDist2D(joint_vn_wn, res=(w_res,v_res))

    fig, ax = plt.subplots(1,1)
    ax.scatter(mc_neurons_prime[:,0], mc_neurons_prime[:,1], s=0.3)
    ax.set_xlim((v_min,v_max))
    ax.set_ylim((w_max,w_min))
    fig.tight_layout()
    plt.show()


    #cj.joint2Di(v1, w1, joint_v1_w1) # independent
    cj.collider(simple_joint_vn_wn, [0,1], c_vw, joint_v_w_vw)
    cj.marginal(joint_v_w_vw, 1, joint_v_vw)
    cj.marginal(joint_v_vw, 0, vw_t)

    plotDist1D(vw_t, range=range(res))

    mc_neurons_joint_v_w_vw = np.array([[mc_neurons_prime[a][0],mc_neurons_prime[a][1],vw([mc_neurons_prime[a][0],mc_neurons_prime[a][1]])] for a in range(num_neurons)])
    mc_neurons_joint_v_vw = np.array([[mc_neurons_prime[a][0],vw([mc_neurons_prime[a][0],mc_neurons_prime[a][1]])] for a in range(num_neurons)])
    mc_neurons_joint_vw = np.array([[vw([mc_neurons_prime[a][0],mc_neurons_prime[a][1]])] for a in range(num_neurons)])
    

    fig, ax = plt.subplots(1,1)
    ax.hist(mc_neurons_joint_v_w_vw[:,2], density=True, bins=res, range=[cj.base(c_vw)[2],cj.base(c_vw)[2]+cj.size(c_vw)[2]], histtype='step')
    fig.tight_layout()
    plt.show()

    cj.collider(joint_v_vw, [0,1], c_v_prime, joint_vn_vw_vnp1)
    cj.marginal(joint_vn_vw_vnp1, 0, marginal_vw_vnp1)
    cj.transpose(marginal_vw_vnp1, marginal_vw_vnp1_t)

    mc_neurons_joint_v0_vw_v1 = np.array([[mc_neurons_joint_v_vw[a][0],mc_neurons_joint_v_vw[a][1],v_prime([mc_neurons_joint_v_vw[a][0],mc_neurons_joint_v_vw[a][1]])] for a in range(num_neurons)])
    mc_neurons_marginal_vw_v1 = np.array([[mc_neurons_joint_v_vw[a][1],v_prime([mc_neurons_joint_v_vw[a][0],mc_neurons_joint_v_vw[a][1]])] for a in range(num_neurons)])
    

    for nn in range(len(mc_neurons_joint_v0_vw_v1)):
        if (mc_neurons_joint_v0_vw_v1[nn][2] > -50.0):
            mc_neurons_joint_v0_vw_v1[nn][2] = -70.6
    
    # Now calcultae v2 including the threshold reset

    vw_v_dist = cj.readDist(marginal_vw_vnp1)

    threshold = -50.0
    reset = -70.6
    threshold_cell = int((threshold - cj.base(vnp1)[0]) / (cj.size(vnp1)[0] / cj.res(vnp1)[0]))
    reset_cell = int((reset - cj.base(vnp1)[0]) / (cj.size(vnp1)[0] / cj.res(vnp1)[0]))
    
    n_mass = [a for a in vw_v_dist]
    total_reset_mass = 0.0
    for j in range(cj.res(marginal_vw_vnp1)[0]): # for each column
        for i in range(cj.res(marginal_vw_vnp1)[1]): 
            index = (i * cj.res(marginal_vw_vnp1)[0]) + j
            reset_index = (reset_cell * cj.res(marginal_vw_vnp1)[0]) + j
            if i >= threshold_cell and vw_v_dist[index] > 0.0:
                n_mass[reset_index] += vw_v_dist[index]
                n_mass[index] = 0.0
                total_reset_mass += vw_v_dist[index]
    
    cj.update(marginal_vw_vnp1, n_mass)

    cj.marginal(marginal_vw_vnp1, 0, vnp1)

    cj.collider(simple_joint_vn_wn, [0,1], c_v_prime_simple, simple_joint_v_w_v1)
    cj.marginal(simple_joint_v_w_v1, 0, simple_joint_w_v1)
    cj.marginal(simple_joint_w_v1, 0, simple_v1)

    cj.conditional(simple_joint_w_v1, [0], wn, simple_v1_given_w)

    

    mc_neurons_vnp1 = np.array([[mc_neurons_marginal_vw_v1[a][0]] for a in range(num_neurons)])
    
    
    cj.marginal(joint_v_w_vw, 0, marginal_wn_vw)

    mc_neurons_marginal_wn_vw = np.array([[mc_neurons_joint_v_w_vw[a][1],mc_neurons_joint_v_w_vw[a][2]] for a in range(num_neurons)])

    fig, ax = plt.subplots(1,1)
    ax.scatter(mc_neurons_marginal_wn_vw[:,0], mc_neurons_marginal_wn_vw[:,1], s=0.3)
    ax.set_xlim((w_min,w_max))
    ax.set_ylim((cj.base(c_vw)[2]+cj.size(c_vw)[2],cj.base(c_vw)[2]))
    fig.tight_layout()
    plt.show()
    

    cj.conditional(marginal_vw_vnp1, [0], vw_t, vnp1_given_vw)
    cj.joint3D(marginal_wn_vw, 1, vnp1_given_vw, joint_wn_vw_vnp1)
    cj.marginal(joint_wn_vw_vnp1, 1, joint_wn_vnp1)

    shuff1 = [a for a in range(num_neurons)]
    #random.shuffle(shuff1)
    mc_neurons_joint_wn_vw_vnp1 = np.array([[mc_neurons_marginal_wn_vw[shuff1[a]][0], mc_neurons_marginal_vw_v1[a][0],mc_neurons_marginal_vw_v1[a][1]] for a in range(num_neurons)])
    mc_neurons_marginal_wn_vnp1 = np.array([[mc_neurons_marginal_wn_vw[shuff1[a]][0], mc_neurons_marginal_vw_v1[a][1]] for a in range(num_neurons)])
    

    fig, ax = plt.subplots(1,1)
    ax.scatter(mc_neurons_marginal_wn_vnp1[:,0], mc_neurons_marginal_wn_vnp1[:,1], s=0.3)
    ax.set_xlim((w_min,w_max))
    ax.set_ylim((v_max,v_min))
    fig.tight_layout()

    cj.conditional(joint_wn_vnp1, [0], wn, vnp1_given_wn)

    cj.fork(wn, 0, vnp1_given_wn, 0, wnp1_given_wn, joint_wn_vnp1_wnp1)
    cj.marginal(joint_wn_vnp1_wnp1, 0, joint_vnp1_wnp1)
    

    # alternative

    cj.marginal(joint_vn_vw_vnp1, 1, joint_vn_vnp1)
    cj.conditional(joint_vn_vnp1, [0], vn, vnp1_given_vn)
    
    cj.joint3D(joint_vn_wn, 0, vnp1_given_vn, joint_vn_wn_vnp1)
    
    cj.marginal(joint_vn_wn_vnp1, 0, joint_wn_vnp1)

    plotDist2D(simple_joint_w_v1, res=(v_res,w_res))
    plotDist2D(joint_wn_vnp1, res=(v_res,w_res))
    plt.show()

    cj.conditional(joint_wn_vnp1, [0], wn, vnp1_given_wn)

    cj.fork(wn, 0, vnp1_given_wn, 0, wnp1_given_wn, joint_wn_vnp1_wnp1)
    cj.marginal(joint_wn_vnp1_wnp1, 0, joint_vnp1_wnp1)

    v_w_dist = cj.readDist(joint_vnp1_wnp1)
    n_mass = [a for a in v_w_dist]
    total_reset_mass = 0.0
    for j in range(cj.res(joint_vnp1_wnp1)[0]): # for each column
        for i in range(cj.res(joint_vnp1_wnp1)[1]): 
            index = (i * cj.res(joint_vnp1_wnp1)[0]) + j
            reset_index = (i * cj.res(joint_vnp1_wnp1)[0]) + reset_cell
            if j >= threshold_cell and v_w_dist[index] > 0.0:
                n_mass[reset_index] += v_w_dist[index]
                n_mass[index] = 0.0
                total_reset_mass += v_w_dist[index]
    
    cj.update(joint_vnp1_wnp1, n_mass)

    cj.marginal(joint_vnp1_wnp1, 1, vnp1)

    ###

    # Alternative alternative
    
    #cj.conditional(marginal_wn_vw, [0], wn, vw_given_wn)
    #cj.fork(wn, 0, vw_given_wn, 0, wnp1_given_wn, joint_wn_vw_wnp1)
    
    #cj.marginal(joint_wn_vw_wnp1, 0, marginal_vw_wnp1)
    #cj.conditional(marginal_vw_wnp1, [0], vw_t, wnp1_given_vw)

    #cj.joint3D(marginal_vw_wnp1, 0, vnp1_given_vw, joint_vw_vnp1_wnp1)
    #cj.marginal(joint_vw_vnp1_wnp1, 0, joint_vnp1_wnp1_t)
    #cj.transpose(joint_vnp1_wnp1_t, joint_vnp1_wnp1)

    #v_w_dist = cj.readDist(joint_vnp1_wnp1)
    #n_mass = [a for a in v_w_dist]
    #total_reset_mass = 0.0
    #for j in range(cj.res(joint_vnp1_wnp1)[0]): # for each column
    #    for i in range(cj.res(joint_vnp1_wnp1)[1]): 
    #        index = (i * cj.res(joint_vnp1_wnp1)[0]) + j
    #        reset_index = (i * cj.res(joint_vnp1_wnp1)[0]) + reset_cell
    #        if j >= threshold_cell and v_w_dist[index] > 0.0:
    #            n_mass[reset_index] += v_w_dist[index]
    #            n_mass[index] = 0.0
    #            total_reset_mass += v_w_dist[index]
    
    #cj.update(joint_vnp1_wnp1, n_mass)

    #cj.marginal(joint_vnp1_wnp1, 1, vnp1)

    ###


    cj.fork(wn, 0, simple_v1_given_w, 0, wnp1_given_wn, simple_joint_w0_v1_w1)
    cj.marginal(simple_joint_w0_v1_w1, 0, simple_joint_vnp1_wnp1)
    
    v_w_dist = cj.readDist(simple_joint_vnp1_wnp1)
    n_mass = [a for a in v_w_dist]
    total_reset_mass = 0.0
    for j in range(cj.res(simple_joint_vnp1_wnp1)[0]): # for each column
        for i in range(cj.res(simple_joint_vnp1_wnp1)[1]): 
            index = (i * cj.res(simple_joint_vnp1_wnp1)[0]) + j
            reset_index = (i * cj.res(simple_joint_vnp1_wnp1)[0]) + reset_cell
            if j >= threshold_cell and v_w_dist[index] > 0.0:
                n_mass[reset_index] += v_w_dist[index]
                n_mass[index] = 0.0
                total_reset_mass += v_w_dist[index]
    
    cj.update(simple_joint_vnp1_wnp1, n_mass)

    cj.marginal(simple_joint_vnp1_wnp1, 1, simple_vnp1)

    shuff1 = [a for a in range(num_neurons)]
    #random.shuffle(shuff1)
    shuff2 = [a for a in range(num_neurons)]
    #random.shuffle(shuff2)
    mc_neurons_prime = np.array([[mc_neurons_marginal_wn_vnp1[shuff2[a]][1],mc_neurons_marginal_w_w1[shuff1[a]][1]] for a in range(num_neurons)])

    for nn in range(len(mc_neurons_prime)):
        if (mc_neurons_prime[nn][0] > -50.0):
            mc_neurons_prime[nn][0] = -70.6

    fig, ax = plt.subplots(1,1)
    ax.scatter(mc_neurons_prime[:,0], mc_neurons_prime[:,1], s=0.3)
    ax.set_xlim((v_min,v_max))
    ax.set_ylim((w_max,w_min))
    fig.tight_layout()

    plotDist2D(joint_vnp1_wnp1, res=(w_res,v_res))
    plotDist2D(simple_joint_vnp1_wnp1, res=(w_res,v_res))
    plt.show()

    # update monte carlo joint

    for nn in range(len(mc_joint_neurons_prime)):
        mc_joint_neurons_prime[nn] = cond(mc_joint_neurons_prime[nn])

        if (mc_joint_neurons_prime[nn][0] > -50.0):
            mc_joint_neurons_prime[nn][0] = -70.6

        mc_joint_neurons_prime[nn][1] += epsp*poisson.rvs(w_rate*timestep) # override w
        
    # plot

    if (iteration % 1 == 0) :
        dist_v = cj.readDist(vnp1)
        dist_w = cj.readDist(wnp1)

        dist_v = [a / ((v_max-v_min)/v_res) for a in dist_v]
        dist_w = [a / ((w_max-w_min)/w_res) for a in dist_w]

        simple_dist_v = cj.readDist(simple_vnp1)
        simple_dist_w = cj.readDist(simple_wn)

        simple_dist_v = [a / ((v_max-v_min)/v_res) for a in simple_dist_v]
        simple_dist_w = [a / ((w_max-w_min)/w_res) for a in simple_dist_w]

        fig, ax = plt.subplots(1,2)
        ax[0].plot(v, dist_v)
        ax[1].plot(w, dist_w)
        ax[0].plot(v, simple_dist_v)
        ax[1].plot(w, simple_dist_w)
        ax[0].hist(mc_neurons_prime[:,0], density=True, bins=v_res, range=[v_min,v_max], histtype='step')
        ax[1].hist(mc_neurons_prime[:,1], density=True, bins=w_res, range=[w_min,w_max], histtype='step')
        ax[0].hist(mc_joint_neurons_prime[:,0], density=True, bins=v_res, range=[v_min,v_max], histtype='step')
        ax[1].hist(mc_joint_neurons_prime[:,1], density=True, bins=w_res, range=[w_min,w_max], histtype='step')
    
        fig.tight_layout()
        plt.show()

    cj.transfer(vnp1,vn)
    cj.transfer(wnp1,wn)
    cj.transfer(joint_vnp1_wnp1, joint_vn_wn)
    cj.transfer(simple_vnp1,simple_vn)
    cj.transfer(simple_wnp1,simple_wn)
    cj.transfer(simple_joint_vnp1_wnp1, simple_joint_vn_wn)