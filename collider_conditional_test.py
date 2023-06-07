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
    u = y[2]

    v_prime = (-g_l*(v - E_l) - w * (v - E_e) - u * (1)) / C
    w_prime = -(w) / tau_e
    u_prime = -(u) / tau_i

    return [v + timestep*v_prime, w + timestep*w_prime, u + timestep*u_prime]

def w_prime(y):
    w = y[0]
    jump = y[1]
    tau_e =10.49
    dt = timestep

    w_prime = -(w) / tau_e
    return w + (dt*w_prime) + jump

def u_prime(y):
    u = y[0]
    jump = y[1]
    tau_i = 10.49
    dt = timestep

    u_prime = -(u) / tau_i
    return u + (dt*u_prime) + jump

def vw(y):
    v = y[0]
    w = y[1]

    E_e = 0.0

    return -w * (v - E_e)

def vu(y):
    v = y[0]
    u = y[1]

    E_i = 0.0 # -75.0

    return  -u * (1)

def vwvu(y):
    vw = y[0]
    vu = y[1]

    return vw + vu

def v_prime(y):
    v = y[0]
    vw_vu = y[1] 

    E_l = -70.6
    C = 281
    g_l = 0.03
    dt = timestep

    v_prime = (-g_l*(v - E_l) + vw_vu) / C

    return v + dt*v_prime

res = 100
v_res = 100
w_res = 100
u_res = 100
I_res = 100

v_max = -40.0
v_min = -80.0
w_max = 50.0 #25.0
w_min = -5.0 #-1.0
u_max = 50.0
u_min = -5.0

# Set up the starting distribution
v = np.linspace(v_min, v_max, v_res)
w = np.linspace(w_min, w_max, w_res)
u = np.linspace(u_min, u_max, u_res)

points = []
for x in range(u_res):
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
updf = [a * ((u_max-u_min)/u_res) for a in norm.pdf(u, 15.0, 2.4)]

vpdf = [a / sum(vpdf) for a in vpdf]
wpdf = [a / sum(wpdf) for a in wpdf]
updf = [a / sum(updf) for a in updf]

v0 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in vpdf])
w0 = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in wpdf])
u0 = cj.newDist([u_min],[(u_max-u_min)],[u_res],[a for a in updf])

w_rate = 4
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

u_rate = 2
ipsp = 0.5
uI_max_events = 10
uI_min_events = -2
uI_max = uI_max_events*ipsp
uI_min = uI_min_events*ipsp
ipsps = np.linspace(uI_min, uI_max, I_res)
uI_events = np.linspace(uI_min_events, uI_max_events, I_res)
uIpdf_final = [0 for a in uI_events]
for i in range(len(uI_events)-1):
    if (int(uI_events[i]) < int(uI_events[i+1])) or (uI_events[i] < 0 and uI_events[i+1] >= 0): # we have just passed a new event
        e = int(uI_events[i+1])
        if e <= 0:
            e = int(uI_events[i])
        diff = uI_events[i+1] - uI_events[i]
        lower_prop = (int(uI_events[i+1]) - uI_events[i]) / diff 
        upper_prop = 1.0 - lower_prop
        uIpdf_final[i] += poisson.pmf(e, u_rate*timestep) * lower_prop
        uIpdf_final[i+1] += poisson.pmf(e, u_rate*timestep) * upper_prop
uIpdf = uIpdf_final
uI = cj.newDist([uI_min], [uI_max], [I_res], [a for a in uIpdf])

c_w_prime = cj.boundedFunction([w_min,wI_min],[(w_max-w_min),(wI_max-wI_min)],[w_res,I_res], w_prime, w_min, (w_max-w_min), w_res)
c_u_prime = cj.boundedFunction([u_min,uI_min],[(u_max-u_min),(uI_max-uI_min)],[u_res,I_res], u_prime, u_min, (u_max-u_min), u_res)
c_vw = cj.function([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], vw, res)
c_vu = cj.function([v_min,u_min],[(v_max-v_min),(u_max-u_min)],[v_res,u_res], vu, res)
c_vwvu = cj.function([cj.base(c_vw)[2],cj.base(c_vu)[2]],[cj.size(c_vw)[2],cj.size(c_vu)[2]],[res,res], vwvu, res)
c_v_prime = cj.boundedFunction([v_min,cj.base(c_vwvu)[2]],[(v_max-v_min),cj.size(c_vwvu)[2]],[v_res,res],v_prime, v_min, (v_max-v_min), v_res)

print(cj.base(c_vw)[2],cj.size(c_vw)[2])
print(cj.base(c_vu)[2],cj.size(c_vu)[2])
print(cj.base(c_vwvu)[2],cj.size(c_vwvu)[2])


joint_w0_wI_w1 = cj.newDist([w_min,wI_min,w_min],[(w_max-w_min),(wI_max-wI_min),(w_max-w_min)],[w_res,I_res,w_res],[a for a in np.zeros(w_res*I_res*w_res)])
joint_u0_uI_u1 = cj.newDist([u_min,uI_min,u_min],[(u_max-u_min),(uI_max-uI_min),(u_max-u_min)],[u_res,I_res,u_res],[a for a in np.zeros(u_res*I_res*u_res)])
joint_w0_wI = cj.newDist([w_min,wI_min],[(w_max-w_min),(wI_max-wI_min)],[w_res,I_res],[a for a in np.zeros(w_res*I_res)])
joint_u0_uI = cj.newDist([u_min,uI_min],[(u_max-u_min),(uI_max-uI_min)],[u_res,I_res],[a for a in np.zeros(u_res*I_res)])

cj.joint2Di(w0, wI, joint_w0_wI)
cj.joint2Di(u0, uI, joint_u0_uI)
num_neurons = 1000

w_inputs = np.array([[poisson.rvs(w_rate*timestep)*epsp] for a in range(num_neurons)])
u_inputs = np.array([[poisson.rvs(u_rate*timestep)*ipsp] for a in range(num_neurons)])

mc_neurons = np.array([[norm.rvs(-70.6, 2.4, 1)[0],norm.rvs(15.0, 2.4, 1)[0],norm.rvs(15.0, 2.4, 1)[0]] for a in range(num_neurons)])

dist_w0_check = cj.readDist(w0)
dist_w0_check = [a / ((w_max-w_min)/w_res) for a in dist_w0_check]
fig, ax = plt.subplots(1, 1)
ax.plot(np.linspace(w_min, w_max, w_res), dist_w0_check)
ax.hist(mc_neurons[:,1], density=True, bins=500, range=[w_min,w_max], histtype='step')
fig.tight_layout()
plt.show()


dist_w0_check = cj.readDist(v0)
dist_w0_check = [a / ((v_max-v_min)/v_res) for a in dist_w0_check]
fig, ax = plt.subplots(1, 1)
ax.plot(np.linspace(v_min, v_max, v_res), dist_w0_check)
ax.hist(mc_neurons[:,0], density=True, bins=500, range=[v_min,v_max], histtype='step')
fig.tight_layout()
plt.show()

cj.collider(joint_w0_wI, [0,1], c_w_prime, joint_w0_wI_w1)
cj.collider(joint_u0_uI, [0,1], c_u_prime, joint_u0_uI_u1)


marginal_wI_w1 = cj.newDist([wI_min,w_min],[(wI_max-wI_min),(w_max-w_min)],[I_res,w_res],[a for a in np.zeros(I_res*w_res)])
marginal_uI_u1 = cj.newDist([uI_min,u_min],[(uI_max-uI_min),(u_max-u_min)],[I_res,u_res],[a for a in np.zeros(I_res*u_res)])
w1 = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in np.zeros(w_res)])
u1 = cj.newDist([u_min],[(u_max-u_min)],[u_res],[a for a in np.zeros(u_res)])

cj.marginal(joint_w0_wI_w1, 0, marginal_wI_w1)
cj.marginal(joint_u0_uI_u1, 0, marginal_uI_u1)

cj.marginal(marginal_wI_w1, 0, w1)
cj.marginal(marginal_uI_u1, 0, u1)

mc_neurons_joint_w_wI_w1 = np.array([[mc_neurons[a][1],w_inputs[a][0],w_prime([mc_neurons[a][1],w_inputs[a][0]])] for a in range(num_neurons)])
mc_neurons_joint_u_uI_u1 = np.array([[mc_neurons[a][2],u_inputs[a][0],u_prime([mc_neurons[a][2],u_inputs[a][0]])] for a in range(num_neurons)])

dist_w0_check = cj.readDist(w1)
dist_w0_check = [a / ((w_max-w_min)/w_res) for a in dist_w0_check]
fig, ax = plt.subplots(1, 1)
ax.plot(np.linspace(w_min, w_max, w_res), dist_w0_check)
ax.hist(mc_neurons_joint_w_wI_w1[:,2], density=True, bins=500, range=[w_min,w_max], histtype='step')
fig.tight_layout()
plt.show()

marginal_w0_w1 = cj.newDist([w_min,w_min],[(w_max-w_min),(w_max-w_min)],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])
marginal_u0_u1 = cj.newDist([u_min,u_min],[(u_max-u_min),(u_max-u_min)],[u_res,u_res],[a for a in np.zeros(u_res*u_res)])
w1_given_w0 = cj.newDist([w_min,w_min],[(w_max-w_min),(w_max-w_min)],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])
u1_given_u0 = cj.newDist([u_min,u_min],[(u_max-u_min),(u_max-u_min)],[u_res,u_res],[a for a in np.zeros(u_res*u_res)])

cj.marginal(joint_w0_wI_w1, 1, marginal_w0_w1)
cj.marginal(joint_u0_uI_u1, 1, marginal_u0_u1)
cj.conditional(marginal_w0_w1, [0], w0, w1_given_w0)
cj.conditional(marginal_u0_u1, [0], u0, u1_given_u0)

joint_v_w = cj.newDistFrom2(v0, w0)
joint_v_u = cj.newDistFrom2(v0, u0)

cj.rescale(joint_v_w)
cj.rescale(joint_v_u)

joint_v_w_vw = cj.newDist([v_min,w_min,cj.base(c_vw)[2]],[(v_max-v_min),(w_max-w_min),cj.size(c_vw)[2]],[v_res,w_res,res], [a for a in np.zeros(v_res*w_res*res)])
joint_v_u_vu = cj.newDist([v_min,u_min,cj.base(c_vu)[2]],[(v_max-v_min),(u_max-u_min),cj.size(c_vu)[2]],[v_res,u_res,res], [a for a in np.zeros(v_res*u_res*res)])

cj.collider(joint_v_w, [0,1], c_vw, joint_v_w_vw)
cj.collider(joint_v_u, [0,1], c_vu, joint_v_u_vu)

joint_v_vw = cj.newDist([v_min, cj.base(c_vw)[2]], [(v_max-v_min), cj.size(c_vw)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])
joint_v_vu = cj.newDist([v_min, cj.base(c_vu)[2]], [(v_max-v_min), cj.size(c_vu)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])

cj.marginal(joint_v_w_vw, 1, joint_v_vw)
cj.marginal(joint_v_u_vu, 1, joint_v_vu)

vw_t = cj.newDist([cj.base(c_vw)[2]], [cj.size(c_vw)[2]], [res], [a for a in np.zeros(res)])

cj.marginal(joint_v_vw, 0, vw_t)


mc_neurons_joint_v_w_vw = np.array([[mc_neurons[a][0],mc_neurons[a][1],vw([mc_neurons[a][0],mc_neurons[a][1]])] for a in range(num_neurons)])
mc_neurons_joint_v_u_vu = np.array([[mc_neurons[a][0],mc_neurons[a][2],vu([mc_neurons[a][0],mc_neurons[a][2]])] for a in range(num_neurons)])

mc_neurons_joint_vw_vu = np.array([[mc_neurons_joint_v_w_vw[a][2],mc_neurons_joint_v_u_vu[a][2]] for a in range(num_neurons)])

dist_w0_check = cj.readDist(vw_t)
dist_w0_check = [a / (cj.size(c_vw)[2]/res) for a in dist_w0_check]
fig, ax = plt.subplots(1, 1)
ax.set_title("vw")
ax.plot(np.linspace(cj.base(c_vw)[2], cj.base(c_vw)[2]+cj.size(c_vw)[2], res), dist_w0_check)
ax.hist(mc_neurons_joint_v_w_vw[:,2], density=True, bins=500, range=[cj.base(c_vw)[2],cj.base(c_vw)[2]+cj.size(c_vw)[2]], histtype='step')
fig.tight_layout()
plt.show()

vw_given_v = cj.newDist([v_min, cj.base(c_vw)[2]], [(v_max-v_min), cj.size(c_vw)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])
vu_given_v = cj.newDist([v_min, cj.base(c_vu)[2]], [(v_max-v_min), cj.size(c_vu)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])

cj.conditional(joint_v_vw, [0], v0, vw_given_v)
cj.conditional(joint_v_vu, [0], v0, vu_given_v)

joint_v_vw_vu = cj.newDist([v_min,cj.base(c_vw)[2],cj.base(c_vu)[2]],[(v_max-v_min),cj.size(c_vw)[2],cj.size(c_vu)[2]],[v_res,res,res], [a for a in np.zeros(v_res*res*res)])
cj.fork(v0, 0, vw_given_v, 0, vu_given_v, joint_v_vw_vu)

joint_vw_vu = cj.newDist([cj.base(c_vw)[2],cj.base(c_vu)[2]],[cj.size(c_vw)[2],cj.size(c_vu)[2]],[res,res], [a for a in np.zeros(res*res)])
cj.marginal(joint_v_vw_vu, 0, joint_vw_vu)

joint_vw_vu_vwvu = cj.newDist([cj.base(c_vw)[2],cj.base(c_vu)[2],cj.base(c_vwvu)[2]],[cj.size(c_vw)[2],cj.size(c_vu)[2],cj.size(c_vwvu)[2]],[res,res,res], [a for a in np.zeros(res*res*res)])
cj.collider(joint_vw_vu, [0,1], c_vwvu, joint_vw_vu_vwvu)

joint_v0_vw = cj.newDist([v_min,cj.base(c_vw)[2]],[(v_max-v_min),cj.size(c_vw)[2]],[v_res,res], [a for a in np.zeros(v_res*res)])
cj.joint2D(v0, 0, vw_given_v, joint_v0_vw)
cj.rescale(joint_v0_vw)

joint_v0_vu = cj.newDist([v_min,cj.base(c_vu)[2]],[(v_max-v_min),cj.size(c_vu)[2]],[v_res,res], [a for a in np.zeros(v_res*res)])
cj.joint2D(v0, 0, vu_given_v, joint_v0_vu)

marginal_vw_vwvu = cj.newDist([cj.base(c_vw)[2],cj.base(c_vwvu)[2]],[cj.size(c_vw)[2],cj.size(c_vwvu)[2]],[res,res], [a for a in np.zeros(res*res)])
marginal_vw = cj.newDist([cj.base(c_vw)[2]],[cj.size(c_vw)[2]],[res], [a for a in np.zeros(res)])

cj.marginal(joint_vw_vu_vwvu, 1, marginal_vw_vwvu)
cj.marginal(marginal_vw_vwvu, 1, marginal_vw)

marginal_vu_vwvu = cj.newDist([cj.base(c_vu)[2],cj.base(c_vwvu)[2]],[cj.size(c_vu)[2],cj.size(c_vwvu)[2]],[res,res], [a for a in np.zeros(res*res)])
marginal_vu = cj.newDist([cj.base(c_vu)[2]],[cj.size(c_vu)[2]],[res], [a for a in np.zeros(res)])

cj.marginal(joint_vw_vu_vwvu, 0, marginal_vu_vwvu)
cj.marginal(marginal_vu_vwvu, 1, marginal_vu)

cj.rescale(joint_vw_vu_vwvu)
cj.rescale(marginal_vu_vwvu)

vwvu_given_vw = cj.newDist([cj.base(c_vw)[2],cj.base(c_vwvu)[2]],[cj.size(c_vw)[2],cj.size(c_vwvu)[2]],[res,res], [a for a in np.zeros(res*res)])
cj.conditional(marginal_vw_vwvu, [0], marginal_vw, vwvu_given_vw)

vwvu_given_vu = cj.newDist([cj.base(c_vu)[2],cj.base(c_vwvu)[2]],[cj.size(c_vu)[2],cj.size(c_vwvu)[2]],[res,res], [a for a in np.zeros(res*res)])
cj.conditional(marginal_vu_vwvu, [0], marginal_vu, vwvu_given_vu)

marginal_vwvu_v1 = cj.newDist([cj.base(c_vwvu)[2],v_min],[cj.size(c_vwvu)[2],(v_max-v_min)],[res,v_res], [a for a in np.zeros(res*v_res)])
v1 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in np.zeros(v_res)])

marginal_vwvu = cj.newDist([cj.base(c_vwvu)[2]],[cj.size(c_vwvu)[2]],[res], [a for a in np.zeros(res)])
joint_v0_vwvu = cj.newDist([v_min,cj.base(c_vwvu)[2]],[(v_max-v_min),cj.size(c_vwvu)[2]],[v_res,res], [a for a in np.zeros(v_res*res)])
joint_v0_vwvu_check = cj.newDist([v_min,cj.base(c_vwvu)[2]],[(v_max-v_min),cj.size(c_vwvu)[2]],[v_res,res], [a for a in np.zeros(v_res*res)])

# Independent v0 and vwvu
cj.marginal(marginal_vw_vwvu, 0, marginal_vwvu)
#cj.joint2Di(v0, marginal_vwvu, joint_v0_vwvu)

mc_neurons_joint_vw_vu_vwvu = np.array([[mc_neurons_joint_v_w_vw[a][2],mc_neurons_joint_v_u_vu[a][2],vwvu([mc_neurons_joint_v_w_vw[a][2],mc_neurons_joint_v_u_vu[a][2]])] for a in range(num_neurons)])


dist_w0_check = cj.readDist(marginal_vwvu)
dist_w0_check = [a / (cj.size(c_vwvu)[2]/res) for a in dist_w0_check]
fig, ax = plt.subplots(1, 1)
ax.set_title("vwvu")
ax.plot(np.linspace(cj.base(c_vwvu)[2], cj.base(c_vwvu)[2]+cj.size(c_vwvu)[2], res), dist_w0_check)
ax.hist(mc_neurons_joint_vw_vu_vwvu[:,2], density=True, bins=500, range=[cj.base(c_vwvu)[2],cj.base(c_vwvu)[2]+cj.size(c_vwvu)[2]], histtype='step')
fig.tight_layout()
plt.show()

# Dependent v0 and vwvu
vw_vu_given_v = cj.newDist([v_min,cj.base(c_vw)[2],cj.base(c_vu)[2]],[(v_max-v_min),cj.size(c_vw)[2],cj.size(c_vu)[2]],[v_res,res,res], [a for a in np.zeros(v_res*res*res)])
vwvu_given_vw_vu = cj.newDist([cj.base(c_vw)[2],cj.base(c_vu)[2],cj.base(c_vwvu)[2]],[cj.size(c_vw)[2],cj.size(c_vu)[2],cj.size(c_vwvu)[2]],[res,res,res], [a for a in np.zeros(res*res*res)])

cj.conditional(joint_v_vw_vu, [0], v0, vw_vu_given_v)
cj.conditional(joint_vw_vu_vwvu, [0,1], joint_vw_vu, vwvu_given_vw_vu)
cj.diamond(v0, vw_vu_given_v, vwvu_given_vw_vu, joint_v0_vwvu)

# Just via vw
joint_v0_vw_vwvu = cj.newDist([v_min,cj.base(c_vw)[2],cj.base(c_vwvu)[2]],[(v_max-v_min),cj.size(c_vw)[2],cj.size(c_vwvu)[2]],[v_res,res,res], [a for a in np.zeros(v_res*res*res)])
cj.joint3D(joint_v0_vw, 1, vwvu_given_vw, joint_v0_vw_vwvu)
cj.marginal(joint_v0_vw_vwvu, 1, joint_v0_vwvu_check)

cj.rescale(joint_v0_vwvu)
cj.rescale(joint_v0_vwvu_check)

#plotDist2D(joint_v0_vwvu)
#plotDist2D(joint_v0_vwvu_check)
#plt.show()


######

joint_v0_vwvu_v1 = cj.newDist([v_min,cj.base(c_vwvu)[2],v_min],[(v_max-v_min),cj.size(c_vwvu)[2],(v_max-v_min)],[v_res,res,v_res], [a for a in np.zeros(v_res*res*v_res)])

cj.collider(joint_v0_vwvu, [0,1], c_v_prime, joint_v0_vwvu_v1)
cj.rescale(joint_v0_vwvu_v1)
cj.marginal(joint_v0_vwvu_v1, 0, marginal_vwvu_v1)
cj.marginal(marginal_vwvu_v1, 0, v1)
cj.rescale(v1)


mc_neurons_joint_v0_vwvu_v1 = np.array([[mc_neurons[a][0],mc_neurons_joint_vw_vu_vwvu[a][2],v_prime([mc_neurons[a][0],mc_neurons_joint_vw_vu_vwvu[a][2]])] for a in range(num_neurons)])

dist_w0_check = cj.readDist(v1)
dist_w0_check = [a / ((v_max-v_min)/v_res) for a in dist_w0_check]
fig, ax = plt.subplots(1, 1)
ax.set_title("v1")
ax.plot(np.linspace(v_min,v_max, v_res), dist_w0_check)
ax.hist(mc_neurons_joint_v0_vwvu_v1[:,2], density=True, bins=500, range=[v_min,v_max], histtype='step')
fig.tight_layout()
plt.show()

v1_given_w0 = cj.newDist([w_min, v_min], [(w_max-w_min), (v_max-v_min)], [w_res,v_res], [a for a in np.zeros(w_res*v_res)])

joint_w0_vw_vwvu = cj.newDist([w_min,cj.base(c_vw)[2],cj.base(c_vwvu)[2]],[(w_max-w_min),cj.size(c_vw)[2],cj.size(c_vwvu)[2]],[w_res,res,res], [a for a in np.zeros(w_res*res*res)])
marginal_w0_vw = cj.newDist([w_min,cj.base(c_vw)[2]],[(w_max-w_min),cj.size(c_vw)[2]],[w_res,res], [a for a in np.zeros(w_res*res)])

cj.marginal(joint_v_w_vw, 0, marginal_w0_vw)
cj.joint3D(marginal_w0_vw, 1, vwvu_given_vw, joint_w0_vw_vwvu)

marginal_w0_vwvu = cj.newDist([w_min,cj.base(c_vwvu)[2]],[(w_max-w_min),cj.size(c_vwvu)[2]],[w_res,res], [a for a in np.zeros(w_res*res)])

cj.marginal(joint_w0_vw_vwvu, 1, marginal_w0_vwvu)

v1_given_vwvu = cj.newDist([cj.base(c_vwvu)[2],v_min],[cj.size(c_vwvu)[2],(v_max-v_min)],[res,v_res], [a for a in np.zeros(res*v_res)])

cj.conditional(marginal_vwvu_v1, [0], marginal_vwvu, v1_given_vwvu)

joint_w0_vwvu_v1 = cj.newDist([w_min,cj.base(c_vwvu)[2],v_min],[(w_max-w_min),cj.size(c_vwvu)[2],(v_max-v_min)],[w_res,res,v_res], [a for a in np.zeros(w_res*res*v_res)])
marginal_w0_v1 = cj.newDist([w_min, v_min], [(w_max-w_min), (v_max-v_min)], [w_res,v_res], [a for a in np.zeros(w_res*v_res)])

cj.joint3D(marginal_w0_vwvu, 1, v1_given_vwvu, joint_w0_vwvu_v1)
cj.marginal(joint_w0_vwvu_v1, 1, marginal_w0_v1)
cj.conditional(marginal_w0_v1, [0], w0, v1_given_w0)

v1_given_u0 = cj.newDist([u_min, v_min], [(u_max-u_min), (v_max-v_min)], [u_res,v_res], [a for a in np.zeros(u_res*v_res)])

joint_u0_vu_vwvu = cj.newDist([u_min,cj.base(c_vu)[2],cj.base(c_vwvu)[2]],[(u_max-u_min),cj.size(c_vu)[2],cj.size(c_vwvu)[2]],[u_res,res,res], [a for a in np.zeros(u_res*res*res)])
marginal_u0_vu = cj.newDist([u_min,cj.base(c_vu)[2]],[(u_max-u_min),cj.size(c_vu)[2]],[u_res,res], [a for a in np.zeros(u_res*res)])

cj.marginal(joint_v_u_vu, 0, marginal_u0_vu)
cj.joint3D(marginal_u0_vu, 1, vwvu_given_vu, joint_u0_vu_vwvu)

marginal_u0_vwvu = cj.newDist([u_min,cj.base(c_vwvu)[2]],[(u_max-u_min),cj.size(c_vwvu)[2]],[u_res,res], [a for a in np.zeros(u_res*res)])

cj.marginal(joint_u0_vu_vwvu, 1, marginal_u0_vwvu)

joint_u0_vwvu_v1 = cj.newDist([u_min,cj.base(c_vwvu)[2],v_min],[(u_max-u_min),cj.size(c_vwvu)[2],(v_max-v_min)],[u_res,res,v_res], [a for a in np.zeros(u_res*res*v_res)])
marginal_u0_v1 = cj.newDist([u_min, v_min], [(u_max-u_min), (v_max-v_min)], [u_res,v_res], [a for a in np.zeros(u_res*v_res)])

cj.joint3D(marginal_u0_vwvu, 1, v1_given_vwvu, joint_u0_vwvu_v1)
cj.marginal(joint_u0_vwvu_v1, 1, marginal_u0_v1)

cj.conditional(marginal_u0_v1, [0], u0, v1_given_u0)

joint_w0_v1_w1 = cj.newDist([w_min,v_min,w_min],[(w_max-w_min),(v_max-v_min),(w_max-w_min)],[w_res,v_res,w_res], [a for a in np.zeros(w_res*v_res*w_res)])
joint_u0_v1_u1 = cj.newDist([u_min,v_min,u_min],[(u_max-u_min),(v_max-v_min),(u_max-u_min)],[u_res,v_res,u_res], [a for a in np.zeros(u_res*v_res*u_res)])
joint_v1_w1 = cj.newDist([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])
joint_v1_u1 = cj.newDist([v_min,u_min],[(v_max-v_min),(u_max-u_min)],[v_res,u_res], [a for a in np.zeros(v_res*u_res)])

cj.fork(w0, 0, v1_given_w0, 0, w1_given_w0, joint_w0_v1_w1)
cj.fork(u0, 0, v1_given_u0, 0, u1_given_u0, joint_u0_v1_u1)

cj.marginal(joint_w0_v1_w1, 0, joint_v1_w1)
cj.marginal(joint_u0_v1_u1, 0, joint_v1_u1)


v1_check = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in np.zeros(v_res)])
cj.marginal(joint_v1_w1, 1, v1_check)

v1_check2 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in np.zeros(v_res)])
cj.marginal(joint_v1_u1, 1, v1_check2)

w1_check = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in np.zeros(w_res)])
cj.marginal(joint_v1_w1, 0, w1_check)

u1_check = cj.newDist([u_min],[(u_max-u_min)],[u_res],[a for a in np.zeros(u_res)])
cj.marginal(joint_v1_u1, 0, u1_check)

mc_neurons_prime = np.array([[mc_neurons_joint_v0_vwvu_v1[a][2],mc_neurons_joint_w_wI_w1[a][2],mc_neurons_joint_u_uI_u1[a][2]] for a in range(num_neurons)])
w_inputs = np.array([[poisson.rvs(w_rate*timestep)*epsp] for a in range(num_neurons)])
u_inputs = np.array([[poisson.rvs(u_rate*timestep)*ipsp] for a in range(num_neurons)])

dist_w0_check = cj.readDist(v1_check)
dist_w0_check = [a / ((v_max-v_min)/v_res) for a in dist_w0_check]
fig, ax = plt.subplots(1, 1)
ax.set_title("v1")
ax.plot(np.linspace(v_min,v_max, v_res), dist_w0_check)
ax.hist(mc_neurons_prime[:,0], density=True, bins=500, range=[v_min,v_max], histtype='step')
fig.tight_layout()
plt.show()

dist_w0_check = cj.readDist(w1_check)
dist_w0_check = [a / ((w_max-w_min)/w_res) for a in dist_w0_check]
fig, ax = plt.subplots(1, 1)
ax.set_title("w1")
ax.plot(np.linspace(w_min,w_max, w_res), dist_w0_check)
ax.hist(mc_neurons_prime[:,1], density=True, bins=500, range=[w_min,w_max], histtype='step')
fig.tight_layout()
plt.show()

print(cj.base(joint_v1_w1), cj.size(joint_v1_w1), cj.base(c_vw), cj.size(c_vw), cj.base(joint_v_w_vw), cj.size(joint_v_w_vw))

cj.collider(joint_v1_w1, [0,1], c_vw, joint_v_w_vw)
cj.collider(joint_v1_u1, [0,1], c_vu, joint_v_u_vu)

cj.marginal(joint_v_w_vw, 1, joint_v_vw)
cj.marginal(joint_v_u_vu, 1, joint_v_vu)
cj.marginal(joint_v_vw, 0, vw_t)


mc_neurons_joint_v_w_vw = np.array([[mc_neurons_prime[a][0],mc_neurons_prime[a][1],vw([mc_neurons_prime[a][0],mc_neurons_prime[a][1]])] for a in range(num_neurons)])
mc_neurons_joint_v_u_vu = np.array([[mc_neurons_prime[a][0],mc_neurons_prime[a][2],vu([mc_neurons_prime[a][0],mc_neurons_prime[a][2]])] for a in range(num_neurons)])

mc_neurons_joint_vw_vu = np.array([[mc_neurons_joint_v_w_vw[a][2],mc_neurons_joint_v_u_vu[a][2]] for a in range(num_neurons)])

dist_w0_check = cj.readDist(vw_t)
dist_w0_check = [a / (cj.size(c_vw)[2]/res) for a in dist_w0_check]
fig, ax = plt.subplots(1, 1)
ax.set_title("vw")
ax.plot(np.linspace(cj.base(c_vw)[2], cj.base(c_vw)[2]+cj.size(c_vw)[2], res), dist_w0_check)
ax.hist(mc_neurons_joint_v_w_vw[:,2], density=True, bins=500, range=[cj.base(c_vw)[2],cj.base(c_vw)[2]+cj.size(c_vw)[2]], histtype='step')
fig.tight_layout()
plt.show()


# Joint Monte Carlo

mc_joint_neurons = np.array([[norm.rvs(-70.6, 2.4, 1)[0],norm.rvs(15.0, 2.4, 1)[0],norm.rvs(15.0, 2.4, 1)[0]] for a in range(num_neurons)])

mc_joint_neurons_prime = np.array([cond(mc_joint_neurons[a]) for a in range(num_neurons)])

for nn in range(len(mc_joint_neurons_prime)):
        if (mc_joint_neurons_prime[nn][0] > -50.0):
            mc_joint_neurons_prime[nn][0] = -70.6

        mc_joint_neurons_prime[nn][1] += epsp*poisson.rvs(w_rate*timestep) # override w
        mc_joint_neurons_prime[nn][2] += ipsp*poisson.rvs(u_rate*timestep)

        
# Timestep 1 seems fine so far. Now to start the loop.

v2 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in np.zeros(v_res)])
w2 = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in np.zeros(w_res)])
u2 = cj.newDist([u_min],[(u_max-u_min)],[u_res],[a for a in np.zeros(u_res)])

joint_v2_w2 = cj.newDist([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])
joint_v2_u2 = cj.newDist([v_min,u_min],[(v_max-v_min),(u_max-u_min)],[v_res,u_res], [a for a in np.zeros(v_res*u_res)])

for iteration in range(1000):

    # Calculate next w and u
    cj.joint2Di(w1, wI, joint_w0_wI)
    cj.joint2Di(u1, uI, joint_u0_uI)

    cj.collider(joint_w0_wI, [0,1], c_w_prime, joint_w0_wI_w1)
    cj.collider(joint_u0_uI, [0,1], c_u_prime, joint_u0_uI_u1)
    
    cj.marginal(joint_w0_wI_w1, 0, marginal_wI_w1)
    cj.marginal(joint_u0_uI_u1, 0, marginal_uI_u1)

    cj.marginal(marginal_wI_w1, 0, w2)
    cj.marginal(marginal_uI_u1, 0, u2)

    mc_neurons_joint_w_wI_w1 = np.array([[mc_neurons_prime[a][1],w_inputs[a][0],w_prime([mc_neurons_prime[a][1],w_inputs[a][0]])] for a in range(num_neurons)])
    mc_neurons_joint_u_uI_u1 = np.array([[mc_neurons_prime[a][2],u_inputs[a][0],u_prime([mc_neurons_prime[a][2],u_inputs[a][0]])] for a in range(num_neurons)])

    dist_w0 = cj.readDist(w1)
    dist_w1 = cj.readDist(w2)
    dist_w0 = [a / ((w_max-w_min)/100) for a in dist_w0]
    dist_w1 = [a / ((w_max-w_min)/100) for a in dist_w1]
    fig, ax = plt.subplots(1, 1)
    ax.set_title("w1, w2")
    ax.plot(np.linspace(w_min, w_max, 100), dist_w0)
    ax.plot(np.linspace(w_min, w_max, 100), dist_w1)
    ax.hist(mc_neurons_joint_w_wI_w1[:,0], density=True, bins=100, range=[w_min,w_max], histtype='step')
    ax.hist(mc_neurons_joint_w_wI_w1[:,2], density=True, bins=100, range=[w_min,w_max], histtype='step')
    fig.tight_layout()
    plt.show()


    # Calc w2 given w1 for later

    cj.marginal(joint_w0_wI_w1, 1, marginal_w0_w1)
    cj.marginal(joint_u0_uI_u1, 1, marginal_u0_u1)
    cj.conditional(marginal_w0_w1, [0], w1, w1_given_w0)
    cj.conditional(marginal_u0_u1, [0], u1, u1_given_u0)

    # Take the joint distirbutions of joint_v_u from the previous iteration and calcualte vw and vu

    cj.collider(joint_v1_w1, [0,1], c_vw, joint_v_w_vw)
    cj.collider(joint_v1_u1, [0,1], c_vu, joint_v_u_vu)

    # Take the two joint distributions and calculate the fork distribution v vw vu
    
    cj.marginal(joint_v_w_vw, 1, joint_v_vw)
    cj.marginal(joint_v_u_vu, 1, joint_v_vu)
    
    cj.conditional(joint_v_vw, [0], v1, vw_given_v)
    cj.conditional(joint_v_vu, [0], v1, vu_given_v)

    cj.fork(v1, 0, vw_given_v, 0, vu_given_v, joint_v_vw_vu)

    # Use the joint vw vu to calculate vwvu

    cj.marginal(joint_v_vw_vu, 0, joint_vw_vu)

    cj.collider(joint_vw_vu, [0,1], c_vwvu, joint_vw_vu_vwvu)

    # Now work with the diamond to calculate joint v vwvu

    #cj.joint2D(v1, 0, vw_given_v, joint_v0_vw)
    #cj.joint2D(v1, 0, vu_given_v, joint_v0_vu) # not needed, why are we doing this?

    # Later, we need vwvu given vw and vwvu given vu
    
    cj.marginal(joint_vw_vu_vwvu, 1, marginal_vw_vwvu)
    cj.marginal(marginal_vw_vwvu, 1, marginal_vw)
    
    cj.marginal(joint_vw_vu_vwvu, 0, marginal_vu_vwvu)
    cj.marginal(marginal_vu_vwvu, 1, marginal_vu)

    mc_neurons_joint_v_w_vw = np.array([[mc_neurons_prime[a][0],mc_neurons_prime[a][1],vw([mc_neurons_prime[a][0],mc_neurons_prime[a][1]])] for a in range(num_neurons)])
    mc_neurons_joint_v_u_vu = np.array([[mc_neurons_prime[a][0],mc_neurons_prime[a][2],vu([mc_neurons_prime[a][0],mc_neurons_prime[a][2]])] for a in range(num_neurons)])

    mc_neurons_joint_vw_vu = np.array([[mc_neurons_joint_v_w_vw[a][2],mc_neurons_joint_v_u_vu[a][2]] for a in range(num_neurons)])

    dist_read = cj.readDist(marginal_vw)
    dist_read = [a / (cj.size(c_vw)[2]/100) for a in dist_read]
    fig, ax = plt.subplots(1, 1)
    ax.set_title("vw")
    ax.plot(np.linspace(cj.base(c_vw)[2], cj.base(c_vw)[2]+cj.size(c_vw)[2], 100), dist_read)
    ax.hist(mc_neurons_joint_vw_vu[:,0], density=True, bins=100, range=[cj.base(c_vw)[2],cj.base(c_vw)[2]+cj.size(c_vw)[2]], histtype='step')
    fig.tight_layout()
    
    dist_read = cj.readDist(marginal_vu)
    dist_read = [a / (cj.size(c_vu)[2]/100) for a in dist_read]
    fig, ax = plt.subplots(1, 1)
    ax.set_title("vu")
    ax.plot(np.linspace(cj.base(c_vu)[2], cj.base(c_vu)[2]+cj.size(c_vu)[2], 100), dist_read)
    ax.hist(mc_neurons_joint_vw_vu[:,1], density=True, bins=100, range=[cj.base(c_vu)[2],cj.base(c_vu)[2]+cj.size(c_vu)[2]], histtype='step')
    fig.tight_layout()
    plt.show()

    mc_neurons_joint_vw_vu_vwvu = np.array([[mc_neurons_joint_v_w_vw[a][2],mc_neurons_joint_v_u_vu[a][2],vwvu([mc_neurons_joint_v_w_vw[a][2],mc_neurons_joint_v_u_vu[a][2]])] for a in range(num_neurons)])

    cj.conditional(marginal_vw_vwvu, [0], marginal_vw, vwvu_given_vw)

    cj.conditional(marginal_vu_vwvu, [0], marginal_vu, vwvu_given_vu)
    
    cj.conditional(joint_v_vw_vu, [0], v1, vw_vu_given_v)
    cj.conditional(joint_vw_vu_vwvu, [0,1], joint_vw_vu, vwvu_given_vw_vu)
    cj.diamond(v1, vw_vu_given_v, vwvu_given_vw_vu, joint_v0_vwvu)

    cj.rescale(joint_v0_vwvu)
    
    cj.collider(joint_v0_vwvu, [0,1], c_v_prime, joint_v0_vwvu_v1)

    plotDist2D(joint_v0_vwvu)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("joint_v0_vwvu")
    ax.scatter(mc_neurons_prime[:,0], mc_neurons_joint_vw_vu_vwvu[:,2])
    ax.set_xlim([v_min,v_max])
    ax.set_ylim([cj.base(c_vwvu)[2],cj.base(c_vwvu)[2]+cj.size(c_vwvu)[2]])
    fig.tight_layout()
    plt.show()

    cj.marginal(joint_v0_vwvu_v1, 0, marginal_vwvu_v1)

    # Now calcultae v2 including the threshold reset

    vwvu_v_dist = cj.readDist(marginal_vwvu_v1)

    threshold = -50.0
    reset = -70.6
    threshold_cell = int((threshold - cj.base(v2)[0]) / (cj.size(v2)[0] / cj.res(v2)[0]))
    reset_cell = int((reset - cj.base(v2)[0]) / (cj.size(v2)[0] / cj.res(v2)[0]))
    
    n_mass = [a for a in vwvu_v_dist]
    total_reset_mass = 0.0
    for j in range(cj.res(marginal_vwvu_v1)[0]): # for each column
        for i in range(cj.res(marginal_vwvu_v1)[1]): 
            index = (i * cj.res(marginal_vwvu_v1)[0]) + j
            reset_index = (reset_cell * cj.res(marginal_vwvu_v1)[0]) + j
            if i >= threshold_cell and vwvu_v_dist[index] > 0.0:
                n_mass[reset_index] += vwvu_v_dist[index]
                n_mass[index] = 0.0
                total_reset_mass += vwvu_v_dist[index]
    
    cj.update(marginal_vwvu_v1, n_mass)

    cj.marginal(marginal_vwvu_v1, 0, v2)

    
    mc_neurons_joint_v0_vwvu_v1 = np.array([[mc_neurons_prime[a][0],mc_neurons_joint_vw_vu_vwvu[a][2],v_prime([mc_neurons_prime[a][0],mc_neurons_joint_vw_vu_vwvu[a][2]])] for a in range(num_neurons)])


    # Now to calculate the new joint v,w and joint v,u

    cj.marginal(joint_v_w_vw, 0, marginal_w0_vw)
    cj.rescale(marginal_w0_vw)
    cj.joint3D(marginal_w0_vw, 1, vwvu_given_vw, joint_w0_vw_vwvu)
    cj.rescale(joint_w0_vw_vwvu)
    
    cj.marginal(joint_w0_vw_vwvu, 1, marginal_w0_vwvu)
    cj.rescale(marginal_w0_vwvu)

    plotDist2D(marginal_w0_vwvu)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("marginal_w0_vwvu")
    print(len(mc_neurons_prime[:,1]), len(mc_neurons_joint_v0_vwvu_v1[:,2]))
    ax.scatter(mc_neurons_prime[:,1], mc_neurons_joint_v0_vwvu_v1[:,1])
    ax.set_xlim([w_min,w_max])
    ax.set_ylim([cj.base(c_vwvu)[2],cj.base(c_vwvu)[2]+cj.size(c_vwvu)[2]])
    fig.tight_layout()

    cj.rescale(marginal_vwvu_v1)
    cj.marginal(marginal_vwvu_v1, 1, marginal_vwvu)

    plotDist2D(marginal_vwvu_v1)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("marginal_vwvu_v1")
    ax.scatter(mc_neurons_joint_v0_vwvu_v1[:,1], mc_neurons_joint_v0_vwvu_v1[:,2])
    ax.set_xlim([cj.base(c_vwvu)[2],cj.base(c_vwvu)[2]+cj.size(c_vwvu)[2]])
    ax.set_ylim([v_min,v_max])
    fig.tight_layout()

    cj.rescale(marginal_vwvu)

    dist_read = cj.readDist(w1)
    dist_read = [a / ((w_max-w_min)/100) for a in dist_read]
    fig, ax = plt.subplots(1, 1)
    ax.set_title("w0_check")
    ax.plot(np.linspace(w_min, w_max, 100), dist_read)
    ax.hist(mc_neurons_prime[:,1], density=True, bins=100, range=[w_min,w_max], histtype='step')
    fig.tight_layout()

    dist_read = cj.readDist(marginal_vwvu)
    dist_read = [a / (cj.size(c_vwvu)[2]/100) for a in dist_read]
    fig, ax = plt.subplots(1, 1)
    ax.set_title("vwvu_check")
    ax.plot(np.linspace(cj.base(c_vwvu)[2], cj.base(c_vwvu)[2]+cj.size(c_vwvu)[2], 100), dist_read)
    ax.hist(mc_neurons_joint_v0_vwvu_v1[:,1], density=True, bins=100, range=[cj.base(c_vwvu)[2],cj.base(c_vwvu)[2]+cj.size(c_vwvu)[2]], histtype='step')
    fig.tight_layout()

    cj.conditional(marginal_vwvu_v1, [0], marginal_vwvu, v1_given_vwvu)

    plotDist2D(v1_given_vwvu)
    
    cj.joint3D(marginal_w0_vwvu, 1, v1_given_vwvu, joint_w0_vwvu_v1)
    cj.rescale(joint_w0_vwvu_v1)
    cj.marginal(joint_w0_vwvu_v1, 1, marginal_w0_v1)
    cj.rescale(marginal_w0_v1)

    plotDist2D(marginal_w0_v1)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("marginal_w0_v1")
    ax.scatter(mc_neurons_prime[:,1], mc_neurons_joint_v0_vwvu_v1[:,2])
    ax.set_xlim([w_min,w_max])
    ax.set_ylim([v_min,v_max])
    fig.tight_layout()
    plt.show()

    cj.conditional(marginal_w0_v1, [0], w1, v1_given_w0)

    cj.marginal(joint_v_u_vu, 0, marginal_u0_vu)
    cj.rescale(marginal_u0_vu)
    cj.joint3D(marginal_u0_vu, 1, vwvu_given_vu, joint_u0_vu_vwvu)
    cj.rescale(joint_u0_vu_vwvu)
    
    cj.marginal(joint_u0_vu_vwvu, 1, marginal_u0_vwvu)
    cj.rescale(marginal_u0_vwvu)
    
    cj.joint3D(marginal_u0_vwvu, 1, v1_given_vwvu, joint_u0_vwvu_v1)
    cj.rescale(joint_u0_vwvu_v1)
    cj.marginal(joint_u0_vwvu_v1, 1, marginal_u0_v1)
    cj.rescale(marginal_u0_v1)

    cj.conditional(marginal_u0_v1, [0], u1, v1_given_u0)

    cj.fork(w1, 0, v1_given_w0, 0, w1_given_w0, joint_w0_v1_w1)
    cj.fork(w1, 0, v1_given_w0, 0, w1_given_w0, joint_w0_v1_w1)
    cj.rescale(joint_w0_v1_w1)
    cj.fork(u1, 0, v1_given_u0, 0, u1_given_u0, joint_u0_v1_u1)
    cj.rescale(joint_u0_v1_u1)

    cj.marginal(joint_w0_v1_w1, 0, joint_v2_w2)
    cj.marginal(joint_u0_v1_u1, 0, joint_v2_u2)

    plotDist2D(joint_v2_w2)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("joint_v2_w2")
    ax.scatter(mc_neurons_prime[:,0], mc_neurons_prime[:,1])
    ax.set_xlim([v_min,v_max])
    ax.set_ylim([w_min,w_max])
    fig.tight_layout()
    plt.show()

    # update monte carlo joint

    for nn in range(len(mc_joint_neurons_prime)):
        mc_joint_neurons_prime[nn] = cond(mc_joint_neurons_prime[nn])

        if (mc_joint_neurons_prime[nn][0] > -50.0):
            mc_joint_neurons_prime[nn][0] = -70.6

        mc_joint_neurons_prime[nn][1] += epsp*poisson.rvs(w_rate*timestep) # override w
        mc_joint_neurons_prime[nn][2] += ipsp*poisson.rvs(u_rate*timestep)
        

    # update monte carlo checker

    mc_neurons_prime = np.array([[mc_neurons_joint_v0_vwvu_v1[a][2],mc_neurons_joint_w_wI_w1[a][2],mc_neurons_joint_u_uI_u1[a][2]] for a in range(num_neurons)])
    w_inputs = np.array([[poisson.rvs(w_rate*timestep)*epsp] for a in range(num_neurons)])
    u_inputs = np.array([[poisson.rvs(u_rate*timestep)*ipsp] for a in range(num_neurons)])
    
    

    

    # plot

    if (iteration % 1 == 0) :
        dist_v = cj.readDist(v2)
        dist_w = cj.readDist(w2)
        dist_u = cj.readDist(u2)

        dist_v = [a / ((v_max-v_min)/v_res) for a in dist_v]
        dist_w = [a / ((w_max-w_min)/w_res) for a in dist_w]
        dist_u = [a / ((u_max-u_min)/u_res) for a in dist_u]

        fig, ax = plt.subplots(2,2)
        ax[0,0].plot(v, dist_v)
        ax[0,1].plot(w, dist_w)
        ax[1,0].plot(u, dist_u)
        ax[0,0].hist(mc_neurons_prime[:,0], density=True, bins=v_res, range=[v_min,v_max], histtype='step')
        ax[0,1].hist(mc_neurons_prime[:,1], density=True, bins=w_res, range=[w_min,w_max], histtype='step')
        ax[1,0].hist(mc_neurons_prime[:,2], density=True, bins=u_res, range=[u_min,u_max], histtype='step')
        ax[0,0].hist(mc_joint_neurons_prime[:,0], density=True, bins=v_res, range=[v_min,v_max], histtype='step')
        ax[0,1].hist(mc_joint_neurons_prime[:,1], density=True, bins=w_res, range=[w_min,w_max], histtype='step')
        ax[1,0].hist(mc_joint_neurons_prime[:,2], density=True, bins=u_res, range=[u_min,u_max], histtype='step')
    
        fig.tight_layout()
        plt.show()

        cj.marginal(joint_v2_w2, 1, v1_check)
        cj.marginal(joint_v2_u2, 1, v1_check2)
        cj.marginal(joint_v2_w2, 0, w1_check)
        cj.marginal(joint_v2_u2, 0, u1_check)

        # Check v1 from vwvu_v1 and v1 from w1_v1 match
        dist_v1_check = cj.readDist(v1_check)
        dist_v1_check2 = cj.readDist(v1_check2)
        dist_v1_check = [a / ((v_max-v_min)/100) for a in dist_v1_check]
        dist_v1_check2 = [a / ((v_max-v_min)/100) for a in dist_v1_check2]
        fig, ax = plt.subplots(1, 1)
        ax.set_title("v check")
        ax.plot(np.linspace(v_min, v_max, 100), dist_v)
        ax.plot(np.linspace(v_min, v_max, 100), dist_v1_check, linestyle="--")
        ax.plot(np.linspace(v_min, v_max, 100), dist_v1_check2, linestyle="-.")
        ax.hist(mc_joint_neurons_prime[:,0], density=True, bins=100, range=[v_min,v_max], histtype='step')
        ax.hist(mc_neurons_prime[:,0], density=True, bins=100, range=[v_min,v_max], histtype='step')
        fig.tight_layout()
        plt.show()

        # Check w1 and w1 from w1_v1 match
        dist_w1_check = cj.readDist(w1)
        dist_w1_check2 = cj.readDist(w2)
        dist_w1_check = [a / ((w_max-w_min)/100) for a in dist_w1_check]
        dist_w1_check2 = [a / ((w_max-w_min)/100) for a in dist_w1_check2]
        fig, ax = plt.subplots(1, 1)
        ax.set_title("w check")
        ax.plot(np.linspace(w_min, w_max, 100), dist_w1_check)
        ax.plot(np.linspace(w_min, w_max, 100), dist_w1_check2, linestyle="--")
        ax.hist(mc_joint_neurons_prime[:,1], density=True, bins=100, range=[w_min,w_max], histtype='step')
        ax.hist(mc_neurons_prime[:,1], density=True, bins=100, range=[w_min,w_max], histtype='step')
        fig.tight_layout()
        plt.show()



    cj.rescale(v2)
    cj.rescale(w2)
    cj.rescale(u2)
    cj.rescale(joint_v2_w2)
    cj.rescale(joint_v2_u2)

    # Transfer v1,w1,u1 -> v0,w0,u0, transfer v2,w2,u2 -> v1,w1,u1

    cj.transfer(v2,v1)
    cj.transfer(w2,w1)
    cj.transfer(u2,u1)
    cj.transfer(joint_v2_w2, joint_v1_w1)
    cj.transfer(joint_v2_u2, joint_v1_u1)