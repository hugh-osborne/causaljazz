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

timestep = 1.0

#### 2D chain tests

def a_to_b(y):
    a = y[0]
    return a + 2  

a_res = 100
b_res = 100

a_max = 50.0
a_min = -50.0

# Set up the starting distribution
a = np.linspace(a_min, a_max, a_res)
    
apdf = [x * ((a_max-a_min)/a_res) for x in norm.pdf(a, -25, 2.4)]

apdf = [x / sum(apdf) for x in apdf]

a0 = cj.newDist([a_min],[(a_max-a_min)],[a_res],[x for x in apdf])

c_a_to_b = cj.function([a_min],[(a_max-a_min)],[a_res], a_to_b, b_res)
#c_a_to_b = cj.boundedFunction([a_min],[(a_max-a_min)],[a_res], a_to_b, a_min, (a_max-a_min), b_res)
joint_a_b = cj.newDist([a_min, cj.base(c_a_to_b)[1]],[(a_max-a_min),cj.size(c_a_to_b)[1]],[a_res,b_res], [a for a in np.zeros(a_res*b_res)])
joint_b = cj.newDist([cj.base(c_a_to_b)[1]],[cj.size(c_a_to_b)[1]],[b_res], [a for a in np.zeros(b_res)])

print(cj.base(c_a_to_b)[1], cj.size(c_a_to_b)[1])
fig, ax = plt.subplots(1, 1)
ax.set_title("a0")

plotDist2D(c_a_to_b)

for i in range(10):
    cj.joint2D(a0, 0, c_a_to_b, joint_a_b)
    cj.marginal(joint_a_b, 0, joint_b)
    #cj.transfer(joint_b, a0)

    dist_a0_check = cj.readDist(a0)
    dist_b_check = cj.readDist(joint_b)
    #dist_a0_check = [a / (100/a_res) for a in dist_a0_check]

    print(len(dist_b_check), len(dist_a0_check))

    ax.plot(np.linspace(cj.base(c_a_to_b)[1], cj.base(c_a_to_b)[1]+cj.size(c_a_to_b)[1], a_res), dist_b_check)
    ax.plot(np.linspace(a_min, a_max, a_res), dist_a0_check)


fig.tight_layout()
plt.show()


##### conditional difference

def a_to_b(y):
    a = y[0]
    return a + 10 

a_max = 50.0
a_min = -50.0
b_max = 50.0
b_min = -50.0

a = np.linspace(a_min, a_max, a_res)
b = np.linspace(b_min, b_max, b_res)

apdf = [x * ((a_max-a_min)/a_res) for x in norm.pdf(a, -25, 2.4)]
bpdf = [x * ((b_max-b_min)/b_res) for x in norm.pdf(b, 15.0, 2.4)]

apdf2 = [x * ((a_max-a_min)/a_res) for x in norm.pdf(a, -2 , 0.4)]
bpdf2 = [x * ((b_max-b_min)/b_res) for x in norm.pdf(b, -25.0, 0.4)]

apdf = [apdf[x] + apdf2[x] for x in range(a_res)]
bpdf = [bpdf[x] + bpdf2[x] for x in range(b_res)]

apdf = [x / sum(apdf) for x in apdf]
bpdf = [x / sum(bpdf) for x in bpdf]

a0 = cj.newDist([a_min],[(a_max-a_min)],[a_res],[x for x in apdf])
b0 = cj.newDist([b_min],[(b_max-b_min)],[b_res],[x for x in bpdf])

joint_a_b = cj.newDist([a_min,b_min],[(a_max-a_min),(b_max-b_min)],[a_res,b_res],[a for a in np.zeros(a_res*b_res)])

c_a_to_b = cj.boundedFunction([a_min],[(a_max-a_min)],[a_res], a_to_b, b_min, (b_max-b_min), b_res)

cj.joint2Di(a0,b0, joint_a_b)
#cj.joint2D(a0, 0, c_a_to_b, joint_a_b)

a_given_b = cj.newDist([a_min,b_min],[(a_max-a_min),(b_max-b_min)],[a_res,b_res],[a for a in np.zeros(a_res*b_res)])

cj.marginal(joint_a_b, 0, b0)
cj.conditional(joint_a_b, [1], b0, a_given_b)

plotDist2D(a_given_b)

marginal_conditional_a = cj.newDist([a_min],[(a_max-a_min)],[a_res],[x for x in apdf])

cj.marginal(a_given_b, 1, marginal_conditional_a)
#cj.rescale(marginal_conditional_a)

cj.marginal(joint_a_b, 1, a0)
plotDist1D(a0)
plotDist1D(b0)
plotDist1D(marginal_conditional_a)
plt.show()

##### Multiple Dependencies test

# We have P(A)P(B|A)P(C|A,B)P(D|B,C) and want to find P(A,E) without calculating the full joint probability. Possible?

a_max = 50.0
a_min = -50.0
b_max = 50.0
b_min = -50.0
c_max = 50.0
c_min = -50.0
d_max = 50.0
d_min = -50.0

a_res = 100
b_res = 100
c_res = 100
d_res = 100

a = np.linspace(a_min, a_max, a_res)
b = np.linspace(b_min, b_max, b_res)
c = np.linspace(c_min, c_max, c_res)
d = np.linspace(d_min, d_max, d_res)

apdf = [x * ((a_max-a_min)/a_res) for x in norm.pdf(a, -25, 2.4)]

apdf2 = [x * ((a_max-a_min)/a_res) for x in norm.pdf(a, -2 , 0.4)]

apdf = [apdf[x] + apdf2[x] for x in range(a_res)]

apdf = [x / sum(apdf) for x in apdf]

a0 = cj.newDist([a_min],[(a_max-a_min)],[a_res],[x for x in apdf])

def a_to_b(y):
    a = y[0]
    return a + 10 

def a_b_to_c(y):
    a = y[0]
    b = y[1]
    return a + b

def b_c_to_d(y):
    b = y[0]
    c = y[1]
    return b - c

c_a_to_b = cj.boundedFunction([a_min],[(a_max-a_min)],[a_res], a_to_b, b_min, (b_max-b_min), b_res)
c_a_b_to_c = cj.boundedFunction([a_min,b_min],[(a_max-a_min),(b_max-b_min)],[a_res,b_res], a_b_to_c, c_min, (c_max-c_min), c_res)
c_b_c_to_d = cj.boundedFunction([b_min,c_min],[(b_max-b_min),(c_max-c_min)],[b_res,c_res], b_c_to_d, d_min, (d_max-d_min), d_res)


joint_a_b = cj.newDist([a_min,b_min],[(a_max-a_min),(b_max-b_min)],[a_res,b_res],[a for a in np.zeros(a_res*b_res)])

cj.joint2D(a0, 0, c_a_to_b, joint_a_b)

joint_a_b_c = cj.newDist([a_min,b_min,c_min],[(a_max-a_min),(b_max-b_min),(c_max-c_min)],[a_res,b_res,c_res],[a for a in np.zeros(a_res*b_res*c_res)])

cj.collider(joint_a_b, [0,1], c_a_b_to_c, joint_a_b_c)

joint_b_c = cj.newDist([b_min,c_min],[(b_max-b_min),(c_max-c_min)],[b_res,c_res],[a for a in np.zeros(b_res*c_res)])

cj.marginal(joint_a_b_c, 0, joint_b_c)

joint_b_c_d = cj.newDist([b_min,c_min,d_min],[(b_max-b_min),(c_max-c_min),(d_max-d_min)],[b_res,c_res,d_res],[a for a in np.zeros(b_res*c_res*d_res)])

cj.collider(joint_b_c, [0,1], c_b_c_to_d, joint_b_c_d)

joint_a_d = cj.newDist([a_min,d_min],[(a_max-a_min),(d_max-d_min)],[a_res,d_res],[a for a in np.zeros(a_res*d_res)])
b_c_given_a = cj.newDist([a_min,b_min,c_min],[(a_max-a_min),(b_max-b_min),(c_max-c_min)],[a_res,b_res,c_res],[a for a in np.zeros(a_res*b_res*c_res)])
d_given_b_c = cj.newDist([b_min,c_min,d_min],[(b_max-b_min),(c_max-c_min),(d_max-d_min)],[b_res,c_res,d_res],[a for a in np.zeros(b_res*c_res*d_res)])

cj.conditional(joint_a_b_c, [0], a0, b_c_given_a)
cj.conditional(joint_b_c_d, [0,1], joint_b_c, d_given_b_c)
cj.diamond(a0, b_c_given_a, d_given_b_c, joint_a_d)

plotDist2D(joint_a_d)

joint_b_d = cj.newDist([b_min,d_min],[(b_max-b_min),(d_max-d_min)],[b_res,d_res],[a for a in np.zeros(b_res*d_res)])
b0 = cj.newDist([b_min],[(b_max-b_min)],[b_res],[x for x in np.zeros(b_res)])
d_given_b = cj.newDist([b_min,d_min],[(b_max-b_min),(d_max-d_min)],[b_res,d_res],[a for a in np.zeros(b_res*d_res)])

cj.marginal(joint_a_b, 0, b0)
cj.marginal(joint_b_c_d, 1, joint_b_d)
cj.conditional(joint_b_d, [0], b0, d_given_b)

joint_a_b_d = cj.newDist([a_min,b_min,d_min],[(a_max-a_min),(b_max-b_min),(d_max-d_min)],[a_res,b_res,d_res],[a for a in np.zeros(a_res*b_res*d_res)])

cj.joint3D(joint_a_b, 1, d_given_b, joint_a_b_d)

cj.marginal(joint_a_b_d, 1, joint_a_d)

plotDist2D(joint_a_d)

plt.show()

##### 3D chain tests

def x_to_y(y):
    x = y[0]
    return x + 2

def y_to_z(y):
    y = y[0]
    return y + 2

x_res = 100
y_res = 100
z_res = 100

x_max = 50.0
x_min = -50.0
y_max = 50.0
y_min = -50.0
z_max = 50.0
z_min = -50.0

# Set up the starting distribution
x = np.linspace(x_min, x_max, x_res)
y = np.linspace(y_min, y_max, y_res)
z = np.linspace(z_min, z_max, z_res)
    
xpdf = [a * ((x_max-x_min)/x_res) for a in norm.pdf(x, -25, 2.4)]

xpdf = [a / sum(xpdf) for a in xpdf]

x0 = cj.newDist([x_min],[(x_max-x_min)],[x_res],[a for a in xpdf])

c_x_to_y = cj.function([x_min],[(x_max-x_min)],[x_res], x_to_y, y_res)
c_y_to_z = cj.function([cj.base(c_x_to_y)[1]],[cj.size(c_x_to_y)[1]],[y_res], y_to_z, z_res)

joint_x_y_z = cj.newDist([x_min,cj.base(c_x_to_y)[1],cj.base(c_y_to_z)[1]],[(x_max-x_min),cj.size(c_x_to_y)[1],cj.size(c_y_to_z)[1]],[x_res,y_res,z_res], [a for a in np.zeros(x_res*y_res*z_res)])

cj.chain(x0, 0, c_x_to_y, 0, c_y_to_z, joint_x_y_z)

marginal_y_z = cj.newDist([cj.base(c_x_to_y)[1],cj.base(c_y_to_z)[1]],[cj.size(c_x_to_y)[1],cj.size(c_y_to_z)[1]],[y_res,z_res], [a for a in np.zeros(y_res*z_res)])
marginal_x_z = cj.newDist([x_min,cj.base(c_y_to_z)[1]],[(x_max-x_min),cj.size(c_y_to_z)[1]],[x_res,z_res], [a for a in np.zeros(x_res*z_res)])
marginal_y = cj.newDist([cj.base(c_x_to_y)[1]],[cj.size(c_x_to_y)[1]],[y_res], [a for a in np.zeros(y_res)])
marginal_z = cj.newDist([cj.base(c_y_to_z)[1]],[cj.size(c_y_to_z)[1]],[z_res], [a for a in np.zeros(z_res)])

cj.marginal(joint_x_y_z, 0, marginal_y_z)
cj.marginal(joint_x_y_z, 1, marginal_x_z)
cj.marginal(marginal_y_z, 1, marginal_y)
cj.marginal(marginal_y_z, 0, marginal_z)

joint_x_y_alt = cj.newDist([x_min,cj.base(c_x_to_y)[1]],[(x_max-x_min),cj.size(c_x_to_y)[1]],[x_res,y_res], [a for a in np.zeros(x_res*y_res)])
joint_x_y_z_alt = cj.newDist([x_min,cj.base(c_x_to_y)[1],cj.base(c_y_to_z)[1]],[(x_max-x_min),cj.size(c_x_to_y)[1],cj.size(c_y_to_z)[1]],[x_res,y_res,z_res], [a for a in np.zeros(x_res*y_res*z_res)])

cj.joint2D(x0, 0, c_x_to_y, joint_x_y_alt)
cj.joint3D(joint_x_y_alt, 1, c_y_to_z, joint_x_y_z_alt)

marginal_y_z_alt = cj.newDist([cj.base(c_x_to_y)[1],cj.base(c_y_to_z)[1]],[cj.size(c_x_to_y)[1],cj.size(c_y_to_z)[1]],[y_res,z_res], [a for a in np.zeros(y_res*z_res)])
marginal_x_z_alt = cj.newDist([x_min,cj.base(c_y_to_z)[1]],[(x_max-x_min),cj.size(c_y_to_z)[1]],[x_res,z_res], [a for a in np.zeros(x_res*z_res)])
marginal_y_alt = cj.newDist([cj.base(c_x_to_y)[1]],[cj.size(c_x_to_y)[1]],[y_res], [a for a in np.zeros(y_res)])
marginal_z_alt = cj.newDist([cj.base(c_y_to_z)[1]],[cj.size(c_y_to_z)[1]],[z_res], [a for a in np.zeros(z_res)])

cj.marginal(joint_x_y_z_alt, 0, marginal_y_z_alt)
cj.marginal(joint_x_y_z_alt, 1, marginal_x_z_alt)
cj.marginal(marginal_y_z_alt, 1, marginal_y_alt)
cj.marginal(marginal_y_z_alt, 0, marginal_z_alt)

dist_marginal_y = cj.readDist(marginal_y)
dist_marginal_z = cj.readDist(marginal_z)
dist_marginal_y_alt = cj.readDist(marginal_y_alt)
dist_marginal_z_alt = cj.readDist(marginal_z_alt)

dist_x = cj.readDist(x0)

fig, ax = plt.subplots(1, 1)
ax.set_title("joint")
ax.plot(np.linspace(x_min, x_max, x_res), dist_x)
ax.plot(np.linspace(cj.base(c_x_to_y)[1], cj.base(c_x_to_y)[1]+cj.size(c_x_to_y)[1], y_res), dist_marginal_y)
ax.plot(np.linspace(cj.base(c_x_to_y)[1], cj.base(c_x_to_y)[1]+cj.size(c_x_to_y)[1], y_res), dist_marginal_y_alt)
ax.plot(np.linspace(cj.base(c_y_to_z)[1], cj.base(c_y_to_z)[1]+cj.size(c_y_to_z)[1], z_res), dist_marginal_z)
ax.plot(np.linspace(cj.base(c_y_to_z)[1], cj.base(c_y_to_z)[1]+cj.size(c_y_to_z)[1], z_res), dist_marginal_z_alt)
fig.tight_layout()
plt.show()

num_neurons = 100
mc_xs = np.array([[norm.rvs(-25, 2.4, 1)[0]] for a in range(num_neurons)])
mc_xs_ys = np.array([[mc_xs[a][0], x_to_y([mc_xs[a][0]])] for a in range(num_neurons)])
mc_ys_zs = np.array([[mc_xs_ys[a][1], y_to_z([mc_xs_ys[a][1]])] for a in range(num_neurons)])
mc_xs_zs = np.array([[mc_xs[a][0], mc_ys_zs[a][0]] for a in range(num_neurons)])

plotDist2D(marginal_y_z_alt)

fig, ax = plt.subplots(1, 1)
ax.set_title("joint_ys_zs")
ax.scatter(mc_ys_zs[:,0], mc_ys_zs[:,1])
ax.set_xlim([y_min,y_max])
ax.set_ylim([z_min,z_max])
fig.tight_layout()
plt.show()

plotDist2D(marginal_x_z_alt)

fig, ax = plt.subplots(1, 1)
ax.set_title("joint_xs_zs")
ax.scatter(mc_xs_zs[:,0], mc_xs_zs[:,1])
ax.set_xlim([x_min,x_max])
ax.set_ylim([z_min,z_max])
fig.tight_layout()
plt.show()

#### Collider tests

