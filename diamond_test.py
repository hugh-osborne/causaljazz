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

def a_to_b(y):
    a = y[0]
    return a*3.

def a_to_c(y):
    a = y[0]
    return a+5.0

def bc_to_d(y):
    b = y[0]
    c = y[1]
    return b - c

a_res = 100
b_res = 100
c_res = 100
d_res = 100

a_max = 50.0
a_min = -50.0

# Set up the starting distribution
a = np.linspace(a_min, a_max, a_res)

points = []
for x in range(c_res):
    points_col = []
    for y in range(b_res):
        points_dep = []
        for z in range(a_res):
            points_dep = points_dep + [(x,y,z)]
        points_col = points_col + [points_dep]
    points = points + [points_col]
    
apdf = [x * ((a_max-a_min)/a_res) for x in norm.pdf(a, -25, 2.4)]

apdf = [x / sum(apdf) for x in apdf]

a0 = cj.newDist([a_min],[(a_max-a_min)],[a_res],[x for x in apdf])

c_a_to_b = cj.function([a_min],[(a_max-a_min)],[a_res], a_to_b, b_res)
c_a_to_c = cj.function([a_min],[(a_max-a_min)],[a_res], a_to_c, c_res)
c_bc_to_d = cj.function([cj.base(c_a_to_b)[1],cj.base(c_a_to_c)[1]],[cj.size(c_a_to_b)[1],cj.size(c_a_to_c)[1]],[b_res,c_res], bc_to_d, d_res)

joint_a_b_c = cj.newDist([a_min,cj.base(c_a_to_b)[1],cj.base(c_a_to_c)[1]],[(a_max-a_min),cj.size(c_a_to_b)[1],cj.size(c_a_to_c)[1]],[a_res,b_res,c_res], [a for a in np.zeros(a_res*b_res*c_res)])
joint_b_c_d = cj.newDist([cj.base(c_a_to_b)[1],cj.base(c_a_to_c)[1],cj.base(c_bc_to_d)[2]],[cj.size(c_a_to_b)[1],cj.size(c_a_to_c)[1],cj.size(c_bc_to_d)[2]],[b_res,c_res,d_res], [a for a in np.zeros(b_res*c_res*d_res)])
joint_b_c = cj.newDist([cj.base(c_a_to_b)[1],cj.base(c_a_to_c)[1]],[cj.size(c_a_to_b)[1],cj.size(c_a_to_c)[1]],[b_res,c_res], [a for a in np.zeros(b_res*c_res)])

cj.fork(a0, 0, c_a_to_b, 0, c_a_to_c, joint_a_b_c)
cj.marginal(joint_a_b_c, 0, joint_b_c)
cj.collider(joint_b_c, [0,1], c_bc_to_d, joint_b_c_d)

b_c_given_a = cj.newDist([a_min,cj.base(c_a_to_b)[1],cj.base(c_a_to_c)[1]],[(a_max-a_min),cj.size(c_a_to_b)[1],cj.size(c_a_to_c)[1]],[a_res,b_res,c_res], [a for a in np.zeros(a_res*b_res*c_res)])
d_given_b_c = cj.newDist([cj.base(c_a_to_b)[1],cj.base(c_a_to_c)[1],cj.base(c_bc_to_d)[2]],[cj.size(c_a_to_b)[1],cj.size(c_a_to_c)[1],cj.size(c_bc_to_d)[2]],[b_res,c_res,d_res], [a for a in np.zeros(b_res*c_res*d_res)])

joint_a_d = cj.newDist([a_min,cj.base(c_bc_to_d)[2]],[(a_max-a_min),cj.size(c_bc_to_d)[2]],[a_res,d_res], [a for a in np.zeros(a_res*d_res)])

cj.conditional(joint_a_b_c, [0], a0, b_c_given_a)
cj.conditional(joint_b_c_d, [0,1], joint_b_c, d_given_b_c)
cj.diamond(a0, b_c_given_a, d_given_b_c, joint_a_d)

plotDist2D(joint_a_d)


# Just via vw
joint_a_b = cj.newDist([a_min,cj.base(c_a_to_b)[1]],[(a_max-a_min),cj.size(c_a_to_b)[1]],[a_res,b_res], [a for a in np.zeros(a_res*b_res)])
joint_b_d = cj.newDist([cj.base(c_a_to_b)[1],cj.base(c_bc_to_d)[2]],[cj.size(c_a_to_b)[1],cj.size(c_bc_to_d)[2]],[b_res,d_res], [a for a in np.zeros(b_res*d_res)])
b0 = cj.newDist([cj.base(c_a_to_b)[1]],[cj.size(c_a_to_b)[1]],[b_res], [a for a in np.zeros(b_res)])
d_given_b = cj.newDist([cj.base(c_a_to_b)[1],cj.base(c_bc_to_d)[2]],[cj.size(c_a_to_b)[1],cj.size(c_bc_to_d)[2]],[b_res,d_res], [a for a in np.zeros(b_res*d_res)])
joint_a_b_d = cj.newDist([a_min,cj.base(c_a_to_b)[1],cj.base(c_bc_to_d)[2]],[(a_max-a_min),cj.size(c_a_to_b)[1],cj.size(c_bc_to_d)[2]],[a_res,b_res,d_res], [a for a in np.zeros(a_res*b_res*d_res)])

cj.marginal(joint_a_b_c, 2, joint_a_b)
cj.marginal(joint_b_c_d, 1, joint_b_d)
cj.marginal(joint_a_b, 0, b0)
cj.conditional(joint_b_d, [0], b0, d_given_b)

cj.joint3D(joint_a_b, 1, d_given_b, joint_a_b_d)
cj.marginal(joint_a_b_d, 1, joint_a_d)

plotDist2D(joint_a_d)
plt.show()