from pycausaljazz import pycausaljazz as cj
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
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
                     s=20, alpha=0.01, color='green')

    ax.set_title("3D Heatmap")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

def plotDist2D(dist, res=(100,100)):
    dist = cj.readDist(dist)
    dist = np.array(dist)
    dist = dist.reshape(res)

    fig, ax = plt.subplots(1, 1)
    plt.imshow(dist, cmap='hot', interpolation='nearest')


def plotDist3D(dist, points, res=(100,100,100)):
    dist = cj.readDist(dist)
    dist = np.array(dist)
    dist = dist.reshape(res)

    plot3D(points, dist)

# Try loading then reading 1D distribution
print("1D")
x = np.linspace(-2.0, 2.0, 100)
what = [a * (4.0/100) for a in norm.pdf(x, 0.0, 0.1)]

id = cj.newDist([-2.0],[4.0],[100],what)
looped = cj.readDist(id)
fig, ax = plt.subplots(1, 1)

ax.set_title('')
ax.plot(x, what)
ax.plot(x, looped, linestyle='--')
fig.tight_layout()
plt.show()

# Try loading then reading 2D distribution
print("2D")
x = np.linspace(-2.0, 2.0, 100)
y = np.linspace(-2.0, 2.0, 100)

xpdf = [a * (4.0/100) for a in norm.pdf(x, 1.0, 0.1)]
ypdf = [a * (4.0/100) for a in norm.pdf(y, 1.0, 0.1)]

joint = []
for x in range(100):
    col = []
    for y in range(100):
        col = col + [xpdf[x]*ypdf[y]]
    joint = joint + [col]

x2 = np.linspace(-2.0, 2.0, 100)
y2 = np.linspace(-2.0, 2.0, 100)
xpdf = [a * (4.0/100) for a in norm.pdf(x2, -1.0, 0.1)]
ypdf = [a * (4.0/100) for a in norm.pdf(y2, -1.0, 0.1)]

for x in range(100):
    for y in range(100):
        joint[x][y] += xpdf[x]*ypdf[y]

joint = np.array(joint)
joint = joint.flatten()

id = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in joint.tolist()])
looped = cj.readDist(id)

fig, ax = plt.subplots(1, 1)

joint = joint.reshape((100,100))
plt.imshow(joint, cmap='hot', interpolation='nearest')


fig, ax = plt.subplots(1, 1)
looped = np.array(looped)
looped = looped.reshape((100,100))
plt.imshow(looped, cmap='hot', interpolation='nearest')

plt.show()

# Plot the marginals

x_marginal = cj.newDist([-2.0],[4.0],[100],[a for a in np.zeros(100)])
y_marginal = cj.newDist([-2.0],[4.0],[100],[a for a in np.zeros(100)])

cj.marginal(id, 0, x_marginal)
cj.marginal(id, 1, y_marginal)

marginal_x = cj.readDist(x_marginal)
marginal_y = cj.readDist(y_marginal)

fig, ax = plt.subplots(1, 1)

ax.set_title('')
ax.plot(np.linspace(-2.0, 2.0, 100), marginal_x)
ax.plot(np.linspace(-2.0, 2.0, 100), marginal_y)
fig.tight_layout()
plt.show()

# Calculate and plot the conditionals

x_given_y = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])
y_given_x = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])

cj.conditional(id, [0], x_marginal, y_given_x)
cj.conditional(id, [1], y_marginal, x_given_y)

dist_x_given_y = cj.readDist(x_given_y)
dist_y_given_x = cj.readDist(y_given_x)

fig, ax = plt.subplots(1, 1)
dist_x_given_y = np.array(dist_x_given_y)
dist_x_given_y = dist_x_given_y.reshape((100,100))
plt.imshow(dist_x_given_y, cmap='hot', interpolation='nearest')

fig, ax = plt.subplots(1, 1)
dist_y_given_x = np.array(dist_y_given_x)
dist_y_given_x = dist_y_given_x.reshape((100,100))
plt.imshow(dist_y_given_x, cmap='hot', interpolation='nearest')

plt.show()

# Multiply conditionals by marginals to get the original joint distribution (chain structure)

joint_from_conditional = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])

cj.chain(x_marginal, 0, y_given_x, joint_from_conditional)

dist_joint_from_conditional = cj.readDist(joint_from_conditional)

fig, ax = plt.subplots(1, 1)
dist_joint_from_conditional = np.array(dist_joint_from_conditional)
dist_joint_from_conditional = dist_joint_from_conditional.reshape((100,100))
plt.imshow(dist_joint_from_conditional, cmap='hot', interpolation='nearest')

plt.show()

joint_from_conditional2 = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])

cj.chain(y_marginal, 1, x_given_y, joint_from_conditional2)

dist_joint_from_conditional2 = cj.readDist(joint_from_conditional2)

fig, ax = plt.subplots(1, 1)
dist_joint_from_conditional2 = np.array(dist_joint_from_conditional2)
dist_joint_from_conditional2 = dist_joint_from_conditional2.reshape((100,100))
plt.imshow(dist_joint_from_conditional2, cmap='hot', interpolation='nearest')

plt.show()

# Try loading then reading 3D distribution

print("3D")
x = np.linspace(-2.0, 2.0, 100)
y = np.linspace(-2.0, 2.0, 100)
z = np.linspace(-2.0, 2.0, 100)

xpdf = [a * (4.0/100) for a in norm.pdf(x, 1.0, 0.1)]
ypdf = [a * (4.0/100) for a in norm.pdf(y, 1.0, 0.1)]
zpdf = [a * (4.0/100) for a in norm.pdf(z, 1.0, 0.1)]

points = []
joint = []
for x in range(100):
    col = []
    points_col = []
    for y in range(100):
        dep = []
        points_dep = []
        for z in range(100):
            dep = dep + [xpdf[x]*ypdf[y]*zpdf[z]]
            points_dep = points_dep + [(x,y,z)]
        col = col + [dep]
        points_col = points_col + [points_dep]
    joint = joint + [col]
    points = points + [points_col]

x2 = np.linspace(-2.0, 2.0, 100)
y2 = np.linspace(-2.0, 2.0, 100)
z2 = np.linspace(-2.0, 2.0, 100)
xpdf = [a * (4.0/100) for a in norm.pdf(x2, -1.0, 0.1)]
ypdf = [a * (4.0/100) for a in norm.pdf(y2, -1.0, 0.1)]
zpdf = [a * (4.0/100) for a in norm.pdf(z2, -1.0, 0.1)]

for x in range(100):
    for y in range(100):
        for z in range(100):
            joint[x][y][z] += xpdf[x]*ypdf[y]*zpdf[z]
            
joint = np.array(joint)
joint = joint.reshape((1000000))

id = cj.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in joint])
looped = cj.readDist(id)

joint = joint.reshape((100,100,100))
looped = np.array(looped)
looped = looped.reshape((100,100,100))


plot3D(points, joint)
plot3D(points, looped)

plt.show()

# marginals

xy_marg = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])
yz_marg = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])
xz_marg = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])

cj.marginal(id, 2, xy_marg)
cj.marginal(id, 0, yz_marg)
cj.marginal(id, 1, xz_marg)

dist_xy_marg = cj.readDist(xy_marg)
dist_yz_marg = cj.readDist(yz_marg)
dist_xz_marg = cj.readDist(xz_marg)

fig, ax = plt.subplots(1, 1)
dist_xy_marg = np.array(dist_xy_marg)
dist_xy_marg = dist_xy_marg.reshape((100,100))
plt.imshow(dist_xy_marg, cmap='hot', interpolation='nearest')

fig, ax = plt.subplots(1, 1)
dist_yz_marg = np.array(dist_yz_marg)
dist_yz_marg = dist_yz_marg.reshape((100,100))
plt.imshow(dist_yz_marg, cmap='hot', interpolation='nearest')

fig, ax = plt.subplots(1, 1)
dist_xz_marg = np.array(dist_xz_marg)
dist_xz_marg = dist_xz_marg.reshape((100,100))
plt.imshow(dist_xz_marg, cmap='hot', interpolation='nearest')

plt.show()

# Conditionals

x_given_yz = cj.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in np.zeros(1000000)])
y_given_xz = cj.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in np.zeros(1000000)])
z_given_xy = cj.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in np.zeros(1000000)])

cj.conditional(id, [1,2], yz_marg, x_given_yz)
cj.conditional(id, [0,2], xz_marg, y_given_xz)
cj.conditional(id, [0,1], xy_marg, z_given_xy)

dist_x_given_yz = cj.readDist(x_given_yz)
dist_y_given_xz = cj.readDist(y_given_xz)
dist_z_given_xy = cj.readDist(z_given_xy)

dist_x_given_yz = np.array(dist_x_given_yz)
dist_x_given_yz = dist_x_given_yz.reshape((100,100,100))

dist_y_given_xz = np.array(dist_y_given_xz)
dist_y_given_xz = dist_y_given_xz.reshape((100,100,100))

dist_z_given_xy = np.array(dist_z_given_xy)
dist_z_given_xy = dist_z_given_xy.reshape((100,100,100))

# Takes forever to plot so commenting out
#plot3D(points, dist_x_given_yz)
#plt.show()
#plot3D(points, dist_y_given_xz)
#plt.show()
#plot3D(points, dist_z_given_xy)
#plt.show()

# Multiply two conditionals and a marginal to get a 3D joint distribution (fork structure)

# First, does the marginal x from xy and xz match? This needs a better test - currently the distribution is too symmetric to tell

x_from_xy_marg = cj.newDist([-2.0],[4.0],[100],[a for a in np.zeros(100)])
x_from_xz_marg = cj.newDist([-2.0],[4.0],[100],[a for a in np.zeros(100)])

cj.marginal(xy_marg, 1, x_from_xy_marg)
cj.marginal(xz_marg, 1, x_from_xz_marg)

dist_x_from_xy_marg = cj.readDist(x_from_xy_marg)
dist_x_from_xz_marg = cj.readDist(x_from_xz_marg)

fig, ax = plt.subplots(1, 1)
ax.set_title('')
ax.plot(np.linspace(-2.0, 2.0, 100), dist_x_from_xy_marg)
ax.plot(np.linspace(-2.0, 2.0, 100), dist_x_from_xz_marg, linestyle='--')
fig.tight_layout()
plt.show()

# Generate the original

y_given_x = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])
z_given_x = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])
joint_from_fork = cj.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in np.zeros(1000000)])

cj.conditional(xy_marg, [0], x_from_xy_marg, y_given_x)
cj.conditional(xz_marg, [0], x_from_xz_marg, z_given_x)
cj.fork(x_from_xy_marg, 0, y_given_x, 0, z_given_x, joint_from_fork)

dist_joint_from_fork = cj.readDist(joint_from_fork)
dist_joint_from_fork = np.array(dist_joint_from_fork)
dist_joint_from_fork = dist_joint_from_fork.reshape((100,100,100))

plot3D(points, dist_joint_from_fork)
plt.show()

# TODO: Add tests for other conditionals where they are transposed

# Multiply a conditional by a marginal to get the original joint distribution (collider structure)

joint_from_collider = cj.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in np.zeros(1000000)])

cj.collider(xy_marg, [0,1], z_given_xy, joint_from_collider)

dist_joint_from_collider = cj.readDist(joint_from_collider)
dist_joint_from_collider = np.array(dist_joint_from_collider)
dist_joint_from_collider = dist_joint_from_collider.reshape((100,100,100))

plot3D(points, dist_joint_from_collider)
plt.show()

# Generate a 2D conditional from a function

def testFunc(y):
    A = y[0]
    return 5*A

testFunc_conditional = cj.function([-2.0],[4.0],[100],testFunc,100)
dist_testFunc = cj.readDist(testFunc_conditional)
dist_testFunc = np.array(dist_testFunc)
dist_testFunc = dist_testFunc.reshape((100,100))

fig, ax = plt.subplots(1, 1)
plt.imshow(dist_testFunc, cmap='hot', interpolation='nearest')

plt.show()

# Generate a 3D conditional from a function

def testFunc2(y):
    A = y[0]
    B = y[1]
    return A+B

testFunc2_conditional = cj.function([-2.0,-2.0],[4.0,4.0],[100,100],testFunc,100)
dist_testFunc2 = cj.readDist(testFunc2_conditional)
dist_testFunc2 = np.array(dist_testFunc2)
dist_testFunc2 = dist_testFunc2.reshape((100,100,100))

plot3D(points, dist_testFunc2)
plt.show()

# OK. Let's try a 3D conductance

# 3D cond as it should be in its entirety - but we're going to split this into separate functions
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

def w_prime(y):
    w = y[0]
    tau_e = 2.728
    dt = 1

    w_prime = -(w) / tau_e
    return w + dt*w_prime

def u_prime(y):
    u = y[0]
    tau_i = 10.49
    dt = 1

    u_prime = -(u) / tau_i
    return u + dt*u_prime

def vw(y):
    v = y[0]
    w = y[1]

    E_e = 0.0

    return -w * (v - E_e)

def vu(y):
    v = y[0]
    u = y[1]

    E_i = -75

    return  -u * (v - E_i)

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
    dt = 1

    v_prime = (-g_l*(v - E_l) + vw_vu) / C

    return v + dt*v_prime

# cond_grid = cj.generate(cond,  [-80.0,-1.0,-5.0], [40.0,26.0,55.0], [150,100,100], -50.4, -70.6, [0,0,0], dt)
# Set up the starting distribution
v = np.linspace(-80.0, -40.0, 100)
w = np.linspace(-1.0, 25.0, 100)
u = np.linspace(-5.0, 50.0, 100)

#[-70.6, 0.001, 0.001]
vpdf = [a * (40.0/100) for a in norm.pdf(v, -70.6, 0.1)]
wpdf = [a * (26.0/100) for a in norm.pdf(w, 20.001, 0.1)]
updf = [a * (55.0/100) for a in norm.pdf(u, 0.001, 0.1)]

v0 = cj.newDist([-80.0],[40.0],[100],[a for a in vpdf])
w0 = cj.newDist([-1.0],[26.0],[100],[a for a in wpdf])
u0 = cj.newDist([-5.0],[55.0],[100],[a for a in updf])

c_w_prime = cj.function([-1.0],[26.0],[100], w_prime, 100)
c_u_prime = cj.function([-5.0],[55.0],[100], u_prime, 100)
c_vw = cj.function([-80.0,-1.0],[40.0,26.0],[100,100], vw, 100)
c_vu = cj.function([-80.0,-5.0],[40.0,55.0],[100,100], vu, 100)

# plot the conditionals so far out of interest
#plotDist2D(c_w_prime)
#plotDist2D(c_u_prime)
#plotDist3D(c_vw, points)
#plotDist3D(c_vu, points)

#plt.show()

c_vwvu = cj.function([cj.base(c_vw)[2],cj.base(c_vu)[2]],[cj.size(c_vw)[2],cj.size(c_vu)[2]],[100,100], vwvu, 100)

#plotDist3D(c_vwvu, points)
#plt.show()

c_v_prime = cj.function([-80.0,cj.base(c_vwvu)[2]],[40.0,cj.size(c_vwvu)[2]],[100,100],v_prime, 100)

#plotDist3D(c_v_prime, points)
#plt.show()

# w1 and u1 are easy to calculate - they're just chains
# TODO: When applying c_w_prime, the resulting distribution w' is likely a different size to the input w
# but we want to maintain a stable size and truncate anything that goes beyond the boundaries
# so we need a function to "resize" a distribution (or force an output size like we did previously)

w0w1 = cj.newDist([-1.0,-1.0],[26.0,26.0],[100,100],[a for a in np.zeros(10000)])
u0u1 = cj.newDist([-5.0,-5.0],[55.0,55.0],[100,100],[a for a in np.zeros(10000)])

cj.chain(w0, 1, c_w_prime, w0w1)
cj.chain(u0, 1, c_u_prime, u0u1)

w1 = cj.newDist([-1.0],[26.0],[100],[a for a in np.zeros(100)])
u1 = cj.newDist([-5.0],[55.0],[100],[a for a in np.zeros(100)])

cj.marginal(w0w1, 1, w1)
cj.marginal(u0u1, 1, u1)

dist_w0 = cj.readDist(w0)
dist_w1 = cj.readDist(w1)

fig, ax = plt.subplots(1, 1)
ax.set_title('')
ax.plot(w, dist_w0)
ax.plot(w, dist_w1, linestyle="--")
fig.tight_layout()
plt.show()

v1 = cj.newDist([-80.0],[40.0],[100],[a for a in np.zeros(100)])

