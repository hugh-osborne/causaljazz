from pycausaljazz import pycausaljazz
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

# Try loading then reading 1D distribution
print("1D")
x = np.linspace(-2.0, 2.0, 100)
what = [a * (4.0/100) for a in norm.pdf(x, 0.0, 0.1)]

id = pycausaljazz.newDist([-2.0],[4.0],[100],what)
looped = pycausaljazz.readDist(id)
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

id = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in joint.tolist()])
looped = pycausaljazz.readDist(id)

fig, ax = plt.subplots(1, 1)

joint = joint.reshape((100,100))
plt.imshow(joint, cmap='hot', interpolation='nearest')


fig, ax = plt.subplots(1, 1)
looped = np.array(looped)
looped = looped.reshape((100,100))
plt.imshow(looped, cmap='hot', interpolation='nearest')

plt.show()

# Plot the marginals

x_marginal = pycausaljazz.newDist([-2.0],[4.0],[100],[a for a in np.zeros(100)])
y_marginal = pycausaljazz.newDist([-2.0],[4.0],[100],[a for a in np.zeros(100)])

pycausaljazz.marginal(id, 0, x_marginal)
pycausaljazz.marginal(id, 1, y_marginal)

marginal_x = pycausaljazz.readDist(x_marginal)
marginal_y = pycausaljazz.readDist(y_marginal)

fig, ax = plt.subplots(1, 1)

ax.set_title('')
ax.plot(np.linspace(-2.0, 2.0, 100), marginal_x)
ax.plot(np.linspace(-2.0, 2.0, 100), marginal_y)
fig.tight_layout()
plt.show()

# Calculate and plot the conditionals

x_given_y = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])
y_given_x = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])

pycausaljazz.conditional(id, [0], x_marginal, y_given_x)
pycausaljazz.conditional(id, [1], y_marginal, x_given_y)

dist_x_given_y = pycausaljazz.readDist(x_given_y)
dist_y_given_x = pycausaljazz.readDist(y_given_x)

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

joint_from_conditional = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])

pycausaljazz.chain(x_marginal, 0, y_given_x, joint_from_conditional)

dist_joint_from_conditional = pycausaljazz.readDist(joint_from_conditional)

fig, ax = plt.subplots(1, 1)
dist_joint_from_conditional = np.array(dist_joint_from_conditional)
dist_joint_from_conditional = dist_joint_from_conditional.reshape((100,100))
plt.imshow(dist_joint_from_conditional, cmap='hot', interpolation='nearest')

plt.show()

joint_from_conditional2 = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])

pycausaljazz.chain(y_marginal, 1, x_given_y, joint_from_conditional2)

dist_joint_from_conditional2 = pycausaljazz.readDist(joint_from_conditional2)

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

id = pycausaljazz.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in joint])
looped = pycausaljazz.readDist(id)

joint = joint.reshape((100,100,100))
looped = np.array(looped)
looped = looped.reshape((100,100,100))


plot3D(points, joint)
plot3D(points, looped)

plt.show()

# marginals

xy_marg = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])
yz_marg = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])
xz_marg = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])

pycausaljazz.marginal(id, 2, xy_marg)
pycausaljazz.marginal(id, 0, yz_marg)
pycausaljazz.marginal(id, 1, xz_marg)

dist_xy_marg = pycausaljazz.readDist(xy_marg)
dist_yz_marg = pycausaljazz.readDist(yz_marg)
dist_xz_marg = pycausaljazz.readDist(xz_marg)

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

x_given_yz = pycausaljazz.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in np.zeros(1000000)])
y_given_xz = pycausaljazz.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in np.zeros(1000000)])
z_given_xy = pycausaljazz.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in np.zeros(1000000)])

pycausaljazz.conditional(id, [1,2], yz_marg, x_given_yz)
pycausaljazz.conditional(id, [0,2], xz_marg, y_given_xz)
pycausaljazz.conditional(id, [0,1], xy_marg, z_given_xy)

dist_x_given_yz = pycausaljazz.readDist(x_given_yz)
dist_y_given_xz = pycausaljazz.readDist(y_given_xz)
dist_z_given_xy = pycausaljazz.readDist(z_given_xy)

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

x_from_xy_marg = pycausaljazz.newDist([-2.0],[4.0],[100],[a for a in np.zeros(100)])
x_from_xz_marg = pycausaljazz.newDist([-2.0],[4.0],[100],[a for a in np.zeros(100)])

pycausaljazz.marginal(xy_marg, 1, x_from_xy_marg)
pycausaljazz.marginal(xz_marg, 1, x_from_xz_marg)

dist_x_from_xy_marg = pycausaljazz.readDist(x_from_xy_marg)
dist_x_from_xz_marg = pycausaljazz.readDist(x_from_xz_marg)

fig, ax = plt.subplots(1, 1)
ax.set_title('')
ax.plot(np.linspace(-2.0, 2.0, 100), dist_x_from_xy_marg)
ax.plot(np.linspace(-2.0, 2.0, 100), dist_x_from_xz_marg, linestyle='--')
fig.tight_layout()
plt.show()

# Generate the original

y_given_x = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])
z_given_x = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])
joint_from_fork = pycausaljazz.newDist([-2.0,-2.0,-2.0],[4.0,4.0,4.0],[100,100,100],[a for a in np.zeros(1000000)])

pycausaljazz.conditional(xy_marg, [0], x_from_xy_marg, y_given_x)
pycausaljazz.conditional(xz_marg, [0], x_from_xz_marg, z_given_x)
pycausaljazz.fork(x_from_xy_marg, 0, y_given_x, 0, z_given_x, joint_from_fork)

dist_joint_from_fork = pycausaljazz.readDist(joint_from_fork)
dist_joint_from_fork = np.array(dist_joint_from_fork)
dist_joint_from_fork = dist_joint_from_fork.reshape((100,100,100))

plot3D(points, dist_joint_from_fork)
plt.show()


# Multiply a conditional by a marginal to get the original joint distribution (collider structure)

