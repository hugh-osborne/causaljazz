from pycausaljazz import pycausaljazz
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Try loading then reading 1D distribution
print("1D")
x = np.linspace(-2.0, 2.0, 100)
what = [a * (4.0/100) for a in norm.pdf(x, 0.0, 0.1)]

id = pycausaljazz.newDist([-2.0],[4.0],[100],what)
looped = pycausaljazz.readDist(id)
fig, ax = plt.subplots(1, 1)

ax.set_title('')
ax.plot(x, what)
ax.plot(x, looped)
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

pycausaljazz.chain(x_marginal, y_given_x, joint_from_conditional)

dist_joint_from_conditional = pycausaljazz.readDist(joint_from_conditional)

fig, ax = plt.subplots(1, 1)
dist_joint_from_conditional = np.array(dist_joint_from_conditional)
dist_joint_from_conditional = dist_joint_from_conditional.reshape((100,100))
plt.imshow(dist_joint_from_conditional, cmap='hot', interpolation='nearest')

plt.show()

joint_from_conditional2 = pycausaljazz.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])

pycausaljazz.chain(y_marginal, x_given_y, joint_from_conditional2)

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

flattened_points_x = []
flattened_points_y = []
flattened_points_z = []
flattened_joint_col = []

flattened_looped_points_x = []
flattened_looped_points_y = []
flattened_looped_points_z = []
flattened_looped_col = []
for x in range(100):
    for y in range(100):
        for z in range(100):
            if (joint[x][y][z] > 0.000001):
                flattened_points_x = flattened_points_x + [points[x][y][z][0]]
                flattened_points_y = flattened_points_y + [points[x][y][z][1]]
                flattened_points_z = flattened_points_z + [points[x][y][z][2]]
                flattened_col = joint[x][y][z]
            if (looped[x][y][z] > 0.000001):
                flattened_looped_points_x = flattened_looped_points_x + [points[x][y][z][0]]
                flattened_looped_points_y = flattened_looped_points_y + [points[x][y][z][1]]
                flattened_looped_points_z = flattened_looped_points_z + [points[x][y][z][2]]
                flattened_looped_col = looped[x][y][z]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

img = ax.scatter(flattened_points_x, flattened_points_y, flattened_points_z, marker='s',
                 s=20, alpha=0.01, color='green')

img = ax.scatter(flattened_looped_points_x, flattened_looped_points_y, flattened_looped_points_z, marker='s',
                 s=20, alpha=0.01, color='red')

ax.set_title("3D Heatmap")
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()

