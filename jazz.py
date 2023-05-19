from pycausaljazz import pycausaljazz as cj
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

test_1D = False
test_2D = False
test_3D = False
test_Function_to_Conditional = False
test_deomposition = True


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

if test_1D:
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

if test_2D:
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

    cj.joint2D(x_marginal, 0, y_given_x, joint_from_conditional)

    dist_joint_from_conditional = cj.readDist(joint_from_conditional)

    fig, ax = plt.subplots(1, 1)
    dist_joint_from_conditional = np.array(dist_joint_from_conditional)
    dist_joint_from_conditional = dist_joint_from_conditional.reshape((100,100))
    plt.imshow(dist_joint_from_conditional, cmap='hot', interpolation='nearest')

    plt.show()

    joint_from_conditional2 = cj.newDist([-2.0,-2.0],[4.0,4.0],[100,100],[a for a in np.zeros(10000)])

    cj.joint2D(y_marginal, 1, x_given_y, joint_from_conditional2)

    dist_joint_from_conditional2 = cj.readDist(joint_from_conditional2)

    fig, ax = plt.subplots(1, 1)
    dist_joint_from_conditional2 = np.array(dist_joint_from_conditional2)
    dist_joint_from_conditional2 = dist_joint_from_conditional2.reshape((100,100))
    plt.imshow(dist_joint_from_conditional2, cmap='hot', interpolation='nearest')

    plt.show()

if test_3D:
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

if test_Function_to_Conditional:
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

    return [v + 0.1*v_prime, w + 0.1*w_prime, u + 0.1*u_prime]

def w_prime(y):
    w = y[0]
    tau_e = 2.728
    dt = 0.1

    w_prime = -(w) / tau_e
    return w + dt*w_prime

def u_prime(y):
    u = y[0]
    tau_i = 10.49
    dt = 0.1

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
    dt = 0.1

    v_prime = (-g_l*(v - E_l) + vw_vu) / C

    return v + dt*v_prime

res = 200
v_res = 200
w_res = 200
u_res = 200

# cond_grid = cj.generate(cond,  [-80.0,-1.0,-5.0], [40.0,26.0,55.0], [150,100,100], -50.4, -70.6, [0,0,0], dt)
# Set up the starting distribution
v = np.linspace(-80.0, -40.0, v_res)
w = np.linspace(-1.0, 25.0, w_res)
u = np.linspace(-5.0, 50.0, u_res)

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
vpdf = [a * (40.0/v_res) for a in norm.pdf(v, -70.6, 0.4)]
wpdf = [a * (26.0/w_res) for a in norm.pdf(w, 13.001, 0.4)]
updf = [a * (55.0/u_res) for a in norm.pdf(u, 0.001, 0.4)]

vpdf = [a / sum(vpdf) for a in vpdf]
wpdf = [a / sum(wpdf) for a in wpdf]
updf = [a / sum(updf) for a in updf]

v0 = cj.newDist([-80.0],[40.0],[v_res],[a for a in vpdf])
w0 = cj.newDist([-1.0],[26.0],[w_res],[a for a in wpdf])
u0 = cj.newDist([-5.0],[55.0],[u_res],[a for a in updf])

# Initialise the monte carlo neurons
mc_neurons = np.array([[norm.rvs(-70.6, 0.4, 1)[0],norm.rvs(13.0, 0.4, 1)[0],norm.rvs(0.0, 0.4, 1)[0]] for a in range(5000)])

c_w_prime = cj.boundedFunction([-1.0],[26.0],[w_res], w_prime, -1.0,  6.0, w_res)
c_u_prime = cj.boundedFunction([-5.0],[55.0],[u_res], u_prime, -5.0, 55.0, u_res)
c_vw = cj.function([-80.0,-1.0],[40.0,26.0],[v_res,w_res], vw, res)
c_vu = cj.function([-80.0,-5.0],[40.0,55.0],[v_res,u_res], vu, res)

c_vwvu = cj.function([cj.base(c_vw)[2],cj.base(c_vu)[2]],[cj.size(c_vw)[2],cj.size(c_vu)[2]],[res,res], vwvu, res)

#plotDist3D(c_vwvu, points)
#plt.show()

c_v_prime = cj.boundedFunction([-80.0,cj.base(c_vwvu)[2]],[40.0,cj.size(c_vwvu)[2]],[v_res,res],v_prime, -80.0, 40.0, v_res)

# w1 and u1 are easy to calculate - they're just chains
print("w1 and u1 are easy to calculate - they're just chains.")

w0w1 = cj.newDist([-1.0,-1.0],[26.0,26.0],[w_res,w_res],[a for a in np.zeros(w_res*w_res)])
u0u1 = cj.newDist([-5.0,-5.0],[55.0,55.0],[u_res,u_res],[a for a in np.zeros(u_res*u_res)])

cj.joint2D(w0, 0, c_w_prime, w0w1)
cj.joint2D(u0, 0, c_u_prime, u0u1)

w1 = cj.newDist([-1.0],[26.0],[w_res],[a for a in np.zeros(w_res)])
u1 = cj.newDist([-5.0],[55.0],[u_res],[a for a in np.zeros(u_res)])

cj.marginal(w0w1, 0, w1)
cj.marginal(u0u1, 0, u1)

#dist_w0 = cj.readDist(w0)
#dist_w1 = cj.readDist(w1)

#fig, ax = plt.subplots(1, 1)
#ax.set_title('')
#ax.plot(w, dist_w0)
#ax.plot(w, dist_w1, linestyle="--")
#fig.tight_layout()
#plt.show()

# Next calculate the distributions from the conditionals wv and uv. v0 and w0 -> vw0 is easy as v0 and w0 are independent. Not so much for v1 and w1 -> vw1.
print("Next calculate the distributions from the conditionals wv and uv. v0 and w0 -> vw0 is easy as v0 and w0 are independent. Not so much for v1 and w1 -> vw1.")

joint_v_w = cj.newDistFrom2(v0, w0)
joint_v_u = cj.newDistFrom2(v0, u0)

joint_v_w_vw = cj.newDist([-80.0,-1.0,cj.base(c_vw)[2]],[40.0,26.0,cj.size(c_vw)[2]],[v_res,w_res,res], [a for a in np.zeros(v_res*w_res*res)])
joint_v_u_vu = cj.newDist([-80.0,-5.0,cj.base(c_vu)[2]],[40.0,55.0,cj.size(c_vu)[2]],[v_res,u_res,res], [a for a in np.zeros(v_res*u_res*res)])

cj.collider(joint_v_w, [0,1], c_vw, joint_v_w_vw)
cj.collider(joint_v_u, [0,1], c_vu, joint_v_u_vu)

#plotDist3D(joint_v_w_vw)
#plt.show()

# To calculate vwvu, we must find the joint probability vw and vu which are dependent (via v)
print("To calculate vwvu, we must find the joint probability vw and vu which are dependent (via v)")

# First find, the v_vw and v_vu marginals of the two joint probabilities we just calculated
print("First find, the v_vw and v_vu marginals of the two joint probabilities we just calculated")
joint_v_vw = cj.newDist([-80.0, cj.base(c_vw)[2]], [40.0, cj.size(c_vw)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])
joint_v_vu = cj.newDist([-80.0, cj.base(c_vu)[2]], [40.0, cj.size(c_vu)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])

cj.marginal(joint_v_w_vw, 1, joint_v_vw)
cj.marginal(joint_v_u_vu, 1, joint_v_vu)

#plotDist2D(joint_v_vw)
#plotDist3D(joint_v_w_vw)
#plt.show()
#plotDist2D(joint_v_vu)
#plotDist3D(joint_v_u_vu)
#plt.show()

#print("joint_v_vw", cj.base(joint_v_vw)[0], cj.size(joint_v_vw)[0], cj.base(joint_v_vw)[1], cj.size(joint_v_vw)[1])
#print("joint_v_vu", cj.base(joint_v_vu)[0], cj.size(joint_v_vu)[0], cj.base(joint_v_vu)[1], cj.size(joint_v_vu)[1])


# Now calculate the conditionals wv|v and uv|v
print("Now calculate the conditionals wv|v and uv|v")
vw_given_v = cj.newDist([-80.0, cj.base(c_vw)[2]], [40.0, cj.size(c_vw)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])
vu_given_v = cj.newDist([-80.0, cj.base(c_vu)[2]], [40.0, cj.size(c_vu)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])

cj.conditional(joint_v_vw, [0], v0, vw_given_v)
cj.conditional(joint_v_vu, [0], v0, vu_given_v)

# Now calculate the joint probability wv uv
print("Now calculate the joint probability wv uv")
joint_v_vw_vu = cj.newDist([-80.0,cj.base(c_vw)[2],cj.base(c_vu)[2]],[40.0,cj.size(c_vw)[2],cj.size(c_vu)[2]],[v_res,res,res], [a for a in np.zeros(v_res*res*res)])

cj.fork(v0, 0, vw_given_v, 0, vu_given_v, joint_v_vw_vu)

joint_vw_vu = cj.newDist([cj.base(c_vw)[2],cj.base(c_vu)[2]],[cj.size(c_vw)[2],cj.size(c_vu)[2]],[res,res], [a for a in np.zeros(res*res)])

cj.marginal(joint_v_vw_vu, 0, joint_vw_vu)

#plotDist2D(joint_vw_vu)
#plotDist3D(joint_v_vw_vu)
#plt.show()

#print("joint_vw_vu", cj.base(joint_vw_vu)[0], cj.size(joint_vw_vu)[0], cj.base(joint_vw_vu)[1], cj.size(joint_vw_vu)[1])

# Now vwvu can be calculated from the joint distribution and function conditional
print("Now vwvu can be calculated from the joint distribution and function conditional")
joint_vw_vu_vwvu = cj.newDist([cj.base(c_vw)[2],cj.base(c_vu)[2],cj.base(c_vwvu)[2]],[cj.size(c_vw)[2],cj.size(c_vu)[2],cj.size(c_vwvu)[2]],[res,res,res], [a for a in np.zeros(res*res*res)])

cj.collider(joint_vw_vu, [0,1], c_vwvu, joint_vw_vu_vwvu)

#plotDist3D(joint_vw_vu_vwvu)
#plt.show()

# Finally, we need to calculate v'. However, as before, vwvu and v0 are not independent. So we start the long road to calculating the joint probability
# We have a fork followed by a collider creating a diamond shape

#            v0                 
#          /    \             
#         /      \            
#       vw        vu                  
#        \       /              
#         \     /               
#           vwvu                
#
# Question: If we use the chain v0 -> vw -> vwvu to calculate the joint distribution (v0,vwvu), will the result be the same if we were to instead
# use the chain v0 -> vu -> vwvu. We want to do this to avoid having to calculate the full 4D joint distribution (v0, vw, vu, vwvu).

# Later, we're actually going to need both sides of this diamond, we'll use vw to calculate v1 here

joint_v0_vw = cj.newDist([-80.0,cj.base(c_vw)[2]],[40.0,cj.size(c_vw)[2]],[v_res,res], [a for a in np.zeros(v_res*res)])

cj.joint2D(v0, 0, vw_given_v, joint_v0_vw)

joint_v0_vu = cj.newDist([-80.0,cj.base(c_vu)[2]],[40.0,cj.size(c_vu)[2]],[v_res,res], [a for a in np.zeros(v_res*res)])

cj.joint2D(v0, 0, vu_given_v, joint_v0_vu)

#plotDist2D(joint_v0_vw)
#plotDist2D(joint_v0_vu)
#plt.show()

#plotDist2D(joint_v0_vw)
#plt.show()

# Now we have P(v0,vw). We need to find P(vwvu|vw) to complete the chain.
print("Now we have P(v0,vw). We need to find P(vwvu|vw) to complete the chain.")
marginal_vw_vwvu = cj.newDist([cj.base(c_vw)[2],cj.base(c_vwvu)[2]],[cj.size(c_vw)[2],cj.size(c_vwvu)[2]],[res,res], [a for a in np.zeros(res*res)])
marginal_vw = cj.newDist([cj.base(c_vw)[2]],[cj.size(c_vw)[2]],[res], [a for a in np.zeros(res)])

cj.marginal(joint_vw_vu_vwvu, 1, marginal_vw_vwvu)
cj.marginal(marginal_vw_vwvu, 1, marginal_vw)

#plotDist3D(joint_vw_vu_vwvu)
#plotDist2D(marginal_vw_vwvu)
#plotDist1D(marginal_vw)
#plt.show()

marginal_vu_vwvu = cj.newDist([cj.base(c_vu)[2],cj.base(c_vwvu)[2]],[cj.size(c_vu)[2],cj.size(c_vwvu)[2]],[res,res], [a for a in np.zeros(res*res)])
marginal_vu = cj.newDist([cj.base(c_vu)[2]],[cj.size(c_vu)[2]],[res], [a for a in np.zeros(res)])

cj.marginal(joint_vw_vu_vwvu, 0, marginal_vu_vwvu)
cj.marginal(marginal_vu_vwvu, 1, marginal_vu)

vwvu_given_vw = cj.newDist([cj.base(c_vw)[2],cj.base(c_vwvu)[2]],[cj.size(c_vw)[2],cj.size(c_vwvu)[2]],[res,res], [a for a in np.zeros(res*res)])

cj.conditional(marginal_vw_vwvu, [0], marginal_vw, vwvu_given_vw)

vwvu_given_vu = cj.newDist([cj.base(c_vu)[2],cj.base(c_vwvu)[2]],[cj.size(c_vu)[2],cj.size(c_vwvu)[2]],[res,res], [a for a in np.zeros(res*res)])

cj.conditional(marginal_vu_vwvu, [0], marginal_vu, vwvu_given_vu)

joint_v0_vw_vwvu = cj.newDist([-80.0,cj.base(c_vw)[2],cj.base(c_vwvu)[2]],[40.0,cj.size(c_vw)[2],cj.size(c_vwvu)[2]],[v_res,res,res], [a for a in np.zeros(v_res*res*res)])

cj.joint3D(joint_v0_vw, 1, vwvu_given_vw, joint_v0_vw_vwvu)

#plotDist3D(joint_v0_vw_vwvu)
#plt.show()

# We have the joint, now find the marginal v0,vwvu which we can use to find v1
print("We have the joint, now find the marginal v0,vwvu which we can use to find v1")
marginal_v0_vwvu = cj.newDist([-80.0,cj.base(c_vwvu)[2]],[40.0,cj.size(c_vwvu)[2]],[v_res,res], [a for a in np.zeros(v_res*res)])

cj.marginal(joint_v0_vw_vwvu, 1, marginal_v0_vwvu)

#plotDist2D(marginal_v0_vwvu)
#plt.show()

print("marginal_v0_vwvu", cj.size(marginal_v0_vwvu)[0], cj.size(marginal_v0_vwvu)[1])

# Now find v1 using the function conditional
print("Now find v1 using the function conditional")
joint_v0_vwvu_v1 = cj.newDist([-80.0,cj.base(c_vwvu)[2],-80.0],[40.0,cj.size(c_vwvu)[2],40.0],[v_res,res,v_res], [a for a in np.zeros(v_res*res*v_res)])

cj.collider(marginal_v0_vwvu, [0,1], c_v_prime, joint_v0_vwvu_v1)


#plotDist3D(joint_v0_vwvu_v1)
#plt.show()

# Grab v1 from the joint distirbution
print("Grab v1 from the joint distirbution")

marginal_vwvu_v1 = cj.newDist([cj.base(c_vwvu)[2],-80.0],[cj.size(c_vwvu)[2],40.0],[res,v_res], [a for a in np.zeros(res*v_res)])
v1 = cj.newDist([-80.0],[40.0],[v_res],[a for a in np.zeros(v_res)])

cj.marginal(joint_v0_vwvu_v1, 0, marginal_vwvu_v1)
cj.marginal(marginal_vwvu_v1, 0, v1)

dist_v0 = cj.readDist(v0)
dist_v1 = cj.readDist(v1)

fig, ax = plt.subplots(1, 1)
ax.set_title('v0 -> v1')
ax.plot(v, dist_v0)
ax.plot(v, dist_v1, linestyle="--")
fig.tight_layout()
plt.show()

# OK, so. We've done one iteration. However, where previously we were able to assume independence between the three variables v0,w0 and u0, 
# we cannot say the same of v1,w1, and u1. Hopefully, we can just adjust for that in the second iteration then all subsequent iterations
# will be the same.

# First, we need to calculate the conditionals v1|w0 and v1|u0

v1_given_w0 = cj.newDist([-1.0, -80.0], [26.0, 40.0], [w_res,v_res], [a for a in np.zeros(w_res*v_res)])
v1_given_u0 = cj.newDist([-5.0, -80.0], [55.0, 40.0], [u_res,v_res], [a for a in np.zeros(u_res*v_res)])

# For these distributions, we have w0 -> vw -> vwvu -> v1 and u0 -> vu -> vwvu -> v1
# For each, find joint dist w0,vw,vwvu using the chain structure, get the marginal w0,vwvu then chain again to calculate w0,vwvu,v1, then marginal to get w0,v1

joint_w0_vw_vwvu = cj.newDist([-1.0,cj.base(c_vw)[2],cj.base(c_vwvu)[2]],[26.0,cj.size(c_vw)[2],cj.size(c_vwvu)[2]],[w_res,res,res], [a for a in np.zeros(w_res*res*res)])
marginal_w0_vw = cj.newDist([-1.0,cj.base(c_vw)[2]],[26.0,cj.size(c_vw)[2]],[w_res,res], [a for a in np.zeros(w_res*res)])

# We've already calculated w0,v0,vw and also vwvu|vw so we can find the marginal then build what we need
cj.marginal(joint_v_w_vw, 0, marginal_w0_vw)
cj.joint3D(marginal_w0_vw, 1, vwvu_given_vw, joint_w0_vw_vwvu)

marginal_w0_vwvu = cj.newDist([-1.0,cj.base(c_vwvu)[2]],[26.0,cj.size(c_vwvu)[2]],[w_res,res], [a for a in np.zeros(w_res*res)])

cj.marginal(joint_w0_vw_vwvu, 1, marginal_w0_vwvu)

# Now get v1|vwvu. We already have marginal_vwvu_v1

marginal_vwvu = cj.newDist([cj.base(c_vwvu)[2]],[cj.size(c_vwvu)[2]],[res], [a for a in np.zeros(res)])

cj.marginal(marginal_vwvu_v1, 1, marginal_vwvu)

v1_given_vwvu = cj.newDist([-80.0,cj.base(c_vwvu)[2]],[40.0,cj.size(c_vwvu)[2]],[v_res,res], [a for a in np.zeros(v_res*res)])

cj.conditional(marginal_vwvu_v1, [0], marginal_vwvu, v1_given_vwvu)

# Finally, calculate the joint w0,vwvu,v1 then find the marginal

joint_w0_vwvu_v1 = cj.newDist([-1.0,cj.base(c_vwvu)[2],-80.0],[26.0,cj.size(c_vwvu)[2],40.0],[w_res,res,v_res], [a for a in np.zeros(w_res*res*v_res)])
marginal_w0_v1 = cj.newDist([-1.0, -80.0], [26.0, 40.0], [w_res,v_res], [a for a in np.zeros(w_res*v_res)])

cj.joint3D(marginal_w0_vwvu, 1, v1_given_vwvu, joint_w0_vwvu_v1)
cj.marginal(joint_w0_vwvu_v1, 1, marginal_w0_v1)
cj.conditional(marginal_w0_v1, [0], w0, v1_given_w0)

# Do it all again for u0

joint_u0_vu_vwvu = cj.newDist([-5.0,cj.base(c_vu)[2],cj.base(c_vwvu)[2]],[55.0,cj.size(c_vu)[2],cj.size(c_vwvu)[2]],[u_res,res,res], [a for a in np.zeros(u_res*res*res)])
marginal_u0_vu = cj.newDist([-5.0,cj.base(c_vu)[2]],[55.0,cj.size(c_vu)[2]],[u_res,res], [a for a in np.zeros(u_res*res)])

# We've already calculated u0,v0,vu and also vwvu|vu so we can find the marginal then build what we need
cj.marginal(joint_v_u_vu, 0, marginal_u0_vu)
cj.joint3D(marginal_u0_vu, 1, vwvu_given_vu, joint_u0_vu_vwvu)

marginal_u0_vwvu = cj.newDist([-5.0,cj.base(c_vwvu)[2]],[55.0,cj.size(c_vwvu)[2]],[u_res,res], [a for a in np.zeros(u_res*res)])

cj.marginal(joint_u0_vu_vwvu, 1, marginal_u0_vwvu)

# Finally, calculate the joint w0,vwvu,v1 then find the marginal

joint_u0_vwvu_v1 = cj.newDist([-5.0,cj.base(c_vwvu)[2],-80.0],[55.0,cj.size(c_vwvu)[2],40.0],[u_res,res,v_res], [a for a in np.zeros(u_res*res*v_res)])
marginal_u0_v1 = cj.newDist([-5.0, -80.0], [55.0, 40.0], [u_res,v_res], [a for a in np.zeros(u_res*v_res)])

cj.joint3D(marginal_u0_vwvu, 1, v1_given_vwvu, joint_u0_vwvu_v1)
cj.marginal(joint_u0_vwvu_v1, 1, marginal_u0_v1)

cj.conditional(marginal_u0_v1, [0], u0, v1_given_u0)

# We can now correctly define the joint distributions v',m' and v',u' and this should be the same each iteration

joint_w0_v1_w1 = cj.newDist([-1.0,-80.0,-1.0],[26.0,40.0,26.0],[w_res,v_res,w_res], [a for a in np.zeros(w_res*v_res*w_res)])
joint_u0_v1_u1 = cj.newDist([-5.0,-80.0,-5.0],[55.0,40.0,55.0],[u_res,v_res,u_res], [a for a in np.zeros(u_res*v_res*u_res)])
joint_v1_w1 = cj.newDist([-80.0,-1.0],[40.0,26.0],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])
joint_v1_u1 = cj.newDist([-80.0,-5.0],[40.0,55.0],[v_res,u_res], [a for a in np.zeros(v_res*u_res)])

cj.fork(w0, 0, v1_given_w0, 0, c_w_prime, joint_w0_v1_w1)
cj.fork(u0, 0, v1_given_u0, 0, c_u_prime, joint_u0_v1_u1)

cj.marginal(joint_w0_v1_w1, 0, joint_v1_w1)
cj.marginal(joint_u0_v1_u1, 0, joint_v1_u1)

# Record v0,w0,u0 and v1,w1,u1 here if we wish.

cj.rescale(v1)
cj.rescale(w1)
cj.rescale(u1)

# For each iteration, let's calculate a new set of variables v2,w2,u2 then shift everything backward

v2 = cj.newDist([-80.0],[40.0],[v_res],[a for a in np.zeros(v_res)])
w2 = cj.newDist([-1.0],[26.0],[w_res],[a for a in np.zeros(w_res)])
u2 = cj.newDist([-5.0],[55.0],[u_res],[a for a in np.zeros(u_res)])

joint_v2_w2 = cj.newDist([-80.0,-1.0],[40.0,26.0],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])
joint_v2_u2 = cj.newDist([-80.0,-5.0],[40.0,55.0],[v_res,u_res], [a for a in np.zeros(v_res*u_res)])

# Lets try 10 iterations
for i in range(100):

    # Calculate v'

    # vw and vu
    cj.collider(joint_v1_w1, [0,1], c_vw, joint_v_w_vw)
    cj.collider(joint_v1_u1, [0,1], c_vu, joint_v_u_vu)

    cj.marginal(joint_v_w_vw, 1, joint_v_vw)
    cj.marginal(joint_v_u_vu, 1, joint_v_vu)

    cj.conditional(joint_v_vw, [0], v1, vw_given_v)
    cj.conditional(joint_v_vu, [0], v1, vu_given_v)

    cj.fork(v1, 0, vw_given_v, 0, vu_given_v, joint_v_vw_vu)
    cj.marginal(joint_v_vw_vu, 0, joint_vw_vu)

    # vwvu
    cj.collider(joint_vw_vu, [0,1], c_vwvu, joint_vw_vu_vwvu)

    # vwvu|v1

    cj.marginal(joint_vw_vu_vwvu, 1, marginal_vw_vwvu)
    cj.marginal(marginal_vw_vwvu, 1, marginal_vw)

    cj.marginal(joint_vw_vu_vwvu, 0, marginal_vu_vwvu)
    cj.marginal(marginal_vu_vwvu, 1, marginal_vu)

    cj.conditional(marginal_vw_vwvu, [0], marginal_vw, vwvu_given_vw)
    cj.conditional(marginal_vu_vwvu, [0], marginal_vu, vwvu_given_vu)
    cj.joint3D(joint_v_vw, 1, vwvu_given_vw, joint_v0_vw_vwvu)

    cj.marginal(joint_v0_vw_vwvu, 1, marginal_v0_vwvu)
    cj.collider(marginal_v0_vwvu, [0,1], c_v_prime, joint_v0_vwvu_v1)

    cj.marginal(joint_v0_vwvu_v1, 0, marginal_vwvu_v1)

    # We want to perform the threshold-reset functionality. However, just doing this to v2 is going
    # to muck things up because we use marginal_vwvu_v1 later on which would have the "unedited"
    # version of v2 in it. 
    # So, we update marginal_vwvu_v1 instead.

    vwvu_v_dist = cj.readDist(marginal_vwvu_v1)

    threshold = -50.0
    reset = -70.6
    threshold_cell = int((threshold - cj.base(v2)[0]) / (cj.size(v2)[0] / cj.res(v2)[0]))
    reset_cell = int((reset - cj.base(v2)[0]) / (cj.size(v2)[0] / cj.res(v2)[0]))
    
    n_mass = [a for a in vwvu_v_dist]
    
    for j in range(cj.res(marginal_vwvu_v1)[0]): # for each column
        reset_mass = 0.0
        for i in range(cj.res(marginal_vwvu_v1)[1]): 
            index = (i * cj.res(marginal_vwvu_v1)[0]) + j
            reset_index = (reset_cell * cj.res(marginal_vwvu_v1)[0]) + j
            if i >= threshold_cell and vwvu_v_dist[index] > 0.0:
                reset_mass += vwvu_v_dist[index]
                n_mass[reset_index] += vwvu_v_dist[index]
                n_mass[index] = 0.0
    
    cj.update(marginal_vwvu_v1, n_mass)

    cj.rescale(marginal_vwvu_v1)

    cj.marginal(marginal_vwvu_v1, 0, v2) # Now both v2 and vwvu_v1 capture the correct distribution

    # Any deviation from the mass summing to 1.0 and we get some nasty blowup or collapse
    #Rescale v2 to sum to 1.0

    

    # Update joint distributions w'v' and u'v'
    cj.marginal(joint_v_w_vw, 0, marginal_w0_vw)
    cj.joint3D(marginal_w0_vw, 1, vwvu_given_vw, joint_w0_vw_vwvu)
    
    cj.marginal(joint_w0_vw_vwvu, 1, marginal_w0_vwvu)
    
    cj.marginal(marginal_vwvu_v1, 1, marginal_vwvu)
    
    cj.conditional(marginal_vwvu_v1, [0], marginal_vwvu, v1_given_vwvu)
    
    cj.joint3D(marginal_w0_vwvu, 1, v1_given_vwvu, joint_w0_vwvu_v1)
    cj.marginal(joint_w0_vwvu_v1, 1, marginal_w0_v1)
    cj.conditional(marginal_w0_v1, [0], w1, v1_given_w0)
    
    cj.marginal(joint_v_u_vu, 0, marginal_u0_vu)
    cj.joint3D(marginal_u0_vu, 1, vwvu_given_vu, joint_u0_vu_vwvu)
    
    cj.marginal(joint_u0_vu_vwvu, 1, marginal_u0_vwvu)
    
    cj.joint3D(marginal_u0_vwvu, 1, v1_given_vwvu, joint_u0_vwvu_v1)
    cj.marginal(joint_u0_vwvu_v1, 1, marginal_u0_v1)
    cj.conditional(marginal_u0_v1, [0], u1, v1_given_u0)
    
    cj.fork(w1, 0, v1_given_w0, 0, c_w_prime, joint_w0_v1_w1)
    cj.fork(u1, 0, v1_given_u0, 0, c_u_prime, joint_u0_v1_u1)

    cj.marginal(joint_w0_v1_w1, 0, joint_v2_w2)
    cj.marginal(joint_u0_v1_u1, 0, joint_v2_u2)

    # Calculate w2 and u2 from the marginals of joint_v2_w2 and joint_v2_u2

    cj.marginal(joint_v2_w2, 0, w2)
    cj.marginal(joint_v2_u2, 0, u2)

    # We want to intervene and keep w2 at a high level. By doing so,
    # w2 and v2 are now independent, so joint_v2_w2 must be updated as well

    # Set new w2
    cj.update(w2, wpdf)

    # Set new joint v2 w2
    cj.joint2Di(v2, w2, joint_v2_w2)

    # Record distributions if you wish here.
    # Also run the monte carlo simulation 

    for nn in range(len(mc_neurons)):
        mc_neurons[nn] = cond(mc_neurons[nn])
        if (mc_neurons[nn][0] > -50.0):
            mc_neurons[nn][0] = -70.6

        mc_neurons[nn][1] = norm.rvs(13.0, 0.4, 1)[0] # override w

    dist_v = cj.readDist(v0)
    dist_w = cj.readDist(w0)
    dist_u = cj.readDist(u0)

    # The monte carlo hist function gives density not mass (booo)
    # so let's just convert to density here

    dist_v = [a / (40.0/v_res) for a in dist_v]
    dist_w = [a / (26.0/w_res) for a in dist_w]
    dist_u = [a / (55.0/u_res) for a in dist_u]

    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(v, dist_v)
    ax[0,0].hist(mc_neurons[:,0], density=True, bins=v_res, range=[-80.0,-40.0], histtype='step')
    ax[0,1].plot(w, dist_w)
    ax[0,1].hist(mc_neurons[:,1], density=True, bins=w_res, range=[-1.0,25.0], histtype='step')
    ax[1,0].plot(u, dist_u)
    ax[1,0].hist(mc_neurons[:,2], density=True, bins=u_res, range=[-5.0,50.0], histtype='step')
    fig.tight_layout()
    plt.show()

    # Transfer v1,w1,u1 -> v0,w0,u0, transfer v2,w2,u2 -> v1,w1,u1
    cj.transfer(v1,v0)
    cj.transfer(w1,w0)
    cj.transfer(u1,u0)
    cj.transfer(v2,v1)
    cj.transfer(w2,w1)
    cj.transfer(u2,u1)
    cj.transfer(joint_v2_w2, joint_v1_w1)
    cj.transfer(joint_v2_u2, joint_v1_u1)