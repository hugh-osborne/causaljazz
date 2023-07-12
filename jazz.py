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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

test_1D = False
test_2D = False
test_3D = False
test_Function_to_Conditional = False
test_deomposition = True

show_miind_redux = True
show_monte_carlo = True


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
    tau_e =2.728
    tau_i = 10.49

    v = y[0]
    w = y[1]
    u = y[2]

    v_prime = (-g_l*(v - E_l) - w * (v - E_e) - u * (v - E_i)) / C
    w_prime = -(w) / tau_e
    u_prime = -(u) / tau_i

    return [v + 0.1*v_prime, w + 0.1*w_prime, u + 0.1*u_prime]

def cond_diff(y):
    E_l = -70.6
    V_thres = -50.4 
    E_e = 0.0
    E_i = -75
    C = 281
    g_l = 0.03
    tau_e =2.728
    tau_i =10.49

    v = y[0]
    w = y[1]
    u = y[2]

    v_prime = (-g_l*(v - E_l) - w * (v - E_e) - u * (v - E_i)) / C
    w_prime = -(w) / tau_e
    u_prime = -(u) / tau_i

    return [v_prime, w_prime, u_prime]

def w_prime(y):
    w = y[0]
    jump = y[1]
    tau_e =2.728
    dt = 0.1

    w_prime = -(w) / tau_e
    return w + (dt*w_prime) + jump

def u_prime(y):
    u = y[0]
    jump = y[1]
    tau_i = 10.49
    dt = 0.1

    u_prime = -(u) / tau_i
    return u + (dt*u_prime) + jump

def vw(y):
    v = y[0]
    w = y[1]

    E_l = -70.6
    g_l = 0.03
    E_e = 0.0

    dt = 0.1
    C = 281

    return v + (dt/C)*(-(g_l*(v - E_l)) - (w * (v - E_e)))

def vu(y):
    v = y[0]
    u = y[1]

    E_i = -75

    dt = 0.1
    C = 281

    return (dt/C)*(-u * (v - E_i))

def v_prime(y):
    vw = y[0]
    vu = y[1] 

    threshold = -50.0
    reset = -70.6

    v = vw + vu

    return v


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
vpdf = [a * ((v_max-v_min)/v_res) for a in norm.pdf(v, -70.6, 0.1)]
wpdf = [a * ((w_max-w_min)/w_res) for a in norm.pdf(w, 0.0, 0.1)]
updf = [a * ((u_max-u_min)/u_res) for a in norm.pdf(u, 0.0, 0.1)]

vpdf = [a / sum(vpdf) for a in vpdf]
wpdf = [a / sum(wpdf) for a in wpdf]
updf = [a / sum(updf) for a in updf]

v0 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in vpdf])
w0 = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in wpdf])
u0 = cj.newDist([u_min],[(u_max-u_min)],[u_res],[a for a in updf])

# Poisson inputs

w_rate = 4
epsp = 0.5
wI_max_events = 5
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
        wIpdf_final[i] += poisson.pmf(e, w_rate*0.1) * lower_prop
        wIpdf_final[i+1] += poisson.pmf(e, w_rate*0.1) * upper_prop
wIpdf = wIpdf_final
wI = cj.newDist([wI_min], [wI_max], [I_res], [a for a in wIpdf])

u_rate = 2
ipsp = 0.5
uI_max_events = 5
uI_min_events = -2
uI_max = wI_max_events*epsp
uI_min = wI_min_events*epsp
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
        uIpdf_final[i] += poisson.pmf(e, u_rate*0.1) * lower_prop
        uIpdf_final[i+1] += poisson.pmf(e, u_rate*0.1) * upper_prop
uIpdf = uIpdf_final
uI = cj.newDist([uI_min], [uI_max], [I_res], [a for a in uIpdf])

# Initialise the monte carlo neurons
mc_neurons = np.array([[norm.rvs(-70.6, 0.1, 1)[0],norm.rvs(0.0, 0.1, 1)[0],norm.rvs(0.0, 0.1, 1)[0]] for a in range(5000)])

# Initialise the MIIND (Redux) simulation
miind_cond_grid = cj.generate(cond_diff,  [v_min,w_min,u_min], [(v_max-v_min),(w_max-w_min),(u_max-u_min)], [v_res,w_res,u_res], -50, -70.6, [0,0,0], 0.1)
cj.init(0.1, False)

pop3 = cj.addPopulation(miind_cond_grid, [-70.6, 0.0, 0.0], 0.0, False)
miind_ex = cj.poisson(pop3, [0,epsp,0])
miind_in = cj.poisson(pop3, [0,0,ipsp])

######

c_w_prime = cj.boundedFunction([w_min,wI_min],[(w_max-w_min),(wI_max-wI_min)],[w_res,I_res], w_prime, w_min, (w_max-w_min), w_res)

c_u_prime = cj.boundedFunction([u_min,uI_min],[(u_max-u_min),(uI_max-uI_min)],[u_res,I_res], u_prime, u_min, (u_max-u_min), u_res)
c_vw = cj.function([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], vw, res)
c_vu = cj.function([v_min,u_min],[(v_max-v_min),(u_max-u_min)],[v_res,u_res], vu, res)

c_v_prime = cj.boundedFunction([cj.base(c_vw)[2],cj.base(c_vu)[2]], [cj.size(c_vw)[2],cj.size(c_vu)[2]], [res,res], v_prime, v_min, (v_max-v_min), v_res)

w0_v0_vw = cj.newDist([w_min, v_min, cj.base(c_vw)[2]], [(w_max-w_min),(v_max-v_min), cj.size(c_vw)[2]], [w_res,v_res,res], [a for a in np.zeros(w_res*v_res*res)])
cj.multiply([w0,v0,c_vw], [[0],[1],[1,0,2]], w0_v0_vw, [0,1,2])

u0_v0_vu = cj.newDist([u_min, v_min, cj.base(c_vu)[2]], [(u_max-u_min),(v_max-v_min), cj.size(c_vu)[2]], [w_res,v_res,res], [a for a in np.zeros(w_res*v_res*res)])
cj.multiply([u0,v0,c_vu], [[0],[1],[1,0,2]], u0_v0_vu, [0,1,2])

v0_vw = cj.newDist([v_min, cj.base(c_vw)[2]], [(v_max-v_min),cj.size(c_vw)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])
v0_vu = cj.newDist([v_min, cj.base(c_vu)[2]], [(v_max-v_min),cj.size(c_vu)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])
vw_given_v0 = cj.newDist([v_min, cj.base(c_vw)[2]], [(v_max-v_min),cj.size(c_vw)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])
vu_given_v0 = cj.newDist([v_min, cj.base(c_vu)[2]], [(v_max-v_min),cj.size(c_vu)[2]], [v_res,res], [a for a in np.zeros(v_res*res)])

cj.marginal(w0_v0_vw, 0, v0_vw)
cj.marginal(u0_v0_vu, 0, v0_vu)
cj.conditional(v0_vw, [0], v0, vw_given_v0)
cj.conditional(v0_vu, [0], v0, vu_given_v0)

w0_vw_vu = cj.newDist([w_min, cj.base(c_vw)[2], cj.base(c_vu)[2]], [(w_max-w_min),cj.size(c_vw)[2],cj.size(c_vu)[2]], [w_res,res,res], [a for a in np.zeros(w_res*res*res)])
u0_vw_vu = cj.newDist([u_min, cj.base(c_vw)[2], cj.base(c_vu)[2]], [(u_max-u_min),cj.size(c_vw)[2],cj.size(c_vu)[2]], [u_res,res,res], [a for a in np.zeros(u_res*res*res)])

cj.multiply([w0_v0_vw, vu_given_v0], [[0,3,1],[3,2]], w0_vw_vu, [0,1,2])
cj.multiply([u0_v0_vu, vw_given_v0], [[0,3,2],[3,1]], u0_vw_vu, [0,1,2])

w0_vw_v1 = cj.newDist([w_min, cj.base(c_vw)[2], v_min], [(w_max-w_min),cj.size(c_vw)[2],(v_max-v_min)], [w_res,res,v_res], [a for a in np.zeros(w_res*res*v_res)]) 
u0_vu_v1 = cj.newDist([u_min, cj.base(c_vu)[2], v_min], [(u_max-u_min),cj.size(c_vu)[2],(v_max-v_min)], [u_res,res,v_res], [a for a in np.zeros(u_res*res*v_res)]) 

cj.multiply([w0_vw_vu, c_v_prime], [[0,1,3],[1,3,2]], w0_vw_v1, [0,1,2])
cj.multiply([u0_vw_vu, c_v_prime], [[0,3,1],[3,1,2]], u0_vu_v1, [0,1,2])

w0_v1 = cj.newDist([w_min, v_min], [(w_max-w_min),(v_max-v_min)], [w_res,v_res], [a for a in np.zeros(w_res*v_res)]) 
u0_v1 = cj.newDist([u_min, v_min], [(u_max-u_min),(v_max-v_min)], [u_res,v_res], [a for a in np.zeros(u_res*v_res)])  
v1_given_w0 = cj.newDist([w_min, v_min], [(w_max-w_min),(v_max-v_min)], [w_res,v_res], [a for a in np.zeros(w_res*v_res)]) 
v1_given_u0 = cj.newDist([u_min, v_min], [(u_max-u_min),(v_max-v_min)], [u_res,v_res], [a for a in np.zeros(u_res*v_res)]) 

cj.marginal(w0_vw_v1, 1, w0_v1)
cj.marginal(u0_vu_v1, 1, u0_v1)
cj.conditional(w0_v1, [0], w0, v1_given_w0)
cj.conditional(u0_v1, [0], u0, v1_given_u0)

w0_v1_w1 = cj.newDist([w_min, v_min, w_min], [(w_max-w_min),(v_max-v_min),(w_max-w_min)], [w_res,v_res,w_res], [a for a in np.zeros(w_res*v_res*w_res)]) 
u0_v1_u1 = cj.newDist([u_min, v_min, u_min], [(u_max-u_min),(v_max-v_min),(u_max-u_min)], [u_res,v_res,u_res], [a for a in np.zeros(u_res*v_res*u_res)]) 

cj.multiply([w0,wI,c_w_prime,v1_given_w0], [[0],[3],[0,3,2],[0,1]], w0_v1_w1, [0,1,2])
cj.multiply([u0,uI,c_u_prime,v1_given_u0], [[0],[3],[0,3,2],[0,1]], u0_v1_u1, [0,1,2])

v1_w1 = cj.newDist([v_min, w_min], [(v_max-v_min),(w_max-w_min)], [v_res,w_res], [a for a in np.zeros(v_res*w_res)]) 
v1_u1 = cj.newDist([v_min, u_min], [(v_max-v_min),(u_max-u_min)], [v_res,u_res], [a for a in np.zeros(v_res*u_res)]) 

w1 = cj.newDist([w_min], [(w_max-w_min)], [w_res], [a for a in np.zeros(w_res)]) 
v1 = cj.newDist([v_min], [(v_max-v_min)], [v_res], [a for a in np.zeros(v_res)]) 
u1 = cj.newDist([u_min], [(u_max-u_min)], [u_res], [a for a in np.zeros(u_res)])

cj.marginal(w0_v1_w1, 0, v1_w1)
cj.marginal(u0_v1_u1, 0, v1_u1)

cj.marginal(v1_w1, 0, w1)
cj.marginal(v1_w1, 1, v1)
cj.marginal(v1_u1, 0, u1)

v2 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in np.zeros(v_res)])
w2 = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in np.zeros(w_res)])
u2 = cj.newDist([u_min],[(u_max-u_min)],[u_res],[a for a in np.zeros(u_res)])

v2_w2 = cj.newDist([v_min,w_min],[(v_max-v_min),(w_max-w_min)],[v_res,w_res], [a for a in np.zeros(v_res*w_res)])
v2_u2 = cj.newDist([v_min,u_min],[(v_max-v_min),(u_max-u_min)],[v_res,u_res], [a for a in np.zeros(v_res*u_res)])

# Start the simulation
cj.start()

# We can use causal jazz distributions to read and manage distributions from the MIIND redux simulation. Nice!
miind_mass_grid = cj.newDist([v_min,w_min,u_min], [(v_max-v_min),(w_max-w_min),(u_max-u_min)], [v_res,w_res,u_res], [a for a in np.zeros(v_res*w_res*u_res)])
miind_marginal_vw = cj.newDist([v_min,w_min], [(v_max-v_min),(w_max-w_min)], [v_res,w_res], [a for a in np.zeros(v_res*w_res)])
miind_marginal_vu = cj.newDist([v_min,u_min], [(v_max-v_min),(u_max-u_min)], [v_res,u_res], [a for a in np.zeros(v_res*u_res)])
miind_marginal_v = cj.newDist([v_min], [(v_max-v_min)], [v_res], [a for a in np.zeros(v_res)])
miind_marginal_w = cj.newDist([w_min], [(w_max-w_min)], [w_res], [a for a in np.zeros(w_res)])
miind_marginal_u = cj.newDist([u_min], [(u_max-u_min)], [u_res], [a for a in np.zeros(u_res)])

# firing rates

monte_carlo_rates = []
jazz_rates = []
miind_redux_rates = []

# Lets try 10 iterations
for iteration in range(1000):
    
    cj.multiply([v1_w1,c_vw], [[1,0],[1,0,2]], w0_v0_vw, [0,1,2])
    cj.multiply([v1_u1,c_vu], [[1,0],[1,0,2]], u0_v0_vu, [0,1,2])

    cj.marginal(w0_v0_vw, 0, v0_vw)
    cj.marginal(u0_v0_vu, 0, v0_vu)
    cj.conditional(v0_vw, [0], v1, vw_given_v0)
    cj.conditional(v0_vu, [0], v1, vu_given_v0)

    cj.multiply([w0_v0_vw, vu_given_v0], [[0,3,1],[3,2]], w0_vw_vu, [0,1,2])
    cj.multiply([u0_v0_vu, vw_given_v0], [[0,3,2],[3,1]], u0_vw_vu, [0,1,2])
    
    cj.multiply([w0_vw_vu, c_v_prime], [[0,1,3],[1,3,2]], w0_vw_v1, [0,1,2])
    cj.multiply([u0_vw_vu, c_v_prime], [[0,3,1],[3,1,2]], u0_vu_v1, [0,1,2])

    cj.marginal(w0_vw_v1, 1, w0_v1)
    cj.marginal(u0_vu_v1, 1, u0_v1)
    cj.conditional(w0_v1, [0], w1, v1_given_w0)
    cj.conditional(u0_v1, [0], u1, v1_given_u0)
    
    cj.multiply([w1,wI,c_w_prime,v1_given_w0], [[0],[3],[0,3,2],[0,1]], w0_v1_w1, [0,1,2])
    cj.multiply([u1,uI,c_u_prime,v1_given_u0], [[0],[3],[0,3,2],[0,1]], u0_v1_u1, [0,1,2])

    cj.marginal(w0_v1_w1, 0, v2_w2)
    cj.marginal(u0_v1_u1, 0, v2_u2)

    threshold = -50.0
    reset = -70.6
    threshold_cell = int((threshold - cj.base(v2)[0]) / (cj.size(v2)[0] / cj.res(v2)[0]))
    reset_cell = int((reset - cj.base(v2)[0]) / (cj.size(v2)[0] / cj.res(v2)[0]))

    dist_v2_w2 = cj.readDist(v2_w2)
    dist_v2_u2 = cj.readDist(v2_u2)
    n_v2_w2_mass = [a for a in dist_v2_w2]
    n_v2_u2_mass = [a for a in dist_v2_u2]
    total_reset_mass = 0.0
    for j in range(cj.res(v2_w2)[1]): # for each column
        for i in range(cj.res(v2_w2)[0]): 
            index = (i * cj.res(v2_w2)[1]) + j
            reset_index = (reset_cell * cj.res(v2_w2)[1]) + j
            if j >= threshold_cell and dist_v2_w2[index] > 0.0:
                n_v2_w2_mass[reset_index] += dist_v2_w2[index]
                n_v2_w2_mass[index] = 0.0
                total_reset_mass += dist_v2_w2[index]

    for j in range(cj.res(v2_u2)[1]): # for each column
        for i in range(cj.res(v2_u2)[0]): 
            index = (i * cj.res(v2_u2)[1]) + j
            reset_index = (reset_cell * cj.res(v2_u2)[1]) + j
            if j >= threshold_cell and dist_v2_u2[index] > 0.0:
                n_v2_u2_mass[reset_index] += dist_v2_u2[index]
                n_v2_u2_mass[index] = 0.0
                total_reset_mass += dist_v2_u2[index]
    
    cj.update(v2_w2, n_v2_w2_mass)
    cj.update(v2_u2, n_v2_u2_mass)

    cj.marginal(v2_w2, 0, w2)
    cj.marginal(v2_w2, 1, v2)
    cj.marginal(v2_u2, 0, u2)

    # Record distributions if you wish here.

    if show_monte_carlo:
        # Also run the monte carlo simulation 
    
        fired_count = 0
        for nn in range(len(mc_neurons)):
            mc_neurons[nn] = cond(mc_neurons[nn])

            if (mc_neurons[nn][0] > -50.0):
                mc_neurons[nn][0] = -70.6
                fired_count+=1
                
            mc_neurons[nn][1] += epsp*poisson.rvs(w_rate*0.1) # override w
            mc_neurons[nn][2] += ipsp*poisson.rvs(u_rate*0.1) # override u

        monte_carlo_rates = monte_carlo_rates + [(fired_count / len(mc_neurons)) / 0.0001]

    if show_miind_redux:
        # Also run the MIIND (Redux) simulation
        cj.postRate(pop3, miind_ex, w_rate)
        cj.postRate(pop3, miind_in, u_rate)

        cj.step()
        #rates1 = rates1 + [cj.readRates()[0]*1000]
        mass = cj.readMass(pop3)
        miind_redux_rates = miind_redux_rates + [cj.readRates()[0]*1000]
    
        cj.update(miind_mass_grid, [a for a in mass])

        cj.marginal(miind_mass_grid, 2, miind_marginal_vw)
        cj.marginal(miind_mass_grid, 1, miind_marginal_vu)
        cj.marginal(miind_marginal_vw, 1, miind_marginal_v)
        cj.marginal(miind_marginal_vw, 0, miind_marginal_w)
        cj.marginal(miind_marginal_vu, 0, miind_marginal_u)

        miind_dist_v = cj.readDist(miind_marginal_v)
        miind_dist_w = cj.readDist(miind_marginal_w)
        miind_dist_u = cj.readDist(miind_marginal_u)

        miind_dist_v = [a / ((v_max-v_min)/v_res) for a in miind_dist_v]
        miind_dist_w = [a / ((w_max-w_min)/w_res) for a in miind_dist_w]
        miind_dist_u = [a / ((u_max-u_min)/u_res) for a in miind_dist_u]

    # The monte carlo hist function gives density not mass (booo)
    # so let's just convert to density here
    
    if (iteration % 10 == 0) :
        dist_v = cj.readDist(v0)
        dist_w = cj.readDist(w0)
        dist_u = cj.readDist(u0)

        dist_v = [a / ((v_max-v_min)/v_res) for a in dist_v]
        dist_w = [a / ((w_max-w_min)/w_res) for a in dist_w]
        dist_u = [a / ((u_max-u_min)/u_res) for a in dist_u]

        fig, ax = plt.subplots(2,2)
        ax[0,0].plot(v, dist_v)
        ax[0,1].plot(w, dist_w)
        ax[1,0].plot(u, dist_u)
        if show_monte_carlo:
            ax[0,0].hist(mc_neurons[:,0], density=True, bins=v_res, range=[v_min,v_max], histtype='step')
            ax[0,1].hist(mc_neurons[:,1], density=True, bins=w_res, range=[w_min,w_max], histtype='step')
            ax[1,0].hist(mc_neurons[:,2], density=True, bins=u_res, range=[u_min,u_max], histtype='step')
        if show_miind_redux:
            ax[0,0].plot(np.linspace(v_min,v_max,v_res), miind_dist_v, linestyle='--')
            ax[0,1].plot(np.linspace(w_min,w_max,w_res), miind_dist_w, linestyle='--')
            ax[1,0].plot(np.linspace(u_min,u_max,u_res), miind_dist_u, linestyle='--')
    
        fig.tight_layout()
        plt.show()

    # Transfer v1,w1,u1 -> v0,w0,u0, transfer v2,w2,u2 -> v1,w1,u1
    cj.transfer(v1,v0)
    cj.transfer(w1,w0)
    cj.transfer(u1,u0)
    cj.transfer(v2,v1)
    cj.transfer(w2,w1)
    cj.transfer(u2,u1)
    cj.transfer(v2_w2, v1_w1)
    cj.transfer(v2_u2, v1_u1)

fig, ax = plt.subplots(1,1)
ax.plot(range(len(jazz_rates)), jazz_rates)
if show_monte_carlo:
    ax.plot(range(len(monte_carlo_rates)-2), monte_carlo_rates[2:])
if show_miind_redux:
    ax.plot(range(len(miind_redux_rates)), miind_redux_rates)
    
fig.tight_layout()
plt.show()

# Shutdown MIIND redux simulation
cj.shutdown()