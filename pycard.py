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

import cupy as cp

class NdGrid:
    def __init__(self, _base, _size, _res, _data=None):
        self.base = _base
        self.size = _size
        self.res = _res
        if _data is not None:
            self.data = cp.asarray(_data,dtype=cp.float32,order='F')

        temp_res_offsets = [1]
        self.res_offsets = self.calcResOffsets(1, temp_res_offsets, self.res)
        
        self.cell_widths = [self.size[a] / self.res[a] for a in range(self.numDimensions())]
        
        self.total_cells = 1
        for r in self.res:
            self.total_cells *= r

    def readData(self):
        return cp.asnumpy(self.data)

    def updateData(self, _data):
        self.data = cp.asarray(_data,dtype=cp.float32,order='F')

    def calcResOffsets(self, count, offsets, res):
        if len(res) == 1:
            return offsets

        count *= res[0]
        offsets = offsets + [count]

        new_res = []
        for i in [1+a for a in range(len(res)-1)]:
            new_res = new_res + [res[i]]

        if len(new_res) == 1:
            return offsets

        return self.calcResOffsets(count, offsets, new_res)

    def numDimensions(self):
        return len(self.base)

    def getCellCoords(self, cell_num):
        coords = [0 for a in range(self.numDimensions())]

        i = self.numDimensions()-1
        while i >= 0:
            coords[i] = int(cell_num / self.res_offsets[i])
            cell_num = cell_num - (coords[i] * self.res_offsets[i])
            i -= 1

        return coords

    def getCellNum(self, coords):
        cell_num = 0
        for i in range(self.numDimensions()):
            cell_num += coords[i] * self.res_offsets[i]

        return cell_num

    def getCellCentroid(self, cell_num):
        coords = self.getCellCoords(cell_num)
        centroid = [0 for a in range(self.numDimensions())]

        for d in range(self.numDimensions()):
            centroid[d] = self.base[d] + ((coords[d]+0.5)*self.cell_widths[d])

        return centroid

    def getContainingCellWeightedCoords(self, point):
        coords = [0 for a in range(self.numDimensions())]
        weights = [1.0 for a in range(self.numDimensions())]

        for d in range(self.numDimensions()):
            exact = (point[d]-self.base[d]) / self.cell_widths[d]
            coords[d] = int(exact)
            weights[d] = exact - coords[d]
            if coords[d] < 0:
                coords[d] = 0
                weights[d] = 1.0
            if coords[d] >= self.res[d]:
                coords[d] = self.res[d]-1
                weights[d] = 1.0

        return coords, weights


grids = []

def newDist(base, size, res, data=None):
    grids = grids + [NdGrid(base,size,res,data)]
    return len(grids)-1

def generateConditionalTransitionCSR(grid_in, func, grid_out):
    transitions = [[] for a in range(grid_out.total_cells)]
    
    offset = 0
    num_transitions = 0
    for r in range(grid_in.total_cells):
        start_point = grid_in.getCellCentroid(r)
        coords,weights = grid_out.getContainingCellWeightedCoords(func(start_point))
        transitions[grid_out.getCellNum(coords)] = transitions[grid_out.getCellNum(coords)] + [(r,1.0)]
        num_transitions += 1

    out_transitions_cells = [a for a in range(num_transitions)]
    out_transitions_props = [a for a in range(num_transitions)]
    out_transitions_counts = [a for a in range(grid_out.total_cells)]
    out_transitions_offsets = [a for a in range(grid_out.total_cells)]

    transition_count = 0
    cell_count = 0
    for t in transitions:
        # Don't worry about weighting just yet just do a single cell
        count = len(t)
        for r in t:
            out_transitions_cells[transition_count] = r[0]
            out_transitions_props[transition_count] = r[1]
            transition_count += 1
        out_transitions_offsets[cell_count] = offset
        out_transitions_counts[cell_count] = count
        offset += count
        cell_count += 1

    return out_transitions_cells, out_transitions_props, out_transitions_counts, out_transitions_offsets 

def generateConditional(grid_in, func, grid_out):
    conditional = [0 for a in range(grid_in.total_cells*grid_out.total_cells)]

    for r in range(grid_in.total_cells):
        start_point = grid_in.getCellCentroid(r)
        coords,weights = grid_out.getContainingCellWeightedCoords(func(start_point))
        conditional[(r*grid_out.total_cells) + grid_out.getCellNum(coords)] = 1.0

    return conditional


# For cases where the number of variables (or the combined size) on either end of the arrow is manageable,
# just use the old MIIND-style transition matrix to go from one set of variables
# to another.
# For example, ABCD -> EF, and ABCD -> EFGH are fine. ABCDEF -> GH is not fine because there's unlikely 
# to be space to hold either the transition matrix, or the joint distribution ABCDEF itself (unless some 
# of those are binary or enumerated variables)


cuda_source = r'''
extern "C"{

__device__ int modulo(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

__global__ void convolveKernel(
    unsigned int num_cells,
    float* grid_out,
    float* grid_in,
    float* kernel,
    unsigned int kernel_width,
    unsigned int dim_stride)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_cells; i += stride) {
        grid_out[i] = 0.0;
        for (int j = 0; j < kernel_width; j++) {
            grid_out[i] += grid_in[modulo(i - ((j - int(kernel_width/2.0)) * dim_stride), num_cells)] * kernel[j];
        }
    }
}

__global__
void applyJointTransition(
    unsigned int num_out_grid_cells,
    float* out_grid,
    unsigned int* transition_cells,
    float* transition_props,
    unsigned int* transitions_counts,
    unsigned int* transitions_offsets,
    float* in_grid) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;   
    
    for (int i = index; i < num_out_grid_cells; i += stride) {
        out_grid[i] = 0.0;
        for (int t = transitions_offsets[i]; t < transitions_offsets[i] + transitions_counts[i]; t++) {
            out_grid[i] += in_grid[transition_cells[t]] * transition_props[t];
        }
    }
}

}'''

cuda_module = cp.RawModule(code=cuda_source)
cuda_function_applyJointTransition = cuda_module.get_function('applyJointTransition')
cuda_function_convolveKernel = cuda_module.get_function('convolveKernel')

# For cases where the dimensionality limit of apply_joint_transition_kernel is hit, we need a new tactic. 
# Don't know what that is yet...

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
I_res = 300

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
wI_min_events = -5
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
wI = cj.newDist([wI_min], [(wI_max-wI_min)], [I_res], [a for a in wIpdf])

# For pymiind, I kernel cell size must match the joint cell size
wI_res = int((wI_max-wI_min) / ((w_max-w_min)/w_res))+1
pymiind_wI = [0 for a in range(wI_res)]
ratio = I_res / wI_res
val_counter = 0.0
pos_counter = 0
for i in range(I_res):
    pos_counter += 1
    if pos_counter > ratio:
        pos_counter = 0
        val_counter += wIpdf[i] * (i % ratio)
        pymiind_wI[int(i/ratio)] = val_counter
        val_counter = wIpdf[i] * (1.0-(i % ratio))
    else:
        val_counter += wIpdf[i]

u_rate = 2
ipsp = 0.5
uI_max_events = 5
uI_min_events = -5
uI_max = wI_max_events*ipsp
uI_min = wI_min_events*ipsp
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
uI = cj.newDist([uI_min], [(uI_max-uI_min)], [I_res], [a for a in uIpdf])

uI_res = int((uI_max-uI_min) / ((u_max-u_min)/u_res))+1
pymiind_uI = [0 for a in range(uI_res)]
ratio = I_res / uI_res
val_counter = 0.0
pos_counter = 0
for i in range(I_res):
    pos_counter += 1
    if pos_counter > ratio:
        pos_counter = 0
        val_counter += uIpdf[i] * (i % ratio)
        pymiind_uI[int(i/ratio)] = val_counter
        val_counter = uIpdf[i] * (1.0-(i % ratio))
    else:
        val_counter += uIpdf[i]

# CPU MIIND

dims = 3
cell_widths = [(v_max-v_min)/v_res, (w_max-w_min)/w_res, (u_max-u_min)/u_res]

cell_dict = [{},{}]

cell_base = [0.0 for a in range(dims)]
first_cell = True
current_dict = 0

for cv in range(v_res):
    for cw in range(w_res):
        for cu in range(u_res):
            val = vpdf[cv]*wpdf[cw]*updf[cu]
            if val > 0.0:
                if first_cell:
                    cell_base = [v_min + cv*cell_widths[0], w_min + cw*cell_widths[1], u_min + cu*cell_widths[2]]
                    first_cell = False
                cell_dict[current_dict][(cv,cw,cu)] = [val, []]

def calcTransitions(centroid, stepped_centroid, coord, d=0, target_coord=[], mass=1.0):

    if len(target_coord) == len(coord):
        return [(mass, target_coord)]

    diff = stepped_centroid[d] - centroid[d]
    cell_lo = coord[d] + int(diff / cell_widths[d])
    cell_hi = cell_lo + 1
    prop_lo = 0.0
    if diff < 0.0: # actually, diff is negative so cell_lo is the upper cell
        cell_hi = cell_lo - 1
        prop_lo = ((diff % cell_widths[d]) / cell_widths[d])
    else:
        prop_lo = 1.0 - ((diff % cell_widths[d]) / cell_widths[d])
    prop_hi = 1.0 - prop_lo
    
    return calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_lo], mass*prop_lo) + calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_hi], mass*prop_hi)

for coord in cell_dict[current_dict]:
    centroid = [0 for a in range(dims)]

    for d in range(dims):
        centroid[d] = cell_base[d] + ((coord[d]+0.5)*cell_widths[d])

    stepped_centroid = cond(centroid)

    cell_dict[current_dict][coord][1] = calcTransitions(centroid, stepped_centroid, coord)

    
w_kernel_dim = 1
w_kernel_transitions = {}
cs = tuple([0 for d in range(dims)])
w_kernel_transitions[cs] = []
for c in range(len(pymiind_wI)):
    if pymiind_wI[c] > 0.0:
        w_kernel_transitions[cs] = w_kernel_transitions[cs] + [[(pymiind_wI[c], [c if d == w_kernel_dim else 0 for d in range(dims)])]]

u_kernel_dim = 2
u_kernel_transitions = {}
cs = tuple([0 for d in range(dims)])
u_kernel_transitions[cs] = []
for c in range(len(pymiind_uI)):
    if pymiind_uI[c] > 0.0:
        u_kernel_transitions[cs] = u_kernel_transitions[cs] + [[(pymiind_uI[c], [c if d == u_kernel_dim else 0 for d in range(dims)])]]

threshold = -50.4
reset = -70.6
threshold_reset_dim = 0
threshold_cell = int((threshold - cell_base[threshold_reset_dim])/cell_widths[threshold_reset_dim])
reset_cell = int((reset - cell_base[threshold_reset_dim])/cell_widths[threshold_reset_dim])

def updateCell(relative, new_cell_dict, cell_dict, transition, coord, func):
    print(transition)
    if relative:
        print(coord, transition)
        for d in range(len(coord)):
            transition[1][d] = coord[d] + transition[1][d]

    if transition[1][threshold_reset_dim] >= threshold_cell:
        transition[1][threshold_reset_dim] = reset_cell


    if tuple(transition[1]) not in new_cell_dict:
        new_cell_dict[tuple(transition[1])] = [0.0,[]]

    try:
        new_cell_dict[tuple(transition[1])][1] = cell_dict[tuple(transition[1])][1]
    except:
        centroid = [0 for a in range(len(coord))]

        for d in range(len(coord)):
            centroid[d] = cell_base[d] + ((transition[1][d]+0.5)*cell_widths[d])

        stepped_centroid = func(centroid)
        new_cell_dict[tuple(transition[1])][1] = calcTransitions(centroid, stepped_centroid, transition[1])

    new_cell_dict[tuple(transition[1])][0] += cell_dict[coord][0]*transition[0]

def calcCellCentroid(coords):
    centroid = [0 for a in range(dims)]

    for d in range(dims):
        centroid[d] = cell_base[d] + ((coords[d]+0.5)*cell_widths[d])

    return centroid

def calcMarginals(cell_dict):
    vs = [{} for d in range(dims)]
    for c in cell_dict:
        for d in range(dims):
            if c[d] not in vs[d]:
                vs[d][c[d]] = cell_dict[c][0]
            else:
                vs[d][c[d]] += cell_dict[c][0]

    final_vs = [[] for d in range(dims)]
    final_vals = [[] for d in range(dims)]

    for d in range(dims):
        for v in vs[d]:
            final_vs[d] = final_vs[d] + [v]
            final_vals[d] = final_vals[d] + [vs[d][v]]

    return final_vs, final_vals

    
# pyMIIND

initial_distribution = np.zeros(v_res*w_res*u_res)
for cv in range(v_res):
    for cw in range(w_res):
        for cu in range(u_res):
            initial_distribution[cv + (v_res*cw) + (v_res*w_res*cu)] = vpdf[cv]*wpdf[cw]*updf[cu]

initial_dist_3D = np.zeros((v_res,w_res,u_res))
for cv in range(v_res):
    for cw in range(w_res):
        for cu in range(u_res):
            initial_dist_3D[cv,cw,cu] = vpdf[cv]*wpdf[cw]*updf[cu]

pymiind_grid_1 = NdGrid([v_min,w_min,u_min], [(v_max-v_min),(w_max-w_min),(u_max-u_min)], [v_res,w_res,u_res], initial_dist_3D)
pymiind_grid_2 = NdGrid([v_min,w_min,u_min], [(v_max-v_min),(w_max-w_min),(u_max-u_min)], [v_res,w_res,u_res], initial_dist_3D)
cond_transitions_cells, cond_transitions_props, cond_transitions_counts, cond_transitions_offsets  = generateConditionalTransitionCSR(pymiind_grid_1,cond,pymiind_grid_2)

cond_cells = cp.asarray(cond_transitions_cells,)
cond_props = cp.asarray(cond_transitions_props,dtype=cp.float32)
cond_counts = cp.asarray(cond_transitions_counts)
cond_offsets = cp.asarray(cond_transitions_offsets)

excitatory_kernel = NdGrid([wI_min], [(wI_max-wI_min)], [wI_res], np.array([a for a in pymiind_wI]))
inhibitory_kernel = NdGrid([uI_min], [(uI_max-uI_min)], [uI_res], np.array([a for a in pymiind_uI]))


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
            index = (j * cj.res(v2_w2)[0]) + i
            reset_index = (j * cj.res(v2_w2)[0]) + reset_cell
            if i >= threshold_cell and dist_v2_w2[index] > 0.0:
                n_v2_w2_mass[reset_index] += dist_v2_w2[index]
                n_v2_w2_mass[index] = 0.0
                total_reset_mass += dist_v2_w2[index]

    for j in range(cj.res(v2_u2)[1]): # for each column
        for i in range(cj.res(v2_u2)[0]): 
            index = (j * cj.res(v2_u2)[0]) + i
            reset_index = (j * cj.res(v2_u2)[0]) + reset_cell
            if i >= threshold_cell and dist_v2_u2[index] > 0.0:
                n_v2_u2_mass[reset_index] += dist_v2_u2[index]
                n_v2_u2_mass[index] = 0.0
                total_reset_mass += dist_v2_u2[index]
    
    cj.update(v2_w2, n_v2_w2_mass)
    cj.update(v2_u2, n_v2_u2_mass)

    cj.marginal(v2_w2, 0, w2)
    cj.marginal(v2_w2, 1, v2)
    cj.marginal(v2_u2, 0, u2)

    # Record distributions if you wish here.

    #if CPUMIIND:
    
    for coord in cell_dict[current_dict]:
        for ts in cell_dict[current_dict][coord][1]:
            updateCell(False, cell_dict[(current_dict+1)%2], cell_dict[current_dict], ts, coord, cond)
        print(w_kernel_transitions)
        for ts in w_kernel_transitions[(0,0,0)]:
            print("ts ", ts, " ts")
            updateCell(True, cell_dict[(current_dict+1)%2], w_kernel_transitions, ts, coord, cond)
        for ts in u_kernel_transitions[(0,0,0)]:
            updateCell(True, cell_dict[(current_dict+1)%2], u_kernel_transitions, ts, coord, cond)

    current_dict = (current_dict+1)%2


    #if pyMIIND:

    cuda_function_applyJointTransition((v_res*w_res*u_res,),(128,),(v_res*w_res*u_res, pymiind_grid_2.data, cond_cells, cond_props, cond_counts, cond_offsets, pymiind_grid_1.data))
    cuda_function_convolveKernel((v_res*w_res*u_res,),(128,), (v_res*w_res*u_res, pymiind_grid_1.data, pymiind_grid_2.data, excitatory_kernel.data, wI_res, v_res))
    cuda_function_convolveKernel((v_res*w_res*u_res,),(128,), (v_res*w_res*u_res, pymiind_grid_2.data, pymiind_grid_1.data, inhibitory_kernel.data, uI_res, v_res*w_res))
    cp.copyto(pymiind_grid_1.data, pymiind_grid_2.data)
    

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
    
    if (iteration % 1 == 0) :
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
        #if pyMIIND:
        py_miind_v = cp.asnumpy(cp.sum(pymiind_grid_2.data, (1,2)))
        py_miind_w = cp.asnumpy(cp.sum(pymiind_grid_2.data, (0,2)))
        py_miind_u = cp.asnumpy(cp.sum(pymiind_grid_2.data, (0,1)))

        py_miind_v = [a / ((v_max-v_min)/v_res) for a in py_miind_v]
        py_miind_w = [a / ((w_max-w_min)/w_res) for a in py_miind_w]
        py_miind_u = [a / ((u_max-u_min)/u_res) for a in py_miind_u]

        ax[0,0].plot(np.linspace(v_min,v_max,v_res), py_miind_v, linestyle='-.')
        ax[0,1].plot(np.linspace(w_min,w_max,w_res), py_miind_w, linestyle='-.')
        ax[1,0].plot(np.linspace(u_min,u_max,u_res), py_miind_u, linestyle='-.')
        #if CPUMIIND:
        mpos, marginals = calcMarginals(cell_dict[current_dict])

        ax[0,0].plot(mpos[0], marginals[0], linestyle='-.')
        ax[0,1].plot(mpos[1], marginals[1], linestyle='-.')
        ax[1,0].plot(mpos[2], marginals[2], linestyle='-.')

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