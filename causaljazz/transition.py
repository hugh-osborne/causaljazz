import numpy as np
import cupy as cp

from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import lil_matrix

class transition_cpu:
    def __init__(self, _func):
        
        # Noise kernels are stored like the values in the cell buffers: a pair.
        # The first value is 1.0 : The full amount of mass in each cell is spread according to the kernel
        # The second value is a list of transitions where the cell coords are relative to the cell to which the kernel will be applied
        self.noise_kernels = []
        
        # The deterministic function upon which the transiions are based
        self.func = _func
        
        # cell_buffer transition values for pmf
        self.transition_buffer = {}

    # Add a noise kernel
    def addNoiseKernel(self, kernel, dimension):
        kernel_transitions = {}
        cs = tuple(np.zeros(self.pmf.dims).tolist())
        kernel_transitions[cs] = []
        for c in range(len(kernel)):
            if kernel[c] > 0.0:
                kernel_transitions[cs] = kernel_transitions[cs] + [(kernel[c], [c-int(len(kernel)/2) if d == dimension else 0 for d in range(self.pmf.dims)])]
        self.noise_kernels = self.noise_kernels + [[1.0,kernel_transitions[cs]]]
        return len(self.noise_kernels) - 1

    # Calculate the transitions for each cell based on the given centroid and centroid after one application of self.func (stepped_centroid)
    # The centroid calculation is not done here because sometimes it's better to batch function application (for example with an ANN)
    def calcTransitions(self, out_pmf, stepped_centroid, d=0, target_coord=[], mass=1.0):
        if len(target_coord) == len(stepped_centroid):
            return [(mass, target_coord)]
        
        t_cell_location = out_pmf.getPointModuloDim(stepped_centroid, d)
        t_cell_lo = out_pmf.findCellCoordsOfPointDim(stepped_centroid, d)
        
        if t_cell_location < 0.5: # cell spreads to cell below
            t_cell_hi = t_cell_lo-1
            t_prop_lo = t_cell_location + 0.5
            t_prop_hi = 1-t_prop_lo
        else:
            t_cell_hi = t_cell_lo+1
            t_prop_hi = t_cell_location - 0.5
            t_prop_lo = 1 - t_prop_hi
            
        return self.calcTransitions(out_pmf, stepped_centroid, d+1, target_coord + [t_cell_lo], mass*t_prop_lo) + self.calcTransitions(out_pmf, stepped_centroid, d+1, target_coord + [t_cell_hi], mass*t_prop_hi)

    # Update the mass of a cell
    def updateCell(self, new_cell_dict, transition, mass):
        t = [a for a in transition[1]]
        if tuple(t) not in new_cell_dict:
            new_cell_dict[tuple(t)] = 0.0    
        new_cell_dict[tuple(t)] += mass*transition[0]
        
    def checkTransitionsMatchBuffer(self, in_pmf, out_pmf):
        new_coords = []
        centroids = [] 
        for coord in in_pmf.cell_buffer.keys():
            if coord not in self.transition_buffer.keys(): # the pmf has been altered elsewhere so we need to calculate the transitions for any new cells
                centroid = [0 for a in range(len(coord))]
                for d in range(len(coord)):
                    centroid[d] = in_pmf.cell_base[d] + ((coord[d]+0.5)*in_pmf.cell_widths[d])
                new_coords = new_coords + [coord]
                centroids = centroids + [centroid]
        
        if len(centroids) > 0:
            shifted_centroids = self.func(centroids)
            
        for c in range(len(new_coords)):
            self.transition_buffer[tuple(new_coords[c])] = self.calcTransitions(out_pmf, shifted_centroids[c])
            
        remove = []
        for key in self.transition_buffer.keys():
            if key not in in_pmf.cell_buffer.keys():
                remove = remove + [key]
        
        for a in remove:        
            self.transition_buffer.pop(a, None)
            
        return new_coords
        
    def applyFunction(self, in_pmf, out_pmf):
        new_coords = self.checkTransitionsMatchBuffer(in_pmf, out_pmf)
        
        # Set the next buffer mass values to 0
        for a in out_pmf.cell_buffer.keys():
            out_pmf.cell_buffer[a] = 0.0
            
        # If there were further changes to pmf_in (new_coords is not empty), add those to pmf_out as well.
        for c in range(len(new_coords)):
            out_pmf.cell_buffer[tuple(new_coords[c])] = 0.0
        
        # Fill the pmf_out cell buffer with the updated mass values
        for t_key, t_val in self.transition_buffer.items():
            for ts in t_val:
                self.updateCell(out_pmf.cell_buffer, ts, in_pmf.cell_buffer[t_key])

        # Remove any cells with a small amount of mass and keep a total to spread back to the remaining population
        remove = []
        mass_summed = 1.0
        for a in out_pmf.cell_buffer.keys():
            if out_pmf.cell_buffer[a] < in_pmf.mass_epsilon:
                remove = remove + [a]
                mass_summed -= out_pmf.cell_buffer[a]

        for a in remove:
            out_pmf.cell_buffer.pop(a, None)
            self.transition_buffer.pop(a, None)
            
        for coord in out_pmf.cell_buffer:
            out_pmf.cell_buffer[coord] /= mass_summed

    def applyNoiseKernel(self, kernel_id, in_pmf, out_pmf):
        new_coords = self.checkTransitionsMatchBuffer(in_pmf, out_pmf)
        
        kernel = self.noise_kernels[kernel_id]
        
        # Set the next buffer mass values to 0
        for a in out_pmf.cell_buffer.keys():
            out_pmf.cell_buffer[a] = 0.0
            
        # If there were further changes to pmf_in (new_coords is not empty), add those to pmf_out as well.
        for c in range(len(new_coords)):
            out_pmf.cell_buffer[tuple(new_coords[c])] = 0.0

        # Apply the kernel
        for coord in self.pmf.cell_buffer:
            for ts in kernel[1]:
                relative_ts_coord = [a for a in ts[1]]
                for d in range(len(relative_ts_coord)):
                    relative_ts_coord[d] = coord[d] + relative_ts_coord[d]
                relative_ts = [ts[0],relative_ts_coord]
                if tuple(relative_ts_coord) not in out_pmf.cell_buffer.keys():
                    out_pmf.cell_buffer[tuple(new_coords[c])] = 0.0
                self.updateCell(out_pmf.cell_buffer, relative_ts, in_pmf.cell_buffer[coord])

        remove = []
        mass_summed = 1.0
        for a in out_pmf.cell_buffer.keys():
            if out_pmf.cell_buffer[a] < out_pmf.mass_epsilon:
                remove = remove + [a]
                mass_summed -= out_pmf.cell_buffer[a]

        for a in remove:
            out_pmf.cell_buffer.pop(a, None)
            self.transition_buffer.pop(a, None)
                
        for coord in out_pmf.cell_buffer:
            out_pmf.cell_buffer[coord] /= mass_summed
            

class transition_gpu:
    def __init__(self, pmf, _func, pmf_out):
        self.pmf = pmf
        self.pmf_out = pmf_out
        self.noise_kernels = []
        self.func = _func

        self.csr = self.generateConditionalTransitionCSR(self.pmf.grid, _func, self.pmf_out.grid)

    def generateConditionalTransitionCSR(self, grid_in, func, grid_out):
        lil_mat = lil_matrix((grid_in.total_cells,grid_out.total_cells))
        start_points = [grid_in.getCellCentroid(r) for r in range(grid_in.total_cells)]
        trans_points = func(start_points)
        
        for r in range(grid_in.total_cells):
            ts = grid_out.calcTransitions(trans_points[r])
            for t in ts:
                out_cell = grid_out.getCellNum(t[1])
                if out_cell < 0:
                    out_cell = 0
                if out_cell >= grid_out.total_cells:
                    out_cell = grid_out.total_cells - 1
                lil_mat[out_cell, r] = t[0]

        return cp_csr_matrix(lil_mat)

    # Currently, just allow 1D arrays for noise and pair it with a dimension. 
    # Later we should allow the definition of ND kernels.
    def addNoiseKernel(self, kernel_data, dimension):
        self.noise_kernels = self.noise_kernels + [(dimension, cp.asarray(kernel_data, dtype=cp.float32))]
        return len(self.noise_kernels)-1

    def applyFunction(self):
        self.pmf_out.grid.updateData(self.csr.dot(self.pmf.grid.data))

    def applyNoiseKernel(self, kernel_id):
        kernel = self.noise_kernels[kernel_id]
        
        dim_order = [i for i in range(self.pmf.grid.numDimensions()) if i != kernel[0]]
        dim_order = dim_order + [kernel[0]]
        inv_order = [a for a in range(self.pmf.grid.numDimensions())]
        d_rep = 0
        for d in range(self.pmf.grid.numDimensions()):
            if d == kernel[0]:
                inv_order[d] = self.pmf.grid.numDimensions()-1
            else:
                inv_order[d] = d_rep
                d_rep += 1

        # Transpose the grid to make the contiguous dimension the same as the desired kernel dimension.
        transposed_grid = self.pmf.grid.getTransposed(dim_order)
        transposed_res = [self.pmf.grid.res[d] for d in dim_order]
        # Apply the convolution.
        transposed_grid = cp.convolve(transposed_grid, kernel[1], mode='same')
        # Transpose back to the original dimension order.
        transposed_grid = cp.ravel(cp.transpose(cp.reshape(transposed_grid, transposed_res , order='C'), inv_order), order='C')
        # Update the next grid.
        self.pmf_out.grid.updateData(transposed_grid)