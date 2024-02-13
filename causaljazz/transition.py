import numpy as np
import cupy as cp

from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import lil_matrix

class transition_cpu:
    def __init__(self, pmf, _func):
        self.pmf = pmf
        
        # Noise kernels are stored like the values in the cell buffers: a pair.
        # The first value is 1.0 : The full amount of mass in each cell is spread according to the kernel
        # The second value is a list of transitions where the cell coords are relative to the cell to which the kernel will be applied
        self.noise_kernels = []
        
        # The deterministic function upon which the transiions are based
        self.func = _func
        
        # cell_buffer transition values for pmf
        self.transition_buffer = {}

        # Calculate the transitions for each cell in the buffer (non-zero cells from the initial distribution)
        centroids = []
        for coord in self.pmf.cell_buffers[self.pmf.current_buffer]:
            centroid = [0 for a in range(self.pmf.dims)]

            for d in range(self.pmf.dims):
                centroid[d] = self.pmf.cell_base[d] + ((coord[d]+0.5)*self.pmf.cell_widths[d])
                
            centroids = centroids + [centroid]
            
        if len(centroids) > 0:
            shifted_centroids = self.func(centroids)

        centroid_count = 0
        for coord in self.pmf.cell_buffers[self.pmf.current_buffer]:
            self.transition_buffer[coord] = self.calcTransitions(centroids[centroid_count], shifted_centroids[centroid_count], coord)
            centroid_count += 1

    # Add a noise kernel
    def addNoiseKernel(self, kernel, dimension):
        kernel_transitions = {}
        cs = tuple(np.zeros(self.pmf.dims).tolist())
        kernel_transitions[cs] = []
        for c in range(len(kernel)):
            if kernel[c] > 0.0:
                kernel_transitions[cs] = kernel_transitions[cs] + [(kernel[c], [c-int(len(kernel)/2) if d == dimension else 0 for d in range(self.pmf.dims)])]
        self.noise_kernels = self.noise_kernels + [[1.0,kernel_transitions[cs]]]

    # Calculate the transitions for each cell based on the given centroid and centroid after one application of self.func (stepped_centroid)
    # The centroid calculation is not done here because sometimes it's better to batch function application (for example with an ANN)
    def calcTransitions(self, centroid, stepped_centroid, coord, d=0, target_coord=[], mass=1.0):
        if len(target_coord) == len(coord):
            return [(mass, target_coord)]

        diff = stepped_centroid[d] - centroid[d]
        cell_lo = coord[d] + int(diff / self.pmf.cell_widths[d])
        cell_hi = cell_lo + 1
        prop_lo = 0.0
        if diff < 0.0: # actually, diff is negative so cell_lo is the upper cell
            cell_hi = cell_lo - 1
            prop_lo = ((diff % self.pmf.cell_widths[d]) / self.pmf.cell_widths[d])
        else:
            prop_lo = 1.0 - ((diff % self.pmf.cell_widths[d]) / self.pmf.cell_widths[d])
        prop_hi = 1.0 - prop_lo
    
        return self.calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_lo], mass*prop_lo) + self.calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_hi], mass*prop_hi)

    # Calculate the centroid of a cell if it doesn't already exist in new_cell_dict (the next cell buffer)
    def calculateCellCentroidForUpdate(self, relative, new_cell_dict, transition, coord):
        t = [a for a in transition[1]]
        if relative:
            for d in range(len(coord)):
                t[d] = coord[d] + t[d]

        if tuple(t) not in new_cell_dict.keys():
            centroid = [0 for a in range(len(coord))]

            for d in range(len(coord)):
                centroid[d] = self.pmf.cell_base[d] + ((t[d]+0.5)*self.pmf.cell_widths[d])
                
            return t,centroid
        
        return None,None

    # Update the mass of a cell
    def updateCell(self, relative, new_cell_dict, transition,  coord, mass):
        t = [a for a in transition[1]]
        if relative:
            for d in range(len(coord)):
                t[d] = coord[d] + t[d]
    
        new_cell_dict[tuple(t)] += mass*transition[0]
        
    def checkTransitionsMatchBuffer(self):
        new_coords = []
        centroids = [] 
        for coord in self.pmf.cell_buffers[self.pmf.current_buffer].keys():
            if coord not in self.transition_buffer.keys(): # the pmf has been altered elsewhere so we need to calculate the transitions for any new cells
                centroid = [0 for a in range(len(coord))]
                for d in range(len(coord)):
                    centroid[d] = self.pmf.cell_base[d] + ((coord[d]+0.5)*self.pmf.cell_widths[d])
                new_coords = new_coords + [coord]
                centroids = centroids + [centroid]
        
        if len(centroids) > 0:
            shifted_centroids = self.func(centroids)
            
        for c in range(len(new_coords)):
            self.transition_buffer[tuple(new_coords[c])] = self.calcTransitions(centroids[c], shifted_centroids[c], new_coords[c])
        
    def applyFunction(self):
        self.checkTransitionsMatchBuffer()
        
        # Set the next buffer mass values to 0
        for a in self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2].keys():
            self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2][a] = 0.0
            
        # Calculate the centroids of all *new* cells which are to be added to the next buffer
        new_coords = []
        centroids = []        
        for coord in self.pmf.cell_buffers[self.pmf.current_buffer].keys():
            for ts in self.transition_buffer[coord]:
                _coord,centroid = self.calculateCellCentroidForUpdate(False, self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2], ts, coord)
                if centroid is not None and _coord is not None:
                    new_coords = new_coords + [_coord]
                    centroids = centroids + [centroid]
                
        # Batch apply the function to the new centroids
        if len(centroids) > 0:
            shifted_centroids = self.func(centroids)
        
        # Build the transitions for each new cell
        for c in range(len(new_coords)):
            self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2][tuple(new_coords[c])] = 0.0
            self.transition_buffer[tuple(new_coords[c])] = self.calcTransitions(centroids[c], shifted_centroids[c], new_coords[c])
    
        # Fill the next buffer with the updated mass values
        for coord in self.pmf.cell_buffers[self.pmf.current_buffer]:
            for ts in self.transition_buffer[coord]:
                self.updateCell(False, self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2], ts, coord, self.pmf.cell_buffers[self.pmf.current_buffer][coord])

        # Remove any cells with a small amount of mass and keep a total to spread back to the remaining population
        remove = []
        mass_summed = 1.0
        for a in self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2].keys():
            if self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2][a] < self.pmf.mass_epsilon:
                remove = remove + [a]
                mass_summed -= self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2][a]

        for a in remove:
            self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2].pop(a, None)
            self.transition_buffer.pop(a, None)
            
        for coord in self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2]:
            self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2][coord] /= mass_summed

        # swap the buffer counter
        self.pmf.current_buffer = (self.pmf.current_buffer+1)%2

    def applyNoiseKernels(self):
        self.checkTransitionsMatchBuffer()
        
        for kernel in self.noise_kernels:
            # Set the next buffer mass values to 0
            for a in self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2].keys():
                self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2][a] = 0.0
                
            # Calculate the centroids of all *new* cells which are to be added to the next buffer
            new_coords = []
            centroids = []        
            for coord in self.pmf.cell_buffers[self.pmf.current_buffer].keys():
                for ts in kernel[1]:
                    _coord,centroid = self.calculateCellCentroidForUpdate(True, self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2], ts, coord)
                    if centroid is not None and _coord is not None:
                        new_coords = new_coords + [_coord]
                        centroids = centroids + [centroid]
                
            # Batch apply the function to the new centroids
            if len(centroids) > 0:
                shifted_centroids = self.func(centroids)
        
            # Build the transitions for each new cell
            for c in range(len(new_coords)):
                self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2][tuple(new_coords[c])] = 0.0
                self.transition_buffer[tuple(new_coords[c])] = self.calcTransitions(centroids[c], shifted_centroids[c], new_coords[c])
    
            for coord in self.pmf.cell_buffers[self.pmf.current_buffer]:
                for ts in kernel[1]:
                    self.updateCell(True, self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2], ts, coord, self.pmf.cell_buffers[self.pmf.current_buffer][coord])

            remove = []
            mass_summed = 1.0
            for a in self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2].keys():
                if self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2][a] < self.pmf.mass_epsilon:
                    remove = remove + [a]
                    mass_summed -= self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2][a]

            for a in remove:
                self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2].pop(a, None)
                self.transition_buffer.pop(a, None)
                
            for coord in self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2]:
                self.pmf.cell_buffers[(self.pmf.current_buffer+1)%2][coord] /= mass_summed

            self.pmf.current_buffer = (self.pmf.current_buffer+1)%2
            

class transition_gpu:
    def __init__(self, pmf, _func):
        self.pmf = pmf
        self.noise_kernels = []
        self.func = _func

        self.csr = self.generateConditionalTransitionCSR(self.pmf.grids[0], _func, self.pmf.grids[1])

    def generateConditionalTransitionCSR(self, grid_in, func, grid_out):
        lil_mat = lil_matrix((grid_in.total_cells,grid_out.total_cells))
        start_points = [grid_in.getCellCentroid(r) for r in range(grid_in.total_cells)]
        trans_points = func(start_points)
        
        for r in range(grid_in.total_cells):
            ts = grid_in.calcTransitions(start_points[r], trans_points[r], grid_in.getCellCoords(r))
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
        self.pmf.grids[(self.pmf.current_grid+1)%2].updateData(self.csr.dot(self.pmf.grids[self.pmf.current_grid].data))
        self.pmf.current_grid = (self.pmf.current_grid+1)%2

    def applyNoiseKernels(self):
        for kernel in self.noise_kernels:
            dim_order = [i for i in range(self.pmf.grids[self.pmf.current_grid].numDimensions()) if i != kernel[0]]
            dim_order = dim_order + [kernel[0]]
            inv_order = [a for a in range(self.pmf.grids[self.pmf.current_grid].numDimensions())]
            d_rep = 0
            for d in range(self.pmf.grids[self.pmf.current_grid].numDimensions()):
                if d == kernel[0]:
                    inv_order[d] = self.pmf.grids[self.pmf.current_grid].numDimensions()-1
                else:
                    inv_order[d] = d_rep
                    d_rep += 1

            # Transpose the grid to make the contiguous dimension the same as the desired kernel dimension.
            transposed_grid = self.pmf.grids[self.pmf.current_grid].getTransposed(dim_order)
            transposed_res = [self.pmf.grids[self.pmf.current_grid].res[d] for d in dim_order]
            # Apply the convolution.
            transposed_grid = cp.convolve(transposed_grid, kernel[1], mode='same')
            # Transpose back to the original dimension order.
            transposed_grid = cp.ravel(cp.transpose(cp.reshape(transposed_grid, transposed_res , order='C'), inv_order), order='C')
            # Update the next grid.
            self.pmf.grids[(self.pmf.current_grid+1)%2].updateData(transposed_grid)
            self.pmf.current_grid = (self.pmf.current_grid+1)%2