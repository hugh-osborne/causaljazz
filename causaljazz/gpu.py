import numpy as np
from .visualiser import Visualiser
from .grid import NdGrid
import matplotlib.pyplot as plt

class pmf:
    def __init__(self, initial_distribution, _base, _size, _res, _vis=None, vis_dimensions=(0,1,2)):
        # The visualiser
        self.visualiser = _vis
        self.vis_dimensions = vis_dimensions
        
        self.grid = NdGrid(_base, _size, _res, initial_distribution)
        
        self.vis_coords = None
        self.vis_centroids = None

    # Do CPU marginal calculation for now. Slow because we need to move the full distribution off card
    def calcMarginals(self):
        final_vals = []
        final_vs = []
        for d in range(self.grid.numDimensions()):
            other_dims = tuple([i for i in range(self.grid.numDimensions()) if i != d])
            final_vals = final_vals + [np.sum(self.grid.readData(), other_dims)]
            final_vs = final_vs + [np.linspace(self.grid.base[d],self.grid.base[d] + self.grid.size[d],self.grid.res[d])]

        return final_vs, final_vals

    def calcMarginal(self, dimensions):
        reduced_grid = NdGrid([self.grid.base[d] for d in dimensions], [self.grid.size[d] for d in dimensions], [self.grid.res[d] for d in dimensions])
        other_dims = tuple([i for i in range(self.grid.numDimensions()) if i not in dimensions])
        final_vals = np.ravel(np.sum(self.grid.readData(), other_dims))
        final_coords = [reduced_grid.getCellCoords(c) for c in range(reduced_grid.total_cells)]
        final_centroids = [reduced_grid.getCellCentroid(c) for c in range(reduced_grid.total_cells)]
        return final_coords, final_centroids, final_vals
    
    # Calculating the coords and centroids in calcMarginal is slow and, for the visualiser, these values stay constant (only the mass values change)
    def calcMarginalForVis(self):
        if self.vis_coords is None and self.vis_centroids is None:
            reduced_grid = NdGrid([self.grid.base[d] for d in self.vis_dimensions], [self.grid.size[d] for d in self.vis_dimensions], [self.grid.res[d] for d in self.vis_dimensions])
            self.vis_coords = [reduced_grid.getCellCoords(c) for c in range(reduced_grid.total_cells)]
            self.vis_centroids = [reduced_grid.getCellCentroid(c) for c in range(reduced_grid.total_cells)]
            
        other_dims = tuple([i for i in range(self.grid.numDimensions()) if i not in self.vis_dimensions])
        final_vals = np.ravel(np.sum(self.grid.readData(), other_dims))
        
        return self.vis_coords, self.vis_centroids, final_vals

    def draw(self):
        if not self.visualiser.beginRendering():
            return
        
        mcoords, mcentroids, mvals = self.calcMarginalForVis()
        
        self.max_mass = 0.0
        for m in mvals:
            self.max_mass = max(self.max_mass, m)
        
        for a in range(len(mvals)):
            self.visualiser.drawCell(mcoords[a], mvals[a] / self.max_mass, origin_location=tuple([0.0 for d in range(len(self.vis_dimensions))]), max_size=tuple([2.0 for d in range(len(self.vis_dimensions))]), max_res=[self.grid.res[d] for d in self.vis_dimensions])
        
        self.visualiser.endRendering()
        
class transition:
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