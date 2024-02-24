import numpy as np
from .visualiser import Visualiser
from .grid import NdGrid
import matplotlib.pyplot as plt

class pmf_cpu:
    def __init__(self, initial_distribution, _base, _cell_widths, _mass_epsilon, _vis=None, vis_dimensions=(0,1,2)):
        # dimensions of the state space
        self.dims = _base.shape[0]
        # origin point of full grid
        self.base = _base
        
        # The visualiser
        self.visualiser = _vis
        self.vis_dimensions = vis_dimensions
        
        # associated cell widths along each dimension
        self.cell_widths = _cell_widths
        
        # A Cell buffer is a dictinoary that represents all cells containing (non-zero) probability mass
        # Each cell has a coordinate in the discretised state space which is the dictionary key.
        # The dictionary value is a tuple that holds the current ammount of probability mass in that cell
        # and a list of transitions.
        # Each transition is a pair with a proportion of the original mass and the coordinates of the cell 
        # that will receive that proportion after a single time step
        #
        # cell_buffers
        self.cell_buffer = {}

        # The base coord within the discretised state space (explained below)
        self.cell_base = np.zeros(self.dims)
        # Due to numerical error, the total mass will be slightly more or less than 1.0 each iteration
        # To overcome this, we need to rescale all cell masses. The numerical error remains of course, 
        # but it can't exponentially increase or decrease and is held in check
        self.mass_summed = 1.0

        # Cells with a mass lower than mass_epsilon will be removed from the cell buffer
        # The mass will be spread among all other cells
        self.mass_epsilon = _mass_epsilon
        
        # Helper values for the visualiser
        # The visualiser needs to keep track of the extent of the cell buffer
        self.coord_extent = np.ones(self.dims) # The number of cells in each dimension direction
        # Cell colour is normalised to the maximum mass value
        self.max_mass = 1.0
        # Other dimension values for centering the buffer in the visualiser
        self.vis_coord_offset = (0,0,0)

        # The initial distribution is given in terms of the full state space (with zero mass values included)
        # but we only care about the non zero cells so we find the first non-zero cell and set that
        # as the "base" coordinate (cell_base)
        # All other non-zero cell coords are given in relation to the cell_base.
        first_cell = True
        cell_base_coords = np.zeros(self.dims)
        for idx, val in np.ndenumerate(initial_distribution):
            if val > 0.0:
                if first_cell:
                    cell_base_coords = idx
                    self.vis_coord_offset = cell_base_coords
                    self.cell_base = _base + (np.multiply(idx,_cell_widths))
                    first_cell = False
                self.cell_buffer[tuple((np.asarray(idx)-cell_base_coords).tolist())] = val
    
    # Todo : This badly needs a test!
    @classmethod
    def buildFromIndependentPmfs(cls, pmfs, _mass_epsilon=None, _vis=None, vis_dimensions=(0,1,2)):
        base = np.concatenate([p.base for p in pmfs], axis=0)
        cell_widths = np.concatenate([p.cell_widths for p in pmfs], axis=0)
        if _mass_epsilon is None:
            _mass_epsilon = np.min([p.mass_epsilon for p in pmfs])
        
        new_pmf = cls([], base, cell_widths, _mass_epsilon, _vis, vis_dimensions)
        
        new_pmf.vis_coord_offset = np.concatenate([p.vis_coord_offset for p in pmfs])
        new_pmf.cell_base = np.concatenate([p.cell_base for p in pmfs])
        
        prev_build_cell_buffer = {}
        for k,v in pmfs[0].cell_buffer.items():
            prev_build_cell_buffer[k] = v
            
        dropped_mass = 0.0
        remove_list = []
        for pmf in pmfs[1:]:
            build_cell_buffer = {}
            for k,v in prev_build_cell_buffer.items():
                for nk, nv in pmf.cell_buffer.items():
                    build_cell_buffer[tuple(np.concatenate([k, nk]))] = v * nv
                    if v*nv < _mass_epsilon:
                        dropped_mass += v*nv
                        remove_list = remove_list + [tuple(np.concatenate([k, nk]))]
            for a in remove_list:
                build_cell_buffer.pop(a, None)
            prev_build_cell_buffer = build_cell_buffer.copy()

        new_pmf.cell_buffer = build_cell_buffer.copy()
        
        return new_pmf
        
    # def copyToGpu(self, pmf, new_base=None, new_size=None):
    #     if new_base is None:
    #         new_base = self.base
            
    #     # First make sure that the target pmf base matches this one - if not, force the target to change
    #     if pmf.grid.base = new_base
            
    #     in_dist = np.zeros(pmf.grid.res)
    #     for coord, mass in self.cell_buffer.items():
    #         grid_coord = 

    #     npmf = pmf.grid.updateData(in_dist)

    def findCellCoordsOfPointDim(self, point, dim):
        if point[dim] >= self.cell_base[dim]:
            return int((point[dim] - self.cell_base[dim]) / self.cell_widths[dim])
        else:
            return int((point[dim] - self.cell_base[dim]) / self.cell_widths[dim]) - 1
        
    
    def getPointModuloDim(self, point, dim):
        if point[dim] >= self.cell_base[dim]:
            p = (point[dim] - self.cell_base[dim]) / self.cell_widths[dim]
            return (p - int(p))
        else:
            p = abs(point[dim] - self.cell_base[dim]) / self.cell_widths[dim]
            return 1.0 - (p - int(p))
    
    def findCellCoordsOfPoint(self, point):
        coords = np.zeros(self.dims)
        for c in range(self.dims):
            coords[c] = self.findCellCoordsOfPointDim(point, c)
          
        return coords
    
    def getPointModulo(self, point):
        modulo = np.zeros(self.dims)
        for c in range(self.dims):
            modulo[c] = self.getPointModuloDim(point, c)
          
        return modulo
    
    def generateInitialDistribtionFromSample(self, points):
        # assert dimension of points matches that of the grid
        mass_per_point = 1.0 / points.shape[0]
        # First cell is the location of the first point.

        self.cell_base = points[0]
        cell_base_coords = self.findCellCoordsOfPoint(points[0])
        self.vis_coord_offset = cell_base_coords
        
        
        for p in points:
            cs = tuple((np.asarray(self.findCellCoordsOfPoint(p))-cell_base_coords).tolist())
            if cs not in self.cell_buffer.keys():
                self.cell_buffer[cs] = mass_per_point
            else:
                self.cell_buffer[cs] += mass_per_point

    def calcCellCentroid(self, coords):
        centroid = [0 for a in range(self.dims)]

        for d in range(self.dims):
            centroid[d] = self.cell_base[d] + ((coords[d]+0.5)*self.cell_widths[d])

        return centroid

    def calcMarginals(self):
        vs = [{} for d in range(self.dims)]
        for c in self.cell_buffer:
            for d in range(self.dims):
                if c[d] not in vs[d]:
                    vs[d][c[d]] = self.cell_buffer[c][0]
                else:
                    vs[d][c[d]] += self.cell_buffer[c][0]

        final_vs = [[] for d in range(self.dims)]
        final_vals = [[] for d in range(self.dims)]

        for d in range(self.dims):
            for v in vs[d]:
                final_vs[d] = final_vs[d] + [self.cell_base[d] + (self.cell_widths[d]*(v))]
                final_vals[d] = final_vals[d] + [vs[d][v]]

        return final_vs, final_vals

    def calcMarginal(self, dimensions):
        vals = {}
        for cell_key, cell_val in self.cell_buffer.items():
            reduced_key = tuple([cell_key[a] for a in range(self.dims) if a in dimensions])
            if reduced_key not in vals:
                vals[reduced_key] = cell_val
            else:
                vals[reduced_key] += cell_val

        final_centroids = [np.zeros(len([a for a in dimensions])) for a in vals]
        final_coords = [k for k in vals.keys()]
        final_vals = [v[1] for v in vals.items()]
        
        i = 0
        for v_key, v_val in vals.items():
            for d in range(len([a for a in dimensions])):
                final_centroids[i][d] = self.cell_base[dimensions[d]] + (self.cell_widths[dimensions[d]]*(v_key[d]+0.5))
            i += 1

        return final_coords, final_centroids, final_vals
    
    def calcMarginalToPmf(self, dimensions, out_pmf):
        out_pmf.cell_buffer.clear()
        
        for cell_key, cell_val in self.cell_buffer.items():
            reduced_key = tuple([cell_key[a] for a in range(self.dims) if a in dimensions])
            if reduced_key not in out_pmf.cell_buffer:
                out_pmf.cell_buffer[reduced_key] = cell_val
            else:
                out_pmf.cell_buffer[reduced_key] += cell_val
                
    def drawContinued(self, grid_min_override=None, grid_max_override=None, grid_res_override=None, vis=None):
        if grid_max_override != None and grid_min_override != None:
            for d in range(len(self.vis_dimensions)):
                if grid_max_override[d] == grid_min_override[d]:
                    grid_max_override[d] += 0.005
                    grid_min_override[d] -= 0.005
        
        mcoords, mcentroids, mvals = self.calcMarginal(self.vis_dimensions)
        self.max_mass = mvals[0]
        max_coords = mcoords[0]
        min_coords = mcoords[0]
        for a in range(len(mvals)):
            #mcoords[a] = [mcoords[a][i] + self.vis_coord_offset[self.vis_dimensions[i]] for i in range(len(self.vis_dimensions))]
            max_coords = tuple([max(max_coords[i],mcoords[a][i]) for i in range(len(self.vis_dimensions))])
            min_coords = tuple([min(min_coords[i],mcoords[a][i]) for i in range(len(self.vis_dimensions))])
            self.max_mass = max(self.max_mass, mvals[a])
        self.coord_extent = tuple([max(10,(max_coords[a]-min_coords[a])+1) for a in range(len(self.vis_dimensions))])
        
        if grid_res_override != None:
            self.coord_extent = grid_res_override
            
        origin = tuple([0.0 for d in range(len(self.vis_dimensions))])
        extent = tuple([2.0 for d in range(len(self.vis_dimensions))])
        
        if grid_min_override != None and grid_max_override != None:
            grid_cell_widths = [(grid_max_override[d] - grid_min_override[d])/grid_res_override[d] for d in range(len(self.vis_dimensions))]

        for a in range(len(mvals)):
            ncoords = [mcoords[a][i]-min_coords[i] for i in range(len(self.vis_dimensions))]
            if grid_min_override != None and grid_max_override != None:
                ncoords = [int(((mcentroids[a][i] - self.cell_base[self.vis_dimensions[i]])-grid_min_override[i])/grid_cell_widths[i]) for i in range(len(self.vis_dimensions))]
            vis.drawCell(ncoords, mvals[a]/self.max_mass, origin_location=origin, max_size=extent, max_res=self.coord_extent)

    def draw(self, grid_min_override=None, grid_max_override=None, grid_res_override=None, vis=None):
        if vis is None:
            vis = self.visualiser
        
        if not vis.beginRendering():
            return
        
        self.drawContinued(grid_min_override, grid_max_override, grid_res_override, vis)

        vis.endRendering()
        
    def sample(self, num_points):
        points = []

        cmf = []        
        prob = 0.0
        for key, val in self.cell_buffer.items():
            prob += val
            cmf = cmf + [(prob, key)]

        for p in range(num_points):
            r = np.random.uniform()
            coord = cmf[-1][1]
            for c in range(len(cmf)):
                if r < cmf[c][0]:
                    coord = cmf[c][1]
                    break
                
            inner_r = [np.random.uniform() for i in range(self.dims)]
            point = [self.calcCellCentroid(coord)[a] + ((inner_r[a]-0.5)*self.cell_widths[a]) for a in range(self.dims)]
            points = points + [point]
            
        return points 


class pmf_gpu:
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