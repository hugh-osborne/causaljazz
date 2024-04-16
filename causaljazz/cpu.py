import numpy as np

class pmf:
    def __init__(self, initial_distribution, _origin, _cell_widths, _mass_epsilon, _vis=None, vis_dimensions=(0,1,2)):
        # dimensions of the state space
        self.dims = _cell_widths.shape[0]
        self.origin = _origin
        
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


        # The initial distribution is given in terms of the full state space (with zero mass values included)
        # but we only care about the non zero cells so we find the first non-zero cell and set that
        # as the "base" coordinate (cell_base)
        # All other non-zero cell coords are given in relation to the cell_base.
        for idx, val in np.ndenumerate(initial_distribution):
            if val > 0.0:
                self.cell_buffer[tuple((np.asarray(idx)).tolist())] = val
    
    # Todo : This badly needs a test!
    @classmethod
    def buildFromIndependentPmfs(cls, pmfs, _mass_epsilon=None, _vis=None, vis_dimensions=(0,1,2)):
        cell_widths = np.concatenate([p.cell_widths for p in pmfs], axis=0)
        origin = np.concatenate([p.origin for p in pmfs], axis=0)
        
        if _mass_epsilon is None:
            _mass_epsilon = np.min([p.mass_epsilon for p in pmfs])
            
        new_pmf = cls([], origin, cell_widths, _mass_epsilon, _vis, vis_dimensions)
        
        prev_build_cell_buffer = {}
        for k,v in pmfs[0].cell_buffer.items():
            prev_build_cell_buffer[k] = v
            
        dropped_mass = 0.0
        for pmf in pmfs[1:]:
            build_cell_buffer = {}
            for k,v in prev_build_cell_buffer.items():
                for nk, nv in pmf.cell_buffer.items():
                    if v*nv < _mass_epsilon:
                        dropped_mass += v*nv
                    else:
                        build_cell_buffer[tuple(np.concatenate([k, nk], axis=0).tolist())] = v * nv
            
            prev_build_cell_buffer = build_cell_buffer.copy()
            
        for k in build_cell_buffer.keys():
            build_cell_buffer[k] /= (1.0 - dropped_mass) 

        new_pmf.cell_buffer = build_cell_buffer.copy()
        
        return new_pmf
    
    @classmethod
    def duplicate(cls, source):
        new_pmf = cls([], source.origin.copy(), source.cell_widths.copy(), source.mass_epsilon, source.visualiser, source.vis_dimensions)
        new_pmf.cell_buffer = source.cell_buffer.copy()
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
        if point[dim] >= 0:
            return int((point[dim] - self.origin[dim]) / self.cell_widths[dim])
        else:
            return int((point[dim] - self.origin[dim]) / self.cell_widths[dim]) - 1
        
    
    def getPointModuloDim(self, point, dim):
        if point[dim] - self.origin[dim] >= 0:
            p = (point[dim] - self.origin[dim]) / self.cell_widths[dim]
            return (p - int(p))
        else:
            p = abs((point[dim] - self.origin[dim])) / self.cell_widths[dim]
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
        self.origin = np.zeros(self.dims)
        
        for p in points:
            cs = tuple((np.asarray(self.findCellCoordsOfPoint(p))).tolist())
            if cs not in self.cell_buffer.keys():
                self.cell_buffer[cs] = mass_per_point
            else:
                self.cell_buffer[cs] += mass_per_point

    def calcCellCentroid(self, coords):
        centroid = [0 for a in range(self.dims)]

        for d in range(self.dims):
            centroid[d] = self.origin[d] + ((coords[d]+0.5)*self.cell_widths[d])

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
                final_vs[d] = final_vs[d] + [self.origin[d] + self.cell_widths[d]*(v)]
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
                final_centroids[i][d] = self.origin[d] + (self.cell_widths[dimensions[d]]*(v_key[d]+0.5))
            i += 1

        return final_coords, final_centroids, final_vals
    
    def calcMarginalToPmf(self, dimensions):
        out_pmf = pmf([], self.origin[dimensions], self.cell_widths[dimensions], self.mass_epsilon, self.visualiser, self.vis_dimensions)
        
        # Set out_pmf cell base to this
        out_pmf.cell_widths = np.array([self.cell_widths[a] for a in dimensions])
        
        for cell_key, cell_val in self.cell_buffer.items():
            reduced_key = tuple([cell_key[a] for a in range(self.dims) if a in dimensions])
            if reduced_key not in out_pmf.cell_buffer:
                out_pmf.cell_buffer[reduced_key] = cell_val
            else:
                out_pmf.cell_buffer[reduced_key] += cell_val
                
        return out_pmf
                
    def drawContinued(self, grid_min_override=None, grid_max_override=None, grid_res_override=None, vis=None, vis_dimensions=None):
        if grid_max_override != None and grid_min_override != None:
            for d in range(len(vis_dimensions)):
                if grid_max_override[d] == grid_min_override[d]:
                    grid_max_override[d] += 0.005
                    grid_min_override[d] -= 0.005
        
        mcoords, mcentroids, mvals = self.calcMarginal(vis_dimensions)
        self.max_mass = mvals[0]
        max_coords = mcoords[0]
        min_coords = mcoords[0]
        for a in range(len(mvals)):
            #mcoords[a] = [mcoords[a][i] + self.vis_coord_offset[self.vis_dimensions[i]] for i in range(len(self.vis_dimensions))]
            max_coords = tuple([max(max_coords[i],mcoords[a][i]) for i in range(len(vis_dimensions))])
            min_coords = tuple([min(min_coords[i],mcoords[a][i]) for i in range(len(vis_dimensions))])
            self.max_mass = max(self.max_mass, mvals[a])
        self.coord_extent = tuple([max(10,(max_coords[a]-min_coords[a])+1) for a in range(len(vis_dimensions))])
        
        if grid_res_override != None:
            self.coord_extent = grid_res_override
            
        origin = tuple([0.0 for d in range(len(vis_dimensions))])
        extent = tuple([2.0 for d in range(len(vis_dimensions))])
        
        if grid_min_override != None and grid_max_override != None:
            grid_cell_widths = [(grid_max_override[d] - grid_min_override[d])/grid_res_override[d] for d in range(len(vis_dimensions))]

        for a in range(len(mvals)):
            ncoords = [mcoords[a][i]-min_coords[i] for i in range(len(vis_dimensions))]
            if grid_min_override != None and grid_max_override != None:
                ncoords = [int((mcentroids[a][i]-grid_min_override[i])/grid_cell_widths[i]) for i in range(len(vis_dimensions))]
                
            multiplier = 100000
            multiplied = multiplier*mvals[a]
            if multiplied > 1.0:
                mass_value = np.log(multiplied)/np.log(multiplier)
                if mass_value < 0:
                    mass_value = 0.0
                if mass_value > 1.0:
                    mass_value = 1.0
                vis.drawCell(ncoords, mass_value, origin_location=origin, max_size=extent, max_res=self.coord_extent)

    def draw(self, grid_min_override=None, grid_max_override=None, grid_res_override=None, vis=None, vis_dimensions=None):
        if vis is None:
            vis = self.visualiser

        if vis_dimensions is None:
            vis_dimensions = self.vis_dimensions
        
        if not vis.beginRendering():
            return
        
        self.drawContinued(grid_min_override, grid_max_override, grid_res_override, vis, vis_dimensions)

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
    
class transition:
    def __init__(self, _func, num_input_dimensions, recursive_dimension=None, transition_epsilon=0.0):
        
        # Noise kernels are stored like the values in the cell buffers: a pair.
        # The first value is 1.0 : The full amount of mass in each cell is spread according to the kernel
        # The second value is a list of transitions where the cell coords are relative to the cell to which the kernel will be applied
        self.noise_kernels = []
        
        # The deterministic function upon which the transiions are based
        self.func = _func
        self.input_pmf_dimensions = [a for a in range(num_input_dimensions)]
        self.input_output_mapping = None
        self.recursive_dimension = recursive_dimension
        self.transition_epsilon = transition_epsilon
        
        # cell_buffer transition values for pmf
        self.transition_buffer = {}

    # Add a noise kernel
    def addNoiseKernel(self, kernel_pmf, centre_coord):
        kernel_transitions = []
        for coord, val in kernel_pmf.cell_buffer.items():
            kernel_transitions = kernel_transitions + [(val,tuple((np.array(coord) - np.array(centre_coord)).tolist()))]
        self.noise_kernels = self.noise_kernels + [kernel_transitions]
        return len(self.noise_kernels) - 1
    
    def calcTransitionsForLastDimension(self, out_pmf, stepped_centroid):
        t_cell_location = out_pmf.getPointModuloDim(stepped_centroid, -1)
        t_cell_lo = out_pmf.findCellCoordsOfPointDim(stepped_centroid, -1)
        
        if t_cell_location < 0.5: # cell spreads to cell below
            t_cell_hi = t_cell_lo-1
            t_prop_lo = t_cell_location + 0.5
            t_prop_hi = 1-t_prop_lo
        else:
            t_cell_hi = t_cell_lo+1
            t_prop_hi = t_cell_location - 0.5
            t_prop_lo = 1 - t_prop_hi
            
        return [(t_prop_lo, [t_cell_lo]), (t_prop_hi, [t_cell_hi])]
    
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
        # If there is a recursive dimension (for example, this function adds noise to a time dependent variable)
        # For small amounts of mass that will end up getting culled, to avoid mass being incorrectly spread across the whole distribution,
        # pass it "back" to the original cell
        if self.recursive_dimension is not None and mass*transition[0] < self.transition_epsilon:
            if self.input_output_mapping is not None:
                t[-1] = t[self.input_output_mapping[self.input_pmf_dimensions[self.recursive_dimension]]]
            else:
                t[-1] = t[self.input_pmf_dimensions[self.recursive_dimension]]
                
        if tuple(t) not in new_cell_dict:
            new_cell_dict[tuple(t)] = 0.0
        new_cell_dict[tuple(t)] += mass*transition[0]
        
    def changeInputDimensions(self, new_dims):
        self.input_pmf_dimensions = [a for a in new_dims]
        
    def changeInputOutputMapping(self, new_mapping):
        self.input_output_mapping = new_mapping.copy()
        
    def checkTransitionsMatchBuffer(self, in_pmf, out_pmf):
        new_coords = []
        centroids = []
        
        check_coords = set(map(tuple,[[coord[c] for c in self.input_pmf_dimensions] for coord in in_pmf.cell_buffer.keys()])) # maybe don't need the map here...

        for coord in check_coords:
            if coord not in self.transition_buffer.keys():
                centroid = [0 for a in range(len(coord))]
                for d in range(len(coord)):
                    centroid[d] = in_pmf.origin[self.input_pmf_dimensions[d]] + ((coord[d]+0.5)*in_pmf.cell_widths[self.input_pmf_dimensions[d]])
                new_coords = new_coords + [coord]
                centroids += [centroid]
        
        if len(centroids) > 0:
            result_values = self.func(centroids)

        for c in range(len(new_coords)):
            self.transition_buffer[new_coords[c]] = self.calcTransitionsForLastDimension(out_pmf, result_values[c])
            
        
    def applyFunction(self, in_pmf, out_pmf):
        self.checkTransitionsMatchBuffer(in_pmf, out_pmf)
        
        # Set the next buffer mass values to 0
        for a in out_pmf.cell_buffer.keys():
            out_pmf.cell_buffer[a] = 0.0
        
        # Fill the pmf_out cell buffer with the updated mass values
        mass_summed = 0.0
        for coord in in_pmf.cell_buffer.keys():
            if in_pmf.cell_buffer[coord] < in_pmf.mass_epsilon:
                continue
            mass_summed += in_pmf.cell_buffer[coord]
            
            for ts in self.transition_buffer[tuple([coord[a] for a in self.input_pmf_dimensions])]:
                if self.input_output_mapping is not None:
                    self.updateCell(out_pmf.cell_buffer, (ts[0], tuple([coord[self.input_output_mapping[c]] for c in self.input_output_mapping.keys()]) + tuple(ts[1])), in_pmf.cell_buffer[coord])
                else:
                    self.updateCell(out_pmf.cell_buffer, (ts[0], coord + tuple(ts[1])), in_pmf.cell_buffer[coord])
                
        mass_sum = 0.0
        for coord in out_pmf.cell_buffer:
            out_pmf.cell_buffer[coord] /= mass_summed
            mass_sum += out_pmf.cell_buffer[coord]


    def applyNoiseKernel(self, kernel_id, in_pmf, out_pmf):
        kernel = self.noise_kernels[kernel_id]
        
        # Set the next buffer mass values to 0
        for a in out_pmf.cell_buffer.keys():
            out_pmf.cell_buffer[a] = 0.0

        # Apply the kernel
        for coord in in_pmf.cell_buffer:
            for ts in kernel:
                relative_ts_coord = [a for a in ts[1]]
                for d in range(len(relative_ts_coord)):
                    relative_ts_coord[d] = coord[d] + relative_ts_coord[d]
                relative_ts = [ts[0],relative_ts_coord]
                if tuple(relative_ts_coord) not in out_pmf.cell_buffer.keys():
                    out_pmf.cell_buffer[tuple(relative_ts_coord)] = 0.0
                self.updateCell(out_pmf.cell_buffer, relative_ts, in_pmf.cell_buffer[coord])

        remove = []
        mass_summed = 1.0
        for a in out_pmf.cell_buffer.keys():
            if out_pmf.cell_buffer[a] < out_pmf.mass_epsilon:
                remove = remove + [a]
                mass_summed -= out_pmf.cell_buffer[a]

        for a in remove:
            out_pmf.cell_buffer.pop(a, None)
                
        for coord in out_pmf.cell_buffer:
            out_pmf.cell_buffer[coord] /= mass_summed