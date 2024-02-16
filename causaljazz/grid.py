import numpy as np
import cupy as cp

from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import lil_matrix

class NdGrid:
    def __init__(self, _base, _size, _res, _data=None):
        self.base = [a for a in _base]
        self.size = [a for a in _size]
        self.res = [a for a in _res]
        if _data is not None:
            self.data = cp.asarray(_data,dtype=cp.float32)
            self.data = cp.ravel(self.data, order='C')
            
        temp_res_offsets = [1]
        r = [a for a in self.res]
        r.reverse()
        self.res_offsets = self.calcResOffsets(1, temp_res_offsets, r)
        self.res_offsets.reverse()
        
        self.cell_widths = [self.size[a] / self.res[a] for a in range(self.numDimensions())]
        
        self.total_cells = 1
        for r in self.res:
            self.total_cells *= r

    def readData(self):
        return cp.asnumpy(cp.reshape(self.data, self.res, order='C'))

    def updateData(self, _data):
        self.data = cp.asarray(_data,dtype=cp.float32)
        self.data = cp.ravel(self.data, order='C')

    def getTransposed(self, _ord):
        return cp.ravel(cp.transpose(cp.reshape(self.data, self.res, order='C'), _ord), order='C')

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

        for i in range(self.numDimensions()):
            coords[i] = int(cell_num / self.res_offsets[i])
            cell_num = cell_num - (coords[i] * self.res_offsets[i])
            
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
    
    def findCellCoordsOfPointDim(self, point, dim):
        if point[dim] >= self.base[dim]:
            return int((point[dim] - self.base[dim]) / self.cell_widths[dim])
        else:
            return int((point[dim] - self.base[dim]) / self.cell_widths[dim]) - 1
        
    def getPointModuloDim(self, point, dim):
        if point[dim] >= self.base[dim]:
            p = (point[dim] - self.base[dim]) / self.cell_widths[dim]
            return (p - int(p))
        else:
            p = abs(point[dim] - self.base[dim]) / self.cell_widths[dim]
            return 1.0 - (p - int(p))
        
    def calcTransitions(self, stepped_centroid, d=0, target_coord=[], mass=1.0):
        if len(target_coord) == len(stepped_centroid):
            return [(mass, target_coord)]

        t_cell_location = self.getPointModuloDim(stepped_centroid, d)
        t_cell_lo = self.findCellCoordsOfPointDim(stepped_centroid, d)
        
        if t_cell_location < 0.5: # cell spreads to cell below
            t_cell_hi = t_cell_lo-1
            t_prop_lo = t_cell_location + 0.5
            t_prop_hi = 1-t_prop_lo
        else:
            t_cell_hi = t_cell_lo+1
            t_prop_hi = t_cell_location - 0.5
            t_prop_lo = 1 - t_prop_hi

        return self.calcTransitions(stepped_centroid, d+1, target_coord + [t_cell_lo], mass*t_prop_lo) + self.calcTransitions(stepped_centroid, d+1, target_coord + [t_cell_hi], mass*t_prop_hi)
