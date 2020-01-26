import ctypes
import numpy as np

def knn(k, pts, libLocation=None):
    if libLocation is not None:
        lib = ctypes.CDLL(libLocation)
    else:
        lib = ctypes.CDLL('libknn.so')
    ndim,npt = np.shape(pts)
    indexes = np.empty(shape=[npt,k], dtype=np.int32, order='C')
    dists = np.empty(shape=[npt,k], dtype=np.float32, order='C')
    cindexes = indexes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    cdists = dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if pts.dtype.name == 'float32':
        cpts = pts.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lib.knn_float(k,ndim,npt,cpts,cindexes,cdists)
    elif pts.dtype.name == 'float':
        cpts = pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.knn_float(k,ndim,npt,cpts,cindexes,cdists)
    return indexes,dists