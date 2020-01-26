import numpy as np
from knn.core import knn

if __name__ == '__main__':
    pts = np.empty(shape=[2,15], dtype=np.float32)
    pts[:,0] = [-4,1]
    pts[:,1] = [1,3]
    pts[:,2] = [-1,1]
    pts[:,3] = [1,-4]
    pts[:,4] = [-3,-6]
    pts[:,5] = [-1,-2]
    pts[:,6] = [4,2]
    pts[:,7] = [2,1]
    pts[:,8] = [-2,-1]
    pts[:,9] = [-2,3]
    pts[:,10] = [1,1]
    pts[:,11] = [2,3]
    pts[:,12] = [3,-2]
    pts[:,13] = [2,-3]
    pts[:,14] = [-4,-3]
    indexes,dists = knn(2,pts,'../../debug/libknn.so')
    dum = 5
