extern "C" {
  /*!
   * K Nearest Neighbor algorithm
   *   int k: number of nearest neighbors
   *   int ndim: number of dimensions in each point
   *   unsigned int npt: number of points
   *   float *pts_in: points in ndim dimensional Euclidian space, arranged in memory as
   *     0dim_0,     1dim_0,     2dim_0     ... ndim_0,
   *     0dim_1,     1dim_1,     2dim_1     ... ndim_1,
   *     ...
   *     0dim_npt-1, 1dim_npt-1, 2dim_npt-1 ... ndim_npt-1
   *   unsigned int* indexes_out: k x npt array of integers specifying each neighbor's index
   *     arranged from closest to furthest as. MEMORY ALLOCATED AND MANAGED BY USER!
   *     pt0_neighbor0 ... pt0_neighbork-1
   *     ...
   *     npt-1_neighbor0 ... npt-1_neighbork-1
   *   float *dists_out: k x npt array of floats specifying distances to each neighbor.
   *     MEMORY ALLOCATED AND MANAGED BY USER!
   */
  void knn_indexes( 
    const int k, 
    const int ndim, 
    const int npt, 
    const float *const pts_in, 
    int *const indexes_out );
  
  void knn_dists( 
    const int k, 
    const int ndim, 
    const int npt, 
    const float *const pts_in, 
    const float *const dists_out );

  void knn_indexes_dists( 
    const int k, 
    const int ndim, 
    const int npt, 
    const float *const pts_in, 
    const int *const indexes_out, 
    const float *const dists_out );

}
