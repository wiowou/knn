extern "C" {
  /*!
   * K Nearest Neighbor algorithm
   *   int k: number of nearest neighbors
   *   int ndim: number of dimensions in each point
   *   unsigned int npt: number of points
   *   float *pts_in: points in ndim dimensional Euclidian space, arranged in memory as
   *     1dim_pt1,   1dim_pt2,   1dim_pt3   ... 1dim_ptn,
   *     2dim_pt1,   2dim_pt2,   2dim_pt3   ... 2dim_ptn,
   *     ...
   *     ndim_pt1,   ndim_pt2,   ndim_pt3   ... ndim_ptn,
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
    float *const dists_out );

  void knn_gpu( 
    const int k, 
    const int ndim, 
    const int npt, 
    const float *const pts_in, 
    int *const indexes_out, 
    float *const dists_out );

}
