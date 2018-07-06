#include "../knn.h"
#include "cub/cub.cuh"
#include <limits>

namespace knn
{
#include "kernel.cuh"

// device data
template <typename T>
struct Device
{
  T *pts;
  T *dimDelt2s; // (pt_i_dim_d - pt_j_dim_d)**2
  T *dimDelt2Sums; //sum of dimDelt2s across the ndim dimensions
  T *dists;
  int* indexes;
  void *tmp; //used by cub::min
  cub::KeyValuePair<int, T> *indexDist;
  T *dist;
};
// host data
struct Host
{
  int k; //number of nearest neighbors
  int ndim; //number of dimensions
  int npt; //number of points
  size_t cubTmpSize; //bytes of storage required by cub::min
  bool findIndexes;
  bool findDistances;
};

template <typename T>
void allocate_device_storage( Host *h, Device<T> *d )
{
  size_t bytes = sizeof(T) * h->npt * h->ndim;
  cudaError_t err = cudaMalloc( &d->pts, bytes );
  bytes = sizeof(T) * h->npt;
  err = cudaMalloc( &d->dimDelt2s, bytes );
  err = cudaMalloc( &d->dimDelt2Sums, bytes );
  d->tmp = NULL;
  if (h->findDistances && !h->findIndexes)
  {
    bytes = sizeof(T) * h->npt * h->k;
    err = cudaMalloc( &d->dists, bytes );
      bytes = sizeof(T);
      err = cudaMalloc( &d->dist, bytes );
      cub::DeviceReduce::Min( 
        d->tmp, 
        h->cubTmpSize, 
        d->dists, 
        d->dist,
        h->npt );
  }
  else
  {
    bytes = sizeof(int) * h->npt * h->k;
    err = cudaMalloc( &d->indexes, bytes);
    bytes = sizeof(cub::KeyValuePair<int, T>);
    err = cudaMalloc( &d->indexDist, bytes );
    cub::DeviceReduce::ArgMin( 
      d->tmp, 
      h->cubTmpSize, 
      d->dists, 
      d->indexDist,
      h->npt );
  }
  return;
}

template <typename T>
void free_device_storage( Device<T> *d )
{
  if (d->pts != NULL) cudaFree( d->pts );
  if (d->dimDelt2s != NULL) cudaFree( d->dimDelt2s );
  if (d->dimDelt2Sums != NULL) cudaFree( d->dimDelt2Sums );
  if (d->dists != NULL) cudaFree( d->dists );
  if (d->indexes != NULL) cudaFree( d->indexes );
  if (d->tmp != NULL) cudaFree( d->tmp );
  if (d->indexDist != NULL) cudaFree( d->indexDist );
  if (d->dist != NULL) cudaFree( d->dist );
  return;
}

template <typename T>
void add_to_dimDelt2Sums( Host *h, Device<T> *d, const int ipt, const int idim )
{
  dim3 bsize (BLOCK_DIMX,1,1);
  dim3 gsize (h->npt/bsize.x,1,1);
  compute_dimDelt2s_kernel<<<gsize,bsize>>>(
    d->dimDelt2s, 
    d->dimDelt2Sums, 
    ipt, 
    idim );
  return;
}

template <typename T>
void find_ith_neighbor( Host *h, Device<T> *d, const int ipt, const int inn )
{
  if (h->findDistances  && !h->findIndexes)
  {
    cub::DeviceReduce::Min( 
      d->tmp, 
      h->cubTmpSize, 
      d->dists, 
      d->dist,
      h->npt );
  }
  else
  {
    cub::DeviceReduce::ArgMin( 
      d->tmp, 
      h->cubTmpSize, 
      d->dists, 
      d->indexDist,
      h->npt );
  }

  return;
}

void zero_dimDelt2Sums()
{
  return;
}

void set_dist_to_max( const int ipt )
{
  return;
}

} /*namespace knn*/

void knn_indexes( 
  const int k, 
  const int ndim, 
  const int npt, 
  const float *const pts_in, 
  int *const indexes_out )
{
  knn::Host hostParams;
  hostParams.k = k;
  hostParams.ndim = ndim;
  hostParams.npt = npt;
  hostParams.cubTmpSize = 0;
  hostParams.findIndexes = true;
  hostParams.findDistances = false;
  knn::Device<float> deviceParams;
  knn::allocate_device_storage<float>( &hostParams, &deviceParams);
  knn::free_device_storage<float>( &deviceParams );
  return;
}

