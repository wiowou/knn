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
  T *dist2s; //distance squared sums
  T *dists;
  int* indexes;
  cub::KeyValuePair<int, T> *indexDist; //cub::ArgMin output
  void *tmp; //cub::ArgMin storage
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
  err = cudaMalloc( &d->dist2s, bytes );
  d->tmp = NULL;
  if (h->findDistances)
  {
    bytes = sizeof(T) * h->npt * h->k;
    err = cudaMalloc( &d->dists, bytes );
  }
  if (h->findIndexes)
  {
    bytes = sizeof(int) * h->npt * h->k;
    err = cudaMalloc( &d->indexes, bytes);
  }
  bytes = sizeof(cub::KeyValuePair<int, T>);
  err = cudaMalloc( &d->indexDist, bytes );
  cub::DeviceReduce::ArgMin( 
    d->tmp, 
    h->cubTmpSize, 
    d->dists, 
    d->indexDist,
    h->npt );
  return;
}

template <typename T>
void free_device_storage( Device<T> *d )
{
  if (d->pts != NULL) cudaFree( d->pts );
  if (d->dist2s != NULL) cudaFree( d->dist2s );
  if (d->dists != NULL) cudaFree( d->dists );
  if (d->indexes != NULL) cudaFree( d->indexes );
  if (d->tmp != NULL) cudaFree( d->tmp );
  if (d->indexDist != NULL) cudaFree( d->indexDist );
  return;
}

template <typename T>
void calc_dist2s( Host *h, Device<T> *d, const int ctrPtIdx, const int idim )
{
  dim3 bsize (BLOCK_DIMX,1,1);
  dim3 gsize (h->npt/bsize.x,1,1);
  gsize.x += h->npt % bsize.x;
  if (idim == 0)
  {
    calc_dist2s_dim0_kernel<T><<<gsize,bsize>>>(
      ctrPtIdx,
      d->pts,
      h->npt,
      d->dist2s );
  }
  else
  {
    calc_dist2s_dimi_kernel<T><<<gsize,bsize>>>(
      ctrPtIdx,
      d->pts + idim*h->npt,
      h->npt,
      d->dist2s );
  }
  return;
}

template <typename T>
void find_ith_neighbor( Host *h, Device<T> *d, const int ctrPtIdx, const int inn )
{
  cub::DeviceReduce::ArgMin( 
    d->tmp, 
    h->cubTmpSize, 
    d->dists, 
    d->indexDist,
    h->npt );
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
  //set host parameters
  knn::Host hostParams;
  hostParams.k = k;
  hostParams.ndim = ndim;
  hostParams.npt = npt;
  hostParams.cubTmpSize = 0;
  hostParams.findIndexes = true;
  hostParams.findDistances = false;
  //allocate device storage
  knn::Device<float> deviceParams;
  knn::allocate_device_storage<float>( &hostParams, &deviceParams);
  //copy data to device memory
  size_t bytes = sizeof(float) * ndim * npt;
  cudaError_t err = cudaMemcpy( 
    deviceParams.pts, 
    pts_in, 
    bytes, 
    cudaMemcpyHostToDevice );
  //brutishly calculate nearest neighbors
  for (int ctrPtIdx = 0; ctrPtIdx < npt; ++ctrPtIdx)
  {
    for (int idim = 0; idim < ndim; ++idim)
    {
      knn::calc_dist2s<float>( &hostParams, &deviceParams, ctrPtIdx, idim );
    }
    for (int inn = 0; inn < k; ++inn)
    {
      knn::find_ith_neighbor<float>( &hostParams, &deviceParams, ctrPtIdx, inn );
    }
  }
  //free device storage
  knn::free_device_storage<float>( &deviceParams );
  return;
}

