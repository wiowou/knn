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
};

template <typename T>
void allocate_device_storage( Host *h, Device<T> *d )
{
  size_t bytes = sizeof(T) * h->npt * h->ndim;
  cudaError_t err = cudaMalloc( &d->pts, bytes );
  bytes = sizeof(T) * h->npt;
  err = cudaMalloc( &d->dist2s, bytes );
  bytes = sizeof(T) * h->npt * h->k;
  err = cudaMalloc( &d->dists, bytes );
  bytes = sizeof(int) * h->npt * h->k;
  err = cudaMalloc( &d->indexes, bytes);
  bytes = sizeof(cub::KeyValuePair<int, T>);
  err = cudaMalloc( &d->indexDist, bytes );
  d->tmp = NULL;
  bytes = 0;
  cub::DeviceReduce::ArgMin( 
    d->tmp, 
    bytes, 
    d->dists, 
    d->indexDist,
    h->npt );
  err = cudaMalloc( &d->tmp, bytes );
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
void calc_dist2s( 
  const Host *h, 
  const int ctrPtIdx, 
  const int idim, 
  Device<T> *d )
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
void find_ith_neighbor( 
  Host *h, 
  const int ctrPtIdx, 
  const int inn,
  const T maxValue,
  Device<T> *d )
{
  cub::DeviceReduce::ArgMin( 
    d->tmp, 
    h->cubTmpSize, 
    d->dist2s, 
    d->indexDist,
    h->npt );
  //set distance of the ith n.n. to maxValue so next one can be found
  //copy to device output arrays
  copy_init_kernel<T><<<1,1>>>(
    maxValue,
    d->dist2s,
    d->indexDist,
    d->indexes + ctrPtIdx*h->k + inn,
    d->dists + ctrPtIdx*h->k + inn );
  return;
}

} /*namespace knn*/

void knn_gpu( 
  const int k, 
  const int ndim, 
  const int npt, 
  const float *const pts_in, 
  int *const indexes_out,
  float *const dists_out )
{
  //set host parameters
  knn::Host hostParams;
  hostParams.k = k;
  hostParams.ndim = ndim;
  hostParams.npt = npt;
  hostParams.cubTmpSize = 0;
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
      knn::calc_dist2s<float>( 
        &hostParams, 
        ctrPtIdx, 
        idim, 
        &deviceParams ); 
    }
    const float maxValue = std::numeric_limits<float>::max();
    //distance to itself will be min if not set to maxValue
    err = cudaMemcpy( 
      deviceParams.dist2s+ctrPtIdx, 
      &maxValue, 
      sizeof(float), 
      cudaMemcpyHostToDevice );
    for (int inn = 0; inn < k; ++inn)
    {
      knn::find_ith_neighbor<float>( 
        &hostParams, 
        ctrPtIdx, 
        inn,
        maxValue,
        &deviceParams ); 
    }
  }
  //copy data back to host memory
  bytes = sizeof(int) * npt * k;
  err = cudaMemcpy(
    indexes_out,
    deviceParams.indexes,
    bytes,
    cudaMemcpyDeviceToHost );
  bytes = sizeof(float) * npt * k;
  err = cudaMemcpy(
    dists_out,
    deviceParams.dists,
    bytes,
    cudaMemcpyDeviceToHost );
  //free device storage
  knn::free_device_storage<float>( &deviceParams );
  return;
}

