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
  float *dists;
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
  size_t cubTmpSize; //nbytes of storage required by cub::min
};

template <typename T>
void allocate_device_storage( Host *h, Device<T> *d )
{
  size_t nbytes = sizeof(T) * h->npt * h->ndim;
  cudaError_t err = cudaMalloc( &d->pts, nbytes );
  nbytes = sizeof(T) * h->npt;
  err = cudaMalloc( &d->dist2s, nbytes );
  nbytes = sizeof(float) * h->npt * h->k;
  err = cudaMalloc( &d->dists, nbytes );
  nbytes = sizeof(int) * h->npt * h->k;
  err = cudaMalloc( &d->indexes, nbytes);
  nbytes = sizeof(cub::KeyValuePair<int, T>);
  err = cudaMalloc( &d->indexDist, nbytes );
  d->tmp = NULL;
  nbytes = 0;
  cub::DeviceReduce::ArgMin( 
    d->tmp, 
    nbytes, 
    d->dist2s, 
    d->indexDist,
    h->npt );
  err = cudaMalloc( &d->tmp, nbytes );
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
  Device<T> *d )
{
  const T maxValue = std::numeric_limits<T>::max();
  //don't want self as nearest nbor. Set distance to self to maxValue
  if (inn == 0)
  {
    cudaError_t err = cudaMemcpy( 
      d->dist2s+ctrPtIdx, 
      &maxValue, 
      sizeof(T), 
      cudaMemcpyHostToDevice );
  }
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

template <typename T>
void knn( 
  const int k, 
  const int ndim, 
  const int npt, 
  const T *const pts_in, 
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
  knn::Device<T> deviceParams;
  knn::allocate_device_storage<T>( &hostParams, &deviceParams);
  //copy input data to device memory
  size_t nbytes = sizeof(T) * ndim * npt;
  cudaError_t err = cudaMemcpy( 
    deviceParams.pts, 
    pts_in, 
    nbytes, 
    cudaMemcpyHostToDevice );
  //brutishly calculate nearest neighbors
  for (int ctrPtIdx = 0; ctrPtIdx < npt; ++ctrPtIdx)
  {
    for (int idim = 0; idim < ndim; ++idim)
    {
      knn::calc_dist2s<T>( 
        &hostParams, 
        ctrPtIdx, 
        idim, 
        &deviceParams ); 
    }
    for (int inn = 0; inn < k; ++inn)
    {
      knn::find_ith_neighbor<T>( 
        &hostParams, 
        ctrPtIdx, 
        inn,
        &deviceParams ); 
    }
  }
  //copy result data to host memory
  nbytes = sizeof(int) * npt * k;
  err = cudaMemcpy(
    indexes_out,
    deviceParams.indexes,
    nbytes,
    cudaMemcpyDeviceToHost );
  nbytes = sizeof(float) * npt * k;
  err = cudaMemcpy(
    dists_out,
    deviceParams.dists,
    nbytes,
    cudaMemcpyDeviceToHost );
  //free device storage
  knn::free_device_storage<T>( &deviceParams );
  return;
}

} /*namespace knn*/

void knn_float( 
  const int k, 
  const int ndim, 
  const int npt, 
  const float *const pts_in, 
  int *const indexes_out,
  float *const dists_out )
{
  knn::knn<float>( k, ndim, npt, pts_in, indexes_out, dists_out );
  return;
}

void knn_double( 
  const int k, 
  const int ndim, 
  const int npt, 
  const double *const pts_in, 
  int *const indexes_out,
  float *const dists_out )
{
  knn::knn<double>( k, ndim, npt, pts_in, indexes_out, dists_out );
  return;
}
