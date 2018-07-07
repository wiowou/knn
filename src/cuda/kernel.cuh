
const int BLOCK_DIMX = 1024;

template <typename T>
__global__ void
calc_dist2s_dimi_kernel(
  const int ctrPtIdx, //indexed from dimension start
  const T* __restrict__ const pts, //pointer to start of dimension
  const int npt,
  T* __restrict__ const dist2s)
{
  if (threadIdx.x+blockIdx.x*blockDim.x < npt)
  {
    const T ptDelta = pts[threadIdx.x+blockIdx.x*blockDim.x] - pts[ctrPtIdx];
    dist2s[threadIdx.x+blockIdx.x*blockDim.x] += ptDelta * ptDelta;
  }
  return;
}

template <typename T>
__global__ void
calc_dist2s_dim0_kernel(
  const int ctrPtIdx, //indexed from dimension start
  const T* __restrict__ const pts, //pointer to start of dimension
  const int npt,
  T* __restrict__ const dist2s)
{
  if (threadIdx.x+blockIdx.x*blockDim.x < npt)
  {
    const T ptDelta = pts[threadIdx.x+blockIdx.x*blockDim.x] - pts[ctrPtIdx];
    dist2s[threadIdx.x+blockIdx.x*blockDim.x] = ptDelta * ptDelta;
  }
  return;
}

template <typename T>
__global__ void
copy_init_kernel(
  const T maxValue,
  T* __restrict__ const dist2, 
  const cub::KeyValuePair<int,T>* __restrict__ const indexDist,
  int* __restrict__ const index, //indexed from point start + inn
  T* __restrict__ const dist ) //indexed from point start + inn
{
  const int minIndex = indexDist->key;
  dist2[minIndex] = maxValue;
  *index = minIndex;
  *dist = sqrt(indexDist->value);
  return;
}
