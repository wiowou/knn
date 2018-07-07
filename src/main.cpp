#include "knn.h"

int main( int argc, char *argv[] )
{
  const int ndim = 2;
  const int npt = 15;
  float *pts = new float[npt*ndim];
  int ipt = 0;
  pts[ipt] = -4; pts[ipt+npt] = 1; ipt++;
  pts[ipt] = 1; pts[ipt+npt] = 3; ipt++;
  pts[ipt] = -1; pts[ipt+npt] = 1; ipt++;
  pts[ipt] = 1; pts[ipt+npt] = -4; ipt++;
  pts[ipt] = -3; pts[ipt+npt] = -6; ipt++;
  pts[ipt] = -1; pts[ipt+npt] = -2; ipt++;
  pts[ipt] = 4; pts[ipt+npt] = 2; ipt++;
  pts[ipt] = 2; pts[ipt+npt] = 1; ipt++;
  pts[ipt] = -2; pts[ipt+npt] = -1; ipt++;
  pts[ipt] = -2; pts[ipt+npt] = 3; ipt++;
  pts[ipt] = 1; pts[ipt+npt] = 1; ipt++;
  pts[ipt] = 2; pts[ipt+npt] = 3; ipt++;
  pts[ipt] = 3; pts[ipt+npt] = -2; ipt++;
  pts[ipt] = 2; pts[ipt+npt] = -3; ipt++;
  pts[ipt] = -4; pts[ipt+npt] = -3; ipt++;
  const int k = 2;
  int *indexes_out = new int[k*npt];
  float *dists_out = new float[k*npt];
  knn_gpu(k, ndim, npt, pts, indexes_out, dists_out);
  int indexes_check[k*npt] = {
    8,9,
    11,10,
    10,8,
    13,5,
    14,3,
    8,3,
    7,11,
    10,11,
    5,2,
    2,0,
    7,1,
    1,7,
    13,3,
    3,12,
    8,4 };
  float dists_check[k*npt];
  for (int i = 0; i < k*npt; ++i)
  {
    if (indexes_out[i] != indexes_check[i])
    {
      return i;
    }
  }
  delete[] pts;
  delete[] indexes_out;
  delete[] dists_out;
  return 0;
}
