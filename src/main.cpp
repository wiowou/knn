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
  knn_indexes(k, ndim, npt, pts, indexes_out);
  return 0;
}
