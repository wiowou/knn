#include "knn.h"

template <typename T>
void set_pts( const int npt, T* pts )
{
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
  return;
}

int check_indexes( const int k, const int npt, int *indexes )
{
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
  for (int i = 0; i < k*npt; ++i)
  {
    if (indexes[i] != indexes_check[i])
    {
      return i + 1;
    }
  }
  return 0;
}

int main( int argc, char *argv[] )
{
  const int ndim = 2;
  const int npt = 15;
  const int k = 2;
  int *indexes_out = new int[k*npt];
  float *dists_out = new float[k*npt];
  float *pts_f = new float[npt*ndim];
  set_pts( npt, pts_f );
  knn_float(k, ndim, npt, pts_f, indexes_out, dists_out);
  check_indexes( k, npt, indexes_out );
  double *pts_d = new double[npt*ndim];
  set_pts( npt, pts_d );
  knn_double(k, ndim, npt, pts_d, indexes_out, dists_out);
  check_indexes( k, npt, indexes_out );
  delete[] pts_f;
  delete[] pts_d;
  delete[] indexes_out;
  delete[] dists_out;
  return 0;
}
