/************************************************
 *Some elementary tests on the use of intrinsics
 ************************************************/

#include<x86intrin.h>
#include<stdlib.h>
#include<stdio.h>
#include<complex.h>

int main()
{

  int i,j,k;

  double d[4]={1.0,2.0,3.0,4.0};

  double z1[2],z2[2],z3[2],z4[2];

  __m128d r1,r2,r3,r4;

  r1 = _mm_load_pd(d);

  r2 = _mm_shuffle_pd(r1,r1,1);

  _mm_store_pd(z1,r1);
  _mm_store_pd(z2,r2);

  printf("z1 %f %f z2 %f %f\n",z1[0],z1[1],z2[0],z2[1]);



  return 0;
}
