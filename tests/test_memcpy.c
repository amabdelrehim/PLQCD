///////////////////////////////////////////////////////////
//comparison of copying an array of complex numbers using
//intrinsics to using the complex.h
///////////////////////////////////////////////////////////

#include<stdlib.h>
#include<stdio.h>
#include"utils.h"
#include <x86intrin.h>
#include<complex.h>

int main(int argc, char** argv){

   if(argc<2){
     printf("usage: %s array_size\n",argv[0]);
     exit(0);}

   int N=atoi(argv[1]);

   int i,j,k;

   __m128d register c1,c2,c3;

   _Complex double *A = (_Complex double *) amalloc(N*sizeof(_Complex double),16);
   _Complex double *B = (_Complex double *) amalloc(N*sizeof(_Complex double),16);


  



   double ts,tf, tsum;

   //some intialization
   for(i=0; i<N; i++){
      A[i]=0.1*i + I*10*i;}


   tsum =0.0;
   for(j=0; j<100; j++)
   { 
      ts=stop_watch(0.0);
      for(i=0; i<N; i++)
      {
         c1=_mm_load_pd((double *) &A[i]);
         _mm_store_pd((double *) &B[i],c1);
      }
      tf=stop_watch(ts);
      tsum += tf;
   }
   printf("SIMD copy time %f \n",tsum/100.00);

   tsum=0.0;
   for(j=0; j<100; j++)
   {
      ts=stop_watch(0.0);
      for(i=0; i<N; i++){
         B[i]=A[i];}
      tf=stop_watch(ts);
      tsum += tf;
   }
   printf("direct copy time %f\n",tsum/100.00);


   tsum =0.0;
   for(j=0; j<100; j++)
   { 
      ts=stop_watch(0.0);
      for(i=0; i<N; i++)
      {
         c1=_mm_load_pd((double *) &A[i]);
         _mm_store_pd((double *) &B[i],c1);
      }
      tf=stop_watch(ts);
      tsum += tf;
   }
   printf("SIMD copy time %f \n",tsum/100.00);

   tsum=0.0;
   for(j=0; j<100; j++)
   {
      ts=stop_watch(0.0);
      for(i=0; i<N; i++){
         B[i]=A[i];}
      tf=stop_watch(ts);
      tsum += tf;
   }
   printf("direct copy time %f\n",tsum/100.00);















   afree(A);
   afree(B);

return 0;
}
