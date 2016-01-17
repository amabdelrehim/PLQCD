/********************************************
 * Tests for MIC intrinsics
 * ******************************************/

#include<stdlib.h>
#include<stdio.h>

#include <x86intrin.h>
#include <xmmintrin.h>
#include <immintrin.h>



int main()
{
   __m512d  t0,t1;

   double d1[8] __attribute__ ((aligned(64)));   
   double d2[8] __attribute__ ((aligned(64)));   
   double d3[8] __attribute__ ((aligned(64)));   

   for(int i=0; i<8; i++)
   {
       d1[i]= i*1.0;

       d2[i]= 0.0;

       d3[i] = d1[i];

   }

   //printf("testing intialization of registers\n");
   //_mm512_store_pd(d1,t0);
   //printf("d1=t0: %f %f %f %f %f %f %f %f  \n",d1[0],d1[1],d1[2],d1[3],d1[4],d1[5],d1[6],d1[7]);
   //_mm512_store_pd(d1,t1);
   //printf("d1=t1: %f %f %f %f %f %f %f %f  \n",d1[0],d1[1],d1[2],d1[3],d1[4],d1[5],d1[6],d1[7]);


   t0 = _mm512_load_pd(d1);

   printf("permute backward\n");
   t1 = (__m512d)  _mm512_permute4f128_epi32 ( (__m512i) t0, 0b00111001);
   _mm512_store_pd(d2,t1);
   printf("d1: %f %f %f %f %f %f %f %f  \n",d1[0],d1[1],d1[2],d1[3],d1[4],d1[5],d1[6],d1[7]);
   printf("d2: %f %f %f %f %f %f %f %f  \n",d2[0],d2[1],d2[2],d2[3],d2[4],d2[5],d2[6],d2[7]);
    
   printf("permute forward\n");
   t1 = (__m512d)  _mm512_permute4f128_epi32 ( (__m512i) t0, 0b10010011);
   _mm512_store_pd(d2,t1);
   printf("d1: %f %f %f %f %f %f %f %f  \n",d1[0],d1[1],d1[2],d1[3],d1[4],d1[5],d1[6],d1[7]);
   printf("d2: %f %f %f %f %f %f %f %f  \n",d2[0],d2[1],d2[2],d2[3],d2[4],d2[5],d2[6],d2[7]);
   


   int __attribute__((aligned(64))) order[16]={0,1,0,1,4,5,6,7,8,9,10,11,12,13,14,15};

   __m512i morder = _mm512_load_epi32(order);


   printf("permuting doubles\n");
   t1 = (__m512d) _mm512_permutevar_epi32 (morder, (__m512i) t0);
   _mm512_store_pd(d2,t1);
   printf("d1: %f %f %f %f %f %f %f %f  \n",d1[0],d1[1],d1[2],d1[3],d1[4],d1[5],d1[6],d1[7]);
   printf("d2: %f %f %f %f %f %f %f %f  \n",d2[0],d2[1],d2[2],d2[3],d2[4],d2[5],d2[6],d2[7]);

 
   return 0;

}
