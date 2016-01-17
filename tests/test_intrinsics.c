/************************************************
 *Some elementary tests on the use of intrinsics
 ************************************************/

#include<x86intrin.h>
#include<stdlib.h>
#include<stdio.h>
#include<complex.h>
#include"su3_intrin_elementary_128.h"
#include"su3_intrin_highlevel_128.h"
#include"utils.h"
#include"su3.h"
#include"su3_256.h"
#include"su3_intrin_elementary_256.h"
#include"su3_intrin_highlevel_256.h"


int main()
{

  int i,j,k;

  double d[4]={1.0,2.0,3.0,4.0};

  double z1[2],z2[2],z3[2],z4[2];

  __m128d r1,r2,r3,r4;

  r1=_mm_load_pd(d);
  
  r2=_mm_load_pd(d+2);

  r1=_mm_add_pd(r1,r2);

  _mm_store_pd(z1,r1);

  printf("sum= %f %f\n",z1[0],z1[1]);

  r1=_mm_add_pd(r1,r2);

  _mm_store_pd(z1,r1);

  printf("sum= %f %f\n",z1[0],z1[1]);

  r1=_mm_sub_pd(r1,r2);

  _mm_store_pd(z1,r1);

  printf("sum= %f %f\n",z1[0],z1[1]);

  r1 = _mm_shuffle_pd(r1,r1,1);

  _mm_store_pd(z1,r1);

  printf("sum= %f %f\n",z1[0],z1[1]);

  r1 = complex_i_add_regs_128(r1,r2);

  _mm_store_pd(z1,r1);

  printf("sum= %f %f\n",z1[0],z1[1]);

  r1 = _mm_load_pd(d);
  r2=  _mm_load_pd(d+2);

  r1 = _mm_addsub_pd(r1,r2);

  _mm_store_pd(z1,r1);

  printf("res= %f %f\n",z1[0],z1[1]);



  //testing complex_i_add_regs_128
  r1 = _mm_load_pd(d);
  r2 = _mm_load_pd(d+2);

  _mm_store_pd(z1,r1);
  _mm_store_pd(z2,r2);

  printf("before the operation: r1 %f %f , r2 %f %f\n",z1[0],z1[1],z2[0],z2[1]);

  r1 = complex_i_add_regs_128(r1,r2);

  _mm_store_pd(z1,r1);
  _mm_store_pd(z2,r2);

  
  printf("after complex_i_add_regs_128: r1 %f %f , r2 %f %f answer: %f %f\n",z1[0],z1[1],z2[0],z2[1],d[0]-d[3],d[1]+d[2]);


  //testing complex_i_sub_regs_128
  r1 = _mm_load_pd(d);
  r2 = _mm_load_pd(d+2);

  _mm_store_pd(z1,r1);
  _mm_store_pd(z2,r2);

  printf("before the operation: r1 %f %f , r2 %f %f\n",z1[0],z1[1],z2[0],z2[1]);

  r1 = complex_i_sub_regs_128(r1,r2);

  _mm_store_pd(z1,r1);
  _mm_store_pd(z2,r2);

  
  printf("after complex_i_sub_regs_128: r1 %f %f , r2 %f %f answer: %f %f\n",z1[0],z1[1],z2[0],z2[1],d[0]+d[3],d[1]-d[2]);


  //testing complex_mul_regs_128
  r1 = _mm_load_pd(d);
  r2 = _mm_load_pd(d+2);

  _mm_store_pd(z1,r1);
  _mm_store_pd(z2,r2);

  printf("before the operation: r1 %f %f , r2 %f %f\n",z1[0],z1[1],z2[0],z2[1]);

  r1 = complex_mul_regs_128(r1,r2);

  _mm_store_pd(z1,r1);
  _mm_store_pd(z2,r2);

  
  printf("after complex_mul_regs_128: r1 %f %f , r2 %f %f answer: %f %f\n",z1[0],z1[1],z2[0],z2[1],d[0]*d[2]-d[1]*d[3],d[0]*d[3]+d[1]*d[2]);


  //testing complex_mul_128
  z1[0]=d[0]; z1[1]=d[1]; z2[0]=d[2]; z2[1]=d[3]; z3[0]=0.0; z3[1]=0.0;
  printf("before the operation: z1 %f %f , z2 %f %f, z3 %f %f\n",z1[0],z1[1],z2[0],z2[1],z3[0],z3[1]);

  complex_mul_128(z3,z1,z2);
  
  printf("after complex_mul_128: z1 %f %f , z2 %f %f , z3 %f %f answer: %f %f\n",z1[0],z1[1],z2[0],z2[1],z3[0],z3[1],d[0]*d[2]-d[1]*d[3],d[1]*d[2]+d[0]*d[3]);


  //testing complex_conj_mul_regs_128
  r1 = _mm_load_pd(d);
  r2 = _mm_load_pd(d+2);

  _mm_store_pd(z1,r1);
  _mm_store_pd(z2,r2);

  printf("before the operation: r1 %f %f , r2 %f %f\n",z1[0],z1[1],z2[0],z2[1]);

  r1 = complex_conj_mul_regs_128(r1,r2);

  _mm_store_pd(z1,r1);
  _mm_store_pd(z2,r2);

  
  printf("after complex_mul_regs_128: r1 %f %f , r2 %f %f answer: %f %f\n",z1[0],z1[1],z2[0],z2[1],d[0]*d[2]+d[1]*d[3],d[0]*d[3]-d[1]*d[2]);


  //testing su3 matrix vector multiply
  su3* u = (su3 *) alloc(sizeof(su3),16);
  su3_vector* v = (su3_vector *) alloc(sizeof(su3_vector),16);
  su3_vector* w = (su3_vector *) alloc(sizeof(su3_vector),16);
  su3_vector* s = (su3_vector *) alloc(sizeof(su3_vector),16);

  u->c00=0.0+0.0*I;
  u->c01=0.0+1.0*I;
  u->c02=0.0+2.0*I;
  u->c10=1.0+0.0*I;
  u->c11=1.0+1.0*I;
  u->c12=1.0+2.0*I;
  u->c20=2.0+0.0*I;
  u->c21=2.0+1.0*I;
  u->c22=2.0+2.0*I;

  v->c0=1.0+1.0*I;
  v->c1=2.0+2.0*I;
  v->c2=3.0+3.0*I;

  __m128d U[3][3],V[3],W[3],S[3];

  intrin_su3_load_128(U,u);
  intrin_vector_load_128(V,v);

  intrin_su3_multiply_128(W,U,V);
  intrin_vector_store_128(w,W);

  intrin_su3_inverse_multiply_128(S,U,V);
  intrin_vector_store_128(s,S);


  printf("w[0]=%f + I* %f \n",creal(w->c0),cimag(w->c0));
  printf("w[1]=%f + I* %f \n",creal(w->c1),cimag(w->c1));
  printf("w[2]=%f + I* %f \n\n\n",creal(w->c2),cimag(w->c2));
  
  
  printf("s[0]=%f + I* %f \n",creal(s->c0),cimag(s->c0));
  printf("s[1]=%f + I* %f \n",creal(s->c1),cimag(s->c1));
  printf("s[2]=%f + I* %f \n",creal(s->c2),cimag(s->c2));
  
  //checking the input

  printf("after calling the su3 matrix vector multiply\n\n");
  intrin_su3_store_128(u,U);
  intrin_vector_store_128(v,V);

  printf("u00=%f + I* %f \n",creal(u->c00),cimag(u->c00));
  printf("u01=%f + I* %f \n",creal(u->c01),cimag(u->c01));
  printf("u02=%f + I* %f \n",creal(u->c02),cimag(u->c02));
  printf("u10=%f + I* %f \n",creal(u->c10),cimag(u->c10));
  printf("u11=%f + I* %f \n",creal(u->c11),cimag(u->c11));
  printf("u12=%f + I* %f \n",creal(u->c12),cimag(u->c12));
  printf("u20=%f + I* %f \n",creal(u->c20),cimag(u->c20));
  printf("u21=%f + I* %f \n",creal(u->c21),cimag(u->c21));
  printf("u22=%f + I* %f \n\n\n",creal(u->c22),cimag(u->c22));
    

  printf("v[0]=%f + I* %f \n",creal(v->c0),cimag(v->c0));
  printf("v[1]=%f + I* %f \n",creal(v->c1),cimag(v->c1));
  printf("v[2]=%f + I* %f \n\n\n",creal(v->c2),cimag(v->c2));

  //testing the avx intrinsics
  complex_pair *q1 = (complex_pair *) alloc(sizeof(complex_pair),64);
  complex_pair *q2 = (complex_pair *) alloc(sizeof(complex_pair),64);
  complex_pair *q3 = (complex_pair *) alloc(sizeof(complex_pair),64);
  complex_pair *q4 = (complex_pair *) alloc(sizeof(complex_pair),64);
  complex_pair *q5 = (complex_pair *) alloc(sizeof(complex_pair),64);
  complex_pair *q6 = (complex_pair *) alloc(sizeof(complex_pair),64);

  __m256d h1,h2,h3,h4,h5,h6;

  q1->lo = 0.1 +0.2*I;
  q1->lo = -0.4+0.7*I;
  q1->hi = 5;
  q1->hi = -7.0*I;

  q2->lo = 0.5 -0.2*I;
  q2->lo = 0.4-0.7*I;
  q2->hi = 5*I;
  q2->hi = 1.0*I;

  h1 = _mm256_load_pd((double *) &(q1->lo));
  h2 = _mm256_load_pd((double *) &(q2->lo));

  h3 = complex_i_add_regs_256(h1,h2);
  _mm256_store_pd((double *) &(q3->lo),h3);

  printf("testing complex_i_add_regs_256\n");
  h3 = complex_i_add_regs_256(h1,h2);
  _mm256_store_pd((double *) &(q3->lo),h3);
  printf("input before the call: q1=%f %f %f %f, q2= %f %f %f %f\n",
         creal(q1->lo),cimag(q1->lo),creal(q1->hi),cimag(q1->hi),
         creal(q2->lo),cimag(q2->lo),creal(q2->hi),cimag(q2->hi));
  _mm256_store_pd((double *) &(q1->lo), h1);
  _mm256_store_pd((double *) &(q2->lo), h2);
  printf("input after the call : q1=%f %f %f %f, q2= %f %f %f %f\n",
         creal(q1->lo),cimag(q1->lo),creal(q1->hi),cimag(q1->hi),
         creal(q2->lo),cimag(q2->lo),creal(q2->hi),cimag(q2->hi));
  printf("result: %f %f %f %f\n",creal(q3->lo),cimag(q3->lo),creal(q3->hi),cimag(q3->hi));
  printf("computed: %f %f %f %f\n",
          creal(q1->lo)-cimag(q2->lo),cimag(q1->lo)+creal(q2->lo),
          creal(q1->hi)-cimag(q2->hi),cimag(q1->hi)+creal(q2->hi));

  printf("testing complex_i_sub_regs_256\n");
  h3 = complex_i_sub_regs_256(h1,h2);
  _mm256_store_pd((double *) &(q3->lo),h3);
  printf("input before the call: q1=%f %f %f %f, q2= %f %f %f %f\n",
         creal(q1->lo),cimag(q1->lo),creal(q1->hi),cimag(q1->hi),
         creal(q2->lo),cimag(q2->lo),creal(q2->hi),cimag(q2->hi));
  _mm256_store_pd((double *) &(q1->lo), h1);
  _mm256_store_pd((double *) &(q2->lo), h2);
  printf("input after the call : q1=%f %f %f %f, q2= %f %f %f %f\n",
         creal(q1->lo),cimag(q1->lo),creal(q1->hi),cimag(q1->hi),
         creal(q2->lo),cimag(q2->lo),creal(q2->hi),cimag(q2->hi));
  printf("result: %f %f %f %f\n",creal(q3->lo),cimag(q3->lo),creal(q3->hi),cimag(q3->hi));
  printf("computed: %f %f %f %f\n",
          creal(q1->lo)+cimag(q2->lo),cimag(q1->lo)-creal(q2->lo),
          creal(q1->hi)+cimag(q2->hi),cimag(q1->hi)-creal(q2->hi));

  printf("testing complex_mul_regs_256\n");
  h3 = complex_mul_regs_256(h1,h2);
  _mm256_store_pd((double *) &(q3->lo),h3);
  printf("input before the call: q1=%f %f %f %f, q2= %f %f %f %f\n",
         creal(q1->lo),cimag(q1->lo),creal(q1->hi),cimag(q1->hi),
         creal(q2->lo),cimag(q2->lo),creal(q2->hi),cimag(q2->hi));
  _mm256_store_pd((double *) &(q1->lo), h1);
  _mm256_store_pd((double *) &(q2->lo), h2);
  printf("input after the call : q1=%f %f %f %f, q2= %f %f %f %f\n",
         creal(q1->lo),cimag(q1->lo),creal(q1->hi),cimag(q1->hi),
         creal(q2->lo),cimag(q2->lo),creal(q2->hi),cimag(q2->hi));
  printf("result: %f %f %f %f\n",creal(q3->lo),cimag(q3->lo),creal(q3->hi),cimag(q3->hi));
  printf("computed: %f %f %f %f\n",
          creal(q1->lo)*creal(q2->lo) - cimag(q1->lo)*cimag(q2->lo) ,creal(q1->lo)*cimag(q2->lo)+cimag(q1->lo)*creal(q2->lo),
          creal(q1->hi)*creal(q2->hi) - cimag(q1->hi)*cimag(q2->hi) ,creal(q1->hi)*cimag(q2->hi)+cimag(q1->hi)*creal(q2->hi));
          

  printf("testing complex_conj_mul_regs_256\n");
  h3 = complex_conj_mul_regs_256(h1,h2);
  _mm256_store_pd((double *) &(q3->lo),h3);
  printf("input before the call: q1=%f %f %f %f, q2= %f %f %f %f\n",
         creal(q1->lo),cimag(q1->lo),creal(q1->hi),cimag(q1->hi),
         creal(q2->lo),cimag(q2->lo),creal(q2->hi),cimag(q2->hi));
  _mm256_store_pd((double *) &(q1->lo), h1);
  _mm256_store_pd((double *) &(q2->lo), h2);
  printf("input after the call : q1=%f %f %f %f, q2= %f %f %f %f\n",
         creal(q1->lo),cimag(q1->lo),creal(q1->hi),cimag(q1->hi),
         creal(q2->lo),cimag(q2->lo),creal(q2->hi),cimag(q2->hi));
  printf("result: %f %f %f %f\n",creal(q3->lo),cimag(q3->lo),creal(q3->hi),cimag(q3->hi));
  printf("computed: %f %f %f %f\n",
          creal(q1->lo)*creal(q2->lo) + cimag(q1->lo)*cimag(q2->lo) ,creal(q1->lo)*cimag(q2->lo)-cimag(q1->lo)*creal(q2->lo),
          creal(q1->hi)*creal(q2->hi) + cimag(q1->hi)*cimag(q2->hi) ,creal(q1->hi)*cimag(q2->hi)-cimag(q1->hi)*creal(q2->hi));
          

  //testing su3 matrix vector multiply
  su3_256* u256 = (su3_256 *) alloc(sizeof(su3_256),64);
  su3_vector_256* v256 = (su3_vector_256 *) alloc(sizeof(su3_vector_256),64);
  su3_vector_256* w256 = (su3_vector_256 *) alloc(sizeof(su3_vector_256),64);
  su3_vector_256* s256 = (su3_vector_256 *) alloc(sizeof(su3_vector_256),64);

  u256->c00.lo=0.0+0.0*I;
  u256->c01.lo=0.0+1.0*I;
  u256->c02.lo=0.0+2.0*I;
  u256->c10.lo=1.0+0.0*I;
  u256->c11.lo=1.0+1.0*I;
  u256->c12.lo=1.0+2.0*I;
  u256->c20.lo=2.0+0.0*I;
  u256->c21.lo=2.0+1.0*I;
  u256->c22.lo=2.0+2.0*I;

  u256->c00.hi=u256->c00.lo;
  u256->c01.hi=u256->c01.lo;
  u256->c02.hi=u256->c02.lo;
  u256->c10.hi=u256->c10.lo;
  u256->c11.hi=u256->c11.lo;
  u256->c12.hi=u256->c12.lo;
  u256->c20.hi=u256->c20.lo;
  u256->c21.hi=u256->c21.lo;
  u256->c22.hi=u256->c22.lo;



  v256->c0.lo=1.0+1.0*I;
  v256->c1.lo=2.0+2.0*I;
  v256->c2.lo=3.0+3.0*I;
  v256->c0.hi=1.0+1.0*I;
  v256->c1.hi=2.0+2.0*I;
  v256->c2.hi=3.0+3.0*I;

  __m256d U256[3][3],V256[3],W256[3],S256[3];

  intrin_su3_load_256(U256,u256);
  intrin_vector_load_256(V256,v256);

  intrin_su3_multiply_256(W256,U256,V256);
  intrin_vector_store_256(w256,W256);

  intrin_su3_inverse_multiply_256(S256,U256,V256);
  intrin_vector_store_256(s256,S256);


  printf("w256[0]=%f  %f %f %f\n",creal(w256->c0.lo),cimag(w256->c0.lo),creal(w256->c0.hi),cimag(w256->c0.hi));
  printf("w256[1]=%f  %f %f %f\n",creal(w256->c1.lo),cimag(w256->c1.lo),creal(w256->c1.hi),cimag(w256->c1.hi));
  printf("w256[2]=%f  %f %f %f\n",creal(w256->c2.lo),cimag(w256->c2.lo),creal(w256->c2.hi),cimag(w256->c2.hi));
  
  
  printf("s256[0]=%f  %f %f %f\n",creal(s256->c0.lo),cimag(s256->c0.lo),creal(s256->c0.hi),cimag(s256->c0.hi));
  printf("s256[1]=%f  %f %f %f\n",creal(s256->c1.lo),cimag(s256->c1.lo),creal(s256->c1.hi),cimag(s256->c1.hi));
  printf("s256[2]=%f  %f %f %f\n",creal(s256->c2.lo),cimag(s256->c2.lo),creal(s256->c2.hi),cimag(s256->c2.hi));
  
  
  //checking the input

  printf("after calling the su3 matrix vector multiply\n\n");
  intrin_su3_store_256(u256,U256);
  intrin_vector_store_256(v256,V256);

  printf("u256_00=%f %f %f %f\n",creal(u256->c00.lo),cimag(u256->c00.lo),creal(u256->c00.hi),cimag(u256->c00.hi));
  printf("u256_01=%f %f %f %f\n",creal(u256->c01.lo),cimag(u256->c01.lo),creal(u256->c01.hi),cimag(u256->c01.hi));
  printf("u256_02=%f %f %f %f\n",creal(u256->c02.lo),cimag(u256->c02.lo),creal(u256->c02.hi),cimag(u256->c02.hi));
  printf("u256_10=%f %f %f %f\n",creal(u256->c10.lo),cimag(u256->c10.lo),creal(u256->c10.hi),cimag(u256->c10.hi));
  printf("u256_11=%f %f %f %f\n",creal(u256->c11.lo),cimag(u256->c11.lo),creal(u256->c11.hi),cimag(u256->c11.hi));
  printf("u256_12=%f %f %f %f\n",creal(u256->c12.lo),cimag(u256->c12.lo),creal(u256->c12.hi),cimag(u256->c12.hi));
  printf("u256_20=%f %f %f %f\n",creal(u256->c20.lo),cimag(u256->c20.lo),creal(u256->c20.hi),cimag(u256->c20.hi));
  printf("u256_21=%f %f %f %f\n",creal(u256->c21.lo),cimag(u256->c21.lo),creal(u256->c21.hi),cimag(u256->c21.hi));
  printf("u256_22=%f %f %f %f\n",creal(u256->c22.lo),cimag(u256->c22.lo),creal(u256->c22.hi),cimag(u256->c22.hi));
    

  printf("v256[0]=%f  %f %f %f\n",creal(v256->c0.lo),cimag(v256->c0.lo),creal(v256->c0.hi),cimag(v256->c0.hi));
  printf("v256[1]=%f  %f %f %f\n",creal(v256->c1.lo),cimag(v256->c1.lo),creal(v256->c1.hi),cimag(v256->c1.hi));
  printf("v256[2]=%f  %f %f %f\n",creal(v256->c2.lo),cimag(v256->c2.lo),creal(v256->c2.hi),cimag(v256->c2.hi));



  return 0;
}
