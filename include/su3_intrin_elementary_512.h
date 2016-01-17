/*********************************************************************************
 * Copyright (C) 2012 Abdou Abdel-Rehim, Giannis Koutsou, Nikos Anastopolous
 * This file is part of the PLQCD library
 * 
 * function declarations and definitions for 512 related structs
 *********************************************************************************/

#ifndef _SU3_INTRIN_ELEMENTARY_512_H_
#define _SU3_INTRIN_ELEMENTARY_512_H_

#include <x86intrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

static double __attribute__((aligned(128))) dsign[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
static __m512d sign;



/*
 This part is not used
typedef union {
   int iv[8];
   double dv[4];
} vector_256_t;
*/
//extern __m256d __neghi_mask, __neglo_mask;


/**
 * Inline functions for elementary complex arithmetic operations, optimized for MIC
 * Need to use -mmic flag in compilation.
 */

/**
 * t0: a0 + b0*I, c0 + d0*I, e0 +f0*I, g0 + h0*I
 * t1: a1 + b1*I, c1 + d1*I, e1 +f1*I, g1 + h1*I
 *
 * return:  t0 + I*t1
 *
 * (a0-b1)+(b0+a1)*I , ...
 */
static inline __m512d complex_i_add_regs_512(__m512d t0, __m512d t1)
{

    //double  __attribute__((aligned(128))) dsign[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    //__m512d sign = _mm512_load_pd(dsign);
    __m512d t2;                                    //t0: a0,b0,c0,d0,...
                                                   //t1: a1,b1,c1,d1,...
    t2 = _mm512_swizzle_pd(t1, _MM_SWIZ_REG_CDAB); //t2: b1,a1,d1,c1,...
    return _mm512_fmadd_pd(t2,sign,t0);
}

/** 
 * t0: a0 + b0*I, c0 + d0*I, e0 +f0*I, g0 + h0*I
 * t1: a1 + b1*I, c1 + d1*I, e1 +f1*I, g1 + h1*I
 *
 * return:  t0 - I*t1
 *
 * (a0+b1)+(b0-a1)*I , ...
 */
static inline __m512d complex_i_sub_regs_512(__m512d t0, __m512d t1)
{
    //double  __attribute__((aligned(128))) dsign[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    //__m512d sign = _mm512_load_pd(dsign);
    __m512d t2;                                    //t0: a0,b0,c0,d0,...
                                                   //t1: a1,b1,c1,d1,...
    t2 = _mm512_swizzle_pd(t0, _MM_SWIZ_REG_CDAB); //t2: b0,a0,d0,c0,...
    t2 = _mm512_fmadd_pd(t1,sign,t2);
    return _mm512_swizzle_pd(t2, _MM_SWIZ_REG_CDAB);
}


/**
 * t0: a0 + b0*I, c0 + d0*I, e0 +f0*I, g0 + h0*I
 * t1: a1 + b1*I, c1 + d1*I, e1 +f1*I, g1 + h1*I
 *
 * return:  t0*t1
 *
 * (a0*a1-b0*b1), (a0*b1+b0*a1), ..... 
 */
static inline __m512d complex_mul_regs_512(__m512d t0, __m512d t1)
{
    //double  __attribute__((aligned(128))) dsign[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    //__m512d sign = _mm512_load_pd(dsign);
    __m512d t2,t3,t4;                    //t0: a0,b0,c0,d0,..
                                        //t1: a1,b1,c1,d1,..

    t2 = _mm512_swizzle_pd(t1, _MM_SWIZ_REG_CDAB); //t2: b1,a1,d1,c1,...
    t3 = _mm512_mask_swizzle_pd(t1,0xaa, t2, _MM_SWIZ_REG_NONE); //t3: a1,a1,c1,c1,..
    t4 = _mm512_mask_swizzle_pd(t1,0x85, t2, _MM_SWIZ_REG_NONE); //t4: b1,b1,d1,d1,..
    t3 = _mm512_mul_pd(t3, t0);         //t3: a0*a1,b0*a1,....
    t4 = _mm512_mul_pd(t4, t0);         //t4: a0*b1,b0*b1,....
    t4 = _mm512_swizzle_pd(t4, _MM_SWIZ_REG_CDAB);  //t4: b0*b1, a0*b1,....
    return _mm512_fmadd_pd(t4,sign,t3);
}


/**
 * t0: a0 + b0*I, c0 + d0*I, e0 +f0*I, g0 + h0*I
 * t1: a1 + b1*I, c1 + d1*I, e1 +f1*I, g1 + h1*I
 *
 * return:  conj(t0)*t1
 *
 * (a0*a1+b0*b1), (a0*b1-b0*a1), ..... 
 */
static inline __m512d complex_conj_mul_regs_512(__m512d t0, __m512d t1)
{
    //double  __attribute__((aligned(128))) dsign[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    //__m512d sign = _mm512_load_pd(dsign);
    __m512d t2,t3,t4;                    //t0: a0,b0,c0,d0,..
                                        //t1: a1,b1,c1,d1,..

    t2 = _mm512_swizzle_pd(t1, _MM_SWIZ_REG_CDAB);               //t2: b1,a1,d1,c1,...
    t3 = _mm512_mask_swizzle_pd(t1,0xaa, t2, _MM_SWIZ_REG_NONE); //t3: a1,a1,c1,c1,..
    t4 = _mm512_mask_swizzle_pd(t1,0x85, t2, _MM_SWIZ_REG_NONE); //t4: b1,b1,d1,d1,..
    t3 = _mm512_mul_pd(t3, t0);                                  //t3: a0*a1,b0*a1,....
    t3 = _mm512_swizzle_pd(t3, _MM_SWIZ_REG_CDAB);               //t3: b0*a1, a0*a1, ....  
    t4 = _mm512_mul_pd(t4, t0);                                  //t4: a0*b1,b0*b1,....
    t4 = _mm512_fmadd_pd(t3,sign,t4);                            //t4: a0*b1-b0*a1, a0*a1+b0*b1, ...
    return _mm512_swizzle_pd(t4, _MM_SWIZ_REG_CDAB);             //t4: b0*b1, a0*b1,....
}

#endif
