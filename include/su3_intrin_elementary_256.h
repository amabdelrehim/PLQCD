/*********************************************************************************
 * Copyright (C) 2012 Abdou Abdel-Rehim, Giannis Koutsou, Nikos Anastopolous
 * This file is part of the PLQCD library
 * 
 * function declarations and definitions for 256 related structs
 *********************************************************************************/

#ifndef _SU3_INTRIN_ELEMENTARY_256_H_
#define _SU3_INTRIN_ELEMENTARY_256_H_

#include <x86intrin.h>

typedef union {
   int iv[8];
   double dv[4];
} vector_256_t;

extern __m256d __neghi_mask, __neglo_mask;

/**
 * Inline functions for elementary complex arithmetic operations, optimized for AVX
 * Need to use -mavx flag in compilation.
 */

/**
 * t0: a+bI, e+fI
 * t1: c+dI, g+hI
 * return: (a+bI) + I(c+dI), (e+fI) + I(g+hI) 
 *      =  (a-d) + (b+c)I,   (e-h) + (f+g)I 
 */
static inline __m256d complex_i_add_regs_256(__m256d t0, __m256d t1)
{
    __m256d t2;
    t2 = _mm256_shuffle_pd(t1, t1, 5); //swap each part of t0 separately
    return _mm256_addsub_pd(t0,t2);
}

/**
 * t0: a+bI, e+fI
 * t1: c+dI, g+hI
 * return: (a+bI) - I(c+dI), (e+fI) - I(g+hI) 
 *      =  (a+d) + (b-c)I,   (e+h) + (f-g)I 
 */
static inline __m256d complex_i_sub_regs_256(__m256d t0, __m256d t1)
{
    __m256d t2;
    t2 = _mm256_shuffle_pd(t0, t0, 5); //swap each part of t0 separately
    t2 = _mm256_addsub_pd(t2,t1);
    return _mm256_shuffle_pd(t2, t2, 5);
}

/**
 * t0: a+bI, e+fI
 * t1: c+dI, g+hI
 * return: (ac-bd) + (ad+bc)I, (eg-fh) + (eh+fg)I
 */
static inline __m256d complex_mul_regs_256(__m256d t0, __m256d t1)
{
    __m256d t2,t3;                      //t0: a,b,e,f    t1:c,d,g,h
    t2 = _mm256_unpacklo_pd(t1,t1);     //t2: c,c,g,g
    t3 = _mm256_unpackhi_pd(t1,t1);     //t3: d,d,h,h
    t2 = _mm256_mul_pd(t2, t0);         //t2: ac,bc,eg,fg
    t3 = _mm256_mul_pd(t3, t0);         //t3: ad,bd,eh,fh
    t3 = _mm256_shuffle_pd(t3, t3, 5);  //t3: bd,ad,fh,eh
    return _mm256_addsub_pd(t2, t3);    //ac-bd,bc+ad,eg-fh,fg+eh
}

/**
 * t0: a+bI, e+fI
 * t1: c+dI, g+hI
 * return: (a-bI)*(c+dI), (e-fI)*(g+hI) 
 *      =  (ac+bd) + (ad-bc)I, (eg+fh) + (eh-fg)I
 */
static inline __m256d complex_conj_mul_regs_256(__m256d t0, __m256d t1)
{
    __m256d t2,t3;                          //t0: a,b,e,f  t1: c,d,g,h
    t2 = _mm256_unpacklo_pd(t1,t1);         //t2: c,c,g,g
    t3 = _mm256_unpackhi_pd(t1,t1);         //t3: d,d,h,h
    t2 = _mm256_mul_pd(t2, t0);             //t2: ac,bc,eg,fg
    t3 = _mm256_mul_pd(t3, t0);             //t3: ad,bd,eh,fh
    t2 = _mm256_shuffle_pd(t2, t2, 5);      //t2: bc,ac,fg,eg
    t3 = _mm256_addsub_pd(t3, t2);          //t3: ad-bc,bd+ac,eh-fg,fh+eg
    return _mm256_shuffle_pd(t3, t3, 5);    //    bd+ac,ad-bc,fh+eg,eh-fg
}


/**
 * Select user-specified 128-bit lanes from 256-bit input arguments and paste them
 * in output result
 *
 * _mm256_permute2f128_pd(a,b,which)
 * arg1: a0(lo) a1(hi) 
 * arg2: b0(lo) b1(hi)
 *
 * return:
 * switch (which)
 *    0x00: a0 a0 
 *    0x01: a1 a0
 *    0x02: b0 a0
 *    0x03: b1 a0
 *    0x10: a0 a1 
 *    0x11: a1 a1 
 *    0x12: b0 a1 
 *    0x13: b1 a1 
 *    0x20: a0 b0 
 *    0x21: a1 b0 
 *    0x22: b0 b0 
 *    0x23: b1 b0 
 *    0x30: a0 b1 
 *    0x31: a1 b1 
 *    0x32: b0 b1 
 *    0x33: b1 b1 
 *
 *    0x08: 0 a0 
 *    0x18: 0 a1 
 *    0x28: 0 b0 
 *    0x38: 0 b1 
 *    0x80: a0 0 
 *    0x81: a1 0 
 *    0x82: b0 0 
 *    0x83: b1 0 
 */

#define intrin256_select_arg1lo_arg1lo(a,b) _mm256_permute2f128_pd(a,b,0x00)
#define intrin256_select_arg1hi_arg1lo(a,b) _mm256_permute2f128_pd(a,b,0x01)
#define intrin256_select_arg2lo_arg1lo(a,b) _mm256_permute2f128_pd(a,b,0x02)
#define intrin256_select_arg2hi_arg1lo(a,b) _mm256_permute2f128_pd(a,b,0x03)

#define intrin256_select_arg1lo_arg1hi(a,b) _mm256_permute2f128_pd(a,b,0x10)
#define intrin256_select_arg1hi_arg1hi(a,b) _mm256_permute2f128_pd(a,b,0x11)
#define intrin256_select_arg2lo_arg1hi(a,b) _mm256_permute2f128_pd(a,b,0x12)
#define intrin256_select_arg2hi_arg1hi(a,b) _mm256_permute2f128_pd(a,b,0x13)

#define intrin256_select_arg1lo_arg2lo(a,b) _mm256_permute2f128_pd(a,b,0x20)
#define intrin256_select_arg1hi_arg2lo(a,b) _mm256_permute2f128_pd(a,b,0x21)
#define intrin256_select_arg2lo_arg2lo(a,b) _mm256_permute2f128_pd(a,b,0x22)
#define intrin256_select_arg2hi_arg2lo(a,b) _mm256_permute2f128_pd(a,b,0x23)

#define intrin256_select_arg1lo_arg2hi(a,b) _mm256_permute2f128_pd(a,b,0x30)
#define intrin256_select_arg1hi_arg2hi(a,b) _mm256_permute2f128_pd(a,b,0x31)
#define intrin256_select_arg2lo_arg2hi(a,b) _mm256_permute2f128_pd(a,b,0x32)
#define intrin256_select_arg2hi_arg2hi(a,b) _mm256_permute2f128_pd(a,b,0x33)

#define intrin256_select_zero_arg1lo(a,b) _mm256_permute2f128_pd(a,b,0x08)
#define intrin256_select_zero_arg1hi(a,b) _mm256_permute2f128_pd(a,b,0x18)
#define intrin256_select_zero_arg2lo(a,b) _mm256_permute2f128_pd(a,b,0x28)
#define intrin256_select_zero_arg2hi(a,b) _mm256_permute2f128_pd(a,b,0x38)

#define intrin256_select_arg1lo_zero(a,b) _mm256_permute2f128_pd(a,b,0x80)
#define intrin256_select_arg1hi_zero(a,b) _mm256_permute2f128_pd(a,b,0x81)
#define intrin256_select_arg2lo_zero(a,b) _mm256_permute2f128_pd(a,b,0x82)
#define intrin256_select_arg2hi_zero(a,b) _mm256_permute2f128_pd(a,b,0x83)

/*
 * Inits the masks for negating the high or low 128-bit parts (both 64-bit elements) of 
 * a 256-bit vector register
 */
static void intrin256_init_negate_masks()
{
    vector_256_t _neghi_mask_256 = { 
        .iv[0] = 0x00000000,
        .iv[1] = 0x00000000,
        .iv[2] = 0x00000000,
        .iv[3] = 0x00000000,
        .iv[4] = 0x00000000,
        .iv[5] = 0x80000000,
        .iv[6] = 0x00000000,
        .iv[7] = 0x80000000
    };
    __neghi_mask = _mm256_load_pd(_neghi_mask_256.dv);
    
    vector_256_t _neglo_mask_256 = {
        .iv[0] = 0x00000000,
        .iv[1] = 0x80000000,
        .iv[2] = 0x00000000,
        .iv[3] = 0x80000000,
        .iv[4] = 0x00000000,
        .iv[5] = 0x00000000,
        .iv[6] = 0x00000000,
        .iv[7] = 0x00000000
    };
    __neglo_mask = _mm256_load_pd(_neglo_mask_256.dv);
}

/**
 * Negates the low 128-bit part (both 64-bit elements) of a
 */
static inline __m256d intrin256_negate_lo(__m256d a)
{
    return _mm256_xor_pd(a, __neglo_mask);
}

/**
 * Negates the high 128-bit part (both 64-bit elements) of a
 */
static inline __m256d intrin256_negate_hi(__m256d a)
{
    return _mm256_xor_pd(a, __neghi_mask);
}


#endif
