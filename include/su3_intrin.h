#ifndef _SU3_INTRIN_H
#define _SU3_INTRIN_H

#include <x86intrin.h>



/**
 * Inline functions for elementary complex arithmetic operations, optimized for SSE3
 */

/**
 * t0: a + bI
 * t1: c + dI
 * return: (a+bI) + I(c+dI) = (a-d) + (b+c)I
 */
static inline __m128d complex_i_add_regs(__m128d t0, __m128d t1)
{
    t1 = _mm_shuffle_pd(t1, t1, 1); //swap t1
    return _mm_addsub_pd(t0,t1);
}

 
/**
 * t0: a + bI
 * t1: c + dI
 * return: (a+bI) - I(c+dI) = (a+d) + (b-c)I
 */
static inline __m128d complex_i_sub_regs(__m128d t0, __m128d t1)
{
    __m128d t2;
    t0 = _mm_shuffle_pd(t0, t0, 1); //swap t0
    t2 = _mm_addsub_pd(t0,t1);
    return _mm_shuffle_pd(t2, t2, 1);
}


/**
 * t0: a + bI
 * t1: c + dI
 * return: (ac-bd) + (ad+bc)I
 */
static inline __m128d complex_mul_regs(__m128d t0, __m128d t1)
{
    __m128d t2;
    t2 = t1;
    t1 = _mm_unpacklo_pd(t1,t1);
    t2 = _mm_unpackhi_pd(t2,t2);
    t1 = _mm_mul_pd(t1, t0); 
    t2 = _mm_mul_pd(t2, t0); 
    t2 = _mm_shuffle_pd(t2, t2, 1); 
    t1 = _mm_addsub_pd(t1, t2);
    return t1;
}

static inline void complex_mul(double *c, double *a, double *b)
{
    __m128d t0,t1,t2;
    t0 = _mm_load_pd(a);
    t1 = _mm_load_pd(b);
    t2 = t1;
    t1 = _mm_unpacklo_pd(t1,t1);
    t2 = _mm_unpackhi_pd(t2,t2);
    t1 = _mm_mul_pd(t1, t0); 
    t2 = _mm_mul_pd(t2, t0); 
    t2 = _mm_shuffle_pd(t2, t2, 1); //swaps the two parts of t2 register 
    t1 = _mm_addsub_pd(t1, t2);
    _mm_store_pd(c,t1);
}


/**
 * t0: a + bI
 * t1: c + dI
 * return: (a-bI)*(c+dI) = (ac+bd) + (ad-bc)I
 */
static inline __m128d complex_conj_mul_regs(__m128d t0, __m128d t1)
{
    __m128d t2;
    t2 = t1;
    t1 = _mm_unpacklo_pd(t1,t1);
    t2 = _mm_unpackhi_pd(t2,t2);
    t1 = _mm_mul_pd(t1, t0); 
    t2 = _mm_mul_pd(t2, t0); 
    t1 = _mm_shuffle_pd(t1, t1, 1); 
    t1 = _mm_addsub_pd(t2, t1);
    t1 = _mm_shuffle_pd(t1, t1, 1); 
    return t1;
}



#endif
