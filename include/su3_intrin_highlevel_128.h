#ifndef _SU3_INTRIN_HIGHLEVEL_128_H_
#define _SU3_INTRIN_HIGHLEVEL_128_H_

#include "su3.h"
#include "su3_intrin_elementary_128.h"

static inline void intrin_vector_load_128(__m128d out[3], su3_vector *v)
{
    //out[0] = _mm_load_pd(v->c0.ve);
    //out[1] = _mm_load_pd(v->c1.ve);
    //out[2] = _mm_load_pd(v->c2.ve);
    out[0] = _mm_load_pd((double *) &(v->c0));
    out[1] = _mm_load_pd((double *) &(v->c1));
    out[2] = _mm_load_pd((double *) &(v->c2));
}
 
static inline void intrin_vector_store_128(su3_vector *v, __m128d in[3])
{
    //_mm_store_pd(v->c0.ve, in[0]);
    //_mm_store_pd(v->c1.ve, in[1]);
    //_mm_store_pd(v->c2.ve, in[2]);
    _mm_store_pd((double *) &(v->c0), in[0]);
    _mm_store_pd((double *) &(v->c1), in[1]);
    _mm_store_pd((double *) &(v->c2), in[2]);
}

static inline void intrin_vector_stream_128(su3_vector *v, __m128d in[3])
{
    //_mm_store_pd(v->c0.ve, in[0]);
    //_mm_store_pd(v->c1.ve, in[1]);
    //_mm_store_pd(v->c2.ve, in[2]);
    _mm_stream_pd((double *) &(v->c0), in[0]);
    _mm_stream_pd((double *) &(v->c1), in[1]);
    _mm_stream_pd((double *) &(v->c2), in[2]);
}

static inline void intrin_su3_load_128(__m128d Ui[3][3], su3 *U) 
{
      Ui[0][0] = _mm_load_pd((double *) &((*U).c00));
      Ui[0][1] = _mm_load_pd((double *) &((*U).c01));
      Ui[0][2] = _mm_load_pd((double *) &((*U).c02));
      Ui[1][0] = _mm_load_pd((double *) &((*U).c10));
      Ui[1][1] = _mm_load_pd((double *) &((*U).c11));
      Ui[1][2] = _mm_load_pd((double *) &((*U).c12));
      Ui[2][0] = _mm_load_pd((double *) &((*U).c20));
      Ui[2][1] = _mm_load_pd((double *) &((*U).c21));
      Ui[2][2] = _mm_load_pd((double *) &((*U).c22));
}


static inline void intrin_su3_store_128(su3* U, __m128d Ui[3][3]) 
{
      _mm_store_pd((double *) &((*U).c00), Ui[0][0]);
      _mm_store_pd((double *) &((*U).c01), Ui[0][1]);
      _mm_store_pd((double *) &((*U).c02), Ui[0][2]);
      _mm_store_pd((double *) &((*U).c10), Ui[1][0]);
      _mm_store_pd((double *) &((*U).c11), Ui[1][1]);
      _mm_store_pd((double *) &((*U).c12), Ui[1][2]);
      _mm_store_pd((double *) &((*U).c20), Ui[2][0]);
      _mm_store_pd((double *) &((*U).c21), Ui[2][1]);
      _mm_store_pd((double *) &((*U).c22), Ui[2][2]);
}







static inline void intrin_vector_sub_128(__m128d out[3], __m128d in1[3], __m128d in2[3])
{
    out[0] = _mm_sub_pd(in1[0], in2[0]);
    out[1] = _mm_sub_pd(in1[1], in2[1]);
    out[2] = _mm_sub_pd(in1[2], in2[2]);
} 
  
static inline void intrin_vector_add_128(__m128d out[3], __m128d in1[3], __m128d in2[3])
{
    out[0] = _mm_add_pd(in1[0], in2[0]);
    out[1] = _mm_add_pd(in1[1], in2[1]);
    out[2] = _mm_add_pd(in1[2], in2[2]);
}

static inline void intrin_vector_i_sub_128(__m128d out[3], __m128d in1[3], __m128d in2[3])
{
    out[0] = complex_i_sub_regs_128(in1[0], in2[0]);
    out[1] = complex_i_sub_regs_128(in1[1], in2[1]);
    out[2] = complex_i_sub_regs_128(in1[2], in2[2]);
}


static inline void intrin_vector_i_add_128(__m128d out[3], __m128d in1[3], __m128d in2[3])
{
    out[0] = complex_i_add_regs_128(in1[0], in2[0]);
    out[1] = complex_i_add_regs_128(in1[1], in2[1]);
    out[2] = complex_i_add_regs_128(in1[2], in2[2]);
}

static inline void intrin_complex_times_vector_128(__m128d out[3], __m128d ka, __m128d chi[3])
{
    out[0] = complex_mul_regs_128(ka, chi[0]);
    out[1] = complex_mul_regs_128(ka, chi[1]);
    out[2] = complex_mul_regs_128(ka, chi[2]);
}

static inline void intrin_complexcjg_times_vector_128(__m128d out[3], __m128d v, __m128d in1[3])
{
    out[0] = complex_conj_mul_regs_128(v, in1[0]);
    out[1] = complex_conj_mul_regs_128(v, in1[1]);
    out[2] = complex_conj_mul_regs_128(v, in1[2]);
}


static inline void intrin_su3_multiply_128(__m128d chi[3], __m128d U[3][3], __m128d psi[3])
{
      __m128d tmp0, tmp1, tmp2;

      // chi_c0 = U_c00 * psi_c0 + U_c01 * psi_c1 + U_c02 * psi_c2; 
      tmp0 = complex_mul_regs_128(U[0][0], psi[0]);
      tmp1 = complex_mul_regs_128(U[0][1], psi[1]);
      tmp2 = complex_mul_regs_128(U[0][2], psi[2]);
      chi[0] = _mm_add_pd(tmp0, tmp1);
      chi[0] = _mm_add_pd(chi[0], tmp2);
      // chi_c1 = U_c10 * psi_c0 + U_c11 * psi_c1 + U_c12 * psi_c2; 
      tmp0 = complex_mul_regs_128(U[1][0], psi[0]);
      tmp1 = complex_mul_regs_128(U[1][1], psi[1]);
      tmp2 = complex_mul_regs_128(U[1][2], psi[2]);
      chi[1] = _mm_add_pd(tmp0, tmp1);
      chi[1] = _mm_add_pd(chi[1], tmp2);
      // chi_c2 = U_c20 * psi_c0 + U_c21 * psi_c1 + U_c22 * psi_c2; 
      tmp0 = complex_mul_regs_128(U[2][0], psi[0]);
      tmp1 = complex_mul_regs_128(U[2][1], psi[1]);
      tmp2 = complex_mul_regs_128(U[2][2], psi[2]);
      chi[2] = _mm_add_pd(tmp0, tmp1);
      chi[2] = _mm_add_pd(chi[2], tmp2);
 
}

static inline void intrin_su3_inverse_multiply_128(__m128d chi[3], __m128d U[3][3], __m128d psi[3])
{
      __m128d tmp0, tmp1, tmp2;

      tmp0 = complex_conj_mul_regs_128(U[0][0], psi[0]);
      tmp1 = complex_conj_mul_regs_128(U[1][0], psi[1]);
      tmp2 = complex_conj_mul_regs_128(U[2][0], psi[2]);
      chi[0] = _mm_add_pd(tmp0, tmp1);
      chi[0] = _mm_add_pd(chi[0], tmp2);
      tmp0 = complex_conj_mul_regs_128(U[0][1], psi[0]);
      tmp1 = complex_conj_mul_regs_128(U[1][1], psi[1]);
      tmp2 = complex_conj_mul_regs_128(U[2][1], psi[2]);
      chi[1] = _mm_add_pd(tmp0, tmp1);
      chi[1] = _mm_add_pd(chi[1], tmp2);
      tmp0 = complex_conj_mul_regs_128(U[0][2], psi[0]);
      tmp1 = complex_conj_mul_regs_128(U[1][2], psi[1]);
      tmp2 = complex_conj_mul_regs_128(U[2][2], psi[2]);
      chi[2] = _mm_add_pd(tmp0, tmp1);
      chi[2] = _mm_add_pd(chi[2], tmp2);
 
}
#endif

