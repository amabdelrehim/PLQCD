#ifndef _SU3_INTRIN_HIGHLEVEL_512_H_
#define _SU3_INTRIN_HIGHLEVEL_512_H_

#include "su3_512.h"
#include "su3_intrin_elementary_512.h"

#ifdef MIC
static inline void intrin_prefetch_su3_512(su3_512 *u) //size 576 
{
    _mm_prefetch((char *) u, _MM_HINT_T1);
    _mm_prefetch((char *) u+64, _MM_HINT_T1);
    _mm_prefetch((char *) u+128, _MM_HINT_T1);
    _mm_prefetch((char *) u+192, _MM_HINT_T1);
    _mm_prefetch((char *) u+256, _MM_HINT_T1);
    _mm_prefetch((char *) u+320, _MM_HINT_T1);
    _mm_prefetch((char *) u+384, _MM_HINT_T1);
    _mm_prefetch((char *) u+448, _MM_HINT_T1);
    _mm_prefetch((char *) u+512, _MM_HINT_T1);
}

static inline void intrin_prefetch_spinor_512(spinor_512 *p) //size 768 
{
    _mm_prefetch((char *) p, _MM_HINT_T1);
    _mm_prefetch((char *) p+64, _MM_HINT_T1);
    _mm_prefetch((char *) p+128, _MM_HINT_T1);
    _mm_prefetch((char *) p+192, _MM_HINT_T1);
    _mm_prefetch((char *) p+256, _MM_HINT_T1);
    _mm_prefetch((char *) p+320, _MM_HINT_T1);
    _mm_prefetch((char *) p+384, _MM_HINT_T1);
    _mm_prefetch((char *) p+448, _MM_HINT_T1);
    _mm_prefetch((char *) p+512, _MM_HINT_T1);
    _mm_prefetch((char *) p+576, _MM_HINT_T1);
    _mm_prefetch((char *) p+640, _MM_HINT_T1);
    _mm_prefetch((char *) p+704, _MM_HINT_T1);
}

static inline void intrin_prefetch_halfspinor_512(halfspinor_512 *hp) //size 384
{
    _mm_prefetch((char *) hp, _MM_HINT_T1);
    _mm_prefetch((char *) hp+64, _MM_HINT_T1);
    _mm_prefetch((char *) hp+128, _MM_HINT_T1);
    _mm_prefetch((char *) hp+192, _MM_HINT_T1);
    _mm_prefetch((char *) hp+256, _MM_HINT_T1);
    _mm_prefetch((char *) hp+320, _MM_HINT_T1);
}
#endif

static inline void intrin_vector_load_512(__m512d out[3], su3_vector_512 *v)
{
    out[0] = _mm512_load_pd((double *) &((v->c0).z[0]));
    out[1] = _mm512_load_pd((double *) &((v->c1).z[0]));
    out[2] = _mm512_load_pd((double *) &((v->c2).z[0]));
}

static inline void intrin_vector_store_512(su3_vector_512 *v, __m512d in[3])
{
    _mm512_store_pd((double *) &((v->c0).z[0]), in[0]);
    _mm512_store_pd((double *) &((v->c1).z[0]), in[1]);
    _mm512_store_pd((double *) &((v->c2).z[0]), in[2]);
}

static inline void intrin_vector_stream_512(su3_vector_512 *v, __m512d in[3])
{
    _mm512_stream_pd((double *) &((v->c0).z[0]), in[0]);
    _mm512_stream_pd((double *) &((v->c1).z[0]), in[1]);
    _mm512_stream_pd((double *) &((v->c2).z[0]), in[2]);
}

static inline void intrin_su3_load_512(__m512d Ui[3][3], su3_512 *U) 
{
      Ui[0][0] = _mm512_load_pd((double *) &((*U).c00.z[0]) );
      Ui[0][1] = _mm512_load_pd((double *) &((*U).c01.z[0]) );
      Ui[0][2] = _mm512_load_pd((double *) &((*U).c02.z[0]) );
      Ui[1][0] = _mm512_load_pd((double *) &((*U).c10.z[0]) );
      Ui[1][1] = _mm512_load_pd((double *) &((*U).c11.z[0]) );
      Ui[1][2] = _mm512_load_pd((double *) &((*U).c12.z[0]) );
      Ui[2][0] = _mm512_load_pd((double *) &((*U).c20.z[0]) );
      Ui[2][1] = _mm512_load_pd((double *) &((*U).c21.z[0]) );
      Ui[2][2] = _mm512_load_pd((double *) &((*U).c22.z[0]) );
}


//load only the first two rows and compute the third row using unitarity
//third_row = first_row cross_product second_row
static inline void intrin_su3_load_short_512(__m512d Ui[3][3], su3_512 *U) 
{
      Ui[0][0] = _mm512_load_pd((double *) &((*U).c00.z[0]) );
      Ui[0][1] = _mm512_load_pd((double *) &((*U).c01.z[0]) );
      Ui[0][2] = _mm512_load_pd((double *) &((*U).c02.z[0]) );
      Ui[1][0] = _mm512_load_pd((double *) &((*U).c10.z[0]) );
      Ui[1][1] = _mm512_load_pd((double *) &((*U).c11.z[0]) );
      Ui[1][2] = _mm512_load_pd((double *) &((*U).c12.z[0]) );

      __m512d register m1,m2;

      m1 = complex_mul_regs_512(Ui[0][1],Ui[1][2]);
      m2 = complex_mul_regs_512(Ui[0][2],Ui[1][1]);
      Ui[2][0] = _mm512_sub_pd(m1,m2);

      m1 = complex_mul_regs_512(Ui[0][2],Ui[1][0]);
      m2 = complex_mul_regs_512(Ui[0][0],Ui[1][2]);
      Ui[2][1] = _mm512_sub_pd(m1,m2);


      m1 = complex_mul_regs_512(Ui[0][0],Ui[1][1]);
      m2 = complex_mul_regs_512(Ui[0][1],Ui[1][0]);
      Ui[2][2] = _mm512_sub_pd(m1,m2);

}











static inline void intrin_su3_store_512(su3_512 *U, __m512d Ui[3][3]) 
{
      _mm512_store_pd((double *) &((*U).c00.z[0]), Ui[0][0] );
      _mm512_store_pd((double *) &((*U).c01.z[0]), Ui[0][1] );
      _mm512_store_pd((double *) &((*U).c02.z[0]), Ui[0][2] );
      _mm512_store_pd((double *) &((*U).c10.z[0]), Ui[1][0] );
      _mm512_store_pd((double *) &((*U).c11.z[0]), Ui[1][1] );
      _mm512_store_pd((double *) &((*U).c12.z[0]), Ui[1][2] );
      _mm512_store_pd((double *) &((*U).c20.z[0]), Ui[2][0] );
      _mm512_store_pd((double *) &((*U).c21.z[0]), Ui[2][1] );
      _mm512_store_pd((double *) &((*U).c22.z[0]), Ui[2][2] );
}




static inline void intrin_vector_sub_512(__m512d out[3], __m512d in1[3], __m512d in2[3])
{
    out[0] = _mm512_sub_pd(in1[0], in2[0]);
    out[1] = _mm512_sub_pd(in1[1], in2[1]);
    out[2] = _mm512_sub_pd(in1[2], in2[2]);
} 
  
static inline void intrin_vector_add_512(__m512d out[3], __m512d in1[3], __m512d in2[3])
{
    out[0] = _mm512_add_pd(in1[0], in2[0]);
    out[1] = _mm512_add_pd(in1[1], in2[1]);
    out[2] = _mm512_add_pd(in1[2], in2[2]);
}
 

static inline void intrin_vector_i_sub_512(__m512d out[3], __m512d in1[3], __m512d in2[3])
{
    out[0] = complex_i_sub_regs_512(in1[0], in2[0]);
    out[1] = complex_i_sub_regs_512(in1[1], in2[1]);
    out[2] = complex_i_sub_regs_512(in1[2], in2[2]);
}


static inline void intrin_vector_i_add_512(__m512d out[3], __m512d in1[3], __m512d in2[3])
{
    out[0] = complex_i_add_regs_512(in1[0], in2[0]);
    out[1] = complex_i_add_regs_512(in1[1], in2[1]);
    out[2] = complex_i_add_regs_512(in1[2], in2[2]);
}


static inline void intrin_su3_multiply_512(__m512d chi[3], __m512d U[3][3], __m512d psi[3])
{
      __m512d tmp0, tmp1, tmp2;

      // chi_c0 = U_c00 * psi_c0 + U_c01 * psi_c1 + U_c02 * psi_c2; 
      tmp0 = complex_mul_regs_512(U[0][0], psi[0]);
      tmp1 = complex_mul_regs_512(U[0][1], psi[1]);
      tmp2 = complex_mul_regs_512(U[0][2], psi[2]);
      chi[0] = _mm512_add_pd(tmp0, tmp1);
      chi[0] = _mm512_add_pd(chi[0], tmp2);
      // chi_c1 = U_c10 * psi_c0 + U_c11 * psi_c1 + U_c12 * psi_c2; 
      tmp0 = complex_mul_regs_512(U[1][0], psi[0]);
      tmp1 = complex_mul_regs_512(U[1][1], psi[1]);
      tmp2 = complex_mul_regs_512(U[1][2], psi[2]);
      chi[1] = _mm512_add_pd(tmp0, tmp1);
      chi[1] = _mm512_add_pd(chi[1], tmp2);
      // chi_c2 = U_c20 * psi_c0 + U_c21 * psi_c1 + U_c22 * psi_c2; 
      tmp0 = complex_mul_regs_512(U[2][0], psi[0]);
      tmp1 = complex_mul_regs_512(U[2][1], psi[1]);
      tmp2 = complex_mul_regs_512(U[2][2], psi[2]);
      chi[2] = _mm512_add_pd(tmp0, tmp1);
      chi[2] = _mm512_add_pd(chi[2], tmp2);
 
}


static inline void intrin_su3_inverse_multiply_512(__m512d chi[3], __m512d U[3][3], __m512d psi[3])
{
      __m512d tmp0, tmp1, tmp2;

      tmp0 = complex_conj_mul_regs_512(U[0][0], psi[0]);
      tmp1 = complex_conj_mul_regs_512(U[1][0], psi[1]);
      tmp2 = complex_conj_mul_regs_512(U[2][0], psi[2]);
      chi[0] = _mm512_add_pd(tmp0, tmp1);
      chi[0] = _mm512_add_pd(chi[0], tmp2);
      tmp0 = complex_conj_mul_regs_512(U[0][1], psi[0]);
      tmp1 = complex_conj_mul_regs_512(U[1][1], psi[1]);
      tmp2 = complex_conj_mul_regs_512(U[2][1], psi[2]);
      chi[1] = _mm512_add_pd(tmp0, tmp1);
      chi[1] = _mm512_add_pd(chi[1], tmp2);
      tmp0 = complex_conj_mul_regs_512(U[0][2], psi[0]);
      tmp1 = complex_conj_mul_regs_512(U[1][2], psi[1]);
      tmp2 = complex_conj_mul_regs_512(U[2][2], psi[2]);
      chi[2] = _mm512_add_pd(tmp0, tmp1);
      chi[2] = _mm512_add_pd(chi[2], tmp2);
}

#endif
