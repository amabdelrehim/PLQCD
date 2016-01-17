#ifndef _SU3_INTRIN_HIGHLEVEL_256_H_
#define _SU3_INTRIN_HIGHLEVEL_256_H_

#include "su3_256.h"
#include "su3_intrin_elementary_256.h"

//add prefetching functions here

/*
#ifdef P4
static inline void intrin_prefetch_su3_256(su3_256 *u) //size 288 
{
    _mm_prefetch((char *) u, _MM_HINT_T1);
    _mm_prefetch((char *) u+128, _MM_HINT_T1);
    _mm_prefetch((char *) u+256, _MM_HINT_T1);
}

static inline void intrin_prefetch_spinor_256(spinor_256 *p) //size 384 
{
    _mm_prefetch((char *) p, _MM_HINT_T1);
    _mm_prefetch((char *) p+128, _MM_HINT_T1);
    _mm_prefetch((char *) p+256, _MM_HINT_T1);
}
static inline void intrin_prefetch_halfspinor_256(halfspinor_256 *hp) //size 192
{
    _mm_prefetch((char *) hp, _MM_HINT_T1);
    _mm_prefetch((char *) hp+128, _MM_HINT_T1);
}
#endif
*/

//#ifdef OPTERON
static inline void intrin_prefetch_su3_256(su3_256 *u)  //size 288
{
    _mm_prefetch((char *) u, _MM_HINT_T1);
    _mm_prefetch((char *) u+64, _MM_HINT_T1);
    _mm_prefetch((char *) u+128, _MM_HINT_T1);
    _mm_prefetch((char *) u+192, _MM_HINT_T1);
    _mm_prefetch((char *) u+256, _MM_HINT_T1);
}

static inline void intrin_prefetch_spinor_256(spinor_256 *p)  //size 384
{
    _mm_prefetch((char *) p, _MM_HINT_T1);
    _mm_prefetch((char *) p+64, _MM_HINT_T1);
    _mm_prefetch((char *) p+128, _MM_HINT_T1);
    _mm_prefetch((char *) p+192, _MM_HINT_T1);
    _mm_prefetch((char *) p+256, _MM_HINT_T1);
    _mm_prefetch((char *) p+320, _MM_HINT_T1);
}
static inline void intrin_prefetch_halfspinor_256(halfspinor_256 *hp) //size 192
{
    _mm_prefetch((char *) hp, _MM_HINT_T1);
    _mm_prefetch((char *) hp+64, _MM_HINT_T1);
    _mm_prefetch((char *) hp+128, _MM_HINT_T1);
}
//#endif





static inline void intrin_vector_load_256(__m256d out[3], su3_vector_256 *v)
{
    //out[0] = _mm256_load_pd(v->c0.lo.ve);
    //out[1] = _mm256_load_pd(v->c1.lo.ve);
    //out[2] = _mm256_load_pd(v->c2.lo.ve);
    out[0] = _mm256_load_pd((double *) &((v->c0).lo));
    out[1] = _mm256_load_pd((double *) &((v->c1).lo));
    out[2] = _mm256_load_pd((double *) &((v->c2).lo));
}

/* 
static inline void intrin_vector_dummy_load_256(__m256d out[3], su3_vector *v)
{
    out[0] = _mm256_load_pd(v->c0.ve);
    out[1] = _mm256_load_pd(v->c1.ve);
    out[2] = _mm256_load_pd(v->c2.ve);
}
*/ 

static inline void intrin_vector_store_256(su3_vector_256 *v, __m256d in[3])
{
    //_mm256_store_pd(v->c0.lo.ve, in[0]);
    //_mm256_store_pd(v->c1.lo.ve, in[1]);
    //_mm256_store_pd(v->c2.lo.ve, in[2]);
    
    _mm256_store_pd((double *) &((v->c0).lo), in[0]);
    _mm256_store_pd((double *) &((v->c1).lo), in[1]);
    _mm256_store_pd((double *) &((v->c2).lo), in[2]);
}

static inline void intrin_vector_stream_256(su3_vector_256 *v, __m256d in[3])
{
    //_mm256_store_pd(v->c0.lo.ve, in[0]);
    //_mm256_store_pd(v->c1.lo.ve, in[1]);
    //_mm256_store_pd(v->c2.lo.ve, in[2]);
    
    _mm256_stream_pd((double *) &((v->c0).lo), in[0]);
    _mm256_stream_pd((double *) &((v->c1).lo), in[1]);
    _mm256_stream_pd((double *) &((v->c2).lo), in[2]);
}

/*
static inline void intrin_vector_dummy_store_256(su3_vector *v, __m256d in[3])
{
    _mm256_store_pd(v->c0.ve, in[0]);
    _mm256_store_pd(v->c1.ve, in[1]);
    _mm256_store_pd(v->c2.ve, in[2]);
}
*/


static inline void intrin_su3_load_256(__m256d Ui[3][3], su3_256 *U) 
{
      //Ui[0][0] = _mm256_load_pd((*U).c00.lo.ve);
      //Ui[0][1] = _mm256_load_pd((*U).c01.lo.ve);
      //Ui[0][2] = _mm256_load_pd((*U).c02.lo.ve);
      //Ui[1][0] = _mm256_load_pd((*U).c10.lo.ve);
      //Ui[1][1] = _mm256_load_pd((*U).c11.lo.ve);
      //Ui[1][2] = _mm256_load_pd((*U).c12.lo.ve);
      //Ui[2][0] = _mm256_load_pd((*U).c20.lo.ve);
      //Ui[2][1] = _mm256_load_pd((*U).c21.lo.ve);
      //Ui[2][2] = _mm256_load_pd((*U).c22.lo.ve);
      Ui[0][0] = _mm256_load_pd((double *) &((*U).c00.lo) );
      Ui[0][1] = _mm256_load_pd((double *) &((*U).c01.lo) );
      Ui[0][2] = _mm256_load_pd((double *) &((*U).c02.lo) );
      Ui[1][0] = _mm256_load_pd((double *) &((*U).c10.lo) );
      Ui[1][1] = _mm256_load_pd((double *) &((*U).c11.lo) );
      Ui[1][2] = _mm256_load_pd((double *) &((*U).c12.lo) );
      Ui[2][0] = _mm256_load_pd((double *) &((*U).c20.lo) );
      Ui[2][1] = _mm256_load_pd((double *) &((*U).c21.lo) );
      Ui[2][2] = _mm256_load_pd((double *) &((*U).c22.lo) );
}


static inline void intrin_su3_store_256(su3_256 *U, __m256d Ui[3][3]) 
{
      //Ui[0][0] = _mm256_load_pd((*U).c00.lo.ve);
      //Ui[0][1] = _mm256_load_pd((*U).c01.lo.ve);
      //Ui[0][2] = _mm256_load_pd((*U).c02.lo.ve);
      //Ui[1][0] = _mm256_load_pd((*U).c10.lo.ve);
      //Ui[1][1] = _mm256_load_pd((*U).c11.lo.ve);
      //Ui[1][2] = _mm256_load_pd((*U).c12.lo.ve);
      //Ui[2][0] = _mm256_load_pd((*U).c20.lo.ve);
      //Ui[2][1] = _mm256_load_pd((*U).c21.lo.ve);
      //Ui[2][2] = _mm256_load_pd((*U).c22.lo.ve);
      _mm256_store_pd((double *) &((*U).c00.lo), Ui[0][0] );
      _mm256_store_pd((double *) &((*U).c01.lo), Ui[0][1] );
      _mm256_store_pd((double *) &((*U).c02.lo), Ui[0][2] );
      _mm256_store_pd((double *) &((*U).c10.lo), Ui[1][0] );
      _mm256_store_pd((double *) &((*U).c11.lo), Ui[1][1] );
      _mm256_store_pd((double *) &((*U).c12.lo), Ui[1][2] );
      _mm256_store_pd((double *) &((*U).c20.lo), Ui[2][0] );
      _mm256_store_pd((double *) &((*U).c21.lo), Ui[2][1] );
      _mm256_store_pd((double *) &((*U).c22.lo), Ui[2][2] );
}







/*
static inline void intrin_su3_dummy_load_256(__m256d Ui[3][3], su3 *U) 
{
      Ui[0][0] = _mm256_load_pd((*U).c00.ve);
      Ui[0][1] = _mm256_load_pd((*U).c01.ve);
      Ui[0][2] = _mm256_load_pd((*U).c02.ve);
      Ui[1][0] = _mm256_load_pd((*U).c10.ve);
      Ui[1][1] = _mm256_load_pd((*U).c11.ve);
      Ui[1][2] = _mm256_load_pd((*U).c12.ve);
      Ui[2][0] = _mm256_load_pd((*U).c20.ve);
      Ui[2][1] = _mm256_load_pd((*U).c21.ve);
      Ui[2][2] = _mm256_load_pd((*U).c22.ve);
}
*/

static inline void intrin_vector_sub_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    out[0] = _mm256_sub_pd(in1[0], in2[0]);
    out[1] = _mm256_sub_pd(in1[1], in2[1]);
    out[2] = _mm256_sub_pd(in1[2], in2[2]);
} 
  
static inline void intrin_vector_add_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    out[0] = _mm256_add_pd(in1[0], in2[0]);
    out[1] = _mm256_add_pd(in1[1], in2[1]);
    out[2] = _mm256_add_pd(in1[2], in2[2]);
}
 

static inline void intrin_vector_i_sub_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    out[0] = complex_i_sub_regs_256(in1[0], in2[0]);
    out[1] = complex_i_sub_regs_256(in1[1], in2[1]);
    out[2] = complex_i_sub_regs_256(in1[2], in2[2]);
}


static inline void intrin_vector_i_add_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    out[0] = complex_i_add_regs_256(in1[0], in2[0]);
    out[1] = complex_i_add_regs_256(in1[1], in2[1]);
    out[2] = complex_i_add_regs_256(in1[2], in2[2]);
}


/**
 * Adds the low 128-bit lanes of the 256-bit elements, and subtracts the high 128-bit lanes
 */
static inline void intrin_vector_addlo_subhi_v2(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    in2[0] = intrin256_negate_hi(in2[0]);
    in2[1] = intrin256_negate_hi(in2[1]);
    in2[2] = intrin256_negate_hi(in2[2]);
   
    intrin_vector_add_256(out, in1, in2); 
}
 
/**
 * Adds the low 128-bit lanes of the 256-bit elements, and subtracts the high 128-bit lanes
 */
static inline void intrin_vector_addlo_subhi_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    __m256d res_lo[3], res_hi[3];

    intrin_vector_add_256(res_lo, in1, in2);
    intrin_vector_sub_256(res_hi, in1, in2);

    out[0] = intrin256_select_arg1lo_arg2hi(res_lo[0], res_hi[0]);
    out[1] = intrin256_select_arg1lo_arg2hi(res_lo[1], res_hi[1]);
    out[2] = intrin256_select_arg1lo_arg2hi(res_lo[2], res_hi[2]);
}
 
/**
 * Subtracts the low 128-bit lanes of the 256-bit elements, and i-adds the high 128-bit lanes
 */
static inline void intrin_vector_sublo_iaddhi_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    __m256d res_lo[3], res_hi[3];

    intrin_vector_sub_256(res_lo, in1, in2);
    intrin_vector_i_add_256(res_hi, in1, in2);

    out[0] = intrin256_select_arg1lo_arg2hi(res_lo[0], res_hi[0]);
    out[1] = intrin256_select_arg1lo_arg2hi(res_lo[1], res_hi[1]);
    out[2] = intrin256_select_arg1lo_arg2hi(res_lo[2], res_hi[2]);
}


 
/**
 * i-Adds the low 128-bit lanes of the 256-bit elements, and i-subtracts the high 128-bit lanes
 */
static inline void intrin_vector_iaddlo_isubhi_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    __m256d res_lo[3], res_hi[3];

    intrin_vector_i_add_256(res_lo, in1, in2);
    intrin_vector_i_sub_256(res_hi, in1, in2);

    out[0] = intrin256_select_arg1lo_arg2hi(res_lo[0], res_hi[0]);
    out[1] = intrin256_select_arg1lo_arg2hi(res_lo[1], res_hi[1]);
    out[2] = intrin256_select_arg1lo_arg2hi(res_lo[2], res_hi[2]);
}
 
/**
 * i-subtracts the low 128-bit lanes of the 256-bit elements, and i-adds the high 128-bit lanes
 */
static inline void intrin_vector_isublo_iaddhi_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    __m256d res_lo[3], res_hi[3];

    intrin_vector_i_sub_256(res_lo, in1, in2);
    intrin_vector_i_add_256(res_hi, in1, in2);

    out[0] = intrin256_select_arg1lo_arg2hi(res_lo[0], res_hi[0]);
    out[1] = intrin256_select_arg1lo_arg2hi(res_lo[1], res_hi[1]);
    out[2] = intrin256_select_arg1lo_arg2hi(res_lo[2], res_hi[2]);
}



static inline void intrin_complex_times_vector_256(__m256d out[3], __m256d ka, __m256d chi[3])
{
    out[0] = complex_mul_regs_256(ka, chi[0]);
    out[1] = complex_mul_regs_256(ka, chi[1]);
    out[2] = complex_mul_regs_256(ka, chi[2]);
} 
static inline void intrin_complexcjg_times_vector_256(__m256d out[3], __m256d v, __m256d in1[3])
{
    out[0] = complex_conj_mul_regs_256(v, in1[0]);
    out[1] = complex_conj_mul_regs_256(v, in1[1]);
    out[2] = complex_conj_mul_regs_256(v, in1[2]);
}
static inline void intrin_su3_multiply_256(__m256d chi[3], __m256d U[3][3], __m256d psi[3])
{
      __m256d tmp0, tmp1, tmp2;

      // chi_c0 = U_c00 * psi_c0 + U_c01 * psi_c1 + U_c02 * psi_c2; 
      tmp0 = complex_mul_regs_256(U[0][0], psi[0]);
      tmp1 = complex_mul_regs_256(U[0][1], psi[1]);
      tmp2 = complex_mul_regs_256(U[0][2], psi[2]);
      chi[0] = _mm256_add_pd(tmp0, tmp1);
      chi[0] = _mm256_add_pd(chi[0], tmp2);
      // chi_c1 = U_c10 * psi_c0 + U_c11 * psi_c1 + U_c12 * psi_c2; 
      tmp0 = complex_mul_regs_256(U[1][0], psi[0]);
      tmp1 = complex_mul_regs_256(U[1][1], psi[1]);
      tmp2 = complex_mul_regs_256(U[1][2], psi[2]);
      chi[1] = _mm256_add_pd(tmp0, tmp1);
      chi[1] = _mm256_add_pd(chi[1], tmp2);
      // chi_c2 = U_c20 * psi_c0 + U_c21 * psi_c1 + U_c22 * psi_c2; 
      tmp0 = complex_mul_regs_256(U[2][0], psi[0]);
      tmp1 = complex_mul_regs_256(U[2][1], psi[1]);
      tmp2 = complex_mul_regs_256(U[2][2], psi[2]);
      chi[2] = _mm256_add_pd(tmp0, tmp1);
      chi[2] = _mm256_add_pd(chi[2], tmp2);
 
}


static inline void intrin_su3_inverse_multiply_256(__m256d chi[3], __m256d U[3][3], __m256d psi[3])
{
      __m256d tmp0, tmp1, tmp2;

      tmp0 = complex_conj_mul_regs_256(U[0][0], psi[0]);
      tmp1 = complex_conj_mul_regs_256(U[1][0], psi[1]);
      tmp2 = complex_conj_mul_regs_256(U[2][0], psi[2]);
      chi[0] = _mm256_add_pd(tmp0, tmp1);
      chi[0] = _mm256_add_pd(chi[0], tmp2);
      tmp0 = complex_conj_mul_regs_256(U[0][1], psi[0]);
      tmp1 = complex_conj_mul_regs_256(U[1][1], psi[1]);
      tmp2 = complex_conj_mul_regs_256(U[2][1], psi[2]);
      chi[1] = _mm256_add_pd(tmp0, tmp1);
      chi[1] = _mm256_add_pd(chi[1], tmp2);
      tmp0 = complex_conj_mul_regs_256(U[0][2], psi[0]);
      tmp1 = complex_conj_mul_regs_256(U[1][2], psi[1]);
      tmp2 = complex_conj_mul_regs_256(U[2][2], psi[2]);
      chi[2] = _mm256_add_pd(tmp0, tmp1);
      chi[2] = _mm256_add_pd(chi[2], tmp2);
}




//these functions are not used in the current version
static inline void intrin_vector_select_arg1hi_arg2lo_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    out[0] = intrin256_select_arg1hi_arg2lo(in1[0], in2[0]);
    out[1] = intrin256_select_arg1hi_arg2lo(in1[1], in2[1]);
    out[2] = intrin256_select_arg1hi_arg2lo(in1[2], in2[2]);
}
 
static inline void intrin_vector_select_arg1lo_arg2lo_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    out[0] = intrin256_select_arg1lo_arg2lo(in1[0], in2[0]);
    out[1] = intrin256_select_arg1lo_arg2lo(in1[1], in2[1]);
    out[2] = intrin256_select_arg1lo_arg2lo(in1[2], in2[2]);
}
 
static inline void intrin_vector_select_arg1hi_arg2hi_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    out[0] = intrin256_select_arg1hi_arg2hi(in1[0], in2[0]);
    out[1] = intrin256_select_arg1hi_arg2hi(in1[1], in2[1]);
    out[2] = intrin256_select_arg1hi_arg2hi(in1[2], in2[2]);
}
    

static inline void intrin_vector_select_arg1lo_arg2hi_256(__m256d out[3], __m256d in1[3], __m256d in2[3])
{
    out[0] = intrin256_select_arg1lo_arg2hi(in1[0], in2[0]);
    out[1] = intrin256_select_arg1lo_arg2hi(in1[1], in2[1]);
    out[2] = intrin256_select_arg1lo_arg2hi(in1[2], in2[2]);
}
 

#endif
