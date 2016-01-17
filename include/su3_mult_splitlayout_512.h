#ifndef _SU3_MULT_SPLITLAYOUT_512_H_
#define _SU3_MULT_SPLITLAYOUT_512_H_

#include<immintrin.h>

/**
 * Returns the real part of the FMA operation:
 * c = c + a*b
 * 
 */
static inline __m512d complex_fma_real_512(__m512d c_re, __m512d c_im, __m512d a_re, __m512d a_im, __m512d b_re, __m512d b_im)
{
  c_re = _mm512_fmsub_pd(a_im, b_im, c_re);
  c_re = _mm512_fmsub_pd(a_re, b_re, c_re);
  
  return c_re;
}                    

/**
 * Returns the imaginary part of the FMA operation:
 * c = c + a*b
 */ 
static inline __m512d complex_fma_imag_512(__m512d c_re, __m512d c_im, __m512d a_re, __m512d a_im, __m512d b_re, __m512d b_im)
{
  c_im = _mm512_fmadd_pd(a_im, b_re, c_im);
  c_im = _mm512_fmadd_pd(a_re, b_im, c_im);

  return c_im;
}     


/** 
 * Performs the multiplication: 
 * chi(3x1) = U(3x3) * psi(3x1)
 *
 * Each vector or matrix element (e.g. chi[0] or U[1][2])
 * is actually a set of 8 complex numbers upon which 
 * operations are performed simultaneously. 
 * Each complex number is stored in split layout, meaning that
 * its real and imaginary parts are stored in different vector 
 * registers.  
 *
 * So, the 8 complex numbers contained in chi[0] will actually be
 * distributed in chi_re[0] and chi_im[0] in split layout.
 * 
 * 
 */
static inline void su3_multiply_splitlayout_512_Nikos(  __m512d chi_re[3],  
                                                  __m512d chi_im[3], 
                                                  __m512d U_re[3][3], 
                                                  __m512d U_im[3][3], 
                                                  __m512d psi_re[3],  
                                                  __m512d psi_im[3], 
						                                      __m512d zero)
{
  chi_re[0] = chi_re[1] = chi_re[2] = chi_im[0] = chi_im[1] = chi_im[2] = zero;

  // x0 = x0 + U00*y0
  chi_re[0] = complex_fma_real_512(chi_re[0], chi_im[0], U_re[0][0], U_im[0][0], psi_re[0], psi_im[0]);
  chi_im[0] = complex_fma_imag_512(chi_re[0], chi_im[0], U_re[0][0], U_im[0][0], psi_re[0], psi_im[0]);
  // x0 = x0 + U01*y1
  chi_re[0] = complex_fma_real_512(chi_re[0], chi_im[0], U_re[0][1], U_im[0][1], psi_re[1], psi_im[1]);
  chi_im[0] = complex_fma_imag_512(chi_re[0], chi_im[0], U_re[0][1], U_im[0][1], psi_re[1], psi_im[1]);
  // x0 = x0 + U02*y2
  chi_re[0] = complex_fma_real_512(chi_re[0], chi_im[0], U_re[0][2], U_im[0][2], psi_re[2], psi_im[2]);
  chi_im[0] = complex_fma_imag_512(chi_re[0], chi_im[0], U_re[0][2], U_im[0][2], psi_re[2], psi_im[2]);

  // x1 = x1 + U10*y0
  chi_re[1] = complex_fma_real_512(chi_re[1], chi_im[1], U_re[1][0], U_im[1][0], psi_re[0], psi_im[0]);
  chi_im[1] = complex_fma_imag_512(chi_re[1], chi_im[1], U_re[1][0], U_im[1][0], psi_re[0], psi_im[0]);
  // x1 = x1 + U11*y1
  chi_re[1] = complex_fma_real_512(chi_re[1], chi_im[1], U_re[1][1], U_im[1][1], psi_re[1], psi_im[1]);
  chi_im[1] = complex_fma_imag_512(chi_re[1], chi_im[1], U_re[1][1], U_im[1][1], psi_re[1], psi_im[1]);
  // x1 = x1 + U12*y2
  chi_re[1] = complex_fma_real_512(chi_re[1], chi_im[1], U_re[1][2], U_im[1][2], psi_re[2], psi_im[2]);
  chi_im[1] = complex_fma_imag_512(chi_re[1], chi_im[1], U_re[1][2], U_im[1][2], psi_re[2], psi_im[2]);  

  // x2 = x2 + U20*y0
  chi_re[2] = complex_fma_real_512(chi_re[2], chi_im[2], U_re[2][0], U_im[2][0], psi_re[0], psi_im[0]);
  chi_im[2] = complex_fma_imag_512(chi_re[2], chi_im[2], U_re[2][0], U_im[2][0], psi_re[0], psi_im[0]);
  // x2 = x2 + U21*y1
  chi_re[2] = complex_fma_real_512(chi_re[2], chi_im[2], U_re[2][1], U_im[2][1], psi_re[1], psi_im[1]);
  chi_im[2] = complex_fma_imag_512(chi_re[2], chi_im[2], U_re[2][1], U_im[2][1], psi_re[1], psi_im[1]);
  // x2 = x2 + U22*y2
  chi_re[2] = complex_fma_real_512(chi_re[2], chi_im[2], U_re[2][2], U_im[2][2], psi_re[2], psi_im[2]);
  chi_im[2] = complex_fma_imag_512(chi_re[2], chi_im[2], U_re[2][2], U_im[2][2], psi_re[2], psi_im[2]);  
                      
}
   
//chi = U * psi

static inline void su3_multiply_splitlayout_512(  __m512d chi_re[3],  
                                                  __m512d chi_im[3], 
                                                  __m512d U_re[3][3], 
                                                  __m512d U_im[3][3], 
                                                  __m512d psi_re[3],  
                                                  __m512d psi_im[3])
{

  chi_re[0] =   _mm512_mul_pd(U_im[0][0],psi_im[0]);
  chi_re[0] = _mm512_fmadd_pd(U_im[0][1],psi_im[1],chi_re[0]);
  chi_re[0] = _mm512_fmadd_pd(U_im[0][2],psi_im[2],chi_re[0]);
  chi_re[0] = _mm512_fmsub_pd(U_re[0][0],psi_re[0],chi_re[0]);
  chi_re[0] = _mm512_fmadd_pd(U_re[0][1],psi_re[1],chi_re[0]);
  chi_re[0] = _mm512_fmadd_pd(U_re[0][2],psi_re[2],chi_re[0]);


  chi_im[0] =   _mm512_mul_pd(U_re[0][0],psi_im[0]);
  chi_im[0] = _mm512_fmadd_pd(U_re[0][1],psi_im[1],chi_im[0]);
  chi_im[0] = _mm512_fmadd_pd(U_re[0][2],psi_im[2],chi_im[0]);
  chi_im[0] = _mm512_fmadd_pd(U_im[0][0],psi_re[0],chi_im[0]);
  chi_im[0] = _mm512_fmadd_pd(U_im[0][1],psi_re[1],chi_im[0]);
  chi_im[0] = _mm512_fmadd_pd(U_im[0][2],psi_re[2],chi_im[0]);

                      
  chi_re[1] =   _mm512_mul_pd(U_im[1][0],psi_im[0]);
  chi_re[1] = _mm512_fmadd_pd(U_im[1][1],psi_im[1],chi_re[1]);
  chi_re[1] = _mm512_fmadd_pd(U_im[1][2],psi_im[2],chi_re[1]);
  chi_re[1] = _mm512_fmsub_pd(U_re[1][0],psi_re[0],chi_re[1]);
  chi_re[1] = _mm512_fmadd_pd(U_re[1][1],psi_re[1],chi_re[1]);
  chi_re[1] = _mm512_fmadd_pd(U_re[1][2],psi_re[2],chi_re[1]);


  chi_im[1] =   _mm512_mul_pd(U_re[1][0],psi_im[0]);
  chi_im[1] = _mm512_fmadd_pd(U_re[1][1],psi_im[1],chi_im[1]);
  chi_im[1] = _mm512_fmadd_pd(U_re[1][2],psi_im[2],chi_im[1]);
  chi_im[1] = _mm512_fmadd_pd(U_im[1][0],psi_re[0],chi_im[1]);
  chi_im[1] = _mm512_fmadd_pd(U_im[1][1],psi_re[1],chi_im[1]);
  chi_im[1] = _mm512_fmadd_pd(U_im[1][2],psi_re[2],chi_im[1]);



                      
  chi_re[2] =   _mm512_mul_pd(U_im[2][0],psi_im[0]);
  chi_re[2] = _mm512_fmadd_pd(U_im[2][1],psi_im[1],chi_re[2]);
  chi_re[2] = _mm512_fmadd_pd(U_im[2][2],psi_im[2],chi_re[2]);
  chi_re[2] = _mm512_fmsub_pd(U_re[2][0],psi_re[0],chi_re[2]);
  chi_re[2] = _mm512_fmadd_pd(U_re[2][1],psi_re[1],chi_re[2]);
  chi_re[2] = _mm512_fmadd_pd(U_re[2][2],psi_re[2],chi_re[2]);


  chi_im[2] =   _mm512_mul_pd(U_re[2][0],psi_im[0]);
  chi_im[2] = _mm512_fmadd_pd(U_re[2][1],psi_im[1],chi_im[2]);
  chi_im[2] = _mm512_fmadd_pd(U_re[2][2],psi_im[2],chi_im[2]);
  chi_im[2] = _mm512_fmadd_pd(U_im[2][0],psi_re[0],chi_im[2]);
  chi_im[2] = _mm512_fmadd_pd(U_im[2][1],psi_re[1],chi_im[2]);
  chi_im[2] = _mm512_fmadd_pd(U_im[2][2],psi_re[2],chi_im[2]);

                      
}
   

//chi = U^dagger * psi

static inline void su3_inverse_multiply_splitlayout_512(  __m512d chi_re[3],  
                                                  __m512d chi_im[3], 
                                                  __m512d U_re[3][3], 
                                                  __m512d U_im[3][3], 
                                                  __m512d psi_re[3],  
                                                  __m512d psi_im[3])
{

  chi_re[0] =   _mm512_mul_pd(U_im[0][0],psi_im[0]);
  chi_re[0] = _mm512_fmadd_pd(U_im[1][0],psi_im[1],chi_re[0]);
  chi_re[0] = _mm512_fmadd_pd(U_im[2][0],psi_im[2],chi_re[0]);
  chi_re[0] = _mm512_fmadd_pd(U_re[0][0],psi_re[0],chi_re[0]);
  chi_re[0] = _mm512_fmadd_pd(U_re[1][0],psi_re[1],chi_re[0]);
  chi_re[0] = _mm512_fmadd_pd(U_re[2][0],psi_re[2],chi_re[0]);


  chi_im[0] =   _mm512_mul_pd(U_im[0][0],psi_re[0]);
  chi_im[0] = _mm512_fmadd_pd(U_im[1][0],psi_re[1],chi_im[0]);
  chi_im[0] = _mm512_fmadd_pd(U_im[2][0],psi_re[2],chi_im[0]);
  chi_im[0] = _mm512_fmsub_pd(U_re[0][0],psi_im[0],chi_im[0]);
  chi_im[0] = _mm512_fmadd_pd(U_re[1][0],psi_im[1],chi_im[0]);
  chi_im[0] = _mm512_fmadd_pd(U_re[2][0],psi_im[2],chi_im[0]);

                      
  chi_re[1] =   _mm512_mul_pd(U_im[0][1],psi_im[0]);
  chi_re[1] = _mm512_fmadd_pd(U_im[1][1],psi_im[1],chi_re[1]);
  chi_re[1] = _mm512_fmadd_pd(U_im[2][1],psi_im[2],chi_re[1]);
  chi_re[1] = _mm512_fmadd_pd(U_re[0][1],psi_re[0],chi_re[1]);
  chi_re[1] = _mm512_fmadd_pd(U_re[1][1],psi_re[1],chi_re[1]);
  chi_re[1] = _mm512_fmadd_pd(U_re[2][1],psi_re[2],chi_re[1]);


  chi_im[1] =   _mm512_mul_pd(U_im[0][1],psi_re[0]);
  chi_im[1] = _mm512_fmadd_pd(U_im[1][1],psi_re[1],chi_im[1]);
  chi_im[1] = _mm512_fmadd_pd(U_im[2][1],psi_re[2],chi_im[1]);
  chi_im[1] = _mm512_fmsub_pd(U_re[0][1],psi_im[0],chi_im[1]);
  chi_im[1] = _mm512_fmadd_pd(U_re[1][1],psi_im[1],chi_im[1]);
  chi_im[1] = _mm512_fmadd_pd(U_re[2][1],psi_im[2],chi_im[1]);

                      
  chi_re[2] =   _mm512_mul_pd(U_im[0][2],psi_im[0]);
  chi_re[2] = _mm512_fmadd_pd(U_im[1][2],psi_im[1],chi_re[2]);
  chi_re[2] = _mm512_fmadd_pd(U_im[2][2],psi_im[2],chi_re[2]);
  chi_re[2] = _mm512_fmadd_pd(U_re[0][2],psi_re[0],chi_re[2]);
  chi_re[2] = _mm512_fmadd_pd(U_re[1][2],psi_re[1],chi_re[2]);
  chi_re[2] = _mm512_fmadd_pd(U_re[2][2],psi_re[2],chi_re[2]);


  chi_im[2] =   _mm512_mul_pd(U_im[0][2],psi_re[0]);
  chi_im[2] = _mm512_fmadd_pd(U_im[1][2],psi_re[1],chi_im[2]);
  chi_im[2] = _mm512_fmadd_pd(U_im[2][2],psi_re[2],chi_im[2]);
  chi_im[2] = _mm512_fmsub_pd(U_re[0][2],psi_im[0],chi_im[2]);
  chi_im[2] = _mm512_fmadd_pd(U_re[1][2],psi_im[1],chi_im[2]);
  chi_im[2] = _mm512_fmadd_pd(U_re[2][2],psi_im[2],chi_im[2]);

                      
                      
}
   
#endif
