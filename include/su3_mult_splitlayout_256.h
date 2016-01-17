#ifndef _SU3_MULT_SPLITLAYOUT_256_H_
#define _SU3_MULT_SPLITLAYOUT_256_H_

#include<immintrin.h>

//chi = U * psi

static inline void su3_multiply_splitlayout_256(  __m256d chi_re[3],  
                                                  __m256d chi_im[3], 
                                                  __m256d U_re[3][3], 
                                                  __m256d U_im[3][3], 
                                                  __m256d psi_re[3],  
                                                  __m256d psi_im[3])
{

  __m256d register t;

  chi_re[0] =   _mm256_mul_pd(U_re[0][0],psi_re[0]);
  t         =   _mm256_mul_pd(U_re[0][1],psi_re[1]);
  chi_re[0] =   _mm256_add_pd(chi_re[0],t);
  t         =   _mm256_mul_pd(U_re[0][2],psi_re[2]);
  chi_re[0] =   _mm256_add_pd(chi_re[0],t);
  t         =   _mm256_mul_pd(U_im[0][0],psi_im[0]);
  chi_re[0] =   _mm256_sub_pd(chi_re[0],t);
  t         =   _mm256_mul_pd(U_im[0][1],psi_im[1]);
  chi_re[0] =   _mm256_sub_pd(chi_re[0],t);
  t         =   _mm256_mul_pd(U_im[0][2],psi_im[2]);
  chi_re[0] =   _mm256_sub_pd(chi_re[0],t);

                      
  chi_im[0] =   _mm256_mul_pd(U_re[0][0],psi_im[0]);
  t         =   _mm256_mul_pd(U_re[0][1],psi_im[1]);
  chi_im[0] =   _mm256_add_pd(chi_im[0],t);
  t         =   _mm256_mul_pd(U_re[0][2],psi_im[2]);
  chi_im[0] =   _mm256_add_pd(chi_im[0],t);
  t         =   _mm256_mul_pd(U_im[0][0],psi_re[0]);
  chi_im[0] =   _mm256_add_pd(chi_im[0],t);
  t         =   _mm256_mul_pd(U_im[0][1],psi_re[1]);
  chi_im[0] =   _mm256_add_pd(chi_im[0],t);
  t         =   _mm256_mul_pd(U_im[0][2],psi_re[2]);
  chi_im[0] =   _mm256_add_pd(chi_im[0],t);

                      
  chi_re[1] =   _mm256_mul_pd(U_re[1][0],psi_re[0]);
  t         =   _mm256_mul_pd(U_re[1][1],psi_re[1]);
  chi_re[1] =   _mm256_add_pd(chi_re[1],t);
  t         =   _mm256_mul_pd(U_re[1][2],psi_re[2]);
  chi_re[1] =   _mm256_add_pd(chi_re[1],t);
  t         =   _mm256_mul_pd(U_im[1][0],psi_im[0]);
  chi_re[1] =   _mm256_sub_pd(chi_re[1],t);
  t         =   _mm256_mul_pd(U_im[1][1],psi_im[1]);
  chi_re[1] =   _mm256_sub_pd(chi_re[1],t);
  t         =   _mm256_mul_pd(U_im[1][2],psi_im[2]);
  chi_re[1] =   _mm256_sub_pd(chi_re[1],t);

                      
  chi_im[1] =   _mm256_mul_pd(U_re[1][0],psi_im[0]);
  t         =   _mm256_mul_pd(U_re[1][1],psi_im[1]);
  chi_im[1] =   _mm256_add_pd(chi_im[1],t);
  t         =   _mm256_mul_pd(U_re[1][2],psi_im[2]);
  chi_im[1] =   _mm256_add_pd(chi_im[1],t);
  t         =   _mm256_mul_pd(U_im[1][0],psi_re[0]);
  chi_im[1] =   _mm256_add_pd(chi_im[1],t);
  t         =   _mm256_mul_pd(U_im[1][1],psi_re[1]);
  chi_im[1] =   _mm256_add_pd(chi_im[1],t);
  t         =   _mm256_mul_pd(U_im[1][2],psi_re[2]);
  chi_im[1] =   _mm256_add_pd(chi_im[1],t);


  //chi[2]                      
  chi_re[2] =   _mm256_mul_pd(U_re[2][0],psi_re[0]);
  t         =   _mm256_mul_pd(U_re[2][1],psi_re[1]);
  chi_re[2] =   _mm256_add_pd(chi_re[2],t);
  t         =   _mm256_mul_pd(U_re[2][2],psi_re[2]);
  chi_re[2] =   _mm256_add_pd(chi_re[2],t);
  t         =   _mm256_mul_pd(U_im[2][0],psi_im[0]);
  chi_re[2] =   _mm256_sub_pd(chi_re[2],t);
  t         =   _mm256_mul_pd(U_im[2][1],psi_im[1]);
  chi_re[2] =   _mm256_sub_pd(chi_re[2],t);
  t         =   _mm256_mul_pd(U_im[2][2],psi_im[2]);
  chi_re[2] =   _mm256_sub_pd(chi_re[2],t);

                      
  chi_im[2] =   _mm256_mul_pd(U_re[2][0],psi_im[0]);
  t         =   _mm256_mul_pd(U_re[2][1],psi_im[1]);
  chi_im[2] =   _mm256_add_pd(chi_im[2],t);
  t         =   _mm256_mul_pd(U_re[2][2],psi_im[2]);
  chi_im[2] =   _mm256_add_pd(chi_im[2],t);
  t         =   _mm256_mul_pd(U_im[2][0],psi_re[0]);
  chi_im[2] =   _mm256_add_pd(chi_im[2],t);
  t         =   _mm256_mul_pd(U_im[2][1],psi_re[1]);
  chi_im[2] =   _mm256_add_pd(chi_im[2],t);
  t         =   _mm256_mul_pd(U_im[2][2],psi_re[2]);
  chi_im[2] =   _mm256_add_pd(chi_im[2],t);

                      
}
   

//chi = U^dagger * psi

static inline void su3_inverse_multiply_splitlayout_256(  
                                           __m256d chi_re[3],  
                                           __m256d chi_im[3], 
                                           __m256d U_re[3][3], 
                                           __m256d U_im[3][3], 
                                           __m256d psi_re[3],  
                                           __m256d psi_im[3])
{


  __m256d register t;


  //chi[0]
  chi_re[0] =   _mm256_mul_pd(U_re[0][0],psi_re[0]);
  t         =   _mm256_mul_pd(U_re[1][0],psi_re[1]);
  chi_re[0] =   _mm256_add_pd(chi_re[0],t);
  t         =   _mm256_mul_pd(U_re[2][0],psi_re[2]);
  chi_re[0] =   _mm256_add_pd(chi_re[0],t);

  t         =   _mm256_mul_pd(U_im[0][0],psi_im[0]);
  chi_re[0] =   _mm256_add_pd(chi_re[0],t);
  t         =   _mm256_mul_pd(U_im[1][0],psi_im[1]);
  chi_re[0] =   _mm256_add_pd(chi_re[0],t);
  t         =   _mm256_mul_pd(U_im[2][0],psi_im[2]);
  chi_re[0] =   _mm256_add_pd(chi_re[0],t);

                      
  chi_im[0] =   _mm256_mul_pd(U_re[0][0],psi_im[0]);
  t         =   _mm256_mul_pd(U_re[1][0],psi_im[1]);
  chi_im[0] =   _mm256_add_pd(chi_im[0],t);
  t         =   _mm256_mul_pd(U_re[2][0],psi_im[2]);
  chi_im[0] =   _mm256_add_pd(chi_im[0],t);

  t         =   _mm256_mul_pd(U_im[0][0],psi_re[0]);
  chi_im[0] =   _mm256_sub_pd(chi_im[0],t);
  t         =   _mm256_mul_pd(U_im[1][0],psi_re[1]);
  chi_im[0] =   _mm256_sub_pd(chi_im[0],t);
  t         =   _mm256_mul_pd(U_im[2][0],psi_re[2]);
  chi_im[0] =   _mm256_sub_pd(chi_im[0],t);

  //chi[1]
  chi_re[1] =   _mm256_mul_pd(U_re[0][1],psi_re[0]);
  t         =   _mm256_mul_pd(U_re[1][1],psi_re[1]);
  chi_re[1] =   _mm256_add_pd(chi_re[1],t);
  t         =   _mm256_mul_pd(U_re[2][1],psi_re[2]);
  chi_re[1] =   _mm256_add_pd(chi_re[1],t);

  t         =   _mm256_mul_pd(U_im[0][1],psi_im[0]);
  chi_re[1] =   _mm256_add_pd(chi_re[1],t);
  t         =   _mm256_mul_pd(U_im[1][1],psi_im[1]);
  chi_re[1] =   _mm256_add_pd(chi_re[1],t);
  t         =   _mm256_mul_pd(U_im[2][1],psi_im[2]);
  chi_re[1] =   _mm256_add_pd(chi_re[1],t);

                      
  chi_im[1] =   _mm256_mul_pd(U_re[0][1],psi_im[0]);
  t         =   _mm256_mul_pd(U_re[1][1],psi_im[1]);
  chi_im[1] =   _mm256_add_pd(chi_im[1],t);
  t         =   _mm256_mul_pd(U_re[2][1],psi_im[2]);
  chi_im[1] =   _mm256_add_pd(chi_im[1],t);

  t         =   _mm256_mul_pd(U_im[0][1],psi_re[0]);
  chi_im[1] =   _mm256_sub_pd(chi_im[1],t);
  t         =   _mm256_mul_pd(U_im[1][1],psi_re[1]);
  chi_im[1] =   _mm256_sub_pd(chi_im[1],t);
  t         =   _mm256_mul_pd(U_im[2][1],psi_re[2]);
  chi_im[1] =   _mm256_sub_pd(chi_im[1],t);

                      
  //chi[2]
  chi_re[2] =   _mm256_mul_pd(U_re[0][2],psi_re[0]);
  t         =   _mm256_mul_pd(U_re[1][2],psi_re[1]);
  chi_re[2] =   _mm256_add_pd(chi_re[2],t);
  t         =   _mm256_mul_pd(U_re[2][2],psi_re[2]);
  chi_re[2] =   _mm256_add_pd(chi_re[2],t);

  t         =   _mm256_mul_pd(U_im[0][2],psi_im[0]);
  chi_re[2] =   _mm256_add_pd(chi_re[2],t);
  t         =   _mm256_mul_pd(U_im[1][2],psi_im[1]);
  chi_re[2] =   _mm256_add_pd(chi_re[2],t);
  t         =   _mm256_mul_pd(U_im[2][2],psi_im[2]);
  chi_re[2] =   _mm256_add_pd(chi_re[2],t);

                      
  chi_im[2] =   _mm256_mul_pd(U_re[0][2],psi_im[0]);
  t         =   _mm256_mul_pd(U_re[1][2],psi_im[1]);
  chi_im[2] =   _mm256_add_pd(chi_im[2],t);
  t         =   _mm256_mul_pd(U_re[2][2],psi_im[2]);
  chi_im[2] =   _mm256_add_pd(chi_im[2],t);

  t         =   _mm256_mul_pd(U_im[0][2],psi_re[0]);
  chi_im[2] =   _mm256_sub_pd(chi_im[2],t);
  t         =   _mm256_mul_pd(U_im[1][2],psi_re[1]);
  chi_im[2] =   _mm256_sub_pd(chi_im[2],t);
  t         =   _mm256_mul_pd(U_im[2][2],psi_re[2]);
  chi_im[2] =   _mm256_sub_pd(chi_im[2],t);

                      
                      

}
   
#endif
