/*************************************************************************
 * Copyright (C) 2012 Abdou M. Abdel-Rehim, G. Koutsou, N. Anastopolous
 * This file is part of PLQCD libarary 
 * functions for the Dirac operator
 *************************************************************************/

#ifndef _DIRAC_H
#define _DIRAC_H

#include"plqcd_global_params.h"
#include"start.h"
#include"utils.h"

struct plqcd_timer{
    double total;
    double computation;
};

//SSE2/3 version with inline assymbly
#ifdef ASSYMBLY
double plqcd_hopping_matrix_eo_sse3_assymbly(spinor *qin, spinor *qout, su3 *u);
double plqcd_hopping_matrix_oe_sse3_assymbly(spinor *qin, spinor *qout, su3 *u);
#endif

//SSE3 with intrinsics
#ifdef SSE3_INTRIN
double plqcd_hopping_matrix_eo_sse3_intrin(spinor *qin, spinor *qout, su3 *u);
double plqcd_hopping_matrix_oe_sse3_intrin(spinor *qin, spinor *qout, su3 *u);
double plqcd_hopping_matrix_eo_sse3_intrin_blocking(spinor *qin, spinor *qout, su3 *u);
double plqcd_hopping_matrix_oe_sse3_intrin_blocking(spinor *qin, spinor *qout, su3 *u);
double plqcd_hopping_matrix_eo_sse3_intrin_nohalfspinor(double *qout_re, double *qout_im, double *u_re, double *u_im, double *qin_re, double *qin_im);
double plqcd_hopping_matrix_oe_sse3_intrin_nohalfspinor(double *qout_re, double *qout_im, double *u_re, double *u_im, double *qin_re, double *qin_im);
#endif

//AVX with intrinsics
#ifdef AVX
double plqcd_hopping_matrix_eo_intrin_256(spinor_256 *qin, spinor_256 *qout, su3_256 *u);
double plqcd_hopping_matrix_oe_intrin_256(spinor_256 *qin, spinor_256 *qout, su3_256 *u);
double plqcd_hopping_matrix_eo_avx_nohalfspinor(double *qout_re, double *qout_im, double *u_re, double *u_im, double *qin_re, double *qin_im);
double plqcd_hopping_matrix_oe_avx_nohalfspinor(double *qout_re, double *qout_im, double *u_re, double *u_im, double *qin_re, double *qin_im);
#endif


//MIC with intrinsics
#ifdef MIC
double plqcd_hopping_matrix_eo_intrin_512(spinor_512 *qin, spinor_512 *qout, su3_512 *u);
double plqcd_hopping_matrix_oe_intrin_512(spinor_512 *qin, spinor_512 *qout, su3_512 *u);
double plqcd_hopping_matrix_eo_single_mic(spinor_512 *qin, spinor_512 *qout, su3_512 *u);
double plqcd_hopping_matrix_oe_single_mic(spinor_512 *qin, spinor_512 *qout, su3_512 *u);
double plqcd_hopping_matrix_eo_single_mic_short(spinor_512 *qin, spinor_512 *qout, su3_512 *u);
double plqcd_hopping_matrix_oe_single_mic_short(spinor_512 *qin, spinor_512 *qout, su3_512 *u);


//note the order of the arguments
double plqcd_hopping_matrix_eo_single_mic_split(double *qout_re, double *qout_im, double *u_re, double *u_im, double *qin_re, double *qin_im);
double plqcd_hopping_matrix_oe_single_mic_split(double *qout_re, double *qout_im, double *u_re, double *u_im, double *qin_re, double *qin_im);

double plqcd_hopping_matrix_eo_single_mic_split_nohalfspinor(double *qout_re, double *qout_im, double *u_re, double *u_im, double *qin_re, double *qin_im);
double plqcd_hopping_matrix_oe_single_mic_split_nohalfspinor(double *qout_re, double *qout_im, double *u_re, double *u_im, double *qin_re, double *qin_im);

double plqcd_hopping_matrix_eo_single_mic_split_nohalfspinor_short(double *qout_re, double *qout_im, double *u_re, double *u_im, double *qin_re, double *qin_im);
double plqcd_hopping_matrix_oe_single_mic_split_nohalfspinor_short(double *qout_re, double *qout_im, double *u_re, double *u_im, double *qin_re, double *qin_im);
#endif

#endif
