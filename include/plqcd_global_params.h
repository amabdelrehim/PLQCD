/*******************************************************************
 * File plqcd_global_params.h
 *
 * Copyright (C) 2012 
 * - A. Abdel-Rehim 
 * - G. Koutsou
 * - N. Anastopoulos
 * - N. Papadopoulou
 *
 * amabdelrehim@gmail.com, g.koutsou@gmail.com
 * 
 * This file is part of the PLQCD library
 * 
 * This file has the definition of the struct _PLQCD_GLOBAL_PARAMS
 * and the global variable "plqcd_g" which will be initialized
 * when PLQCD is intialized. This struct will be included in other
 * parts of the code such as the code which calculates the action
 * of the hopping matrix on a given input fermion field.
 ******************************************************************/

#ifndef PLQCD_GLOBAL_PARAMS_H_
#define PLQCD_GLOBAL_PARAMS_H_

#include<stdlib.h>
#include<stdbool.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<complex.h>
#include"mpi.h"
#include"su3.h"

#ifdef SSE3_INTRIN
#include"su3_intrin_elementary_128.h"
#include"su3_intrin_highlevel_128.h"
#include"su3_mult_splitlayout_128.h"
#endif

#ifdef AVX
#include"su3_256.h"
#include"su3_intrin_elementary_256.h"
#include"su3_intrin_highlevel_256.h"
#include"su3_mult_splitlayout_256.h"
#endif

#ifdef MIC
#include"su3_512.h"
#include"su3_intrin_elementary_512.h"
#include"su3_intrin_highlevel_512.h"
#include"su3_mult_splitlayout_512.h"
#endif

#include"utils.h"

#include"sse.h"


#ifdef _OPENMP
#include<omp.h>
#endif

/*
 * Global structure containing the global parameters
 * of PLQCD. Description of these parameters is given below.
 */

struct _PLQCD_GLOBAL_PARAMS{

  
    //MPI and openMP params
    int nprocs[4];      /*number of MPI processes in each direction
                         *nprocs[i] must be either 1 or even number
                         *default values are 1
                         */
                            
    int  nthread;       /*number of openmp threads that should be used
                         *default is 1
                         */

    int cpr[4];         /*cartesian co-ordinates of my process on the grid of processes*/

    int npr[8];         /*process id's of the 8 neighbouring processes. 
                         *npr[2*i]   is the id of the neighbouring process in the -ve i_th direction 
                         *npr[2*i+1] is the id of the neighbouring process in the +ve i_th direction 
                         *for i=0,1,2,3
                         */

    //Geometry
    int LX,LY,LZ,LT;    /*global sizes of the lattice in the x,y,z,and t directions(0,1,2,3 dirctions).
                         *default is 4
                         */ 
                 
    int latdims[4];     /*local lattice size for each MPI process in each direction
                         *latdims[i] must be an even number >= 4
                         */
                       
    int VOLUME;         /*Total volume of the local lattice=latdims[0]*latdims[1]*latdims[2]*latdims[3]*/


    int Vmic_split;     /* VOLUME/8 */

    int Vavx_split;     /* VOLUME/4 */

    int Vsse_split;     /* VOLUME/2 */

    int face[4];        /*face[i] is the number of sites on the boundary in the i_th direction:
                         *if nprocs[i]=1, face[i]=0, otherwise it is the size of this boundary
                         */

    int *ipt_eo;        /*For a site on the local lattice with co-ordinates x0,x1,x2,x3
                         *and with a lexiographic index:
                         *iy=x0+latdim[0]*x1+latdim[0]*latdim[1]*x2+latdim[0]*latdim[1]*latdim[2]*x3,
                         *ix=ipt[iy] is the even-odd index of that site such that the VOLUME/2
                         *even sites are counted first.
                         */

    int *ipt_lex;       /*gives the lexiographic index of an even-odd index (inverse of ipt_eo)*/

    int *nn_bnde[8];
    int *nn_bndo[8];    /*nearest neighbours of the boundary sites on the neaighbour processes
                         *nn_bnde[2*mu][i] is the index of the nearest neighbour site for the 
                         *even boundary site i at the -mu boundary on the nearest neighbour
                         *process in the -mu direction. Similarly, nn_bnde[2*mu+1][i] is the index
                         *for the nn site of the boundary site i at the +mu boundary on the 
                         *nearest neighbour process in the +mu direction. nn_bndo is defined similarly 
                         *but for odd boundary sites. Note that i runs from 0 to face[mu]/2, i.e. 
                         *i is just a counter of the boundary sites. However values of nn_bnde[][i]
                         *are the even-odd index of the nearest neighbour site. This is different
                         *from iup and idn in that they only deal with boundary sites and also that 
                         *they give the nearest neighbour site as if there was only a single process 
                         *in the mu directions.   
                         */


    int **iup;          /*iup[ix][mu] is the index of the nearest neighbour site to the 
                         *site ix in the +mu direction. If ix is a boundary site 
                         *on the +mu boundary, then iup points to the buffer of the spinor field.
                         */
                
    int **idn;          /*idn[ix][mu] is the index of the nearest neighbour 
                         *site in the -mu direction. If the site ix is a boundary 
                         *site on the -mu boundary, then idn points to the buffer of the spinor.
                         */



    int *ipt_eo_yzt;    //even-odd index of the points w..r.t the yzt block such that even points counted first
    int *ipt_lex_yzt;   //lexiographic index of the points in the yzt block

    int *ipt_eo_sse_split;  //even-odd index w.r.t Vsse_split
    int *ipt_lex_sse_split; //lexiographic index w.r.t. Vsse_split
    int **iup_sse_split;    //nearest neighbours in the + direction w.r.t. Vsse_split
    int **idn_sse_split;    //nearest neighbours in the - direction w.r.t. Vsse_split



    #ifdef AVX_SPLIT
    int *ipt_eo_avx_split;  //even-odd index w.r.t Vavx_split
    int *ipt_lex_avx_split; //lexiographic index w.r.t. Vavx_split
    int **iup_avx_split;    //nearest neighbours in the + direction w.r.t. Vavx_split
    int **idn_avx_split;    //nearest neighbours in the - direction w.r.t. Vavx_split
    #endif




    //global half spinor fields used with the application of the hopping matrix
    //These are of sizes VOLUME/2+face[i]

    halfspinor *phip[4];     /*phip will be used to compute the terms (1-gamma_j)psi(x)
                             */
    halfspinor *phim[4];     /*phim will be used to compute the terms inv(U_j(x))(1+gamma_j)psi(x)
                             */
    #ifdef MIC
    halfspinor_512 *phip512[4]; //buffers of size V/4 used in case of MIC
    halfspinor_512 *phim512[4];
    #ifdef MIC_SPLIT
    //arrays to navigate the lattice for the MIC with split layout 
    int *ipt_eo_mic_split;  //even-odd index w.r.t Vmic_split
    int *ipt_lex_mic_split; //lexiographic index w.r.t. Vmic_split
    int **iup_mic_split;  //nearest neighbours in the + direction w.r.t. Vmic_split
    int **idn_mic_split;  //nearest neighbours in the - direction w.r.t. Vmic_split


    double *phip512_re[4];
    double *phip512_im[4];
    double *phim512_re[4];
    double *phim512_im[4];
    #endif
    #endif


    //Memory alignment                          
    int ALIGN;       /*parameter for memory alignment. Memory will be aligned at 2^ALIGN boundary. default is 4*/
 

    //operator parameters
    //add here parameters for the Dirac operator

    //linear solver parameters
    //add here parameters for the linear solver

    int init;    /*if init=0, means plqcd is not initialized, otherwise it is.*/

}plqcd_g;

#endif



