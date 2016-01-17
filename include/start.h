/*******************************************************************************
 * File start.h
 * Copyright (C) 2012 
 * - A.M. Abdel-Rehim 
 * - G. Koutsou
 * - N. Anastopoulos
 * - N. Papadopoulou
 *
 * amabdelrehim@gmail.com, g.koutsou@gmail.com
 * 
 * This file is part of the PLQCD library
 * 
 * functions needed for intializations
 *******************************************************************************/

#ifndef _START_H
#define _START_H

#include"plqcd_global_params.h"

/*-Initializes MPI and openMP
 *-Reads input data from file and initialize global arrays that 
 * describes the lattice geometry. This should be the first function to be 
 * called before using PLQCD.
 *-Return 0 if it was successful or non-zero value if it fails
 * argc and argv are passed from the main calling program
 * it is expected that argc >=1 with argv[1] the name of the input file which 
 * has the global paramters read from.
 */
int init_plqcd(int argc, char* argv[]); 


/*Finalize MPI communications and do other tasks that the program might need.
 */
void finalize_plqcd();



/*single index for the process with co-ordinates 
 *n[0],n[1],n[2],n[3] on the grid of processes 
 ***********************************************/
int ipr(int n[]);


/*computes cartesian coordiantes n[] of a process on the 
 *grid of processes given its id value ip where ip=0,..,NPROCS-1 
 */
void comp_cpr(int ip, int n[]);

/*set indcies of the nearest neghibour process in the 
 *-ve and +ve 4 directions.
 *
 * npr[2*mu] is the index of the neighbour process
 * in the -mu direction.
 * 
 * npr[2*mu+1] is the index of the neighbour process
 * in the +mu direction.
 */
void set_npr(void);


/*given the lexiographic index of the site
 *it computes the cartesian coordinates of the 
 *site w.r.t. the local lattice.
 */
void lex2cart(int ix,int xcor[]);

/*compute the lexiographic index of a site given the 
 *coordinates w.r.t. the local lattice.    
 */
int cart2lex(int xcor[]);

/*sets the geometry arrays of the lattice
 */
void set_geometry();


/*given global coordinates of a site, compute a single index for 
  the process id that contain that site and the even-odd index of that
  site on that process.
 */

void find_site(int xg[], int *pid, int *xeo); 


//print global parametrs stored in the plqcd_g struct to a file
void print_params(FILE* );
 

#endif
