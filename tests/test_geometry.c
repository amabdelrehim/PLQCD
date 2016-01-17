/*********************************************************************************
 * File test_geometry.c
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
 * testing the initialization of plqcd
 *********************************************************************************/

#include"plqcd.h"

/***************************************************
 *test code for the geometry setup of the lattice 
 **************************************************/
int main(int argc, char **argv)
{

   if(argc < 2){
     fprintf(stderr,"Error. Name of input file to read the parameters is needed\n");
     fprintf(stderr,"Usage: %s input_fname\n2",argv[1]);
   }

   //initialize plqcd
   int init_status;

   init_status = init_plqcd(argc,argv);   
   
   if(init_status != 0)
     printf("Error initializing plqcd\n");

   char ofname[128];

   char buff[128];

   int proc_id = ipr(plqcd_g.cpr);

   strcpy(ofname,"test_geometry_output.procgrid.");

   sprintf(buff,"%d-%d-%d-%d.proc.%d",plqcd_g.nprocs[0],plqcd_g.nprocs[1],plqcd_g.nprocs[2],plqcd_g.nprocs[3],proc_id);

   strcat(ofname,buff);

   //printf("hello from proc %d\n",proc_id);

   //printf("proc %d ofname %s \n",proc_id,ofname);
      
   FILE *ofp=fopen(ofname,"w");

   print_params(ofp);

   MPI_Finalize();

   return 0;
}
