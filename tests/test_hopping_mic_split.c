/*****************************************************************
 * Copyright (C) 2012 Abdou Abdel-Rehim
 * This file is part of the PLQCD library
 *****************************************************************/

#include"plqcd.h"
#include<stdlib.h>


/***************************************************
 * Testing the hopping matrix
 ***************************************************/


int main(int argc, char **argv)
{
   //initialize plqcd
   int init_status;

   if(argc < 3){
     fprintf(stderr,"Error. Must pass the name of the input file and the number of multiplications to be performed \n");
     fprintf(stderr,"Usage: %s input_file_name Nmul\n",argv[0]);
     exit(1);
   }

   init_status = init_plqcd(argc,argv);   
   
   if(init_status != 0)
     printf("Error initializing plqcd\n");

   int proc_id;
   int Nmul;
   proc_id = ipr(plqcd_g.cpr);

   Nmul=atoi(argv[2]); 

   int NPROCS=plqcd_g.nprocs[0]*plqcd_g.nprocs[1]*plqcd_g.nprocs[2]*plqcd_g.nprocs[3];

   char ofname[128];

   char buff[128];

   strcpy(ofname,"test_hopping_output.procgrid.");

   sprintf(buff,"%d-%d-%d-%d.nthreads.%d.proc.%d",plqcd_g.nprocs[0],plqcd_g.nprocs[1],plqcd_g.nprocs[2],plqcd_g.nprocs[3],plqcd_g.nthread,proc_id);

   strcat(ofname,buff);
      
   FILE *ofp;

   if(proc_id==0)
   {
      ofp=fopen(ofname,"w");
      fprintf(ofp,"INPUT GLOBALS:\n");
      fprintf(ofp,"----------------\n");
      fprintf(ofp,"NPROC0 %d, NPROC1 %d, NPROC2 %d, NPROC3 %d, NTHREAD %d\n",plqcd_g.nprocs[0],plqcd_g.nprocs[1],plqcd_g.nprocs[2],plqcd_g.nprocs[3], plqcd_g.nthread);
      fprintf(ofp,"L0 %d, L1 %d, L2 %d, L3 %d\n\n",plqcd_g.latdims[0],plqcd_g.latdims[1],plqcd_g.latdims[2],plqcd_g.latdims[3]);
      //printf("sizeof(spinor) %ld, sizeof(halfspinor) %ld, sizeof(su3) %ld \n",sizeof(spinor),sizeof(halfspinor),sizeof(su3));
   }


   int nthr;
   #ifdef _OPENMP
   #pragma omp parallel
   {
      nthr=omp_get_num_threads();
      if(omp_get_thread_num() == 0)
        if(proc_id==0)
          fprintf(ofp,"Number of threads as returned by openmp %d\n",nthr);
   }
   #endif


   double *pin_re  = (double *) _mm_malloc(plqcd_g.VOLUME*12*sizeof(double), 64);
   double *pin_im  = (double *) _mm_malloc(plqcd_g.VOLUME*12*sizeof(double), 64);
   double *pout_re = (double *) _mm_malloc(plqcd_g.VOLUME*12*sizeof(double), 64);
   double *pout_im = (double *) _mm_malloc(plqcd_g.VOLUME*12*sizeof(double), 64);
   double *u_re    = (double *) _mm_malloc(plqcd_g.VOLUME*36*sizeof(double), 64);
   double *u_im    = (double *) _mm_malloc(plqcd_g.VOLUME*36*sizeof(double), 64);
   if( (pin_re==NULL) || (pin_im==NULL) || (pout_re==NULL) || (pout_re==NULL) || (u_re==NULL) || (u_re==NULL) )
   {
       fprintf(stderr,"ERROR: insufficient memory for input/output spinors and gauge field inside test_hopping_mic_split .\n");
       exit(2);
   }

   //intialize the random number generator by a seed equals to the process rank
   srand((unsigned int) proc_id);

 
   //Initialize the input spinor and gauge links to random numbers 
   
   for(int i=0; i<plqcd_g.VOLUME; i++)
   {
      for(int j=0; j<12; j++)
      {
         pin_re[j+i*12]= rand() / (double) RAND_MAX;
         pin_im[j+i*12]= rand() / (double) RAND_MAX;
         pout_re[j+i*12]= rand() / (double) RAND_MAX;
         pout_im[j+i*12]= rand() / (double) RAND_MAX;
      }

      for(int k=0; k<36; k++)
      {
         u_re[k+i*36] = rand() / (double) RAND_MAX;
         u_im[k+i*36] = rand() / (double) RAND_MAX;
      }
   }


   double total,t1=0.0,t2=0.0,mytotal;
   int  matvecs;


   #ifdef MIC 
   #ifdef MIC_SPLIT
/*
   matvecs=0;
   total=0.0;
   mytotal =0.0;
   //printf("Hi there\n");
   t1=plqcd_hopping_matrix_eo_single_mic_split(pout_re,pout_im,u_re,u_im,pin_re,pin_im);
   printf("hello there from thread %d\n",omp_get_thread_num());
   while(mytotal < 30)
   {
      MPI_Barrier(MPI_COMM_WORLD); 
      for(int i=0; i<Nmul; i++)
      {
         t1=plqcd_hopping_matrix_eo_single_mic_split(pout_re, pout_im, u_re, u_im, pin_re, pin_im);
         t1=plqcd_hopping_matrix_eo_single_mic_split(pout_re, pout_im, u_re, u_im, pin_re, pin_im);
         mytotal += t1+t2;
      }
      matvecs += 2*Nmul;
   }
   
   MPI_Reduce(&mytotal,&total,1,MPI_DOUBLE,MPI_SUM,0, MPI_COMM_WORLD);
   MPI_Bcast(&total,1,MPI_DOUBLE,0, MPI_COMM_WORLD);
  

   if (proc_id==0)
   {
     total /= (double)(NPROCS);
   }
    

   if(proc_id==0)
   {
     fprintf(ofp,"mic with split layout:\n");
     fprintf(ofp,"------------------------------------------\n");
     fprintf(ofp,"test_hopping\tmult\t%d\ttotal(sec)\t%lf\tMFlops/process\t%lf\n",
             matvecs,total,(double )matvecs*plqcd_g.VOLUME/2.0*1200/total/1e+6);
   }

*/

   matvecs=0;
   total=0.0;
   mytotal =0.0;
   t1=plqcd_hopping_matrix_eo_single_mic_split_nohalfspinor(pout_re,pout_im,u_re,u_im,pin_re,pin_im);
   printf("hello there from thread %d\n",omp_get_thread_num());
   while(mytotal < 30)
   {
      MPI_Barrier(MPI_COMM_WORLD); 
      for(int i=0; i<Nmul; i++)
      {
         t1=plqcd_hopping_matrix_eo_single_mic_split_nohalfspinor(pout_re, pout_im, u_re, u_im, pin_re, pin_im);
         t1=plqcd_hopping_matrix_eo_single_mic_split_nohalfspinor(pout_re, pout_im, u_re, u_im, pin_re, pin_im);
         mytotal += t1+t2;
      }
      matvecs += 2*Nmul;
   }
   
   MPI_Reduce(&mytotal,&total,1,MPI_DOUBLE,MPI_SUM,0, MPI_COMM_WORLD);
   MPI_Bcast(&total,1,MPI_DOUBLE,0, MPI_COMM_WORLD);
  

   if (proc_id==0)
   {
     total /= (double)(NPROCS);
   }
    

   if(proc_id==0)
   {
     fprintf(ofp,"mic with split layout no-halfspinor:\n");
     fprintf(ofp,"------------------------------------------\n");
     fprintf(ofp,"test_hopping\tmult\t%d\ttotal(sec)\t%lf\tMFlops/process\t%lf\n",
             matvecs,total,(double )matvecs*plqcd_g.VOLUME/2.0*1200/total/1e+6);
   }

   #endif`
   #endif //MIC & MIC_SPLIT

   finalize_plqcd();

   return 0;
}
