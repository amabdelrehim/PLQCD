/*****************************************************************
 * Copyright (C) 2012 Abdou Abdel-Rehim
 * This file is part of the PLQCD library
 * testing the performance of elementary processes on the MIC
 *****************************************************************/

#include"plqcd.h"
#include<stdlib.h>

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
   }


   int nthr;
   #ifdef _OPENMP
   #pragma omp parallel
   {
      nthr=omp_get_num_threads();
      if(proc_id==0)
        fprintf(ofp,"Number of threads as returned by openmp %d\n",nthr);
   }
   #endif

   //intialize the random number generator by a seed equals to the process rank
   srand((unsigned int) proc_id);

   //allocate spze for the gauge links
   su3_512 *u512 = (su3_512 *) amalloc(plqcd_g.VOLUME*sizeof(su3_512),plqcd_g.ALIGN); 
   su3_vector_512 *p1 = (su3_vector_512 *) amalloc(plqcd_g.VOLUME*sizeof(su3_vector_512),plqcd_g.ALIGN);
   su3_vector_512 *p2 = (su3_vector_512 *) amalloc(plqcd_g.VOLUME*sizeof(su3_vector_512),plqcd_g.ALIGN);
   su3_vector_512 *p3 = (su3_vector_512 *) amalloc(plqcd_g.VOLUME*sizeof(su3_vector_512),plqcd_g.ALIGN);

   if( (u512 == NULL) || (p1 == NULL) || (p2 == NULL) || (p3 == NULL) ){
     printf("couldn't allocate the needed memory\n");
     return 0;
   }

  
   //Initialize the input
   for(int i=0; i<plqcd_g.VOLUME; i++)
   {
       for(int j=0; j<4;  j++)
       {
         p1[i].c0.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         p1[i].c1.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         p1[i].c2.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         p2[i].c0.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         p2[i].c1.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         p2[i].c2.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         p3[i].c0.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         p3[i].c1.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         p3[i].c2.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;

         
         u512[i].c00.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         u512[i].c01.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         u512[i].c02.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         u512[i].c10.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         u512[i].c11.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         u512[i].c12.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         u512[i].c20.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         u512[i].c21.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
         u512[i].c22.z[j]= rand()/(double)RAND_MAX +I* rand()/(double) RAND_MAX;
       }
   }


   double t1,total;
   int    matvecs;

#ifdef MIC

   matvecs=0;
   total=0.0;

   //while(total < 10)
   //{
   //   for(int i=0; i<Nmul; i++)
   //   {
         t1=stop_watch(0.0);
         #ifdef _OPENMP
         #pragma omp parallel
         {
         #endif

         //double __attribute__((aligned(128))) dsign[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
         //__m512d sign = _mm512_load_pd(dsign);


            __m512d U[3][3], g1[3],g2[3],g3[3];
            #ifdef _OPENMP
            #pragma omp for
            #endif
            for(int j=0; j< plqcd_g.VOLUME; j++)
            {
                for(int i=0; i<Nmul; i++)
                {
                   //intrin_su3_load_512(U,&u512[j]);
                   //intrin_vector_load_512(g1,&p1[j]);
                   //intrin_vector_load_512(g2,&p2[j]);
                   //intrin_su3_multiply_512(gout,U,gin);
                   //intrin_vector_add_512(g3,g1,g2);
                   g1[0] = _mm512_load_pd(&p1[j].c0);
                   g1[1] = _mm512_load_pd(&p1[j].c1);
                   g1[2] = _mm512_load_pd(&p1[j].c2);
                   g2[0] = _mm512_load_pd(&p2[j].c0);
                   g2[1] = _mm512_load_pd(&p2[j].c1);
                   g2[2] = _mm512_load_pd(&p2[j].c2);
                   g3[0] = _mm512_add_pd(g1[0],g2[0]);
                   g3[1] = _mm512_add_pd(g1[1],g2[1]);
                   g3[2] = _mm512_add_pd(g1[2],g2[2]);
                   _mm512_store_pd(&p3[j].c0,g3[0]);
                   _mm512_store_pd(&p3[j].c1,g3[1]);
                   _mm512_store_pd(&p3[j].c2,g3[3]);
                   intrin_vector_store_512(&p3[j],g3);
                }
            }
         #ifdef _OPENMP
         }
         #endif
         t1 = stop_watch(t1);
         total += t1;
      //}
      //matvecs += Nmul*plqcd_g.VOLUME;
      matvecs = plqcd_g.VOLUME;
   //}
   
   fprintf(ofp,"a+b mic version:\n");
   fprintf(ofp,"------------------------------------------\n");
   fprintf(ofp,"test_elementary\tmult\t%d\ttotal(sec)\t%lf\tMFlops/process\t%lf\n",
             //matvecs,total,matvecs*3*66.0/total/1e+6);
             matvecs,total,matvecs*3*8/total/1e+6);

#endif //MIC

   finalize_plqcd();

   return 0;
}
