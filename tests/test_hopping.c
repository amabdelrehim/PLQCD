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
   int i,j,k,Nmul;
   proc_id = ipr(plqcd_g.cpr);

   Nmul=atoi(argv[2]); 

#if 0
   //Intialize the ranlux random number generator
   start_ranlux(0,1); 
#endif

   int NPROCS=plqcd_g.nprocs[0]*plqcd_g.nprocs[1]*plqcd_g.nprocs[2]*plqcd_g.nprocs[3];

   char ofname[128];

   char buff[128];

   strcpy(ofname,"test_hopping_output.procgrid.");

   sprintf(buff,"%d-%d-%d-%d.nthreads.%d.proc.%d",plqcd_g.nprocs[0],plqcd_g.nprocs[1],plqcd_g.nprocs[2],plqcd_g.nprocs[3],plqcd_g.nthread,proc_id);



   strcat(ofname,buff);

      
   FILE *ofp;

   //FILE *ofp_source;

   //if(proc_id==0)
   //{
   //     ofp_source = fopen("test_rand_vals.out","w");
   //}

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


   /*****************************************************
    *Testing the Dirac operator interface
    ****************************************************/


   

   spinor *pin= (spinor *) amalloc(plqcd_g.VOLUME*sizeof(spinor), plqcd_g.ALIGN);
   if(pin==NULL)
   {
       fprintf(stderr,"ERROR: insufficient memory for spinor pin.\n");
       exit(2);
   }

   spinor *pout= (spinor *) amalloc(plqcd_g.VOLUME*sizeof(spinor), plqcd_g.ALIGN);
   if(pout==NULL)
   {
       fprintf(stderr,"ERROR: insufficient memory for spinor pout.\n");
       exit(2);
   }

   su3 *ufield= (su3 *) amalloc(4*plqcd_g.VOLUME*sizeof(su3), plqcd_g.ALIGN);
   if(ufield==NULL)
   {
       fprintf(stderr,"ERROR: insufficient memory for gauge field ufield.\n");
       exit(2);
   }


   //256 arrays
#ifdef AVX
   spinor_256 *pin_256= (spinor_256 *) amalloc(plqcd_g.VOLUME/2*sizeof(spinor_256), plqcd_g.ALIGN);
   if(pin_256==NULL)
   {
       fprintf(stderr,"ERROR: insufficient memory for spinor pin_256.\n");
       exit(2);
   }

   
   spinor_256 *pout_256= (spinor_256 *) amalloc(plqcd_g.VOLUME/2*sizeof(spinor_256), plqcd_g.ALIGN);
   if(pout_256==NULL)
   {
       fprintf(stderr,"ERROR: insufficient memory for spinor pout_256.\n");
       exit(2);
   }


   su3_256 *ufield_256= (su3_256 *) amalloc(4*plqcd_g.VOLUME/2*sizeof(su3_256), plqcd_g.ALIGN);

   if(ufield_256==NULL)
   {
       fprintf(stderr,"ERROR: insufficient memory for gauge field ufield_256.\n");
       exit(2);
   }
#endif


   //512 arrays
#ifdef MIC
   spinor_512 *pin_512= (spinor_512 *) amalloc(plqcd_g.VOLUME/4*sizeof(spinor_512), plqcd_g.ALIGN);
   if(pin_512==NULL)
   {
       fprintf(stderr,"ERROR: insufficient memory for spinor pin_512.\n");
       exit(2);
   }

   
   spinor_512 *pout_512= (spinor_512 *) amalloc(plqcd_g.VOLUME/4*sizeof(spinor_512), plqcd_g.ALIGN);
   if(pout_512==NULL)
   {
       fprintf(stderr,"ERROR: insufficient memory for spinor pout_512.\n");
       exit(2);
   }


   su3_512 *ufield_512= (su3_512 *) amalloc(4*plqcd_g.VOLUME/4*sizeof(su3_512), plqcd_g.ALIGN);

   if(ufield_512==NULL)
   {
       fprintf(stderr,"ERROR: insufficient memory for gauge field ufield_512.\n");
       exit(2);
   }
#endif





   //intialize the random number generator by a seed equals to the process rank
   srand((unsigned int) proc_id);

 
   //Initialize the input spinor and gauge links to random numbers 



   //intialize the random number generator by a seed equals to the process rank
   srand((unsigned int) proc_id);

 
   //Initialize the input spinor and gauge links to random numbers 
   double ru[18];
   double rs[24];
   
   for(i=0; i<plqcd_g.VOLUME; i++)
   {
       //ranlxd(rs,24);
       for(j=0; j<24; j++)
       {
          rs[j]= rand() / (double)RAND_MAX;
          //fprintf(stderr,"rs[%d]=%lf\n",j,rs[j]);
       }

       pin[i].s0.c0=rs[0]+I*rs[1];
       pin[i].s0.c1=rs[2]+I*rs[3];
       pin[i].s0.c2=rs[4]+I*rs[5];
       pin[i].s1.c0=rs[6]+I*rs[7];
       pin[i].s1.c1=rs[8]+I*rs[9];
       pin[i].s1.c2=rs[10]+I*rs[11];
       pin[i].s2.c0=rs[12]+I*rs[13];
       pin[i].s2.c1=rs[14]+I*rs[15];
       pin[i].s2.c2=rs[16]+I*rs[17];
       pin[i].s3.c0=rs[18]+I*rs[19];
       pin[i].s3.c1=rs[20]+I*rs[21];
       pin[i].s3.c2=rs[22]+I*rs[23];


       //ranlxd(rs,24);
       for(j=0; j<24; j++)
          rs[j]= rand() / (double)RAND_MAX;

       pout[i].s0.c0=rs[0]+I*rs[1];
       pout[i].s0.c1=rs[2]+I*rs[3];
       pout[i].s0.c2=rs[4]+I*rs[5];
       pout[i].s1.c0=rs[6]+I*rs[7];
       pout[i].s1.c1=rs[8]+I*rs[9];
       pout[i].s1.c2=rs[10]+I*rs[11];
       pout[i].s2.c0=rs[12]+I*rs[13];
       pout[i].s2.c1=rs[14]+I*rs[15];
       pout[i].s2.c2=rs[16]+I*rs[17];
       pout[i].s3.c0=rs[18]+I*rs[19];
       pout[i].s3.c1=rs[20]+I*rs[21];
       pout[i].s3.c2=rs[22]+I*rs[23];

       for(j=0; j<4; j++)
       {
           //ranlxd(ru,18);
           for(k=0; k<18; k++)
           {
              ru[k]= rand() / (double)RAND_MAX;
              //fprintf(stderr,"ru[%d]=%lf\n",k,ru[k]);
           }

           
           ufield[4*i+j].c00=ru[0]+I*ru[1];
           ufield[4*i+j].c01=ru[2]+I*ru[3];
           ufield[4*i+j].c02=ru[4]+I*ru[5];
           ufield[4*i+j].c10=ru[6]+I*ru[7];
           ufield[4*i+j].c11=ru[8]+I*ru[9];
           ufield[4*i+j].c12=ru[10]+I*ru[11];
           ufield[4*i+j].c20=ru[12]+I*ru[13];
           ufield[4*i+j].c21=ru[14]+I*ru[15];
           ufield[4*i+j].c22=ru[16]+I*ru[17];
       }

   }

#ifdef AVX
   for(i=0; i<plqcd_g.VOLUME; i +=2)
   {   
       for(j=0; j<4; j++)
          copy_su3_to_su3_256(ufield_256+4*i/2+j, ufield+4*i+j, ufield+4*(i+1)+j);

       copy_spinor_to_spinor_256(pin_256+i/2, pin+i, pin+i+1);
       copy_spinor_to_spinor_256(pout_256+i/2, pout+i, pout+i+1);
   }
#endif

#ifdef MIC
   for(i=0; i<plqcd_g.VOLUME; i +=4)
   {   
       for(j=0; j<4; j++)
          copy_su3_to_su3_512(ufield_512+4*i/4+j, ufield+4*i+j, ufield+4*(i+1)+j, ufield+4*(i+2)+j, ufield+4*(i+3)+j);

       copy_spinor_to_spinor_512(pin_512+i/4, pin+i, pin+i+1, pin+i+2, pin+i+3);
       copy_spinor_to_spinor_512(pout_512+i/4, pout+i, pout+i+1, pout+i+2, pout+i+3);
   }
#endif


   double total,t1=0.0,t2=0.0,mytotal;
   int  matvecs;


#ifdef ASSYMBLY
   //---------------------------------------------
   //1: non-blocking assymbly/c version
   //---------------------------------------------
   matvecs=0;
   total=0.0;
   mytotal =0.0;

   while(mytotal < 30)
   {
      MPI_Barrier(MPI_COMM_WORLD); 
      for(i=0; i<Nmul; i++)
      {
         t1=plqcd_hopping_matrix_eo_sse3_assymbly(pin,pout,ufield);
         t2=plqcd_hopping_matrix_oe_sse3_assymbly(pin,pout,ufield);
         mytotal += t1+t2;
      }
      matvecs += Nmul;
   }
   
   MPI_Reduce(&mytotal,&total,1,MPI_DOUBLE,MPI_SUM,0, MPI_COMM_WORLD);
   MPI_Bcast(&total,1,MPI_DOUBLE,0, MPI_COMM_WORLD);
  

   if (proc_id==0)
   {
     total /= (double)(NPROCS);
   }
    

   if(proc_id==0)
   {
     fprintf(ofp,"non-blocking assymbly/c version:\n");
     fprintf(ofp,"------------------------------------------\n");
     fprintf(ofp,"test_hopping\tmult\t%d\ttotal(sec)\t%lf\tMFlops/process\t%lf\n",
             matvecs,total,matvecs*plqcd_g.VOLUME/2.0*1200/total/1e+6);
   }
#endif


#ifdef SSE3_INTRIN
   //---------------------------------------------
   //1: non-blocking sse3 with intrinsics version
   //---------------------------------------------
   matvecs=0;
   total=0.0;
   mytotal =0.0;

   while(mytotal < 30)
   {
      MPI_Barrier(MPI_COMM_WORLD); 
      for(i=0; i<Nmul; i++)
      {
         t1=plqcd_hopping_matrix_eo_sse3_intrin(pin,pout,ufield);
         t2=plqcd_hopping_matrix_oe_sse3_intrin(pin,pout,ufield);
         mytotal += t1+t2;
      }
      matvecs += Nmul;
   }
   
   MPI_Reduce(&mytotal,&total,1,MPI_DOUBLE,MPI_SUM,0, MPI_COMM_WORLD);
   MPI_Bcast(&total,1,MPI_DOUBLE,0, MPI_COMM_WORLD);
  

   if (proc_id==0)
   {
     total /= (double)(NPROCS);
   }
    

   if(proc_id==0)
   {
     fprintf(ofp,"non-blocking sse3 with intrinsics version:\n");
     fprintf(ofp,"------------------------------------------\n");
     fprintf(ofp,"test_hopping\tmult\t%d\ttotal(sec)\t%lf\tMFlops/process\t%lf\n",
             matvecs,total,matvecs*plqcd_g.VOLUME/2.0*1200/total/1e+6);
   }



   //---------------------------------------------
   //2: blocking sse3 with intrinsics version
   //---------------------------------------------
   matvecs=0;
   total=0.0;
   mytotal =0.0;

   while(mytotal < 30)
   {
      MPI_Barrier(MPI_COMM_WORLD); 
      for(i=0; i<Nmul; i++)
      {
         t1=plqcd_hopping_matrix_eo_sse3_intrin_blocking(pin,pout,ufield);
         t2=plqcd_hopping_matrix_oe_sse3_intrin_blocking(pin,pout,ufield);
         mytotal += t1+t2;
      }
      matvecs += Nmul;
   }
   
   MPI_Reduce(&mytotal,&total,1,MPI_DOUBLE,MPI_SUM,0, MPI_COMM_WORLD);
   MPI_Bcast(&total,1,MPI_DOUBLE,0, MPI_COMM_WORLD);
  

   if (proc_id==0)
   {
     total /= (double)(NPROCS);
   }
    

   if(proc_id==0)
   {
     fprintf(ofp,"blocking sse3 with intrinsics version:\n");
     fprintf(ofp,"------------------------------------------\n");
     fprintf(ofp,"test_hopping\tmult\t%d\ttotal(sec)\t%lf\tMFlops/process\t%lf\n",
             matvecs,total,matvecs*plqcd_g.VOLUME/2.0*1200/total/1e+6);
   }
#endif


#ifdef AVX
   //---------------------------------------------
   //2: avx version
   //---------------------------------------------
   matvecs=0;
   total=0.0;
   mytotal =0.0;

   t1=plqcd_hopping_matrix_eo_intrin_256(pin_256,pout_256,ufield_256);
   while(mytotal < 30)
   {
      MPI_Barrier(MPI_COMM_WORLD); 
      for(i=0; i<Nmul; i++)
      {
         t1=plqcd_hopping_matrix_eo_intrin_256(pin_256,pout_256,ufield_256);
         t2=plqcd_hopping_matrix_oe_intrin_256(pin_256,pout_256,ufield_256);
         mytotal += t1+t2;
      }
      matvecs += Nmul;
   }
   
   MPI_Reduce(&mytotal,&total,1,MPI_DOUBLE,MPI_SUM,0, MPI_COMM_WORLD);
   MPI_Bcast(&total,1,MPI_DOUBLE,0, MPI_COMM_WORLD);
  

   if (proc_id==0)
   {
     total /= (double)(NPROCS);
   }
    

   if(proc_id==0)
   {
     fprintf(ofp,"avxversion:\n");
     fprintf(ofp,"------------------------------------------\n");
     fprintf(ofp,"test_hopping\tmult\t%d\ttotal(sec)\t%lf\tMFlops/process\t%lf\n",
             matvecs,total,matvecs*plqcd_g.VOLUME/2.0*1200/total/1e+6);
   }
#endif


#ifdef MIC

#ifdef TEST_HOPPING_MIC
   //---------------------------------------------
   //3: MIC version full su3 matrix
   //---------------------------------------------
   matvecs=0;
   total=0.0;
   mytotal =0.0;

   t1=plqcd_hopping_matrix_eo_single_mic(pin_512,pout_512,ufield_512);

   while(mytotal < 30)
   {
      MPI_Barrier(MPI_COMM_WORLD); 
      for(i=0; i<Nmul; i++)
      {
         //t1=plqcd_hopping_matrix_eo_intrin_512(pin_512,pout_512,ufield_512);
         //t2=plqcd_hopping_matrix_oe_intrin_512(pin_512,pout_512,ufield_512);
         t1=plqcd_hopping_matrix_eo_single_mic(pin_512,pout_512,ufield_512);
         t2=plqcd_hopping_matrix_eo_single_mic(pin_512,pout_512,ufield_512);
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
     fprintf(ofp,"mic version, 3x3 links:\n");
     fprintf(ofp,"------------------------------------------\n");
     fprintf(ofp,"test_hopping\tmult\t%d\ttotal(sec)\t%lf\tMFlops/process\t%lf\n",
             matvecs,total,(double )matvecs*plqcd_g.VOLUME/2.0*1200/total/1e+6);
   }


   //---------------------------------------------
   //3: MIC version full reduced su3 storage
   //---------------------------------------------
   matvecs=0;
   total=0.0;
   mytotal =0.0;

   t1=plqcd_hopping_matrix_eo_single_mic_short(pin_512,pout_512,ufield_512);

   while(mytotal < 30)
   {
      MPI_Barrier(MPI_COMM_WORLD); 
      for(i=0; i<Nmul; i++)
      {
         //t1=plqcd_hopping_matrix_eo_intrin_512(pin_512,pout_512,ufield_512);
         //t2=plqcd_hopping_matrix_oe_intrin_512(pin_512,pout_512,ufield_512);
         t1=plqcd_hopping_matrix_eo_single_mic_short(pin_512,pout_512,ufield_512);
         t2=plqcd_hopping_matrix_eo_single_mic_short(pin_512,pout_512,ufield_512);
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
     fprintf(ofp,"mic version, 2x3 links:\n");
     fprintf(ofp,"------------------------------------------\n");
     fprintf(ofp,"test_hopping\tmult\t%d\ttotal(sec)\t%lf\tMFlops/process\t%lf\n",
             matvecs,total,(double )matvecs*plqcd_g.VOLUME/2.0*1200/total/1e+6);
   }

#endif

#ifdef TEST_SU3MUL_MIC

   matvecs=0;
   total=0.0;
   mytotal =0.0;

   //while(mytotal < 10)
   //{
      MPI_Barrier(MPI_COMM_WORLD); 
      for(i=0; i<Nmul; i++)
      {
         t1=stop_watch(0.0);
 
         #ifdef _OPENMP
         #pragma omp parallel
         {
         #endif        
            __m512d U[3][3], gin[3],gout[3];
            su3_512 *u0;
            su3_vector_512 *hin,*hout;
            #ifdef _OPENMP
            #pragma omp for
            #endif
            for(j=0; j< plqcd_g.VOLUME/4; j++)
            {
                u0  = &ufield_512[4*j];
                hin = &pin_512[j].s0;
                hout= &pout_512[j].s0;

                intrin_su3_load_512(U,u0);
                intrin_vector_load_512(gin,hin);
                intrin_su3_multiply_512(gout,U,gin);
                intrin_vector_store_512(hout,gout);

                u0++;
                hin++;
                hout++;

                intrin_su3_load_512(U,u0);
                intrin_vector_load_512(gin,hin);
                intrin_su3_multiply_512(gout,U,gin);
                intrin_vector_store_512(hout,gout);
                u0++;
                hin++;
                hout++;

                intrin_su3_load_512(U,u0);
                intrin_vector_load_512(gin,hin);
                intrin_su3_multiply_512(gout,U,gin);
                intrin_vector_store_512(hout,gout);
                u0++;
                hin++;
                hout++;

                intrin_su3_load_512(U,u0);
                intrin_vector_load_512(gin,hin);
                intrin_su3_multiply_512(gout,U,gin);
                intrin_vector_store_512(hout,gout);

           }
         #ifdef _OPENMP
         }
         #endif
      
         t2 = stop_watch(t1);
         mytotal += t2;
      }
      matvecs += 4*Nmul*plqcd_g.VOLUME;
   //}
   
   MPI_Reduce(&mytotal,&total,1,MPI_DOUBLE,MPI_SUM,0, MPI_COMM_WORLD);
   MPI_Bcast(&total,1,MPI_DOUBLE,0, MPI_COMM_WORLD);
  
   if (proc_id==0)
   {
     total /= (double)(NPROCS);
   }
    

   if(proc_id==0)
   {
     fprintf(ofp,"su3mul mic version:\n");
     fprintf(ofp,"------------------------------------------\n");
     fprintf(ofp,"test_hopping\tmult\t%d\ttotal(sec)\t%lf\tMFlops/process\t%lf\n",
             matvecs,total,matvecs*66.0/total/1e+6);
   }
#endif

#endif //MIC

   finalize_plqcd();

   return 0;
}
