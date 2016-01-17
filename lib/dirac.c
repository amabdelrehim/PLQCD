/*********************************************************************************
 * Copyright (C) 2012 Abdou Abdel-Rehim, Giannis Koutsou, Nikos Anastopolous
 * This file is part of the PLQCD library
 * Hopping Matrix interface
 *********************************************************************************/


#include"dirac.h"
#include<sys/time.h>


static int itags=0,tags[8];
static MPI_Request snd_req[8],rcv_req[8];
static void get_tags(void)
{
   int i;

   if (itags == 0)
   {
      for(i=0; i<8; i++)
         tags[i]=mpi_permanent_tag();

      itags=1;
   }
}


//=========================================================================================
//==============================  0   =====================================================
//==========using either plan C or SSE2,3 with assymbly====================================
//=========================================================================================

#ifdef ASSYMBLY
//======================================EO=================================================
double plqcd_hopping_matrix_eo_sse3_assymbly(spinor *qin, spinor *qout, su3 *u)
{
   
   //variables defined for the master thread
   int snd_err[8],rcv_err[8];
   MPI_Status snd_mpi_stat[8],rcv_mpi_stat[8];
   if(itags==0)
      get_tags(); //get the permanent tags for communications

   double ti,tf;  //timing variables

   ti=stop_watch(0.0);  //start timer


   //start the openmp parallel reigon
   #ifdef _OPENMP
   #pragma omp parallel
   {
   #endif
      //all declared variables from this point are private for each thread
      //this helps to avoid race conditions
      int i,j,k,mu,V,face[4];
      V = plqcd_g.VOLUME;

      for(i=0; i<4; i++)
         face[i] = plqcd_g.face[i];


      su3_vector p   __attribute__ ((aligned (16))) ;
      su3_vector ap0 __attribute__ ((aligned (16))) ;
      su3_vector ap1 __attribute__ ((aligned (16))) ;
      su3_vector q0  __attribute__ ((aligned (16)));
      su3_vector q1  __attribute__ ((aligned (16)));
      su3_vector q2  __attribute__ ((aligned (16)));
      su3_vector q3  __attribute__ ((aligned (16)));
   
      int iup0,iup1,iup2,iup3,idn0,idn1,idn2,idn3;    
  
      su3 *ub0;

      //-------------------------------------------------
      // compute (1-gamma_mu)qin terms and store in phip
      //-------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=V/2; i < V; i++)
      {
         #ifdef SSE3
         _prefetch_spinor(qin+i);
         #endif
         idn0=plqcd_g.idn[i][0];
         idn1=plqcd_g.idn[i][1];
         idn2=plqcd_g.idn[i][2];
         idn3=plqcd_g.idn[i][3];

         //-- 0 direction ---
         #ifdef SSE3
         _prefetch_halfspinor(&plqcd_g.phip[1][idn1]);
         #endif
         _vector_add(plqcd_g.phip[0][idn0].s0, qin[i].s0, qin[i].s2);
         _vector_add(plqcd_g.phip[0][idn0].s1, qin[i].s1, qin[i].s3);
  
         //-- 1 direction --
         #ifdef SSE3
         _prefetch_halfspinor(&plqcd_g.phip[2][idn2]);
         #endif
         _vector_i_add(plqcd_g.phip[1][idn1].s0, qin[i].s0, qin[i].s3);
         _vector_i_add(plqcd_g.phip[1][idn1].s1, qin[i].s1, qin[i].s2);

         //-- 2 direction --
         #ifdef SSE3
         _prefetch_halfspinor(&plqcd_g.phip[3][idn3]);
         #endif
         _vector_add(plqcd_g.phip[2][idn2].s0, qin[i].s0, qin[i].s3);
         _vector_sub(plqcd_g.phip[2][idn2].s1, qin[i].s1, qin[i].s2);

         //-- 3 direction --
         _vector_i_add(plqcd_g.phip[3][idn3].s0, qin[i].s0, qin[i].s2);
         _vector_i_sub(plqcd_g.phip[3][idn3].s1, qin[i].s1, qin[i].s3);

      }


      //start sending the buffers to the nearest neighbours in the -ve directions (only master thread)
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
        for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
            snd_err[mu] = MPI_Isend(&plqcd_g.phip[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu], MPI_COMM_WORLD, &snd_req[mu]);
            rcv_err[mu] = MPI_Irecv(&plqcd_g.phip[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_req[mu]);
            if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
              fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
              exit(1);}
          }
        }
      #ifdef _OPENMP
      }
      #endif //end of the master sub-reigon for MPI

      //---------------------------------------------------------
      // compute U^dagger*(1+gamma_mu)qin terms and store in phim
      //---------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for 
      #endif  
      for(i=V/2; i < V; i++)
      {
         #ifdef SSE3
         _prefetch_spinor(qin+i);
         #endif
         iup0=plqcd_g.iup[i][0];
         iup1=plqcd_g.iup[i][1];
         iup2=plqcd_g.iup[i][2];
         iup3=plqcd_g.iup[i][3];
         #ifdef SSE3
         _prefetch_halfspinor(&plqcd_g.phim[0][iup0]);
         #endif
         ub0 = u+4*i;

         //-- 0 direction ---
         
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[1][iup1]);
         #endif
         _vector_sub(p,qin[i].s0,qin[i].s2);
         _su3_inverse_multiply(plqcd_g.phim[0][iup0].s0,*ub0,p);
         _vector_sub(p,qin[i].s1,qin[i].s3);
         _su3_inverse_multiply(plqcd_g.phim[0][iup0].s1,*ub0,p);
         ub0++;
  
         //-- 1 direction --
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[2][iup2]);
         #endif
         _vector_i_sub(p,qin[i].s0,qin[i].s3);
         _su3_inverse_multiply(plqcd_g.phim[1][iup1].s0,*ub0,p);
         _vector_i_sub(p,qin[i].s1,qin[i].s2);
         _su3_inverse_multiply(plqcd_g.phim[1][iup1].s1,*ub0,p);
         ub0++;

         //-- 2 direction --
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[3][iup3]);
         #endif
         _vector_sub(p,qin[i].s0,qin[i].s3);
         _su3_inverse_multiply(plqcd_g.phim[2][iup2].s0,*ub0,p);
         _vector_add(p,qin[i].s1,qin[i].s2);
         _su3_inverse_multiply(plqcd_g.phim[2][iup2].s1,*ub0,p);
         ub0++;

         //-- 3 direction --
         _vector_i_sub(p,qin[i].s0,qin[i].s2);
         _su3_inverse_multiply(plqcd_g.phim[3][iup3].s0,*ub0,p);
         _vector_i_add(p,qin[i].s1,qin[i].s3);
         _su3_inverse_multiply(plqcd_g.phim[3][iup3].s1,*ub0,p);
      }

      //start sending the buffers to the nearest neighbours in the +ve directions (only the master thread)
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
         for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
           snd_err[mu+4] = MPI_Isend(&plqcd_g.phim[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD, &snd_req[mu+4]);
           rcv_err[mu+4] = MPI_Irecv(&plqcd_g.phim[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_req[mu+4]);
           if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
             fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
             exit(1);}
            }
         }

         //-------------------------------------------------------
         //complete computation of the U_mu*(1-gamma_m)*qin terms
         //-------------------------------------------------------

         //wait for the communications of phip to finish
         for(mu=0; mu<4; mu++)
         {
           if(plqcd_g.nprocs[mu]>1)
           {   
             rcv_err[mu]=MPI_Wait(&rcv_req[mu],&rcv_mpi_stat[mu]);
             snd_err[mu]=MPI_Wait(&snd_req[mu],&snd_mpi_stat[mu]);
             if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
              fprintf(stderr,"Error in MPI_Wait\n");
             exit(1);}
            }
         }
      #ifdef _OPENMP
      }
      #endif //end of the master sub-reigon




      //copy the exchanged boundaries to the correspoding locations on the local phip fields
        for(mu=0; mu<4; mu++)
        { 
          if(plqcd_g.nprocs[mu] > 1)
          {   
             #ifdef _OPENMP
             #pragma omp for
             #endif  
             for(i=0; i< face[mu]/2; i++)
             {
                //can we prefetch here
                j=V/2+face[mu]/2+i;
                k=plqcd_g.nn_bndo[2*mu][i];
                _vector_assign(plqcd_g.phip[mu][k].s0 , plqcd_g.phip[mu][j].s0);
                _vector_assign(plqcd_g.phip[mu][k].s1 , plqcd_g.phip[mu][j].s1);
             }
          }
        }


      //start building the results from the forward pieces
      #ifdef _OPENMP
      #pragma omp for
      #endif 
      for(i=0; i< V/2; i++)
      {
         ub0 = &u[4*i];
   
         // +0
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[1][i]);
         #endif
         _su3_multiply(q0, *ub0, plqcd_g.phip[0][i].s0);
         _su3_multiply(q1, *ub0, plqcd_g.phip[0][i].s1);
         _vector_assign(q2, q0);
         _vector_assign(q3, q1);
         ub0++;

         // +1
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[2][i]);
         #endif
         _su3_multiply(ap0, *ub0, plqcd_g.phip[1][i].s0);
         _su3_multiply(ap1, *ub0, plqcd_g.phip[1][i].s1);
         _vector_add_assign(q0,ap0);
         _vector_add_assign(q1,ap1);
         _vector_add_i_mul(q2, -1.0, ap1);
         _vector_add_i_mul(q3, -1.0, ap0);
         ub0++;        

         // +2
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[3][i]);
         #endif
         _su3_multiply(ap0, *ub0, plqcd_g.phip[2][i].s0);
         _su3_multiply(ap1, *ub0, plqcd_g.phip[2][i].s1);
         _vector_add_assign(q0,ap0);
         _vector_add_assign(q1,ap1);
         _vector_sub_assign(q2,ap1);
         _vector_add_assign(q3,ap0);
         ub0++;        


         #ifdef SSE3
         _prefetch_spinor(qout+i);
         #endif
         // +3
         _su3_multiply(ap0, *ub0, plqcd_g.phip[3][i].s0);
         _su3_multiply(ap1, *ub0, plqcd_g.phip[3][i].s1);
         _vector_add_assign(q0,ap0);
         _vector_add_assign(q1,ap1);
         _vector_add_i_mul(q2,-1.0, ap0);
         _vector_add_i_mul(q3, 1.0, ap1);
 
         //store the result
         _vector_assign(qout[i].s0,q0);
         _vector_assign(qout[i].s1,q1);
         _vector_assign(qout[i].s2,q2);
         _vector_assign(qout[i].s3,q3);         

      } 

      //wait for the communications of phim to finish
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
        for(mu=0; mu<4; mu++)
        {
           if(plqcd_g.nprocs[mu]>1)
           {   
              rcv_err[mu+4]=MPI_Wait(&rcv_req[mu+4],&rcv_mpi_stat[mu+4]);
              snd_err[mu+4]=MPI_Wait(&snd_req[mu+4],&snd_mpi_stat[mu+4]);
              if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
                fprintf(stderr,"Error in MPI_Wait\n");
                exit(1);}
           }
        }
      #ifdef _OPENMP
      }
      #endif //end of the master subreigon

      //---------------------------------------------------------------------
      //finish computation of the U^dagger*(1+gamma_mu)*qin
      //---------------------------------------------------------------------
      //copy the exchanged boundaries to the correspoding locations on the local phim fields
         for(mu=0; mu<4; mu++)
         { 
            if(plqcd_g.nprocs[mu] > 1)
            {   
               #ifdef _OPENMP
               #pragma omp for
               #endif
               for(i=0; i< face[mu]/2; i++)
               {
                 j=V/2+face[mu]/2+i;
                 k=plqcd_g.nn_bndo[2*mu+1][i];
                 _vector_assign(plqcd_g.phim[mu][k].s0, plqcd_g.phim[mu][j].s0);
                 _vector_assign(plqcd_g.phim[mu][k].s1, plqcd_g.phim[mu][j].s1);
               }
            }
          }

       #ifdef _OPENMP
       #pragma omp for
       #endif
       for(i=0; i< V/2; i++)
       {
          // 0 direction
         #ifdef SSE3
          _prefetch_halfspinor(&plqcd_g.phim[1][i]);
         #endif
          _vector_add_assign(qout[i].s0, plqcd_g.phim[0][i].s0);
          _vector_add_assign(qout[i].s1, plqcd_g.phim[0][i].s1);
          _vector_minus_assign(qout[i].s2, plqcd_g.phim[0][i].s0);
          _vector_minus_assign(qout[i].s3, plqcd_g.phim[0][i].s1);

          // 1 direction
         #ifdef SSE3
          _prefetch_halfspinor(&plqcd_g.phim[2][i]);
         #endif
          _vector_add_assign(qout[i].s0, plqcd_g.phim[1][i].s0);
          _vector_add_assign(qout[i].s1, plqcd_g.phim[1][i].s1);
          _vector_add_i_mul(qout[i].s2, 1.0,plqcd_g.phim[1][i].s1);
          _vector_add_i_mul(qout[i].s3, 1.0, plqcd_g.phim[1][i].s0);

          // 2 direction
         #ifdef SSE3
          _prefetch_halfspinor(&plqcd_g.phim[3][i]);
         #endif
          _vector_add_assign(qout[i].s0, plqcd_g.phim[2][i].s0);
          _vector_add_assign(qout[i].s1,  plqcd_g.phim[2][i].s1 );
          _vector_add_assign(qout[i].s2,  plqcd_g.phim[2][i].s1);
          _vector_sub_assign(qout[i].s3,   plqcd_g.phim[2][i].s0);

          //3 direction
          _vector_add_assign(qout[i].s0, plqcd_g.phim[3][i].s0);
          _vector_add_assign(qout[i].s1, plqcd_g.phim[3][i].s1);
          _vector_i_add_assign(qout[i].s2, plqcd_g.phim[3][i].s0);
          _vector_i_sub_assign(qout[i].s3, plqcd_g.phim[3][i].s1);     
      }
   #ifdef _OPENMP
   }
   #endif //end of the openmp parallel reigon

   //time it
   tf=stop_watch(ti);

   return tf;
}


//============================
//========== OE ==============
//============================
double plqcd_hopping_matrix_oe_sse3_assymbly(spinor *qin, spinor *qout, su3 *u)
{
   
   //variables defined for the master thread
   int snd_err[8],rcv_err[8];
   MPI_Status snd_mpi_stat[8],rcv_mpi_stat[8];
   if(itags==0)
      get_tags(); //get the permanent tags for communications

   double ti,tf;  //timing variables

   ti=stop_watch(0.0);  //start timer


   //start the openmp parallel reigon
   #ifdef _OPENMP
   #pragma omp parallel
   {
   #endif
      //all declared variables from this point are private for each thread
      //this helps to avoid race conditions
      int i,j,k,mu,V,face[4];
      V = plqcd_g.VOLUME;

      for(i=0; i<4; i++)
         face[i] = plqcd_g.face[i];


      su3_vector p   __attribute__ ((aligned (16))) ;
      su3_vector ap0 __attribute__ ((aligned (16))) ;
      su3_vector ap1 __attribute__ ((aligned (16))) ;
      su3_vector q0  __attribute__ ((aligned (16)));
      su3_vector q1  __attribute__ ((aligned (16)));
      su3_vector q2  __attribute__ ((aligned (16)));
      su3_vector q3  __attribute__ ((aligned (16)));
   
      int iup0,iup1,iup2,iup3,idn0,idn1,idn2,idn3;    
  
      su3 *ub0;

      //-------------------------------------------------
      // compute (1-gamma_mu)qin terms and store in phip
      //-------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=0; i < V/2; i++)
      {
         #ifdef SSE3
         _prefetch_spinor(qin+i);
         #endif

         idn0=plqcd_g.idn[i][0]-V/2;
         idn1=plqcd_g.idn[i][1]-V/2;
         idn2=plqcd_g.idn[i][2]-V/2;
         idn3=plqcd_g.idn[i][3]-V/2;

         //-- 0 direction ---
         #ifdef SSE3
         _prefetch_halfspinor(&plqcd_g.phip[1][idn1]);
         #endif
         _vector_add(plqcd_g.phip[0][idn0].s0, qin[i].s0, qin[i].s2);
         _vector_add(plqcd_g.phip[0][idn0].s1, qin[i].s1, qin[i].s3);
  
         //-- 1 direction --
         #ifdef SSE3
         _prefetch_halfspinor(&plqcd_g.phip[2][idn2]);
         #endif
         _vector_i_add(plqcd_g.phip[1][idn1].s0, qin[i].s0, qin[i].s3);
         _vector_i_add(plqcd_g.phip[1][idn1].s1, qin[i].s1, qin[i].s2);

         //-- 2 direction --
         #ifdef SSE3
         _prefetch_halfspinor(&plqcd_g.phip[3][idn3]);
         #endif
         _vector_add(plqcd_g.phip[2][idn2].s0, qin[i].s0, qin[i].s3);
         _vector_sub(plqcd_g.phip[2][idn2].s1, qin[i].s1, qin[i].s2);

         //-- 3 direction --
         _vector_i_add(plqcd_g.phip[3][idn3].s0, qin[i].s0, qin[i].s2);
         _vector_i_sub(plqcd_g.phip[3][idn3].s1, qin[i].s1, qin[i].s3);

      }


      //start sending the buffers to the nearest neighbours in the -ve directions (only master thread)
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
        for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
            snd_err[mu] = MPI_Isend(&plqcd_g.phip[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu], MPI_COMM_WORLD, &snd_req[mu]);
            rcv_err[mu] = MPI_Irecv(&plqcd_g.phip[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_req[mu]);
            if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
              fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
              exit(1);}
          }
        }
      #ifdef _OPENMP
      }
      #endif //end of the master sub-reigon for MPI

      //---------------------------------------------------------
      // compute U^dagger*(1+gamma_mu)qin terms and store in phim
      //---------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for 
      #endif  
      for(i=0; i < V/2; i++)
      {
         #ifdef SSE3
         _prefetch_spinor(qin+i);
         #endif
         iup0=plqcd_g.iup[i][0]-V/2;
         iup1=plqcd_g.iup[i][1]-V/2;
         iup2=plqcd_g.iup[i][2]-V/2;
         iup3=plqcd_g.iup[i][3]-V/2;

         #ifdef SSE3
         _prefetch_halfspinor(&plqcd_g.phim[0][iup0]);
         #endif

         ub0 = u+4*i;

         //-- 0 direction ---
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[1][iup1]);
         #endif
         _vector_sub(p,qin[i].s0,qin[i].s2);
         _su3_inverse_multiply(plqcd_g.phim[0][iup0].s0,*ub0,p);
         _vector_sub(p,qin[i].s1,qin[i].s3);
         _su3_inverse_multiply(plqcd_g.phim[0][iup0].s1,*ub0,p);
         ub0++;
  
         //-- 1 direction --
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[2][iup2]);
         #endif
         _vector_i_sub(p,qin[i].s0,qin[i].s3);
         _su3_inverse_multiply(plqcd_g.phim[1][iup1].s0,*ub0,p);
         _vector_i_sub(p,qin[i].s1,qin[i].s2);
         _su3_inverse_multiply(plqcd_g.phim[1][iup1].s1,*ub0,p);
         ub0++;

         //-- 2 direction --
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[3][iup3]);
         #endif
         _vector_sub(p,qin[i].s0,qin[i].s3);
         _su3_inverse_multiply(plqcd_g.phim[2][iup2].s0,*ub0,p);
         _vector_add(p,qin[i].s1,qin[i].s2);
         _su3_inverse_multiply(plqcd_g.phim[2][iup2].s1,*ub0,p);
         ub0++;

         //-- 3 direction --
         _vector_i_sub(p,qin[i].s0,qin[i].s2);
         _su3_inverse_multiply(plqcd_g.phim[3][iup3].s0,*ub0,p);
         _vector_i_add(p,qin[i].s1,qin[i].s3);
         _su3_inverse_multiply(plqcd_g.phim[3][iup3].s1,*ub0,p);
      }

      //start sending the buffers to the nearest neighbours in the +ve directions (only the master thread)
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
         for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
           snd_err[mu+4] = MPI_Isend(&plqcd_g.phim[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD, &snd_req[mu+4]);
           rcv_err[mu+4] = MPI_Irecv(&plqcd_g.phim[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_req[mu+4]);
           if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
             fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
             exit(1);}
            }
         }

         //-------------------------------------------------------
         //complete computation of the U_mu*(1-gamma_m)*qin terms
         //-------------------------------------------------------

         //wait for the communications of phip to finish
         for(mu=0; mu<4; mu++)
         {
           if(plqcd_g.nprocs[mu]>1)
           {   
             rcv_err[mu]=MPI_Wait(&rcv_req[mu],&rcv_mpi_stat[mu]);
             snd_err[mu]=MPI_Wait(&snd_req[mu],&snd_mpi_stat[mu]);
             if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
              fprintf(stderr,"Error in MPI_Wait\n");
             exit(1);}
            }
         }
      #ifdef _OPENMP
      }
      #endif //end of the master sub-reigon




      //copy the exchanged boundaries to the correspoding locations on the local phip fields
        for(mu=0; mu<4; mu++)
        { 
          if(plqcd_g.nprocs[mu] > 1)
          {   
             #ifdef _OPENMP
             #pragma omp for
             #endif  
             for(i=0; i< face[mu]/2; i++)
             {
                //can we prefetch here
                j=V/2+face[mu]/2+i;
                k=plqcd_g.nn_bnde[2*mu][i]-V/2;
                _vector_assign(plqcd_g.phip[mu][k].s0 , plqcd_g.phip[mu][j].s0);
                _vector_assign(plqcd_g.phip[mu][k].s1 , plqcd_g.phip[mu][j].s1);
             }
          }
        }


      //start building the results from the forward pieces
      #ifdef _OPENMP
      #pragma omp for
      #endif 
      for(i=0; i< V/2; i++)
      {
         ub0 = &u[4*i];
         j=i+V/2;

         // +0
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[1][i]);
        #endif
         _su3_multiply(q0, *ub0, plqcd_g.phip[0][i].s0);
         _su3_multiply(q1, *ub0, plqcd_g.phip[0][i].s1);
         _vector_assign(q2, q0);
         _vector_assign(q3, q1);
         ub0++;

         // +1
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[2][i]);
         #endif
         _su3_multiply(ap0, *ub0, plqcd_g.phip[1][i].s0);
         _su3_multiply(ap1, *ub0, plqcd_g.phip[1][i].s1);
         _vector_add_assign(q0,ap0);
         _vector_add_assign(q1,ap1);
         _vector_add_i_mul(q2, -1.0, ap1);
         _vector_add_i_mul(q3, -1.0, ap0);
         ub0++;        

         // +2
         #ifdef SSE3
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[3][i]);
         #endif
         _su3_multiply(ap0, *ub0, plqcd_g.phip[2][i].s0);
         _su3_multiply(ap1, *ub0, plqcd_g.phip[2][i].s1);
         _vector_add_assign(q0,ap0);
         _vector_add_assign(q1,ap1);
         _vector_sub_assign(q2,ap1);
         _vector_add_assign(q3,ap0);
         ub0++;        


         #ifdef SSE3
         _prefetch_spinor(qout+i);
         #endif
         // +3
         _su3_multiply(ap0, *ub0, plqcd_g.phip[3][i].s0);
         _su3_multiply(ap1, *ub0, plqcd_g.phip[3][i].s1);
         _vector_add_assign(q0,ap0);
         _vector_add_assign(q1,ap1);
         _vector_add_i_mul(q2,-1.0, ap0);
         _vector_add_i_mul(q3, 1.0, ap1);
 
         //store the result
         _vector_assign(qout[j].s0,q0);
         _vector_assign(qout[j].s1,q1);
         _vector_assign(qout[j].s2,q2);
         _vector_assign(qout[j].s3,q3);         

      } 

      //wait for the communications of phim to finish
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
        for(mu=0; mu<4; mu++)
        {
           if(plqcd_g.nprocs[mu]>1)
           {   
              rcv_err[mu+4]=MPI_Wait(&rcv_req[mu+4],&rcv_mpi_stat[mu+4]);
              snd_err[mu+4]=MPI_Wait(&snd_req[mu+4],&snd_mpi_stat[mu+4]);
              if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
                fprintf(stderr,"Error in MPI_Wait\n");
                exit(1);}
           }
        }
      #ifdef _OPENMP
      }
      #endif //end of the master subreigon

      //---------------------------------------------------------------------
      //finish computation of the U^dagger*(1+gamma_mu)*qin
      //---------------------------------------------------------------------
      //copy the exchanged boundaries to the correspoding locations on the local phim fields
         for(mu=0; mu<4; mu++)
         { 
            if(plqcd_g.nprocs[mu] > 1)
            {   
               #ifdef _OPENMP
               #pragma omp for
               #endif
               for(i=0; i< face[mu]/2; i++)
               {
                 j=V/2+face[mu]/2+i;
                 k=plqcd_g.nn_bnde[2*mu+1][i]-V/2;
                 _vector_assign(plqcd_g.phim[mu][k].s0, plqcd_g.phim[mu][j].s0);
                 _vector_assign(plqcd_g.phim[mu][k].s1, plqcd_g.phim[mu][j].s1);
               }
            }
          }

       #ifdef _OPENMP
       #pragma omp for
       #endif
       for(i=0; i< V/2; i++)
       {
          j=i+V/2;
          // 0 direction
         #ifdef SSE3
          _prefetch_halfspinor(&plqcd_g.phim[1][i]);
         #endif
          _vector_add_assign(qout[j].s0, plqcd_g.phim[0][i].s0);
          _vector_add_assign(qout[j].s1, plqcd_g.phim[0][i].s1);
          _vector_minus_assign(qout[j].s2, plqcd_g.phim[0][i].s0);
          _vector_minus_assign(qout[j].s3, plqcd_g.phim[0][i].s1);

          // 1 direction
         #ifdef SSE3
          _prefetch_halfspinor(&plqcd_g.phim[2][i]);
         #endif
          _vector_add_assign(qout[j].s0, plqcd_g.phim[1][i].s0);
          _vector_add_assign(qout[j].s1, plqcd_g.phim[1][i].s1);
          _vector_add_i_mul(qout[j].s2, 1.0,plqcd_g.phim[1][i].s1);
          _vector_add_i_mul(qout[j].s3, 1.0, plqcd_g.phim[1][i].s0);

          // 2 direction
         #ifdef SSE3
          _prefetch_halfspinor(&plqcd_g.phim[3][i]);
         #endif
          _vector_add_assign(qout[j].s0, plqcd_g.phim[2][i].s0);
          _vector_add_assign(qout[j].s1,  plqcd_g.phim[2][i].s1 );
          _vector_add_assign(qout[j].s2,  plqcd_g.phim[2][i].s1);
          _vector_sub_assign(qout[j].s3,   plqcd_g.phim[2][i].s0);

          //3 direction
          _vector_add_assign(qout[j].s0, plqcd_g.phim[3][i].s0);
          _vector_add_assign(qout[j].s1, plqcd_g.phim[3][i].s1);
          _vector_i_add_assign(qout[j].s2, plqcd_g.phim[3][i].s0);
          _vector_i_sub_assign(qout[j].s3, plqcd_g.phim[3][i].s1);     
      }
   #ifdef _OPENMP
   }
   #endif //end of the openmp parallel reigon

   //time it
   tf=stop_watch(ti);

   return tf;
}
#endif //#ifdef ASSYMBLY

#ifdef SSE3_INTRIN
//========================================================================================
//=======================================  1   ===========================================
//===================              SSE2,3 with intrinsics       ==========================
//========================================================================================

//===================================EO===================================================
double plqcd_hopping_matrix_eo_sse3_intrin(spinor *qin, spinor *qout, su3 *u)
{
   
   
   int snd_err[8],rcv_err[8];
   MPI_Status snd_mpi_stat[8],rcv_mpi_stat[8];
   if(itags==0)
      get_tags(); //get the permanent tags for communications

   double ts; //timer

   ts=stop_watch(0.0);
   
   #ifdef _OPENMP
   #pragma omp parallel
   {
   #endif
      int i,j,k,mu,V,face[4];
      V = plqcd_g.VOLUME;
      for(i=0; i<4; i++)
        face[i] = plqcd_g.face[i];
   
      __m128d qins0[3],qins1[3],qins2[3],qins3[3], U[3][3];
      __m128d in1[3],out[3];
      __m128d map0[3],map1[3],mq0[3],mq1[3],mq2[3],mq3[3];

      int iup0,iup1,iup2,iup3,idn0,idn1,idn2,idn3;    
      su3 *ub0;

      //-------------------------------------------------
      // compute (1-gamma_mu)qin terms and store in phip
      //-------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=V/2; i < V; i++)
      {
         _prefetch_spinor(qin+i);
         idn0=plqcd_g.idn[i][0];
         idn1=plqcd_g.idn[i][1];
         idn2=plqcd_g.idn[i][2];
         idn3=plqcd_g.idn[i][3];


         intrin_vector_load_128(qins0,&qin[i].s0);
         intrin_vector_load_128(qins1,&qin[i].s1);
         intrin_vector_load_128(qins2,&qin[i].s2);
         intrin_vector_load_128(qins3,&qin[i].s3);

         //-- 0 direction ---
         _prefetch_halfspinor(&plqcd_g.phip[1][idn1]);
         //_vector_add(plqcd_g.phip[0][idn0].s0, qin[i].s0, qin[i].s2);
         //_vector_add(plqcd_g.phip[0][idn0].s1, qin[i].s1, qin[i].s3);
         intrin_vector_add_128(out,qins0,qins2);
         //intrin_vector_store_128(&plqcd_g.phip[0][idn0].s0,out);
         intrin_vector_store_128(&plqcd_g.phip[0][idn0].s0,out);
         intrin_vector_add_128(out,qins1,qins3);
         //intrin_vector_store_128(&plqcd_g.phip[0][idn0].s1,out);
         intrin_vector_store_128(&plqcd_g.phip[0][idn0].s1,out);
  
         //-- 1 direction --
         _prefetch_halfspinor(&plqcd_g.phip[2][idn2]);
         //_vector_i_add(plqcd_g.phip[1][idn1].s0, qin[i].s0, qin[i].s3);
         //_vector_i_add(plqcd_g.phip[1][idn1].s1, qin[i].s1, qin[i].s2);
         intrin_vector_i_add_128(out,qins0,qins3);
         intrin_vector_store_128(&plqcd_g.phip[1][idn1].s0,out);
         intrin_vector_add_128(out,qins1,qins2);
         intrin_vector_store_128(&plqcd_g.phip[1][idn1].s1,out);

         //-- 2 direction --
         _prefetch_halfspinor(&plqcd_g.phip[3][idn3]);
         //_vector_add(plqcd_g.phip[2][idn2].s0, qin[i].s0, qin[i].s3);
         //_vector_sub(plqcd_g.phip[2][idn2].s1, qin[i].s1, qin[i].s2);
         intrin_vector_add_128(out,qins0,qins3);
         intrin_vector_store_128(&plqcd_g.phip[2][idn2].s0,out);
         intrin_vector_sub_128(out,qins1,qins2);
         intrin_vector_store_128(&plqcd_g.phip[2][idn2].s1,out);

         //-- 3 direction --
         //_vector_i_add(plqcd_g.phip[3][idn3].s0, qin[i].s0, qin[i].s2);
         //_vector_i_sub(plqcd_g.phip[3][idn3].s1, qin[i].s1, qin[i].s3);
         intrin_vector_i_add_128(out,qins0,qins2);
         intrin_vector_store_128(&plqcd_g.phip[3][idn3].s0,out);
         intrin_vector_i_sub_128(out,qins1,qins3);
         intrin_vector_store_128(&plqcd_g.phip[3][idn3].s1,out);
      }


      //start sending the buffers to the nearest neighbours in the -ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
         for(mu=0; mu < 4; mu++){  
           if(plqcd_g.nprocs[mu] > 1){  
             snd_err[mu] = MPI_Isend(&plqcd_g.phip[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu], MPI_COMM_WORLD, &snd_req[mu]);
             rcv_err[mu] = MPI_Irecv(&plqcd_g.phip[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_req[mu]);
             if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
                    fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
                    exit(1);}
            }
         }
      #ifdef _OPENMP
      }
      #endif
    
      //--------------------------------------------------------------
      // compute U^dagger*(1+gamma_mu)qin terms and stroe in phim
      //--------------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=V/2; i < V; i++)
      {
        _prefetch_spinor(qin+i);
        iup0=plqcd_g.iup[i][0];
        iup1=plqcd_g.iup[i][1];
        iup2=plqcd_g.iup[i][2];
        iup3=plqcd_g.iup[i][3];

        _prefetch_halfspinor(&plqcd_g.phim[0][iup0]);

        ub0 = &u[4*i];

        intrin_vector_load_128(qins0,&qin[i].s0);
        intrin_vector_load_128(qins1,&qin[i].s1);
        intrin_vector_load_128(qins2,&qin[i].s2);
        intrin_vector_load_128(qins3,&qin[i].s3);

      
        //-- 0 direction ---
        _prefetch_su3(ub0+1);
        _prefetch_halfspinor(&plqcd_g.phim[1][iup1]);
        //_vector_sub(p,qin[i].s0,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s0,*ub0,p);
        //_vector_sub(p,qin[i].s1,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_sub_128(in1,qins0,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[0][iup0].s0,out);
        intrin_vector_sub_128(in1,qins1,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[0][iup0].s1,out);



        ub0++;
  
        //-- 1 direction --
        _prefetch_su3(ub0+1);
        _prefetch_halfspinor(&plqcd_g.phim[2][iup2]);
        //_vector_i_sub(p,qin[i].s0,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s0,*ub0,p);
        //_vector_i_sub(p,qin[i].s1,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_i_sub_128(in1,qins0,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[1][iup1].s0,out);
        intrin_vector_i_sub_128(in1,qins1,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[1][iup1].s1,out);
        ub0++;

        //-- 2 direction --
        _prefetch_su3(ub0+1);
        _prefetch_halfspinor(&plqcd_g.phim[3][iup3]);
        //_vector_sub(p,qin[i].s0,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s0,*ub0,p);
        //_vector_add(p,qin[i].s1,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_sub_128(in1,qins0,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[2][iup2].s0,out);
        intrin_vector_add_128(in1,qins1,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[2][iup2].s1,out);
        ub0++;

        //-- 3 direction --
        //_vector_i_sub(p,qin[i].s0,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s0,*ub0,p);
        //_vector_i_add(p,qin[i].s1,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_i_sub_128(in1,qins0,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[3][iup3].s0,out);
        intrin_vector_i_add_128(in1,qins1,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[3][iup3].s1,out);
      }

      //start sending the buffers to the nearest neighbours in the +ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
         for(mu=0; mu < 4; mu++){  
            if(plqcd_g.nprocs[mu] > 1){  
               snd_err[mu+4] = MPI_Isend(&plqcd_g.phim[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD, &snd_req[mu+4]);
               rcv_err[mu+4] = MPI_Irecv(&plqcd_g.phim[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_req[mu+4]);
               if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
                      fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
                      exit(1);}
               }
         }
      
         //wait for the communications of phip to finish
         for(mu=0; mu<4; mu++)
         {
            if(plqcd_g.nprocs[mu]>1)
            {   
               rcv_err[mu]=MPI_Wait(&rcv_req[mu],&rcv_mpi_stat[mu]);
               snd_err[mu]=MPI_Wait(&snd_req[mu],&snd_mpi_stat[mu]);
               if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
                 fprintf(stderr,"Error in MPI_Wait\n");
                 exit(1);}
            }
         }
      #ifdef _OPENMP
      }
      #endif  //end of the master sub-reigon
        
      //-------------------------------------------------------
      //complete computation of the U_mu*(1-gamma_m)*qin terms
      //-------------------------------------------------------
      //copy the exchanged boundaries to the correspoding locations on the local phip fields
      for(mu=0; mu<4; mu++)
      { 
          #ifdef _OPENMP 
          #pragma omp for 
          #endif  
          for(i=0; i< face[mu]/2; i++)
          {
              //can we prefetch here
              j=V/2+face[mu]/2+i;
              k=plqcd_g.nn_bndo[2*mu][i];
              _vector_assign(plqcd_g.phip[mu][k].s0 , plqcd_g.phip[mu][j].s0);
              _vector_assign(plqcd_g.phip[mu][k].s1 , plqcd_g.phip[mu][j].s1);
          }
      }


      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=0; i< V/2; i++)
      {

         ub0 = u+4*i;
         // +0
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[1][i]);
         //_su3_multiply(q0, *ub0, plqcd_g.phip[0][i].s0);
         //_su3_multiply(q1, *ub0, plqcd_g.phip[0][i].s1);
         //_vector_assign(q2, q0);
         //_vector_assign(q3, q1);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[0][i].s0);
         intrin_su3_multiply_128(mq0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[0][i].s1);
         intrin_su3_multiply_128(mq1,U,in1);
         for(k=0; k<3; k++){
            mq2[k]= mq0[k];
            mq3[k]= mq1[k];}
       
         ub0++;

         // +1
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[2][i]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[1][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[1][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2, -1.0, ap1);
         //_vector_add_i_mul(q3, -1.0, ap0);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[1][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[1][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]); 
            mq2[k] = complex_i_sub_regs_128(mq2[k],map1[k]);
            mq3[k] = complex_i_sub_regs_128(mq3[k],map0[k]);}

         ub0++;        

         // +2
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[3][i]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[2][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[2][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_sub_assign(q2,ap1);
         //_vector_add_assign(q3,ap0);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[2][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[2][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]);
            mq2[k] = _mm_sub_pd(mq2[k],map1[k]);
            mq3[k] = _mm_add_pd(mq3[k],map0[k]);}

         ub0++;        


         _prefetch_spinor(&qout[i]);
         // +3
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[3][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[3][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2,-1.0, ap0);
         //_vector_add_i_mul(q3, 1.0, ap1);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[3][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[3][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]); 
            mq2[k] = complex_i_sub_regs_128(mq2[k],map0[k]);
            mq3[k] = complex_i_add_regs_128(mq3[k],map1[k]);}

 
       //store the result
       //_vector_assign(qout[i].s0,q0);
       //_vector_assign(qout[i].s1,q1);
       //_vector_assign(qout[i].s2,q2);
       //_vector_assign(qout[i].s3,q3);
       intrin_vector_store_128(&qout[i].s0,mq0);         
       intrin_vector_store_128(&qout[i].s1,mq1);         
       intrin_vector_store_128(&qout[i].s2,mq1);         
       intrin_vector_store_128(&qout[i].s3,mq1);         

     } 


   //wait for the communications of phim to finish
   #ifdef _OPENMP
   #pragma omp master
   {
   #endif
      for(mu=0; mu<4; mu++)
      {
        if(plqcd_g.nprocs[mu]>1)
        {   
           rcv_err[mu+4]=MPI_Wait(&rcv_req[mu+4],&rcv_mpi_stat[mu+4]);
           snd_err[mu+4]=MPI_Wait(&snd_req[mu+4],&snd_mpi_stat[mu+4]);
           if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
             fprintf(stderr,"Error in MPI_Wait\n");
             exit(1);}
        }
      }
   #ifdef _OPENMP
   }
   #endif
   
   //copy the exchanged boundaries to the correspoding locations on the local phim fields

   for(mu=0; mu<4; mu++)
   { 
       #ifdef _OPENMP
       #pragma omp for
       #endif
       for(i=0; i< face[mu]/2; i++)
       {
           j=V/2+face[mu]/2+i;
           k=plqcd_g.nn_bndo[2*mu+1][i];
           _vector_assign(plqcd_g.phim[mu][k].s0, plqcd_g.phim[mu][j].s0);
           _vector_assign(plqcd_g.phim[mu][k].s1, plqcd_g.phim[mu][j].s1);
       }
   }


   //---------------------------------------------------------------------
   //finish computation of the U^dagger*(1+gamma_mu)*qin
   //---------------------------------------------------------------------
   #ifdef _OPENMP
   #pragma omp for 
   #endif
   for(i=0; i< V/2; i++)
   {
       intrin_vector_load_128(mq0,&qout[i].s0);
       intrin_vector_load_128(mq1,&qout[i].s1);
       intrin_vector_load_128(mq2,&qout[i].s2);
       intrin_vector_load_128(mq3,&qout[i].s3);
       // 0 direction
       _prefetch_halfspinor(&plqcd_g.phim[1][i]);
       //_vector_add_assign(qout[i].s0, plqcd_g.phim[0][i].s0);
       //_vector_add_assign(qout[i].s1, plqcd_g.phim[0][i].s1);
       //_vector_minus_assign(qout[i].s2, plqcd_g.phim[0][i].s0);
       //_vector_minus_assign(qout[i].s3, plqcd_g.phim[0][i].s1);
       intrin_vector_load_128(map0,&plqcd_g.phim[0][i].s0);
       intrin_vector_load_128(map1,&plqcd_g.phim[0][i].s1);
       intrin_vector_add_128(mq0,mq0,map0);
       intrin_vector_add_128(mq1,mq1,map1);
       intrin_vector_sub_128(mq2,mq2,map0);
       intrin_vector_sub_128(mq3,mq3,map1);


       // 1 direction
       _prefetch_halfspinor(&plqcd_g.phim[2][i]);
       //_vector_add_assign(qout[i].s0, plqcd_g.phim[1][i].s0);
       //_vector_add_assign(qout[i].s1, plqcd_g.phim[1][i].s1);
       //_vector_add_i_mul(qout[i].s2, 1.0,plqcd_g.phim[1][i].s1);
       //_vector_add_i_mul(qout[i].s3, 1.0, plqcd_g.phim[1][i].s0);
       intrin_vector_load_128(map0,&plqcd_g.phim[1][i].s0);
       intrin_vector_load_128(map1,&plqcd_g.phim[1][i].s1);
       intrin_vector_add_128(mq0,mq0,map0);
       intrin_vector_add_128(mq1,mq1,map1);
       intrin_vector_i_add_128(mq2,mq2,map1);
       intrin_vector_i_add_128(mq3,mq3,map0);


       // 2 direction
       _prefetch_halfspinor(&plqcd_g.phim[3][i]);
       //_vector_add_assign(qout[i].s0, plqcd_g.phim[2][i].s0);
       //_vector_add_assign(qout[i].s1,  plqcd_g.phim[2][i].s1 );
       //_vector_add_assign(qout[i].s2,  plqcd_g.phim[2][i].s1);
       //_vector_sub_assign(qout[i].s3,   plqcd_g.phim[2][i].s0);
       intrin_vector_load_128(map0,&plqcd_g.phim[2][i].s0);
       intrin_vector_load_128(map1,&plqcd_g.phim[2][i].s1);
       intrin_vector_add_128(mq0,mq0,map0);
       intrin_vector_add_128(mq1,mq1,map1);
       intrin_vector_add_128(mq2,mq2,map1);
       intrin_vector_sub_128(mq3,mq3,map0);


       //3 direction
       //_vector_add_assign(qout[i].s0, plqcd_g.phim[3][i].s0);
       //_vector_add_assign(qout[i].s1, plqcd_g.phim[3][i].s1);
       //_vector_i_add_assign(qout[i].s2, plqcd_g.phim[3][i].s0);
       //_vector_i_sub_assign(qout[i].s3, plqcd_g.phim[3][i].s1);     
       intrin_vector_load_128(map0,&plqcd_g.phim[3][i].s0);
       intrin_vector_load_128(map1,&plqcd_g.phim[3][i].s1);
       intrin_vector_add_128(mq0,mq0,map0);
       intrin_vector_add_128(mq1,mq1,map1);
       intrin_vector_i_add_128(mq2,mq2,map0);
       intrin_vector_i_add_128(mq3,mq3,map1);


       //store the final result
       intrin_vector_store_128(&qout[i].s0,mq0);
       intrin_vector_store_128(&qout[i].s1,mq1);
       intrin_vector_store_128(&qout[i].s2,mq2);
       intrin_vector_store_128(&qout[i].s3,mq3);

   }
#ifdef _OPENMP
}
#endif //endif of the openmp parallel reigon


   return stop_watch(ts);
}



//===================================OE===================================================
double plqcd_hopping_matrix_oe_sse3_intrin(spinor *qin, spinor *qout, su3 *u)
{
   
   
   int snd_err[8],rcv_err[8];
   MPI_Status snd_mpi_stat[8],rcv_mpi_stat[8];
   if(itags==0)
      get_tags(); //get the permanent tags for communications

   double ts; //timer

   ts=stop_watch(0.0);


   #ifdef _OPENMP
   #pragma omp parallel
   {
   #endif
      int i,j,k,mu,V,face[4];
      V = plqcd_g.VOLUME;
      for(i=0; i<4; i++)
        face[i] = plqcd_g.face[i];
   
      __m128d qins0[3],qins1[3],qins2[3],qins3[3], U[3][3];
      __m128d in1[3],out[3];
      __m128d map0[3],map1[3],mq0[3],mq1[3],mq2[3],mq3[3];

      int iup0,iup1,iup2,iup3,idn0,idn1,idn2,idn3;    
      su3 *ub0;

      //-------------------------------------------------
      // compute (1-gamma_mu)qin terms and store in phip
      //-------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=0; i < V/2; i++)
      {
         _prefetch_spinor(qin+i);
         idn0=plqcd_g.idn[i][0]-V/2;
         idn1=plqcd_g.idn[i][1]-V/2;
         idn2=plqcd_g.idn[i][2]-V/2;
         idn3=plqcd_g.idn[i][3]-V/2;


         intrin_vector_load_128(qins0,&qin[i].s0);
         intrin_vector_load_128(qins1,&qin[i].s1);
         intrin_vector_load_128(qins2,&qin[i].s2);
         intrin_vector_load_128(qins3,&qin[i].s3);

         //-- 0 direction ---
         _prefetch_halfspinor(&plqcd_g.phip[1][idn1]);
         //_vector_add(plqcd_g.phip[0][idn0].s0, qin[i].s0, qin[i].s2);
         //_vector_add(plqcd_g.phip[0][idn0].s1, qin[i].s1, qin[i].s3);
         intrin_vector_add_128(out,qins0,qins2);
         //intrin_vector_store_128(&plqcd_g.phip[0][idn0].s0,out);
         intrin_vector_store_128(&plqcd_g.phip[0][idn0].s0,out);
         intrin_vector_add_128(out,qins1,qins3);
         //intrin_vector_store_128(&plqcd_g.phip[0][idn0].s1,out);
         intrin_vector_store_128(&plqcd_g.phip[0][idn0].s1,out);
  
         //-- 1 direction --
         _prefetch_halfspinor(&plqcd_g.phip[2][idn2]);
         //_vector_i_add(plqcd_g.phip[1][idn1].s0, qin[i].s0, qin[i].s3);
         //_vector_i_add(plqcd_g.phip[1][idn1].s1, qin[i].s1, qin[i].s2);
         intrin_vector_i_add_128(out,qins0,qins3);
         intrin_vector_store_128(&plqcd_g.phip[1][idn1].s0,out);
         intrin_vector_add_128(out,qins1,qins2);
         intrin_vector_store_128(&plqcd_g.phip[1][idn1].s1,out);

         //-- 2 direction --
         _prefetch_halfspinor(&plqcd_g.phip[3][idn3]);
         //_vector_add(plqcd_g.phip[2][idn2].s0, qin[i].s0, qin[i].s3);
         //_vector_sub(plqcd_g.phip[2][idn2].s1, qin[i].s1, qin[i].s2);
         intrin_vector_add_128(out,qins0,qins3);
         intrin_vector_store_128(&plqcd_g.phip[2][idn2].s0,out);
         intrin_vector_sub_128(out,qins1,qins2);
         intrin_vector_store_128(&plqcd_g.phip[2][idn2].s1,out);

         //-- 3 direction --
         //_vector_i_add(plqcd_g.phip[3][idn3].s0, qin[i].s0, qin[i].s2);
         //_vector_i_sub(plqcd_g.phip[3][idn3].s1, qin[i].s1, qin[i].s3);
         intrin_vector_i_add_128(out,qins0,qins2);
         intrin_vector_store_128(&plqcd_g.phip[3][idn3].s0,out);
         intrin_vector_i_sub_128(out,qins1,qins3);
         intrin_vector_store_128(&plqcd_g.phip[3][idn3].s1,out);
      }


      //start sending the buffers to the nearest neighbours in the -ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
            for(mu=0; mu < 4; mu++){  
              if(plqcd_g.nprocs[mu] > 1){  
                 snd_err[mu] = MPI_Isend(&plqcd_g.phip[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu], MPI_COMM_WORLD, &snd_req[mu]);
                 rcv_err[mu] = MPI_Irecv(&plqcd_g.phip[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_req[mu]);
                 if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
                    fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
                    exit(1);}
              }
           }
      #ifdef _OPENMP
      }
      #endif
    
      //--------------------------------------------------------------
      // compute U^dagger*(1+gamma_mu)qin terms and stroe in phim
      //--------------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=0; i < V/2; i++)
      {
        _prefetch_spinor(qin+i);
        iup0=plqcd_g.iup[i][0]-V/2;
        iup1=plqcd_g.iup[i][1]-V/2;
        iup2=plqcd_g.iup[i][2]-V/2;
        iup3=plqcd_g.iup[i][3]-V/2;

        _prefetch_halfspinor(&plqcd_g.phim[0][iup0]);

        ub0 = &u[4*i];

        intrin_vector_load_128(qins0,&qin[i].s0);
        intrin_vector_load_128(qins1,&qin[i].s1);
        intrin_vector_load_128(qins2,&qin[i].s2);
        intrin_vector_load_128(qins3,&qin[i].s3);

      
        //-- 0 direction ---
        _prefetch_su3(ub0+1);
        _prefetch_halfspinor(&plqcd_g.phim[1][iup1]);
        //_vector_sub(p,qin[i].s0,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s0,*ub0,p);
        //_vector_sub(p,qin[i].s1,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_sub_128(in1,qins0,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[0][iup0].s0,out);
        intrin_vector_sub_128(in1,qins1,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[0][iup0].s1,out);



        ub0++;
  
        //-- 1 direction --
        _prefetch_su3(ub0+1);
        _prefetch_halfspinor(&plqcd_g.phim[2][iup2]);
        //_vector_i_sub(p,qin[i].s0,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s0,*ub0,p);
        //_vector_i_sub(p,qin[i].s1,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_i_sub_128(in1,qins0,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[1][iup1].s0,out);
        intrin_vector_i_sub_128(in1,qins1,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[1][iup1].s1,out);
        ub0++;

        //-- 2 direction --
        _prefetch_su3(ub0+1);
        _prefetch_halfspinor(&plqcd_g.phim[3][iup3]);
        //_vector_sub(p,qin[i].s0,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s0,*ub0,p);
        //_vector_add(p,qin[i].s1,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_sub_128(in1,qins0,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[2][iup2].s0,out);
        intrin_vector_add_128(in1,qins1,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[2][iup2].s1,out);
        ub0++;

        //-- 3 direction --
        //_vector_i_sub(p,qin[i].s0,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s0,*ub0,p);
        //_vector_i_add(p,qin[i].s1,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_i_sub_128(in1,qins0,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[3][iup3].s0,out);
        intrin_vector_i_add_128(in1,qins1,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[3][iup3].s1,out);
      }

      //start sending the buffers to the nearest neighbours in the +ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
            for(mu=0; mu < 4; mu++){  
               if(plqcd_g.nprocs[mu] > 1){  
                  snd_err[mu+4] = MPI_Isend(&plqcd_g.phim[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD, &snd_req[mu+4]);
                  rcv_err[mu+4] = MPI_Irecv(&plqcd_g.phim[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_req[mu+4]);
                  if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
                      fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
                      exit(1);}
               }
            }
      


            //wait for the communications of phip to finish
            for(mu=0; mu<4; mu++)
            {
               if(plqcd_g.nprocs[mu]>1)
               {   
                  rcv_err[mu]=MPI_Wait(&rcv_req[mu],&rcv_mpi_stat[mu]);
                  snd_err[mu]=MPI_Wait(&snd_req[mu],&snd_mpi_stat[mu]);
                  if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
                    fprintf(stderr,"Error in MPI_Wait\n");
                    exit(1);}
               }
            }
      #ifdef _OPENMP
      }
      #endif  //end of the master sub-reigon
        
      //-------------------------------------------------------
      //complete computation of the U_mu*(1-gamma_m)*qin terms
      //-------------------------------------------------------
      //copy the exchanged boundaries to the correspoding locations on the local phip fields 
        for(mu=0; mu<4; mu++)
         { 
               #ifdef _OPENMP 
               #pragma omp for 
               #endif  
               for(i=0; i< face[mu]/2; i++)
               {
                   //can we prefetch here
                   j=V/2+face[mu]/2+i;
                   k=plqcd_g.nn_bnde[2*mu][i]-V/2;
                   _vector_assign(plqcd_g.phip[mu][k].s0 , plqcd_g.phip[mu][j].s0);
                   _vector_assign(plqcd_g.phip[mu][k].s1 , plqcd_g.phip[mu][j].s1);
               }
         }


      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=0; i< V/2; i++)
      {
         j=i+V/2;
         ub0 = u+4*i;
         // +0
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[1][i]);
         //_su3_multiply(q0, *ub0, plqcd_g.phip[0][i].s0);
         //_su3_multiply(q1, *ub0, plqcd_g.phip[0][i].s1);
         //_vector_assign(q2, q0);
         //_vector_assign(q3, q1);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[0][i].s0);
         intrin_su3_multiply_128(mq0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[0][i].s1);
         intrin_su3_multiply_128(mq1,U,in1);
         for(k=0; k<3; k++){
            mq2[k]= mq0[k];
            mq3[k]= mq1[k];}
       
         ub0++;

         // +1
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[2][i]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[1][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[1][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2, -1.0, ap1);
         //_vector_add_i_mul(q3, -1.0, ap0);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[1][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[1][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]); 
            mq2[k] = complex_i_sub_regs_128(mq2[k],map1[k]);
            mq3[k] = complex_i_sub_regs_128(mq3[k],map0[k]);}

         ub0++;        

         // +2
         _prefetch_su3(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[3][i]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[2][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[2][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_sub_assign(q2,ap1);
         //_vector_add_assign(q3,ap0);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[2][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[2][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]);
            mq2[k] = _mm_sub_pd(mq2[k],map1[k]);
            mq3[k] = _mm_add_pd(mq3[k],map0[k]);}

         ub0++;        


         _prefetch_spinor(&qout[j]);
         // +3
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[3][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[3][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2,-1.0, ap0);
         //_vector_add_i_mul(q3, 1.0, ap1);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[3][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[3][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]); 
            mq2[k] = complex_i_sub_regs_128(mq2[k],map0[k]);
            mq3[k] = complex_i_add_regs_128(mq3[k],map1[k]);}

 
       //store the result
       //_vector_assign(qout[i].s0,q0);
       //_vector_assign(qout[i].s1,q1);
       //_vector_assign(qout[i].s2,q2);
       //_vector_assign(qout[i].s3,q3);
       intrin_vector_store_128(&qout[j].s0,mq0);         
       intrin_vector_store_128(&qout[j].s1,mq1);         
       intrin_vector_store_128(&qout[j].s2,mq1);         
       intrin_vector_store_128(&qout[j].s3,mq1);         

     } 


   //wait for the communications of phim to finish
   #ifdef _OPENMP
   #pragma omp master
   {
   #endif
         for(mu=0; mu<4; mu++)
         {
           if(plqcd_g.nprocs[mu]>1)
           {   
              rcv_err[mu+4]=MPI_Wait(&rcv_req[mu+4],&rcv_mpi_stat[mu+4]);
              snd_err[mu+4]=MPI_Wait(&snd_req[mu+4],&snd_mpi_stat[mu+4]);
              if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
                fprintf(stderr,"Error in MPI_Wait\n");
                exit(1);}
           }
         }
   #ifdef _OPENMP
   }
   #endif
   
   //copy the exchanged boundaries to the correspoding locations on the local phim fields
     for(mu=0; mu<4; mu++)
     { 
           #ifdef _OPENMP
           #pragma omp for
           #endif
           for(i=0; i< face[mu]/2; i++)
           {
              j=V/2+face[mu]/2+i;
              k=plqcd_g.nn_bnde[2*mu+1][i]-V/2;
              _vector_assign(plqcd_g.phim[mu][k].s0, plqcd_g.phim[mu][j].s0);
              _vector_assign(plqcd_g.phim[mu][k].s1, plqcd_g.phim[mu][j].s1);
           }
      }


   //---------------------------------------------------------------------
   //finish computation of the U^dagger*(1+gamma_mu)*qin
   //---------------------------------------------------------------------
   #ifdef _OPENMP
   #pragma omp for 
   #endif
   for(i=0; i< V/2; i++)
   {
       j=i+V/2;
       intrin_vector_load_128(mq0,&qout[j].s0);
       intrin_vector_load_128(mq1,&qout[j].s1);
       intrin_vector_load_128(mq2,&qout[j].s2);
       intrin_vector_load_128(mq3,&qout[j].s3);
       // 0 direction
       _prefetch_halfspinor(&plqcd_g.phim[1][i]);
       //_vector_add_assign(qout[i].s0, plqcd_g.phim[0][i].s0);
       //_vector_add_assign(qout[i].s1, plqcd_g.phim[0][i].s1);
       //_vector_minus_assign(qout[i].s2, plqcd_g.phim[0][i].s0);
       //_vector_minus_assign(qout[i].s3, plqcd_g.phim[0][i].s1);
       intrin_vector_load_128(map0,&plqcd_g.phim[0][i].s0);
       intrin_vector_load_128(map1,&plqcd_g.phim[0][i].s1);
       intrin_vector_add_128(mq0,mq0,map0);
       intrin_vector_add_128(mq1,mq1,map1);
       intrin_vector_sub_128(mq2,mq2,map0);
       intrin_vector_sub_128(mq3,mq3,map1);


       // 1 direction
       _prefetch_halfspinor(&plqcd_g.phim[2][i]);
       //_vector_add_assign(qout[i].s0, plqcd_g.phim[1][i].s0);
       //_vector_add_assign(qout[i].s1, plqcd_g.phim[1][i].s1);
       //_vector_add_i_mul(qout[i].s2, 1.0,plqcd_g.phim[1][i].s1);
       //_vector_add_i_mul(qout[i].s3, 1.0, plqcd_g.phim[1][i].s0);
       intrin_vector_load_128(map0,&plqcd_g.phim[1][i].s0);
       intrin_vector_load_128(map1,&plqcd_g.phim[1][i].s1);
       intrin_vector_add_128(mq0,mq0,map0);
       intrin_vector_add_128(mq1,mq1,map1);
       intrin_vector_i_add_128(mq2,mq2,map1);
       intrin_vector_i_add_128(mq3,mq3,map0);


       // 2 direction
       _prefetch_halfspinor(&plqcd_g.phim[3][i]);
       //_vector_add_assign(qout[i].s0, plqcd_g.phim[2][i].s0);
       //_vector_add_assign(qout[i].s1,  plqcd_g.phim[2][i].s1 );
       //_vector_add_assign(qout[i].s2,  plqcd_g.phim[2][i].s1);
       //_vector_sub_assign(qout[i].s3,   plqcd_g.phim[2][i].s0);
       intrin_vector_load_128(map0,&plqcd_g.phim[2][i].s0);
       intrin_vector_load_128(map1,&plqcd_g.phim[2][i].s1);
       intrin_vector_add_128(mq0,mq0,map0);
       intrin_vector_add_128(mq1,mq1,map1);
       intrin_vector_add_128(mq2,mq2,map1);
       intrin_vector_sub_128(mq3,mq3,map0);


       //3 direction
       //_vector_add_assign(qout[i].s0, plqcd_g.phim[3][i].s0);
       //_vector_add_assign(qout[i].s1, plqcd_g.phim[3][i].s1);
       //_vector_i_add_assign(qout[i].s2, plqcd_g.phim[3][i].s0);
       //_vector_i_sub_assign(qout[i].s3, plqcd_g.phim[3][i].s1);     
       intrin_vector_load_128(map0,&plqcd_g.phim[3][i].s0);
       intrin_vector_load_128(map1,&plqcd_g.phim[3][i].s1);
       intrin_vector_add_128(mq0,mq0,map0);
       intrin_vector_add_128(mq1,mq1,map1);
       intrin_vector_i_add_128(mq2,mq2,map0);
       intrin_vector_i_add_128(mq3,mq3,map1);


       //store the final result
       intrin_vector_store_128(&qout[j].s0,mq0);
       intrin_vector_store_128(&qout[j].s1,mq1);
       intrin_vector_store_128(&qout[j].s2,mq2);
       intrin_vector_store_128(&qout[j].s3,mq3);

   }
#ifdef _OPENMP
}
#endif //end of the openmp parallel reigon

   return stop_watch(ts);
}

//========================================================================================
//============================  Blocking Version =========================================
//main advantage: access of qin and qout only once
//disadvantage: no overlap of communications and computations
//may payoff for small lattices 
//===================================EO===================================================
double plqcd_hopping_matrix_eo_sse3_intrin_blocking(spinor *qin, spinor *qout, su3 *u)
{
   
   
   int snd_err[8],rcv_err[8];
   MPI_Status rcv_mpi_stat[8];
   if(itags==0)
      get_tags(); //get the permanent tags for communications

   double ts; //timer

   ts=stop_watch(0.0);
   
   #ifdef _OPENMP
   #pragma omp parallel
   {
   #endif
      int i,j,k,mu,V,face[4];
      V = plqcd_g.VOLUME;
      for(i=0; i<4; i++)
        face[i] = plqcd_g.face[i];
   
      __m128d qins0[3],qins1[3],qins2[3],qins3[3], U[3][3];
      __m128d in1[3],out[3];
      __m128d map0[3],map1[3],mq0[3],mq1[3],mq2[3],mq3[3];

      int iup0,iup1,iup2,iup3,idn0,idn1,idn2,idn3;    
      su3 *ub0;

      //---------------------------------------------------------------
      // compute (1-gamma_mu)qin terms and store in phip 
      // compute U^dagger*(1+gamma_mu)qin terms and stroe in phim
      //---------------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=V/2; i < V; i++)
      {
         _prefetch_spinor(qin+i);
         idn0=plqcd_g.idn[i][0];
         idn1=plqcd_g.idn[i][1];
         idn2=plqcd_g.idn[i][2];
         idn3=plqcd_g.idn[i][3];
         iup0=plqcd_g.iup[i][0];
         iup1=plqcd_g.iup[i][1];
         iup2=plqcd_g.iup[i][2];
         iup3=plqcd_g.iup[i][3];
         ub0 = &u[4*i];

         _prefetch_halfspinor(&plqcd_g.phip[0][idn0]);
      

         intrin_vector_load_128(qins0,&qin[i].s0);
         intrin_vector_load_128(qins1,&qin[i].s1);
         intrin_vector_load_128(qins2,&qin[i].s2);
         intrin_vector_load_128(qins3,&qin[i].s3);

         //-- +0 direction ---
         _prefetch_halfspinor(&plqcd_g.phim[0][iup0]);
         _prefetch_su3(ub0);
         //_vector_add(plqcd_g.phip[0][idn0].s0, qin[i].s0, qin[i].s2);
         //_vector_add(plqcd_g.phip[0][idn0].s1, qin[i].s1, qin[i].s3);
         intrin_vector_add_128(out,qins0,qins2);
         intrin_vector_store_128(&plqcd_g.phip[0][idn0].s0,out);
         intrin_vector_add_128(out,qins1,qins3);
         intrin_vector_store_128(&plqcd_g.phip[0][idn0].s1,out);
  

         //-- -0 direction ---
         _prefetch_halfspinor(&plqcd_g.phip[1][idn1]);
         //_vector_sub(p,qin[i].s0,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s0,*ub0,p);
         //_vector_sub(p,qin[i].s1,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s1,*ub0,p);
         intrin_su3_load_128(U,ub0);
         intrin_vector_sub_128(in1,qins0,qins2);
         intrin_su3_inverse_multiply_128(out,U,in1);
         intrin_vector_store_128(&plqcd_g.phim[0][iup0].s0,out);
         intrin_vector_sub_128(in1,qins1,qins3);
         intrin_su3_inverse_multiply_128(out,U,in1);
         intrin_vector_store_128(&plqcd_g.phim[0][iup0].s1,out);
         ub0++;


         //-- +1 direction --
         _prefetch_halfspinor(&plqcd_g.phim[1][iup1]);
         _prefetch_su3(ub0);
         //_vector_i_add(plqcd_g.phip[1][idn1].s0, qin[i].s0, qin[i].s3);
         //_vector_i_add(plqcd_g.phip[1][idn1].s1, qin[i].s1, qin[i].s2);
         intrin_vector_i_add_128(out,qins0,qins3);
         intrin_vector_store_128(&plqcd_g.phip[1][idn1].s0,out);
         intrin_vector_add_128(out,qins1,qins2);
         intrin_vector_store_128(&plqcd_g.phip[1][idn1].s1,out);


        //-- -1 direction --
         _prefetch_halfspinor(&plqcd_g.phip[2][idn2]);
        //_vector_i_sub(p,qin[i].s0,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s0,*ub0,p);
        //_vector_i_sub(p,qin[i].s1,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_i_sub_128(in1,qins0,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[1][iup1].s0,out);
        intrin_vector_i_sub_128(in1,qins1,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[1][iup1].s1,out);
        ub0++;


        //-- +2 direction --
        _prefetch_halfspinor(&plqcd_g.phim[2][iup2]);
        _prefetch_su3(ub0);
        //_vector_add(plqcd_g.phip[2][idn2].s0, qin[i].s0, qin[i].s3);
        //_vector_sub(plqcd_g.phip[2][idn2].s1, qin[i].s1, qin[i].s2);
        intrin_vector_add_128(out,qins0,qins3);
        intrin_vector_store_128(&plqcd_g.phip[2][idn2].s0,out);
        intrin_vector_sub_128(out,qins1,qins2);
        intrin_vector_store_128(&plqcd_g.phip[2][idn2].s1,out);


        //-- -2 direction --
        _prefetch_halfspinor(&plqcd_g.phip[3][idn3]);
        //_vector_sub(p,qin[i].s0,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s0,*ub0,p);
        //_vector_add(p,qin[i].s1,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_sub_128(in1,qins0,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[2][iup2].s0,out);
        intrin_vector_add_128(in1,qins1,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[2][iup2].s1,out);
        ub0++;


         //-- +3 direction --
         _prefetch_halfspinor(&plqcd_g.phim[3][iup3]);
         _prefetch_su3(ub0);
         //_vector_i_add(plqcd_g.phip[3][idn3].s0, qin[i].s0, qin[i].s2);
         //_vector_i_sub(plqcd_g.phip[3][idn3].s1, qin[i].s1, qin[i].s3);
         intrin_vector_i_add_128(out,qins0,qins2);
         intrin_vector_store_128(&plqcd_g.phip[3][idn3].s0,out);
         intrin_vector_i_sub_128(out,qins1,qins3);
         intrin_vector_store_128(&plqcd_g.phip[3][idn3].s1,out);

        //-- -3 direction --
        //_vector_i_sub(p,qin[i].s0,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s0,*ub0,p);
        //_vector_i_add(p,qin[i].s1,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_i_sub_128(in1,qins0,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[3][iup3].s0,out);
        intrin_vector_i_add_128(in1,qins1,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[3][iup3].s1,out);

      }


      //send buffers to the nearest neighbours
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
         for(mu=0; mu < 4; mu++){  
           if(plqcd_g.nprocs[mu] > 1){  
             snd_err[mu]   = MPI_Send(&plqcd_g.phip[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu],  MPI_COMM_WORLD);
             rcv_err[mu]   = MPI_Recv(&plqcd_g.phip[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_mpi_stat[mu]);
             snd_err[mu+4] = MPI_Send(&plqcd_g.phim[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD);
             rcv_err[mu+4] = MPI_Recv(&plqcd_g.phim[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_mpi_stat[mu+4]);

             if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) || (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS)  ){
                    fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
                    exit(1);}
            }
         }
      #ifdef _OPENMP
      }
      #endif
    
      //------------------------------------------------------------------------------------
      //copy the exchanged boundaries to the correspoding locations on the local phi fields
      //------------------------------------------------------------------------------------
      for(mu=0; mu<4; mu++)
      { 
          #ifdef _OPENMP 
          #pragma omp for 
          #endif  
          for(i=0; i< face[mu]/2; i++)
          {
              //can we prefetch here
              j=V/2+face[mu]/2+i;
              k=plqcd_g.nn_bndo[2*mu][i];
              _vector_assign(plqcd_g.phip[mu][k].s0 , plqcd_g.phip[mu][j].s0);
              _vector_assign(plqcd_g.phip[mu][k].s1 , plqcd_g.phip[mu][j].s1);
              k=plqcd_g.nn_bndo[2*mu+1][i];
              _vector_assign(plqcd_g.phim[mu][k].s0, plqcd_g.phim[mu][j].s0);
              _vector_assign(plqcd_g.phim[mu][k].s1, plqcd_g.phim[mu][j].s1);
          }
      }

      //------------------------
      //compute the final result
      //------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=0; i< V/2; i++)
      {

         ub0 = u+4*i;

         // +0
         _prefetch_halfspinor(&plqcd_g.phim[0][i]);
         //_su3_multiply(q0, *ub0, plqcd_g.phip[0][i].s0);
         //_su3_multiply(q1, *ub0, plqcd_g.phip[0][i].s1);
         //_vector_assign(q2, q0);
         //_vector_assign(q3, q1);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[0][i].s0);
         intrin_su3_multiply_128(mq0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[0][i].s1);
         intrin_su3_multiply_128(mq1,U,in1);
         for(k=0; k<3; k++){
            mq2[k]= mq0[k];
            mq3[k]= mq1[k];}
         ub0++;



         // -0 direction
         _prefetch_halfspinor(&plqcd_g.phip[1][i]);
         _prefetch_su3(ub0);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[0][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[0][i].s1);
         //_vector_minus_assign(qout[i].s2, plqcd_g.phim[0][i].s0);
         //_vector_minus_assign(qout[i].s3, plqcd_g.phim[0][i].s1);
         intrin_vector_load_128(map0,&plqcd_g.phim[0][i].s0);
         intrin_vector_load_128(map1,&plqcd_g.phim[0][i].s1);
         intrin_vector_add_128(mq0,mq0,map0);
         intrin_vector_add_128(mq1,mq1,map1);
         intrin_vector_sub_128(mq2,mq2,map0);
         intrin_vector_sub_128(mq3,mq3,map1);


         // +1
         _prefetch_halfspinor(&plqcd_g.phim[1][i]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[1][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[1][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2, -1.0, ap1);
         //_vector_add_i_mul(q3, -1.0, ap0);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[1][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[1][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]); 
            mq2[k] = complex_i_sub_regs_128(mq2[k],map1[k]);
            mq3[k] = complex_i_sub_regs_128(mq3[k],map0[k]);}

         ub0++;        


         // -1 direction
         _prefetch_halfspinor(&plqcd_g.phip[2][i]);
         _prefetch_su3(ub0);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[1][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s2, 1.0,plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s3, 1.0, plqcd_g.phim[1][i].s0);
         intrin_vector_load_128(map0,&plqcd_g.phim[1][i].s0);
         intrin_vector_load_128(map1,&plqcd_g.phim[1][i].s1);
         intrin_vector_add_128(mq0,mq0,map0);
         intrin_vector_add_128(mq1,mq1,map1);
         intrin_vector_i_add_128(mq2,mq2,map1);
         intrin_vector_i_add_128(mq3,mq3,map0);

         // +2
         _prefetch_halfspinor(&plqcd_g.phim[2][i]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[2][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[2][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_sub_assign(q2,ap1);
         //_vector_add_assign(q3,ap0);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[2][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[2][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]);
            mq2[k] = _mm_sub_pd(mq2[k],map1[k]);
            mq3[k] = _mm_add_pd(mq3[k],map0[k]);}

         ub0++;        


         // -2 direction
         _prefetch_halfspinor(&plqcd_g.phip[3][i]);
         _prefetch_su3(ub0);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[2][i].s0);
         //_vector_add_assign(qout[i].s1,  plqcd_g.phim[2][i].s1 );
         //_vector_add_assign(qout[i].s2,  plqcd_g.phim[2][i].s1);
         //_vector_sub_assign(qout[i].s3,   plqcd_g.phim[2][i].s0);
         intrin_vector_load_128(map0,&plqcd_g.phim[2][i].s0);
         intrin_vector_load_128(map1,&plqcd_g.phim[2][i].s1);
         intrin_vector_add_128(mq0,mq0,map0);
         intrin_vector_add_128(mq1,mq1,map1);
         intrin_vector_add_128(mq2,mq2,map1);
         intrin_vector_sub_128(mq3,mq3,map0);

         // +3
         _prefetch_halfspinor(&plqcd_g.phim[3][i]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[3][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[3][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2,-1.0, ap0);
         //_vector_add_i_mul(q3, 1.0, ap1);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[3][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[3][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]); 
            mq2[k] = complex_i_sub_regs_128(mq2[k],map0[k]);
            mq3[k] = complex_i_add_regs_128(mq3[k],map1[k]);}


         _prefetch_spinor(&qout[i]);
         //3 direction
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[3][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[3][i].s1);
         //_vector_i_add_assign(qout[i].s2, plqcd_g.phim[3][i].s0);
         //_vector_i_sub_assign(qout[i].s3, plqcd_g.phim[3][i].s1);     
         intrin_vector_load_128(map0,&plqcd_g.phim[3][i].s0);
         intrin_vector_load_128(map1,&plqcd_g.phim[3][i].s1);
         intrin_vector_add_128(mq0,mq0,map0);
         intrin_vector_add_128(mq1,mq1,map1);
         intrin_vector_i_add_128(mq2,mq2,map0);
         intrin_vector_i_add_128(mq3,mq3,map1);

         //store the result
         //_vector_assign(qout[i].s0,q0);
         //_vector_assign(qout[i].s1,q1);
         //_vector_assign(qout[i].s2,q2);
         //_vector_assign(qout[i].s3,q3);
         intrin_vector_store_128(&qout[i].s0,mq0);         
         intrin_vector_store_128(&qout[i].s1,mq1);         
         intrin_vector_store_128(&qout[i].s2,mq1);         
         intrin_vector_store_128(&qout[i].s3,mq1);         

     } 
#ifdef _OPENMP
}
#endif //endif of the openmp parallel reigon


   return stop_watch(ts);
}


//===================================OE===================================================
double plqcd_hopping_matrix_oe_sse3_intrin_blocking(spinor *qin, spinor *qout, su3 *u)
{
   
   
   int snd_err[8],rcv_err[8];
   MPI_Status rcv_mpi_stat[8];
   if(itags==0)
      get_tags(); //get the permanent tags for communications

   double ts; //timer

   ts=stop_watch(0.0);
   
   #ifdef _OPENMP
   #pragma omp parallel
   {
   #endif
      int i,j,k,mu,V,face[4];
      V = plqcd_g.VOLUME;
      for(i=0; i<4; i++)
        face[i] = plqcd_g.face[i];
   
      __m128d qins0[3],qins1[3],qins2[3],qins3[3], U[3][3];
      __m128d in1[3],out[3];
      __m128d map0[3],map1[3],mq0[3],mq1[3],mq2[3],mq3[3];

      int iup0,iup1,iup2,iup3,idn0,idn1,idn2,idn3;    
      su3 *ub0;

      //---------------------------------------------------------------
      // compute (1-gamma_mu)qin terms and store in phip 
      // compute U^dagger*(1+gamma_mu)qin terms and stroe in phim
      //---------------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=0; i < V/2; i++)
      {
         _prefetch_spinor(qin+i);
         idn0=plqcd_g.idn[i][0]-V/2;
         idn1=plqcd_g.idn[i][1]-V/2;
         idn2=plqcd_g.idn[i][2]-V/2;
         idn3=plqcd_g.idn[i][3]-V/2;
         iup0=plqcd_g.iup[i][0]-V/2;
         iup1=plqcd_g.iup[i][1]-V/2;
         iup2=plqcd_g.iup[i][2]-V/2;
         iup3=plqcd_g.iup[i][3]-V/2;
         ub0 = &u[4*i];

         _prefetch_halfspinor(&plqcd_g.phip[0][idn0]);
      

         intrin_vector_load_128(qins0,&qin[i].s0);
         intrin_vector_load_128(qins1,&qin[i].s1);
         intrin_vector_load_128(qins2,&qin[i].s2);
         intrin_vector_load_128(qins3,&qin[i].s3);

         //-- +0 direction ---
         _prefetch_halfspinor(&plqcd_g.phim[0][iup0]);
         _prefetch_su3(ub0);
         //_vector_add(plqcd_g.phip[0][idn0].s0, qin[i].s0, qin[i].s2);
         //_vector_add(plqcd_g.phip[0][idn0].s1, qin[i].s1, qin[i].s3);
         intrin_vector_add_128(out,qins0,qins2);
         intrin_vector_store_128(&plqcd_g.phip[0][idn0].s0,out);
         intrin_vector_add_128(out,qins1,qins3);
         intrin_vector_store_128(&plqcd_g.phip[0][idn0].s1,out);
  

         //-- -0 direction ---
         _prefetch_halfspinor(&plqcd_g.phip[1][idn1]);
         //_vector_sub(p,qin[i].s0,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s0,*ub0,p);
         //_vector_sub(p,qin[i].s1,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s1,*ub0,p);
         intrin_su3_load_128(U,ub0);
         intrin_vector_sub_128(in1,qins0,qins2);
         intrin_su3_inverse_multiply_128(out,U,in1);
         intrin_vector_store_128(&plqcd_g.phim[0][iup0].s0,out);
         intrin_vector_sub_128(in1,qins1,qins3);
         intrin_su3_inverse_multiply_128(out,U,in1);
         intrin_vector_store_128(&plqcd_g.phim[0][iup0].s1,out);
         ub0++;


         //-- +1 direction --
         _prefetch_halfspinor(&plqcd_g.phim[1][iup1]);
         _prefetch_su3(ub0);
         //_vector_i_add(plqcd_g.phip[1][idn1].s0, qin[i].s0, qin[i].s3);
         //_vector_i_add(plqcd_g.phip[1][idn1].s1, qin[i].s1, qin[i].s2);
         intrin_vector_i_add_128(out,qins0,qins3);
         intrin_vector_store_128(&plqcd_g.phip[1][idn1].s0,out);
         intrin_vector_add_128(out,qins1,qins2);
         intrin_vector_store_128(&plqcd_g.phip[1][idn1].s1,out);


        //-- -1 direction --
         _prefetch_halfspinor(&plqcd_g.phip[2][idn2]);
        //_vector_i_sub(p,qin[i].s0,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s0,*ub0,p);
        //_vector_i_sub(p,qin[i].s1,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_i_sub_128(in1,qins0,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[1][iup1].s0,out);
        intrin_vector_i_sub_128(in1,qins1,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[1][iup1].s1,out);
        ub0++;


        //-- +2 direction --
        _prefetch_halfspinor(&plqcd_g.phim[2][iup2]);
        _prefetch_su3(ub0);
        //_vector_add(plqcd_g.phip[2][idn2].s0, qin[i].s0, qin[i].s3);
        //_vector_sub(plqcd_g.phip[2][idn2].s1, qin[i].s1, qin[i].s2);
        intrin_vector_add_128(out,qins0,qins3);
        intrin_vector_store_128(&plqcd_g.phip[2][idn2].s0,out);
        intrin_vector_sub_128(out,qins1,qins2);
        intrin_vector_store_128(&plqcd_g.phip[2][idn2].s1,out);


        //-- -2 direction --
        _prefetch_halfspinor(&plqcd_g.phip[3][idn3]);
        //_vector_sub(p,qin[i].s0,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s0,*ub0,p);
        //_vector_add(p,qin[i].s1,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_sub_128(in1,qins0,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[2][iup2].s0,out);
        intrin_vector_add_128(in1,qins1,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[2][iup2].s1,out);
        ub0++;


         //-- +3 direction --
         _prefetch_halfspinor(&plqcd_g.phim[3][iup3]);
         _prefetch_su3(ub0);
         //_vector_i_add(plqcd_g.phip[3][idn3].s0, qin[i].s0, qin[i].s2);
         //_vector_i_sub(plqcd_g.phip[3][idn3].s1, qin[i].s1, qin[i].s3);
         intrin_vector_i_add_128(out,qins0,qins2);
         intrin_vector_store_128(&plqcd_g.phip[3][idn3].s0,out);
         intrin_vector_i_sub_128(out,qins1,qins3);
         intrin_vector_store_128(&plqcd_g.phip[3][idn3].s1,out);

        //-- -3 direction --
        //_vector_i_sub(p,qin[i].s0,qin[i].s2);
        //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s0,*ub0,p);
        //_vector_i_add(p,qin[i].s1,qin[i].s3);
        //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s1,*ub0,p);
        intrin_su3_load_128(U,ub0);
        intrin_vector_i_sub_128(in1,qins0,qins2);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[3][iup3].s0,out);
        intrin_vector_i_add_128(in1,qins1,qins3);
        intrin_su3_inverse_multiply_128(out,U,in1);
        intrin_vector_store_128(&plqcd_g.phim[3][iup3].s1,out);

      }


      //send buffers to the nearest neighbours
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
         for(mu=0; mu < 4; mu++){  
           if(plqcd_g.nprocs[mu] > 1){  
             snd_err[mu]   = MPI_Send(&plqcd_g.phip[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu],  MPI_COMM_WORLD);
             rcv_err[mu]   = MPI_Recv(&plqcd_g.phip[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_mpi_stat[mu]);
             snd_err[mu+4] = MPI_Send(&plqcd_g.phim[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD);
             rcv_err[mu+4] = MPI_Recv(&plqcd_g.phim[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_mpi_stat[mu]);

             if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) || (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS)  ){
                    fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
                    exit(1);}
            }
         }
      #ifdef _OPENMP
      }
      #endif
    
      //------------------------------------------------------------------------------------
      //copy the exchanged boundaries to the correspoding locations on the local phi fields
      //------------------------------------------------------------------------------------
      for(mu=0; mu<4; mu++)
      { 
          #ifdef _OPENMP 
          #pragma omp for 
          #endif  
          for(i=0; i< face[mu]/2; i++)
          {
              //can we prefetch here
              j=V/2+face[mu]/2+i;
              k=plqcd_g.nn_bnde[2*mu][i]-V/2;
              _vector_assign(plqcd_g.phip[mu][k].s0 , plqcd_g.phip[mu][j].s0);
              _vector_assign(plqcd_g.phip[mu][k].s1 , plqcd_g.phip[mu][j].s1);
              k=plqcd_g.nn_bnde[2*mu+1][i]-V/2;
              _vector_assign(plqcd_g.phim[mu][k].s0, plqcd_g.phim[mu][j].s0);
              _vector_assign(plqcd_g.phim[mu][k].s1, plqcd_g.phim[mu][j].s1);
          }
      }

      //------------------------
      //compute the final result
      //------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=0; i< V/2; i++)
      {

         j=i+V/2;
         ub0 = u+4*i;

         // +0
         _prefetch_halfspinor(&plqcd_g.phim[0][i]);
         //_su3_multiply(q0, *ub0, plqcd_g.phip[0][i].s0);
         //_su3_multiply(q1, *ub0, plqcd_g.phip[0][i].s1);
         //_vector_assign(q2, q0);
         //_vector_assign(q3, q1);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[0][i].s0);
         intrin_su3_multiply_128(mq0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[0][i].s1);
         intrin_su3_multiply_128(mq1,U,in1);
         for(k=0; k<3; k++){
            mq2[k]= mq0[k];
            mq3[k]= mq1[k];}
         ub0++;



         // -0 direction
         _prefetch_halfspinor(&plqcd_g.phip[1][i]);
         _prefetch_su3(ub0);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[0][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[0][i].s1);
         //_vector_minus_assign(qout[i].s2, plqcd_g.phim[0][i].s0);
         //_vector_minus_assign(qout[i].s3, plqcd_g.phim[0][i].s1);
         intrin_vector_load_128(map0,&plqcd_g.phim[0][i].s0);
         intrin_vector_load_128(map1,&plqcd_g.phim[0][i].s1);
         intrin_vector_add_128(mq0,mq0,map0);
         intrin_vector_add_128(mq1,mq1,map1);
         intrin_vector_sub_128(mq2,mq2,map0);
         intrin_vector_sub_128(mq3,mq3,map1);


         // +1
         _prefetch_halfspinor(&plqcd_g.phim[1][i]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[1][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[1][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2, -1.0, ap1);
         //_vector_add_i_mul(q3, -1.0, ap0);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[1][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[1][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]); 
            mq2[k] = complex_i_sub_regs_128(mq2[k],map1[k]);
            mq3[k] = complex_i_sub_regs_128(mq3[k],map0[k]);}

         ub0++;        


         // -1 direction
         _prefetch_halfspinor(&plqcd_g.phip[2][i]);
         _prefetch_su3(ub0);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[1][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s2, 1.0,plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s3, 1.0, plqcd_g.phim[1][i].s0);
         intrin_vector_load_128(map0,&plqcd_g.phim[1][i].s0);
         intrin_vector_load_128(map1,&plqcd_g.phim[1][i].s1);
         intrin_vector_add_128(mq0,mq0,map0);
         intrin_vector_add_128(mq1,mq1,map1);
         intrin_vector_i_add_128(mq2,mq2,map1);
         intrin_vector_i_add_128(mq3,mq3,map0);

         // +2
         _prefetch_halfspinor(&plqcd_g.phim[2][i]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[2][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[2][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_sub_assign(q2,ap1);
         //_vector_add_assign(q3,ap0);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[2][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[2][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]);
            mq2[k] = _mm_sub_pd(mq2[k],map1[k]);
            mq3[k] = _mm_add_pd(mq3[k],map0[k]);}

         ub0++;        


         // -2 direction
         _prefetch_halfspinor(&plqcd_g.phip[3][i]);
         _prefetch_su3(ub0);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[2][i].s0);
         //_vector_add_assign(qout[i].s1,  plqcd_g.phim[2][i].s1 );
         //_vector_add_assign(qout[i].s2,  plqcd_g.phim[2][i].s1);
         //_vector_sub_assign(qout[i].s3,   plqcd_g.phim[2][i].s0);
         intrin_vector_load_128(map0,&plqcd_g.phim[2][i].s0);
         intrin_vector_load_128(map1,&plqcd_g.phim[2][i].s1);
         intrin_vector_add_128(mq0,mq0,map0);
         intrin_vector_add_128(mq1,mq1,map1);
         intrin_vector_add_128(mq2,mq2,map1);
         intrin_vector_sub_128(mq3,mq3,map0);

         // +3
         _prefetch_halfspinor(&plqcd_g.phim[3][i]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[3][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[3][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2,-1.0, ap0);
         //_vector_add_i_mul(q3, 1.0, ap1);
         intrin_su3_load_128(U,ub0);
         intrin_vector_load_128(in1,&plqcd_g.phip[3][i].s0);
         intrin_su3_multiply_128(map0,U,in1);
         intrin_vector_load_128(in1,&plqcd_g.phip[3][i].s1);
         intrin_su3_multiply_128(map1,U,in1);
         for(k=0; k<3; k++){
            mq0[k] = _mm_add_pd(mq0[k],map0[k]);
            mq1[k] = _mm_add_pd(mq1[k],map1[k]); 
            mq2[k] = complex_i_sub_regs_128(mq2[k],map0[k]);
            mq3[k] = complex_i_add_regs_128(mq3[k],map1[k]);}


         _prefetch_spinor(&qout[j]);
         //3 direction
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[3][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[3][i].s1);
         //_vector_i_add_assign(qout[i].s2, plqcd_g.phim[3][i].s0);
         //_vector_i_sub_assign(qout[i].s3, plqcd_g.phim[3][i].s1);     
         intrin_vector_load_128(map0,&plqcd_g.phim[3][i].s0);
         intrin_vector_load_128(map1,&plqcd_g.phim[3][i].s1);
         intrin_vector_add_128(mq0,mq0,map0);
         intrin_vector_add_128(mq1,mq1,map1);
         intrin_vector_i_add_128(mq2,mq2,map0);
         intrin_vector_i_add_128(mq3,mq3,map1);

         //store the result
         //_vector_assign(qout[i].s0,q0);
         //_vector_assign(qout[i].s1,q1);
         //_vector_assign(qout[i].s2,q2);
         //_vector_assign(qout[i].s3,q3);
         intrin_vector_store_128(&qout[j].s0,mq0);         
         intrin_vector_store_128(&qout[j].s1,mq1);         
         intrin_vector_store_128(&qout[j].s2,mq1);         
         intrin_vector_store_128(&qout[j].s3,mq1);         

     } 
#ifdef _OPENMP
}
#endif //endif of the openmp parallel reigon


   return stop_watch(ts);
}



//======================================================
//sse with openmp only, split real and imaginary, double
//No halfspinor, loop over the output spinor 
//in exactly the same way the operaotr is written 
//================Even-Odd==============================
double plqcd_hopping_matrix_eo_sse3_intrin_nohalfspinor(
                                double *qout_re, 
                                double *qout_im, 
                                double *u_re, 
                                double *u_im,
                                double *qin_re, 
                                double *qin_im) 
{
   double ts;          //timer
   ts=stop_watch(0.0); //start
   #ifdef _OPENMP
   #pragma omp parallel 
   {
   #endif
      int V,Vsse_split,Vyzt,lx;
      V = plqcd_g.VOLUME;
      Vsse_split = plqcd_g.Vsse_split;
      Vyzt = plqcd_g.latdims[1]*plqcd_g.latdims[2]*plqcd_g.latdims[3];
      lx=plqcd_g.latdims[0]/2;
      if((plqcd_g.latdims[0]%4) != 0 ){ 
        fprintf(stderr,"lattice size in the 0 direction must be a factor of 4\n");
        exit(1);
      }
      
      __m128d  mqin_re[12],mqin_im[12], mqout_re[12],mqout_im[12];
      __m128d  U_re[3][3], U_im[3][3];
      __m128d  out1_re[3],out2_re[3],out3_re[3],out4_re[3];
      __m128d  out1_im[3],out2_im[3],out3_im[3],out4_im[3];
      __m128d  map0_re[3],map0_im[3],map1_re[3],map1_im[3];
      __m128d  register m1,m2;

      int ix,iy,it,iz,is,xcor[4],ipt;

      int L[4];
      for(int i=0; i<4; i++)
         L[i] = plqcd_g.latdims[i];

      //needed pointers
      //input spinor, links and output spinor
      double *sx_re,*sx_im,*ux_re,*ux_im,*gx_re,*gx_im;

      //-----------------------------------------------------
      // compute the result
      //----------------------------------------------------
      #ifdef _OPENMP
      //#pragma omp for schedule(static,20) //use this if you want to play with how work is shared among threads 
      #pragma omp for 
      #endif   
      for(int is=0; is < Vyzt; is++) 
      {
         iy = is%L[1];
         iz = (is/L[1]) %L[2];
         it = (is/L[1]/L[2])%L[3];

         for(ix = (iy+iz+it)%2; ix < lx; ix +=2)  //result is on even sites
         {
            xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
            ipt=plqcd_g.ipt_eo_sse_split[cart2lex(xcor)];

            gx_re = qout_re + ipt*24;
            gx_im = qout_im + ipt*24;

            //================
            //====  +0  ======
            //================

            ux_re = u_re + ipt*72;
            ux_im = u_im + ipt*72;

            sx_re = qin_re + plqcd_g.iup_sse_split[ipt][0]*24;
            sx_im = qin_im + plqcd_g.iup_sse_split[ipt][0]*24;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm_load_pd(sx_re);
            mqin_re[1]  = _mm_load_pd(sx_re+2);
            mqin_re[2]  = _mm_load_pd(sx_re+4);
            mqin_im[0]  = _mm_load_pd(sx_im);
            mqin_im[1]  = _mm_load_pd(sx_im+2);
            mqin_im[2]  = _mm_load_pd(sx_im+4);


            mqin_re[9]   = _mm_load_pd(sx_re+18);
            mqin_re[10]  = _mm_load_pd(sx_re+20);
            mqin_re[11]  = _mm_load_pd(sx_re+22);
            mqin_im[9]   = _mm_load_pd(sx_im+18);
            mqin_im[10]  = _mm_load_pd(sx_im+20);
            mqin_im[11]  = _mm_load_pd(sx_im+22);


            out1_re[0] = _mm_sub_pd(mqin_re[0],mqin_im[9]);            
            out1_re[1] = _mm_sub_pd(mqin_re[1],mqin_im[10]);            
            out1_re[2] = _mm_sub_pd(mqin_re[2],mqin_im[11]);            
            out1_im[0] = _mm_add_pd(mqin_im[0],mqin_re[9]);            
            out1_im[1] = _mm_add_pd(mqin_im[1],mqin_re[10]);            
            out1_im[2] = _mm_add_pd(mqin_im[2],mqin_re[11]);


            mqin_re[3]  = _mm_load_pd(sx_re+6);
            mqin_re[4]  = _mm_load_pd(sx_re+8);
            mqin_re[5]  = _mm_load_pd(sx_re+10);
            mqin_im[3]  = _mm_load_pd(sx_im+6);
            mqin_im[4]  = _mm_load_pd(sx_im+8);
            mqin_im[5]  = _mm_load_pd(sx_im+10);


            mqin_re[6]  = _mm_load_pd(sx_re+12);
            mqin_re[7]  = _mm_load_pd(sx_re+14);
            mqin_re[8]  = _mm_load_pd(sx_re+16);
            mqin_im[6]  = _mm_load_pd(sx_im+12);
            mqin_im[7]  = _mm_load_pd(sx_im+14);
            mqin_im[8]  = _mm_load_pd(sx_im+16);


            out2_re[0] = _mm_sub_pd(mqin_re[3],mqin_im[6]);            
            out2_re[1] = _mm_sub_pd(mqin_re[4],mqin_im[7]);            
            out2_re[2] = _mm_sub_pd(mqin_re[5],mqin_im[8]);            
            out2_im[0] = _mm_add_pd(mqin_im[3],mqin_re[6]);            
            out2_im[1] = _mm_add_pd(mqin_im[4],mqin_re[7]);            
            out2_im[2] = _mm_add_pd(mqin_im[5],mqin_re[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 2 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm_load_pd(ux_re+jc*2+ic*6);
                  U_im[ic][jc] = _mm_load_pd(ux_im+jc*2+ic*6);
               }

            su3_multiply_splitlayout_128(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_128(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            //store the result, shuffle if on the boundary
            if(ix == 0)
            {
               for(int i=0; i<3; i++)
               {  
                  out3_re[i]= _mm_shuffle_pd( map0_re[i], map0_re[i], 1);
                  out3_im[i]= _mm_shuffle_pd( map0_im[i], map0_im[i], 1);
                  out4_re[i]= _mm_shuffle_pd( map1_re[i], map1_re[i], 1);
                  out4_im[i]= _mm_shuffle_pd( map1_im[i], map1_im[i], 1);
               }
              
               mqout_re[0] = _mm_add_pd(mqout_re[0] ,out3_re[0]);
               mqout_re[1] = _mm_add_pd(mqout_re[1] ,out3_re[1]);
               mqout_re[2] = _mm_add_pd(mqout_re[2] ,out3_re[2]);
               mqout_re[3] = _mm_add_pd(mqout_re[3] ,out4_re[0]);
               mqout_re[4] = _mm_add_pd(mqout_re[4] ,out4_re[1]);
               mqout_re[5] = _mm_add_pd(mqout_re[5] ,out4_re[2]);

               mqout_re[6] = _mm_add_pd(mqout_re[6] ,out4_im[0]);
               mqout_re[7] = _mm_add_pd(mqout_re[7] ,out4_im[1]);
               mqout_re[8] = _mm_add_pd(mqout_re[8] ,out4_im[2]);
               mqout_re[9] = _mm_add_pd(mqout_re[9] ,out3_im[0]);
               mqout_re[10]= _mm_add_pd(mqout_re[10],out3_im[1]);
               mqout_re[11]= _mm_add_pd(mqout_re[11],out3_im[2]);

               mqout_im[0] = _mm_add_pd(mqout_im[0] ,out3_im[0]);
               mqout_im[1] = _mm_add_pd(mqout_im[1] ,out3_im[1]);
               mqout_im[2] = _mm_add_pd(mqout_im[2] ,out3_im[2]);
               mqout_im[3] = _mm_add_pd(mqout_im[3] ,out4_im[0]);
               mqout_im[4] = _mm_add_pd(mqout_im[4] ,out4_im[1]);
               mqout_im[5] = _mm_add_pd(mqout_im[5] ,out4_im[2]);

               mqout_im[6] = _mm_sub_pd(mqout_im[6] ,out4_re[0]);
               mqout_im[7] = _mm_sub_pd(mqout_im[7] ,out4_re[1]);
               mqout_im[8] = _mm_sub_pd(mqout_im[8] ,out4_re[2]);
               mqout_im[9] = _mm_sub_pd(mqout_im[9] ,out3_re[0]);
               mqout_im[10]= _mm_sub_pd(mqout_im[10],out3_re[1]);
               mqout_im[11]= _mm_sub_pd(mqout_im[11],out3_re[2]);

            }
            else
            {
               mqout_re[0] = _mm_add_pd(mqout_re[0] ,map0_re[0]);
               mqout_re[1] = _mm_add_pd(mqout_re[1] ,map0_re[1]);
               mqout_re[2] = _mm_add_pd(mqout_re[2] ,map0_re[2]);
               mqout_re[3] = _mm_add_pd(mqout_re[3] ,map1_re[0]);
               mqout_re[4] = _mm_add_pd(mqout_re[4] ,map1_re[1]);
               mqout_re[5] = _mm_add_pd(mqout_re[5] ,map1_re[2]);

               mqout_re[6] = _mm_add_pd(mqout_re[6] ,map1_im[0]);
               mqout_re[7] = _mm_add_pd(mqout_re[7] ,map1_im[1]);
               mqout_re[8] = _mm_add_pd(mqout_re[8] ,map1_im[2]);
               mqout_re[9] = _mm_add_pd(mqout_re[9] ,map0_im[0]);
               mqout_re[10]= _mm_add_pd(mqout_re[10],map0_im[1]);
               mqout_re[11]= _mm_add_pd(mqout_re[11],map0_im[2]);

               mqout_im[0] = _mm_add_pd(mqout_im[0] ,map0_im[0]);
               mqout_im[1] = _mm_add_pd(mqout_im[1] ,map0_im[1]);
               mqout_im[2] = _mm_add_pd(mqout_im[2] ,map0_im[2]);
               mqout_im[3] = _mm_add_pd(mqout_im[3] ,map1_im[0]);
               mqout_im[4] = _mm_add_pd(mqout_im[4] ,map1_im[1]);
               mqout_im[5] = _mm_add_pd(mqout_im[5] ,map1_im[2]);

               mqout_im[6] = _mm_sub_pd(mqout_im[6] ,map1_re[0]);
               mqout_im[7] = _mm_sub_pd(mqout_im[7] ,map1_re[1]);
               mqout_im[8] = _mm_sub_pd(mqout_im[8] ,map1_re[2]);
               mqout_im[9] = _mm_sub_pd(mqout_im[9] ,map0_re[0]);
               mqout_im[10]= _mm_sub_pd(mqout_im[10],map0_re[1]);
               mqout_im[11]= _mm_sub_pd(mqout_im[11],map0_re[2]);
            } 

            
            //===================
            //===== +1 ==========
            //===================
      
            ux_re += 18;
            ux_im += 18;

            sx_re = qin_re + plqcd_g.iup_sse_split[ipt][1]*24;
            sx_im = qin_im + plqcd_g.iup_sse_split[ipt][1]*24;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm_load_pd(sx_re);
            mqin_re[1]  = _mm_load_pd(sx_re+2);
            mqin_re[2]  = _mm_load_pd(sx_re+4);
            mqin_im[0]  = _mm_load_pd(sx_im);
            mqin_im[1]  = _mm_load_pd(sx_im+2);
            mqin_im[2]  = _mm_load_pd(sx_im+4);


            mqin_re[9]   = _mm_load_pd(sx_re+18);
            mqin_re[10]  = _mm_load_pd(sx_re+20);
            mqin_re[11]  = _mm_load_pd(sx_re+22);
            mqin_im[9]   = _mm_load_pd(sx_im+18);
            mqin_im[10]  = _mm_load_pd(sx_im+20);
            mqin_im[11]  = _mm_load_pd(sx_im+22);


            out1_re[0] = _mm_add_pd(mqin_re[0],mqin_re[9]);            
            out1_re[1] = _mm_add_pd(mqin_re[1],mqin_re[10]);            
            out1_re[2] = _mm_add_pd(mqin_re[2],mqin_re[11]);            
            out1_im[0] = _mm_add_pd(mqin_im[0],mqin_im[9]);            
            out1_im[1] = _mm_add_pd(mqin_im[1],mqin_im[10]);            
            out1_im[2] = _mm_add_pd(mqin_im[2],mqin_im[11]);

            mqin_re[3]  = _mm_load_pd(sx_re+6);
            mqin_re[4]  = _mm_load_pd(sx_re+8);
            mqin_re[5]  = _mm_load_pd(sx_re+10);
            mqin_im[3]  = _mm_load_pd(sx_im+6);
            mqin_im[4]  = _mm_load_pd(sx_im+8);
            mqin_im[5]  = _mm_load_pd(sx_im+10);


            mqin_re[6]  = _mm_load_pd(sx_re+12);
            mqin_re[7]  = _mm_load_pd(sx_re+14);
            mqin_re[8]  = _mm_load_pd(sx_re+16);
            mqin_im[6]  = _mm_load_pd(sx_im+12);
            mqin_im[7]  = _mm_load_pd(sx_im+14);
            mqin_im[8]  = _mm_load_pd(sx_im+16);


            out2_re[0] = _mm_sub_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm_sub_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm_sub_pd(mqin_re[5],mqin_re[8]);            
            out2_im[0] = _mm_sub_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm_sub_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm_sub_pd(mqin_im[5],mqin_im[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm_load_pd(ux_re+jc*2+ic*6);
                  U_im[ic][jc] = _mm_load_pd(ux_im+jc*2+ic*6);
               }
            
            su3_multiply_splitlayout_128(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_128(map1_re,map1_im,U_re,U_im,out2_re,out2_im);

            mqout_re[0] = _mm_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm_sub_pd(mqout_re[6] ,map1_re[0]);
            mqout_re[7] = _mm_sub_pd(mqout_re[7] ,map1_re[1]);
            mqout_re[8] = _mm_sub_pd(mqout_re[8] ,map1_re[2]);
            mqout_re[9] = _mm_add_pd(mqout_re[9] ,map0_re[0]);
            mqout_re[10]= _mm_add_pd(mqout_re[10],map0_re[1]);
            mqout_re[11]= _mm_add_pd(mqout_re[11],map0_re[2]);

            mqout_im[0] = _mm_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm_sub_pd(mqout_im[6] ,map1_im[0]);
            mqout_im[7] = _mm_sub_pd(mqout_im[7] ,map1_im[1]);
            mqout_im[8] = _mm_sub_pd(mqout_im[8] ,map1_im[2]);
            mqout_im[9] = _mm_add_pd(mqout_im[9] ,map0_im[0]);
            mqout_im[10]= _mm_add_pd(mqout_im[10],map0_im[1]);
            mqout_im[11]= _mm_add_pd(mqout_im[11],map0_im[2]);




            //===================
            //===== +2 ==========
            //===================
      
            ux_re += 18;
            ux_im += 18;

            sx_re = qin_re + plqcd_g.iup_sse_split[ipt][2]*24;
            sx_im = qin_im + plqcd_g.iup_sse_split[ipt][2]*24;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm_load_pd(sx_re);
            mqin_re[1]  = _mm_load_pd(sx_re+4);
            mqin_re[2]  = _mm_load_pd(sx_re+8);
            mqin_im[0]  = _mm_load_pd(sx_im);
            mqin_im[1]  = _mm_load_pd(sx_im+4);
            mqin_im[2]  = _mm_load_pd(sx_im+8);


            mqin_re[6]  = _mm_load_pd(sx_re+12);
            mqin_re[7]  = _mm_load_pd(sx_re+14);
            mqin_re[8]  = _mm_load_pd(sx_re+16);
            mqin_im[6]  = _mm_load_pd(sx_im+12);
            mqin_im[7]  = _mm_load_pd(sx_im+14);
            mqin_im[8]  = _mm_load_pd(sx_im+16);


            out1_re[0] = _mm_sub_pd(mqin_re[0],mqin_im[6]);            
            out1_re[1] = _mm_sub_pd(mqin_re[1],mqin_im[7]);            
            out1_re[2] = _mm_sub_pd(mqin_re[2],mqin_im[8]);            
            out1_im[0] = _mm_add_pd(mqin_im[0],mqin_re[6]);            
            out1_im[1] = _mm_add_pd(mqin_im[1],mqin_re[7]);            
            out1_im[2] = _mm_add_pd(mqin_im[2],mqin_re[8]);

            mqin_re[3]  = _mm_load_pd(sx_re+6);
            mqin_re[4]  = _mm_load_pd(sx_re+8);
            mqin_re[5]  = _mm_load_pd(sx_re+10);
            mqin_im[3]  = _mm_load_pd(sx_im+6);
            mqin_im[4]  = _mm_load_pd(sx_im+8);
            mqin_im[5]  = _mm_load_pd(sx_im+10);


            mqin_re[9]   = _mm_load_pd(sx_re+18);
            mqin_re[10]  = _mm_load_pd(sx_re+20);
            mqin_re[11]  = _mm_load_pd(sx_re+22);
            mqin_im[9]   = _mm_load_pd(sx_im+18);
            mqin_im[10]  = _mm_load_pd(sx_im+20);
            mqin_im[11]  = _mm_load_pd(sx_im+22);


            out2_re[0] = _mm_add_pd(mqin_re[3],mqin_im[9]);            
            out2_re[1] = _mm_add_pd(mqin_re[4],mqin_im[10]);            
            out2_re[2] = _mm_add_pd(mqin_re[5],mqin_im[11]);            
            out2_im[0] = _mm_sub_pd(mqin_im[3],mqin_re[9]);            
            out2_im[1] = _mm_sub_pd(mqin_im[4],mqin_re[10]);            
            out2_im[2] = _mm_sub_pd(mqin_im[5],mqin_re[11]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm_load_pd(ux_re+jc*2+ic*6);
                  U_im[ic][jc] = _mm_load_pd(ux_im+jc*2+ic*6);
               }
            
            su3_multiply_splitlayout_128(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_128(map1_re,map1_im,U_re,U_im,out2_re,out2_im);

            mqout_re[0] = _mm_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm_add_pd(mqout_re[6] ,map0_im[0]);
            mqout_re[7] = _mm_add_pd(mqout_re[7] ,map0_im[1]);
            mqout_re[8] = _mm_add_pd(mqout_re[8] ,map0_im[2]);
            mqout_re[9] = _mm_sub_pd(mqout_re[9] ,map1_im[0]);
            mqout_re[10]= _mm_sub_pd(mqout_re[10],map1_im[1]);
            mqout_re[11]= _mm_sub_pd(mqout_re[11],map1_im[2]);

            mqout_im[0] = _mm_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm_sub_pd(mqout_im[6] ,map0_re[0]);
            mqout_im[7] = _mm_sub_pd(mqout_im[7] ,map0_re[1]);
            mqout_im[8] = _mm_sub_pd(mqout_im[8] ,map0_re[2]);
            mqout_im[9] = _mm_add_pd(mqout_im[9] ,map1_re[0]);
            mqout_im[10]= _mm_add_pd(mqout_im[10],map1_re[1]);
            mqout_im[11]= _mm_add_pd(mqout_im[11],map1_re[2]);



            //===================
            //===== +3 ==========
            //===================
      
            ux_re += 18;
            ux_im += 18;

            sx_re = qin_re + plqcd_g.iup_sse_split[ipt][3]*24;
            sx_im = qin_im + plqcd_g.iup_sse_split[ipt][3]*24;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm_load_pd(sx_re);
            mqin_re[1]  = _mm_load_pd(sx_re+2);
            mqin_re[2]  = _mm_load_pd(sx_re+4);
            mqin_im[0]  = _mm_load_pd(sx_im);
            mqin_im[1]  = _mm_load_pd(sx_im+2);
            mqin_im[2]  = _mm_load_pd(sx_im+4);


            mqin_re[6]  = _mm_load_pd(sx_re+12);
            mqin_re[7]  = _mm_load_pd(sx_re+14);
            mqin_re[8]  = _mm_load_pd(sx_re+16);
            mqin_im[6]  = _mm_load_pd(sx_im+12);
            mqin_im[7]  = _mm_load_pd(sx_im+14);
            mqin_im[8]  = _mm_load_pd(sx_im+16);


            out1_re[0] = _mm_add_pd(mqin_re[0],mqin_re[6]);            
            out1_re[1] = _mm_add_pd(mqin_re[1],mqin_re[7]);            
            out1_re[2] = _mm_add_pd(mqin_re[2],mqin_re[8]);            
            out1_im[0] = _mm_add_pd(mqin_im[0],mqin_im[6]);            
            out1_im[1] = _mm_add_pd(mqin_im[1],mqin_im[7]);            
            out1_im[2] = _mm_add_pd(mqin_im[2],mqin_im[8]);

            mqin_re[3]  = _mm_load_pd(sx_re+6);
            mqin_re[4]  = _mm_load_pd(sx_re+8);
            mqin_re[5]  = _mm_load_pd(sx_re+10);
            mqin_im[3]  = _mm_load_pd(sx_im+6);
            mqin_im[4]  = _mm_load_pd(sx_im+8);
            mqin_im[5]  = _mm_load_pd(sx_im+10);


            mqin_re[9]   = _mm_load_pd(sx_re+18);
            mqin_re[10]  = _mm_load_pd(sx_re+20);
            mqin_re[11]  = _mm_load_pd(sx_re+22);
            mqin_im[9]   = _mm_load_pd(sx_im+18);
            mqin_im[10]  = _mm_load_pd(sx_im+20);
            mqin_im[11]  = _mm_load_pd(sx_im+22);


            out2_re[0] = _mm_add_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm_add_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm_add_pd(mqin_re[5],mqin_re[8]);            
            out2_im[0] = _mm_add_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm_add_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm_add_pd(mqin_im[5],mqin_im[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm_load_pd(ux_re+jc*2+ic*6);
                  U_im[ic][jc] = _mm_load_pd(ux_im+jc*2+ic*6);
               }
            
            su3_multiply_splitlayout_128(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_128(map1_re,map1_im,U_re,U_im,out2_re,out2_im);

            mqout_re[0] = _mm_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm_add_pd(mqout_re[6] ,map0_re[0]);
            mqout_re[7] = _mm_add_pd(mqout_re[7] ,map0_re[1]);
            mqout_re[8] = _mm_add_pd(mqout_re[8] ,map0_re[2]);
            mqout_re[9] = _mm_add_pd(mqout_re[9] ,map1_re[0]);
            mqout_re[10]= _mm_add_pd(mqout_re[10],map1_re[1]);
            mqout_re[11]= _mm_add_pd(mqout_re[11],map1_re[2]);

            mqout_im[0] = _mm_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm_add_pd(mqout_im[6] ,map0_im[0]);
            mqout_im[7] = _mm_add_pd(mqout_im[7] ,map0_im[1]);
            mqout_im[8] = _mm_add_pd(mqout_im[8] ,map0_im[2]);
            mqout_im[9] = _mm_add_pd(mqout_im[9] ,map1_im[0]);
            mqout_im[10]= _mm_add_pd(mqout_im[10],map1_im[1]);
            mqout_im[11]= _mm_add_pd(mqout_im[11],map1_im[2]);



            //=======================
            //===== -0  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_sse_split[ipt][0]*72;
            ux_im = u_im + plqcd_g.idn_sse_split[ipt][0]*72;

            sx_re = qin_re + plqcd_g.idn_sse_split[ipt][0]*24;
            sx_im = qin_im + plqcd_g.idn_sse_split[ipt][0]*24;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm_load_pd(sx_re);
            mqin_re[1]  = _mm_load_pd(sx_re+2);
            mqin_re[2]  = _mm_load_pd(sx_re+4);
            mqin_im[0]  = _mm_load_pd(sx_im);
            mqin_im[1]  = _mm_load_pd(sx_im+2);
            mqin_im[2]  = _mm_load_pd(sx_im+4);


            mqin_re[9]   = _mm_load_pd(sx_re+18);
            mqin_re[10]  = _mm_load_pd(sx_re+20);
            mqin_re[11]  = _mm_load_pd(sx_re+22);
            mqin_im[9]   = _mm_load_pd(sx_im+18);
            mqin_im[10]  = _mm_load_pd(sx_im+20);
            mqin_im[11]  = _mm_load_pd(sx_im+22);


            out1_re[0] = _mm_add_pd(mqin_re[0],mqin_im[9]);            
            out1_re[1] = _mm_add_pd(mqin_re[1],mqin_im[10]);            
            out1_re[2] = _mm_add_pd(mqin_re[2],mqin_im[11]);            
            out1_im[0] = _mm_sub_pd(mqin_im[0],mqin_re[9]);            
            out1_im[1] = _mm_sub_pd(mqin_im[1],mqin_re[10]);            
            out1_im[2] = _mm_sub_pd(mqin_im[2],mqin_re[11]);


            mqin_re[3]  = _mm_load_pd(sx_re+6);
            mqin_re[4]  = _mm_load_pd(sx_re+8);
            mqin_re[5]  = _mm_load_pd(sx_re+10);
            mqin_im[3]  = _mm_load_pd(sx_im+6);
            mqin_im[4]  = _mm_load_pd(sx_im+8);
            mqin_im[5]  = _mm_load_pd(sx_im+10);


            mqin_re[6]  = _mm_load_pd(sx_re+12);
            mqin_re[7]  = _mm_load_pd(sx_re+14);
            mqin_re[8]  = _mm_load_pd(sx_re+16);
            mqin_im[6]  = _mm_load_pd(sx_im+12);
            mqin_im[7]  = _mm_load_pd(sx_im+14);
            mqin_im[8]  = _mm_load_pd(sx_im+16);


            out2_re[0] = _mm_add_pd(mqin_re[3],mqin_im[6]);            
            out2_re[1] = _mm_add_pd(mqin_re[4],mqin_im[7]);            
            out2_re[2] = _mm_add_pd(mqin_re[5],mqin_im[8]);            
            out2_im[0] = _mm_sub_pd(mqin_im[3],mqin_re[6]);            
            out2_im[1] = _mm_sub_pd(mqin_im[4],mqin_re[7]);            
            out2_im[2] = _mm_sub_pd(mqin_im[5],mqin_re[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm_load_pd(ux_re+jc*2+ic*6);
                  U_im[ic][jc] = _mm_load_pd(ux_im+jc*2+ic*6);
               }
            
            su3_inverse_multiply_splitlayout_128(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_128(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            //store the result, shuffle if on the boundary
            if(ix == lx-1)
            {
               for(int i=0; i<3; i++)
               {  
                  out3_re[i]= _mm_shuffle_pd ( map0_re[i], map0_re[i], 1);
                  out3_im[i]= _mm_shuffle_pd ( map0_im[i], map0_im[i], 1);
                  out4_re[i]= _mm_shuffle_pd ( map1_re[i], map1_re[i], 1);
                  out4_im[i]= _mm_shuffle_pd ( map1_im[i], map1_im[i], 1);
               }
              
               mqout_re[0] = _mm_add_pd(mqout_re[0] ,out3_re[0]);
               mqout_re[1] = _mm_add_pd(mqout_re[1] ,out3_re[1]);
               mqout_re[2] = _mm_add_pd(mqout_re[2] ,out3_re[2]);
               mqout_re[3] = _mm_add_pd(mqout_re[3] ,out4_re[0]);
               mqout_re[4] = _mm_add_pd(mqout_re[4] ,out4_re[1]);
               mqout_re[5] = _mm_add_pd(mqout_re[5] ,out4_re[2]);

               mqout_re[6] = _mm_sub_pd(mqout_re[6] ,out4_im[0]);
               mqout_re[7] = _mm_sub_pd(mqout_re[7] ,out4_im[1]);
               mqout_re[8] = _mm_sub_pd(mqout_re[8] ,out4_im[2]);
               mqout_re[9] = _mm_sub_pd(mqout_re[9] ,out3_im[0]);
               mqout_re[10]= _mm_sub_pd(mqout_re[10],out3_im[1]);
               mqout_re[11]= _mm_sub_pd(mqout_re[11],out3_im[2]);

               mqout_im[0] = _mm_add_pd(mqout_im[0] ,out3_im[0]);
               mqout_im[1] = _mm_add_pd(mqout_im[1] ,out3_im[1]);
               mqout_im[2] = _mm_add_pd(mqout_im[2] ,out3_im[2]);
               mqout_im[3] = _mm_add_pd(mqout_im[3] ,out4_im[0]);
               mqout_im[4] = _mm_add_pd(mqout_im[4] ,out4_im[1]);
               mqout_im[5] = _mm_add_pd(mqout_im[5] ,out4_im[2]);

               mqout_im[6] = _mm_add_pd(mqout_im[6] ,out4_re[0]);
               mqout_im[7] = _mm_add_pd(mqout_im[7] ,out4_re[1]);
               mqout_im[8] = _mm_add_pd(mqout_im[8] ,out4_re[2]);
               mqout_im[9] = _mm_add_pd(mqout_im[9] ,out3_re[0]);
               mqout_im[10]= _mm_add_pd(mqout_im[10],out3_re[1]);
               mqout_im[11]= _mm_add_pd(mqout_im[11],out3_re[2]);

            }
            else
            {
               mqout_re[0] = _mm_add_pd(mqout_re[0] ,map0_re[0]);
               mqout_re[1] = _mm_add_pd(mqout_re[1] ,map0_re[1]);
               mqout_re[2] = _mm_add_pd(mqout_re[2] ,map0_re[2]);
               mqout_re[3] = _mm_add_pd(mqout_re[3] ,map1_re[0]);
               mqout_re[4] = _mm_add_pd(mqout_re[4] ,map1_re[1]);
               mqout_re[5] = _mm_add_pd(mqout_re[5] ,map1_re[2]);

               mqout_re[6] = _mm_sub_pd(mqout_re[6] ,map1_im[0]);
               mqout_re[7] = _mm_sub_pd(mqout_re[7] ,map1_im[1]);
               mqout_re[8] = _mm_sub_pd(mqout_re[8] ,map1_im[2]);
               mqout_re[9] = _mm_sub_pd(mqout_re[9] ,map0_im[0]);
               mqout_re[10]= _mm_sub_pd(mqout_re[10],map0_im[1]);
               mqout_re[11]= _mm_sub_pd(mqout_re[11],map0_im[2]);

               mqout_im[0] = _mm_add_pd(mqout_im[0] ,map0_im[0]);
               mqout_im[1] = _mm_add_pd(mqout_im[1] ,map0_im[1]);
               mqout_im[2] = _mm_add_pd(mqout_im[2] ,map0_im[2]);
               mqout_im[3] = _mm_add_pd(mqout_im[3] ,map1_im[0]);
               mqout_im[4] = _mm_add_pd(mqout_im[4] ,map1_im[1]);
               mqout_im[5] = _mm_add_pd(mqout_im[5] ,map1_im[2]);

               mqout_im[6] = _mm_add_pd(mqout_im[6] ,map1_re[0]);
               mqout_im[7] = _mm_add_pd(mqout_im[7] ,map1_re[1]);
               mqout_im[8] = _mm_add_pd(mqout_im[8] ,map1_re[2]);
               mqout_im[9] = _mm_add_pd(mqout_im[9] ,map0_re[0]);
               mqout_im[10]= _mm_add_pd(mqout_im[10],map0_re[1]);
               mqout_im[11]= _mm_add_pd(mqout_im[11],map0_re[2]);
            } 

            //=======================
            //===== -1  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_sse_split[ipt][1]*72;
            ux_im = u_im + plqcd_g.idn_sse_split[ipt][1]*72;

            sx_re = qin_re + plqcd_g.idn_sse_split[ipt][1]*24;
            sx_im = qin_im + plqcd_g.idn_sse_split[ipt][1]*24;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm_load_pd(sx_re);
            mqin_re[1]  = _mm_load_pd(sx_re+2);
            mqin_re[2]  = _mm_load_pd(sx_re+4);
            mqin_im[0]  = _mm_load_pd(sx_im);
            mqin_im[1]  = _mm_load_pd(sx_im+2);
            mqin_im[2]  = _mm_load_pd(sx_im+4);


            mqin_re[9]   = _mm_load_pd(sx_re+18);
            mqin_re[10]  = _mm_load_pd(sx_re+20);
            mqin_re[11]  = _mm_load_pd(sx_re+22);
            mqin_im[9]   = _mm_load_pd(sx_im+18);
            mqin_im[10]  = _mm_load_pd(sx_im+20);
            mqin_im[11]  = _mm_load_pd(sx_im+22);


            out1_re[0] = _mm_sub_pd(mqin_re[0],mqin_re[9]);            
            out1_re[1] = _mm_sub_pd(mqin_re[1],mqin_re[10]);            
            out1_re[2] = _mm_sub_pd(mqin_re[2],mqin_re[11]);            
            out1_im[0] = _mm_sub_pd(mqin_im[0],mqin_im[9]);            
            out1_im[1] = _mm_sub_pd(mqin_im[1],mqin_im[10]);            
            out1_im[2] = _mm_sub_pd(mqin_im[2],mqin_im[11]);


            mqin_re[3]  = _mm_load_pd(sx_re+6);
            mqin_re[4]  = _mm_load_pd(sx_re+8);
            mqin_re[5]  = _mm_load_pd(sx_re+10);
            mqin_im[3]  = _mm_load_pd(sx_im+6);
            mqin_im[4]  = _mm_load_pd(sx_im+8);
            mqin_im[5]  = _mm_load_pd(sx_im+10);


            mqin_re[6]  = _mm_load_pd(sx_re+12);
            mqin_re[7]  = _mm_load_pd(sx_re+14);
            mqin_re[8]  = _mm_load_pd(sx_re+16);
            mqin_im[6]  = _mm_load_pd(sx_im+12);
            mqin_im[7]  = _mm_load_pd(sx_im+14);
            mqin_im[8]  = _mm_load_pd(sx_im+16);


            out2_re[0] = _mm_add_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm_add_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm_add_pd(mqin_re[5],mqin_re[8]);            
            out2_im[0] = _mm_add_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm_add_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm_add_pd(mqin_im[5],mqin_im[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm_load_pd(ux_re+jc*2+ic*6);
                  U_im[ic][jc] = _mm_load_pd(ux_im+jc*2+ic*6);
               }
            
            su3_inverse_multiply_splitlayout_128(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_128(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            mqout_re[0] = _mm_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm_add_pd(mqout_re[6] ,map1_re[0]);
            mqout_re[7] = _mm_add_pd(mqout_re[7] ,map1_re[1]);
            mqout_re[8] = _mm_add_pd(mqout_re[8] ,map1_re[2]);
            mqout_re[9] = _mm_sub_pd(mqout_re[9] ,map0_re[0]);
            mqout_re[10]= _mm_sub_pd(mqout_re[10],map0_re[1]);
            mqout_re[11]= _mm_sub_pd(mqout_re[11],map0_re[2]);

            mqout_im[0] = _mm_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm_add_pd(mqout_im[6] ,map1_im[0]);
            mqout_im[7] = _mm_add_pd(mqout_im[7] ,map1_im[1]);
            mqout_im[8] = _mm_add_pd(mqout_im[8] ,map1_im[2]);
            mqout_im[9] = _mm_sub_pd(mqout_im[9] ,map0_im[0]);
            mqout_im[10]= _mm_sub_pd(mqout_im[10],map0_im[1]);
            mqout_im[11]= _mm_sub_pd(mqout_im[11],map0_im[2]);
            
            //=======================
            //===== -2  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_sse_split[ipt][2]*72;
            ux_im = u_im + plqcd_g.idn_sse_split[ipt][2]*72;

            sx_re = qin_re + plqcd_g.idn_sse_split[ipt][2]*24;
            sx_im = qin_im + plqcd_g.idn_sse_split[ipt][2]*24;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm_load_pd(sx_re);
            mqin_re[1]  = _mm_load_pd(sx_re+2);
            mqin_re[2]  = _mm_load_pd(sx_re+4);
            mqin_im[0]  = _mm_load_pd(sx_im);
            mqin_im[1]  = _mm_load_pd(sx_im+2);
            mqin_im[2]  = _mm_load_pd(sx_im+4);


            mqin_re[6]   = _mm_load_pd(sx_re+12);
            mqin_re[7]   = _mm_load_pd(sx_re+14);
            mqin_re[8]   = _mm_load_pd(sx_re+16);
            mqin_im[6]   = _mm_load_pd(sx_im+12);
            mqin_im[7]   = _mm_load_pd(sx_im+14);
            mqin_im[8]   = _mm_load_pd(sx_im+16);


            out1_re[0] = _mm_add_pd(mqin_re[0],mqin_im[6]);            
            out1_re[1] = _mm_add_pd(mqin_re[1],mqin_im[7]);            
            out1_re[2] = _mm_add_pd(mqin_re[2],mqin_im[8]);            
            out1_im[0] = _mm_sub_pd(mqin_im[0],mqin_re[6]);            
            out1_im[1] = _mm_sub_pd(mqin_im[1],mqin_re[7]);            
            out1_im[2] = _mm_sub_pd(mqin_im[2],mqin_re[8]);


            mqin_re[3]  = _mm_load_pd(sx_re+6);
            mqin_re[4]  = _mm_load_pd(sx_re+8);
            mqin_re[5]  = _mm_load_pd(sx_re+10);
            mqin_im[3]  = _mm_load_pd(sx_im+6);
            mqin_im[4]  = _mm_load_pd(sx_im+8);
            mqin_im[5]  = _mm_load_pd(sx_im+10);


            mqin_re[9]   = _mm_load_pd(sx_re+18);
            mqin_re[10]  = _mm_load_pd(sx_re+20);
            mqin_re[11]  = _mm_load_pd(sx_re+22);
            mqin_im[9]   = _mm_load_pd(sx_im+18);
            mqin_im[10]  = _mm_load_pd(sx_im+20);
            mqin_im[11]  = _mm_load_pd(sx_im+22);


            out2_re[0] = _mm_sub_pd(mqin_re[3],mqin_im[9]);            
            out2_re[1] = _mm_sub_pd(mqin_re[4],mqin_im[10]);            
            out2_re[2] = _mm_sub_pd(mqin_re[5],mqin_im[11]);            
            out2_im[0] = _mm_add_pd(mqin_im[3],mqin_re[9]);            
            out2_im[1] = _mm_add_pd(mqin_im[4],mqin_re[10]);            
            out2_im[2] = _mm_add_pd(mqin_im[5],mqin_re[11]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm_load_pd(ux_re+jc*2+ic*6);
                  U_im[ic][jc] = _mm_load_pd(ux_im+jc*2+ic*6);
               }
            
            su3_inverse_multiply_splitlayout_128(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_128(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            mqout_re[0] = _mm_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm_sub_pd(mqout_re[6] ,map0_im[0]);
            mqout_re[7] = _mm_sub_pd(mqout_re[7] ,map0_im[1]);
            mqout_re[8] = _mm_sub_pd(mqout_re[8] ,map0_im[2]);
            mqout_re[9] = _mm_add_pd(mqout_re[9] ,map1_im[0]);
            mqout_re[10]= _mm_add_pd(mqout_re[10],map1_im[1]);
            mqout_re[11]= _mm_add_pd(mqout_re[11],map1_im[2]);

            mqout_im[0] = _mm_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm_add_pd(mqout_im[6] ,map0_re[0]);
            mqout_im[7] = _mm_add_pd(mqout_im[7] ,map0_re[1]);
            mqout_im[8] = _mm_add_pd(mqout_im[8] ,map0_re[2]);
            mqout_im[9] = _mm_sub_pd(mqout_im[9] ,map1_re[0]);
            mqout_im[10]= _mm_sub_pd(mqout_im[10],map1_re[1]);
            mqout_im[11]= _mm_sub_pd(mqout_im[11],map1_re[2]);


            //=======================
            //===== -3  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_sse_split[ipt][3]*72;
            ux_im = u_im + plqcd_g.idn_sse_split[ipt][3]*72;

            sx_re = qin_re + plqcd_g.idn_sse_split[ipt][3]*24;
            sx_im = qin_im + plqcd_g.idn_sse_split[ipt][3]*24;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm_load_pd(sx_re);
            mqin_re[1]  = _mm_load_pd(sx_re+2);
            mqin_re[2]  = _mm_load_pd(sx_re+4);
            mqin_im[0]  = _mm_load_pd(sx_im);
            mqin_im[1]  = _mm_load_pd(sx_im+2);
            mqin_im[2]  = _mm_load_pd(sx_im+4);


            mqin_re[6]   = _mm_load_pd(sx_re+12);
            mqin_re[7]   = _mm_load_pd(sx_re+14);
            mqin_re[8]   = _mm_load_pd(sx_re+16);
            mqin_im[6]   = _mm_load_pd(sx_im+12);
            mqin_im[7]   = _mm_load_pd(sx_im+14);
            mqin_im[8]   = _mm_load_pd(sx_im+16);


            out1_re[0] = _mm_sub_pd(mqin_re[0],mqin_re[6]);            
            out1_re[1] = _mm_sub_pd(mqin_re[1],mqin_re[7]);            
            out1_re[2] = _mm_sub_pd(mqin_re[2],mqin_re[8]);            
            out1_im[0] = _mm_sub_pd(mqin_im[0],mqin_im[6]);            
            out1_im[1] = _mm_sub_pd(mqin_im[1],mqin_im[7]);            
            out1_im[2] = _mm_sub_pd(mqin_im[2],mqin_im[8]);


            mqin_re[3]  = _mm_load_pd(sx_re+6);
            mqin_re[4]  = _mm_load_pd(sx_re+8);
            mqin_re[5]  = _mm_load_pd(sx_re+10);
            mqin_im[3]  = _mm_load_pd(sx_im+6);
            mqin_im[4]  = _mm_load_pd(sx_im+8);
            mqin_im[5]  = _mm_load_pd(sx_im+10);


            mqin_re[9]   = _mm_load_pd(sx_re+18);
            mqin_re[10]  = _mm_load_pd(sx_re+20);
            mqin_re[11]  = _mm_load_pd(sx_re+22);
            mqin_im[9]   = _mm_load_pd(sx_im+18);
            mqin_im[10]  = _mm_load_pd(sx_im+20);
            mqin_im[11]  = _mm_load_pd(sx_im+22);


            out2_re[0] = _mm_sub_pd(mqin_re[3],mqin_re[9]);            
            out2_re[1] = _mm_sub_pd(mqin_re[4],mqin_re[10]);            
            out2_re[2] = _mm_sub_pd(mqin_re[5],mqin_re[11]);            
            out2_im[0] = _mm_sub_pd(mqin_im[3],mqin_im[9]);            
            out2_im[1] = _mm_sub_pd(mqin_im[4],mqin_im[10]);            
            out2_im[2] = _mm_sub_pd(mqin_im[5],mqin_im[11]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm_load_pd(ux_re+jc*2+ic*6);
                  U_im[ic][jc] = _mm_load_pd(ux_im+jc*2+ic*6);
               }
            
            su3_inverse_multiply_splitlayout_128(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_128(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            mqout_re[0] = _mm_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm_sub_pd(mqout_re[6] ,map0_re[0]);
            mqout_re[7] = _mm_sub_pd(mqout_re[7] ,map0_re[1]);
            mqout_re[8] = _mm_sub_pd(mqout_re[8] ,map0_re[2]);
            mqout_re[9] = _mm_sub_pd(mqout_re[9] ,map1_re[0]);
            mqout_re[10]= _mm_sub_pd(mqout_re[10],map1_re[1]);
            mqout_re[11]= _mm_sub_pd(mqout_re[11],map1_re[2]);

            mqout_im[0] = _mm_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm_sub_pd(mqout_im[6] ,map0_im[0]);
            mqout_im[7] = _mm_sub_pd(mqout_im[7] ,map0_im[1]);
            mqout_im[8] = _mm_sub_pd(mqout_im[8] ,map0_im[2]);
            mqout_im[9] = _mm_sub_pd(mqout_im[9] ,map1_im[0]);
            mqout_im[10]= _mm_sub_pd(mqout_im[10],map1_im[1]);
            mqout_im[11]= _mm_sub_pd(mqout_im[11],map1_im[2]);



            //_mm512_stream_pd(gx_re       , mqout_re[0]);
            _mm_store_pd(gx_re       , mqout_re[0]);
            _mm_store_pd(gx_re+2     , mqout_re[1]);
            _mm_store_pd(gx_re+4    , mqout_re[2]);
            _mm_store_pd(gx_re+6    , mqout_re[3]);
            _mm_store_pd(gx_re+8    , mqout_re[4]);
            _mm_store_pd(gx_re+10    , mqout_re[5]);
            _mm_store_pd(gx_re+12    , mqout_re[6]);
            _mm_store_pd(gx_re+14    , mqout_re[7]);
            _mm_store_pd(gx_re+16    , mqout_re[8]);
            _mm_store_pd(gx_re+18    , mqout_re[9]);
            _mm_store_pd(gx_re+20    , mqout_re[10]);
            _mm_store_pd(gx_re+22    , mqout_re[11]);

            _mm_store_pd(gx_im       , mqout_im[0]);
            _mm_store_pd(gx_im+2     , mqout_im[1]);
            _mm_store_pd(gx_im+4    , mqout_im[2]);
            _mm_store_pd(gx_im+6    , mqout_im[3]);
            _mm_store_pd(gx_im+8    , mqout_im[4]);
            _mm_store_pd(gx_im+10    , mqout_im[5]);
            _mm_store_pd(gx_im+12    , mqout_im[6]);
            _mm_store_pd(gx_im+14    , mqout_im[7]);
            _mm_store_pd(gx_im+16    , mqout_im[8]);
            _mm_store_pd(gx_im+18    , mqout_im[9]);
            _mm_store_pd(gx_im+20    , mqout_im[10]);
            _mm_store_pd(gx_im+22    , mqout_im[11]);

      }

   }
#ifdef _OPENMP
}
#endif  //end of the openmp parallel reigon

   
   return stop_watch(ts);
}















#endif
//==============================================================================================





//==============================================================================================

#ifdef AVX
//========================================================================================
//=======================================  2   ===========================================
//=======================           AVX with intrinsics       ============================
//========================================================================================

//===================================EO===================================================
double plqcd_hopping_matrix_eo_intrin_256(spinor_256 *qin, spinor_256 *qout, su3_256 *u)
{

   int snd_err[8],rcv_err[8];
   MPI_Status rcv_mpi_stat[8],snd_mpi_stat[8];
   if(itags==0)
      get_tags(); //get the permanent tags for communications

   double ts; //timer

   ts=stop_watch(0.0);
   
   #ifdef _OPENMP
   #pragma omp parallel
   {
   #endif
      int i,j,k,mu,V,face[4];
      V = plqcd_g.VOLUME;
      for(i=0; i<4; i++)
        face[i] = plqcd_g.face[i];
   
      __m256d qins0[3],qins1[3],qins2[3],qins3[3], U[3][3];
      __m256d in1[3],out[3];
      __m256d map0[3],map1[3],mq0[3],mq1[3],mq2[3],mq3[3];

      su3_vector_256 v0_256  __attribute__ ((aligned (32))) ;
      su3_vector_256 v1_256  __attribute__ ((aligned (32))) ;

      //check if the dimensions allows for using compact data representation
      if((V%4) != 0 ){ //V/2 must be a factor of 2
          fprintf(stderr,"Volume must be a factor of 4\n");
          exit(1);
      }

    
      int iup0[2],iup1[2],iup2[2],iup3[2],idn0[2],idn1[2],idn2[2],idn3[2];    
  
      su3_256 *ub0;


      //------------------------------------------------
      // compute (1-gamma_mu)qin terms and store in phip
      //------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for 
      #endif  
      //for(i=V/2; i < V; i++)
      for(i=V/2; i < V; i +=2)
      {
         k=i/2;
         //intrin_prefetch_spinor_256(&qin[k]);
         idn0[0]=plqcd_g.idn[i][0];
         idn1[0]=plqcd_g.idn[i][1];
         idn2[0]=plqcd_g.idn[i][2];
         idn3[0]=plqcd_g.idn[i][3];
         idn0[1]=plqcd_g.idn[i+1][0];
         idn1[1]=plqcd_g.idn[i+1][1];
         idn2[1]=plqcd_g.idn[i+1][2];
         idn3[1]=plqcd_g.idn[i+1][3];


         intrin_vector_load_256(qins0,&qin[k].s0);
         intrin_vector_load_256(qins1,&qin[k].s1);
         intrin_vector_load_256(qins2,&qin[k].s2);
         intrin_vector_load_256(qins3,&qin[k].s3);

         //-- 0 direction ---
         _prefetch_halfspinor(&plqcd_g.phip[1][idn1[0]]);
         _prefetch_halfspinor(&plqcd_g.phip[1][idn1[1]]);
         //_vector_add(plqcd_g.phip[0][idn0].s0, qin[i].s0, qin[i].s2);
         //_vector_add(plqcd_g.phip[0][idn0].s1, qin[i].s1, qin[i].s3);
         intrin_vector_add_256(out,qins0,qins2);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[0][idn0[0]].s0, &plqcd_g.phip[0][idn0[1]].s0, &v1_256);
         intrin_vector_add_256(out,qins1,qins3);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[0][idn0[0]].s1, &plqcd_g.phip[0][idn0[1]].s1, &v1_256);

        
         //-- 1 direction --
         _prefetch_halfspinor(&plqcd_g.phip[2][idn2[0]]);
         _prefetch_halfspinor(&plqcd_g.phip[2][idn2[1]]);
         //_vector_i_add(plqcd_g.phip[1][idn1].s0, qin[i].s0, qin[i].s3);
         //_vector_i_add(plqcd_g.phip[1][idn1].s1, qin[i].s1, qin[i].s2);
         intrin_vector_i_add_256(out,qins0,qins3);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[1][idn1[0]].s0, &plqcd_g.phip[1][idn1[1]].s0, &v1_256);
         intrin_vector_i_add_256(out,qins1,qins2);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[1][idn1[0]].s1, &plqcd_g.phip[1][idn1[1]].s1, &v1_256);

         //-- 2 direction --
         _prefetch_halfspinor(&plqcd_g.phip[3][idn3[0]]);
         _prefetch_halfspinor(&plqcd_g.phip[3][idn3[1]]);
         //_vector_add(plqcd_g.phip[2][idn2].s0, qin[i].s0, qin[i].s3);
         //_vector_sub(plqcd_g.phip[2][idn2].s1, qin[i].s1, qin[i].s2);
         intrin_vector_add_256(out,qins0,qins3);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[2][idn2[0]].s0, &plqcd_g.phip[2][idn2[1]].s0, &v1_256);
         intrin_vector_sub_256(out,qins1,qins2);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[2][idn2[0]].s1, &plqcd_g.phip[2][idn2[1]].s1, &v1_256);

         //-- 3 direction --
         //_vector_i_add(plqcd_g.phip[3][idn3].s0, qin[i].s0, qin[i].s2);
         //_vector_i_sub(plqcd_g.phip[3][idn3].s1, qin[i].s1, qin[i].s3);
         intrin_vector_i_add_256(out,qins0,qins2);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[3][idn3[0]].s0, &plqcd_g.phip[3][idn3[1]].s0, &v1_256);
         intrin_vector_i_sub_256(out,qins1,qins3);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[3][idn3[0]].s1, &plqcd_g.phip[3][idn3[1]].s1, &v1_256);
      }


      //start sending the buffers to the nearest neighbours in the -ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
            snd_err[mu] = MPI_Isend(&plqcd_g.phip[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu], MPI_COMM_WORLD, &snd_req[mu]);
            rcv_err[mu] = MPI_Irecv(&plqcd_g.phip[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_req[mu]);
            if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif


      //--------------------------------
      // U^dagger*(1+gamma_mu)qin terms
      //--------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=V/2; i < V; i +=2)
      {
         k=i/2;
         intrin_prefetch_spinor_256(&qin[k]);
         iup0[0]=plqcd_g.iup[i][0];
         iup1[0]=plqcd_g.iup[i][1];
         iup2[0]=plqcd_g.iup[i][2];
         iup3[0]=plqcd_g.iup[i][3];
         iup0[1]=plqcd_g.iup[i+1][0];
         iup1[1]=plqcd_g.iup[i+1][1];
         iup2[1]=plqcd_g.iup[i+1][2];
         iup3[1]=plqcd_g.iup[i+1][3];

         _prefetch_halfspinor(&plqcd_g.phim[0][iup0[0]]);
         _prefetch_halfspinor(&plqcd_g.phim[0][iup0[1]]);

         ub0 = &u[4*k];
      
         intrin_vector_load_256(qins0,&qin[k].s0);
         intrin_vector_load_256(qins1,&qin[k].s1);
         intrin_vector_load_256(qins2,&qin[k].s2);
         intrin_vector_load_256(qins3,&qin[k].s3);

         //-- 0 direction ---
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[1][iup1[0]]);
         _prefetch_halfspinor(&plqcd_g.phim[1][iup1[1]]);
         //_vector_sub(p,qin[i].s0,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s0,*ub0,p);
         //_vector_sub(p,qin[i].s1,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s1,*ub0,p);
         intrin_su3_load_256(U,ub0);
         intrin_vector_sub_256(in1,qins0,qins2);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[0][iup0[0]].s0, &plqcd_g.phim[0][iup0[1]].s0, &v1_256);
         intrin_vector_sub_256(in1,qins1,qins3);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[0][iup0[0]].s1, &plqcd_g.phim[0][iup0[1]].s1, &v1_256);
         ub0++;
  
         //-- 1 direction --
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[2][iup2[0]]);
         _prefetch_halfspinor(&plqcd_g.phim[2][iup2[1]]);
         //_vector_i_sub(p,qin[i].s0,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s0,*ub0,p);
         //_vector_i_sub(p,qin[i].s1,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s1,*ub0,p);
         intrin_su3_load_256(U,ub0);
         intrin_vector_i_sub_256(in1,qins0,qins3);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[1][iup1[0]].s0, &plqcd_g.phim[1][iup1[1]].s0, &v1_256);
         intrin_vector_i_sub_256(in1,qins1,qins2);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[1][iup1[0]].s1, &plqcd_g.phim[1][iup1[1]].s1, &v1_256);
         ub0++;

         //-- 2 direction --
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[3][iup3[0]]);
         _prefetch_halfspinor(&plqcd_g.phim[3][iup3[1]]);
         //_vector_sub(p,qin[i].s0,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s0,*ub0,p);
         //_vector_add(p,qin[i].s1,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s1,*ub0,p);
         intrin_su3_load_256(U,ub0);
         intrin_vector_sub_256(in1,qins0,qins3);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup2[0]].s0, &plqcd_g.phim[2][iup2[1]].s0, &v1_256);
         intrin_vector_add_256(in1,qins1,qins2);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup2[0]].s1, &plqcd_g.phim[2][iup2[1]].s1, &v1_256);
         ub0++;

         //-- 3 direction --
         //_vector_i_sub(p,qin[i].s0,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s0,*ub0,p);
         //_vector_i_add(p,qin[i].s1,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s1,*ub0,p);
         intrin_su3_load_256(U,ub0);
         intrin_vector_i_sub_256(in1,qins0,qins2);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup3[0]].s0, &plqcd_g.phim[2][iup3[1]].s0, &v1_256);
         intrin_vector_i_add_256(in1,qins1,qins3);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[3][iup3[0]].s1, &plqcd_g.phim[3][iup3[1]].s1, &v1_256);
      }

      //start sending the buffers to the nearest neighbours in the +ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
            snd_err[mu+4] = MPI_Isend(&plqcd_g.phim[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD, &snd_req[mu+4]);
            rcv_err[mu+4] = MPI_Irecv(&plqcd_g.phim[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_req[mu+4]);
            if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
               exit(1);}
         }
      }
      
      //wait for the communications of phip to finish
      for(mu=0; mu<4; mu++)
      {
         if(plqcd_g.nprocs[mu]>1)
         {   
            rcv_err[mu]=MPI_Wait(&rcv_req[mu],&rcv_mpi_stat[mu]);
            snd_err[mu]=MPI_Wait(&snd_req[mu],&snd_mpi_stat[mu]);
            if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Wait\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif
     

      //-------------------------------------------------------
      //complete computation of the U_mu*(1-gamma_m)*qin terms
      //-------------------------------------------------------
      //copy the exchanged boundaries to the correspoding locations on the local phip fields
      for(mu=0; mu<4; mu++)
      { 
         if(plqcd_g.nprocs[mu] > 1)
         {  
            #ifdef _OPENMP 
            #pragma omp for 
            #endif  
            for(i=0; i< face[mu]/2; i++)
            {
                //can we prefetch here
                j=V/2+face[mu]/2+i;
                k=plqcd_g.nn_bndo[2*mu][i];
                _vector_assign(plqcd_g.phip[mu][k].s0 , plqcd_g.phip[mu][j].s0);
                _vector_assign(plqcd_g.phip[mu][k].s1 , plqcd_g.phip[mu][j].s1);
            }
         }
      } 



      #ifdef _OPENMP
      #pragma omp for 
      #endif 
      for(i=0; i< V/2; i +=2)
      {

         k=i/2;
         ub0 = &u[4*k];
   
         // +0
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[1][i]);
         _prefetch_halfspinor(&plqcd_g.phip[1][i+1]);
         //_su3_multiply(q0, *ub0, plqcd_g.phip[0][i].s0);
         //_su3_multiply(q1, *ub0, plqcd_g.phip[0][i].s1);
         //_vector_assign(q2, q0);
         //_vector_assign(q3, q1);
         intrin_su3_load_256(U,ub0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[0][i].s0, &plqcd_g.phip[0][i+1].s0);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(mq0,U,in1);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[0][i].s1, &plqcd_g.phip[0][i+1].s1);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(mq1,U,in1);
         for(j=0; j<3; j++){
            mq2[j]= mq0[j];
            mq3[j]= mq1[j];}
         ub0++;

         // +1
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[2][i]);
         _prefetch_halfspinor(&plqcd_g.phip[2][i+1]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[1][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[1][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2, -1.0, ap1);
         //_vector_add_i_mul(q3, -1.0, ap0);
         intrin_su3_load_256(U,ub0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[1][i].s0, &plqcd_g.phip[1][i+1].s0);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map0,U,in1);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[1][i].s1, &plqcd_g.phip[1][i+1].s1);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm256_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm256_add_pd(mq1[j],map1[j]);
            mq2[j] = complex_i_sub_regs_256(mq2[j],map1[j]);
            mq3[j] = complex_i_sub_regs_256(mq3[j],map0[j]);}
         ub0++;        

         // +2
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[3][i]);
         _prefetch_halfspinor(&plqcd_g.phip[3][i+1]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[2][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[2][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_sub_assign(q2,ap1);
         //_vector_add_assign(q3,ap0);
         intrin_su3_load_256(U,ub0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[2][i].s0, &plqcd_g.phip[2][i+1].s0);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map0,U,in1);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[2][i].s1, &plqcd_g.phip[2][i+1].s1);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm256_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm256_add_pd(mq1[j],map1[j]);
            mq2[j] = _mm256_sub_pd(mq2[j],map1[j]);
            mq3[j] = _mm256_add_pd(mq3[j],map0[j]);}
         ub0++;        


         intrin_prefetch_spinor_256(&qout[i]);
         // +3
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[3][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[3][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2,-1.0, ap0);
         //_vector_add_i_mul(q3, 1.0, ap1);
         intrin_su3_load_256(U,ub0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[3][i].s0, &plqcd_g.phip[3][i+1].s0);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map0,U,in1);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[3][i].s1, &plqcd_g.phip[3][i+1].s1);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm256_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm256_add_pd(mq1[j],map1[j]);
            mq2[j] = complex_i_sub_regs_256(mq2[j],map0[j]);
            mq3[j] = complex_i_add_regs_256(mq3[j],map1[j]);}

 
         //store the result
         //_vector_assign(qout[i].s0,q0);
         //_vector_assign(qout[i].s1,q1);
         //_vector_assign(qout[i].s2,q2);
         //_vector_assign(qout[i].s3,q3);
         intrin_vector_store_256(&qout[k].s0,mq0);         
         intrin_vector_store_256(&qout[k].s1,mq1);         
         intrin_vector_store_256(&qout[k].s2,mq1);         
         intrin_vector_store_256(&qout[k].s3,mq1);         

      } 

      //wait for the communications of phim to finish
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu<4; mu++)
      {
         if(plqcd_g.nprocs[mu]>1)
         {   
            rcv_err[mu+4]=MPI_Wait(&rcv_req[mu+4],&rcv_mpi_stat[mu+4]);
            snd_err[mu+4]=MPI_Wait(&snd_req[mu+4],&snd_mpi_stat[mu+4]);
            if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Wait\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif

      //copy the exchanged boundaries to the correspoding locations on the local phim fields
      for(mu=0; mu<4; mu++)
      { 
         if(plqcd_g.nprocs[mu] > 1)
         {
            #ifdef _OPENMP   
            #pragma omp for
            #endif
            for(i=0; i< face[mu]/2; i++)
            {
               j=V/2+face[mu]/2+i;
               k=plqcd_g.nn_bndo[2*mu+1][i];
               _vector_assign(plqcd_g.phim[mu][k].s0, plqcd_g.phim[mu][j].s0);
               _vector_assign(plqcd_g.phim[mu][k].s1, plqcd_g.phim[mu][j].s1);
            }
         }
      }


      //---------------------------------------------------------------------
      //finish computation of the U^dagger*(1+gamma_mu)*qin
      //---------------------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif
      for(i=0; i< V/2; i +=2)
      {

         k=i/2;
         intrin_vector_load_256(mq0,&qout[k].s0);
         intrin_vector_load_256(mq1,&qout[k].s1);
         intrin_vector_load_256(mq2,&qout[k].s2);
         intrin_vector_load_256(mq3,&qout[k].s3);

         // 0 direction
         _prefetch_halfspinor(&plqcd_g.phim[1][i]);
         _prefetch_halfspinor(&plqcd_g.phim[1][i+1]);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[0][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[0][i].s1);
         //_vector_minus_assign(qout[i].s2, plqcd_g.phim[0][i].s0);
         //_vector_minus_assign(qout[i].s3, plqcd_g.phim[0][i].s1);
         copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[0][i].s0, &plqcd_g.phim[0][i+1].s0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[0][i].s1, &plqcd_g.phim[0][i+1].s1);
         intrin_vector_load_256(map0,&v0_256);
         intrin_vector_load_256(map1,&v1_256);
         intrin_vector_add_256(mq0,mq0,map0);
         intrin_vector_add_256(mq1,mq1,map1);
         intrin_vector_sub_256(mq2,mq2,map0);
         intrin_vector_sub_256(mq3,mq3,map1);

         // 1 direction
         _prefetch_halfspinor(&plqcd_g.phim[2][i]);
         _prefetch_halfspinor(&plqcd_g.phim[2][i+1]);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[1][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s2, 1.0,plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s3, 1.0, plqcd_g.phim[1][i].s0);
         copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[1][i].s0, &plqcd_g.phim[1][i+1].s0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[1][i].s1, &plqcd_g.phim[1][i+1].s1);
         intrin_vector_load_256(map0,&v0_256);
         intrin_vector_load_256(map1,&v1_256);
         intrin_vector_add_256(mq0,mq0,map0);
         intrin_vector_add_256(mq1,mq1,map1);
         intrin_vector_i_add_256(mq2,mq2,map1);
         intrin_vector_i_add_256(mq3,mq3,map0);
        

         // 2 direction
         _prefetch_halfspinor(&plqcd_g.phim[3][i]);
         _prefetch_halfspinor(&plqcd_g.phim[3][i+1]);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[2][i].s0);
         //_vector_add_assign(qout[i].s1,  plqcd_g.phim[2][i].s1 );
         //_vector_add_assign(qout[i].s2,  plqcd_g.phim[2][i].s1);
         //_vector_sub_assign(qout[i].s3,   plqcd_g.phim[2][i].s0);
         copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[2][i].s0, &plqcd_g.phim[2][i+1].s0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[2][i].s1, &plqcd_g.phim[2][i+1].s1);
         intrin_vector_load_256(map0,&v0_256);
         intrin_vector_load_256(map1,&v1_256);
         intrin_vector_add_256(mq0,mq0,map0);
         intrin_vector_add_256(mq1,mq1,map1);
         intrin_vector_add_256(mq2,mq2,map1);
         intrin_vector_sub_256(mq3,mq3,map0);
         

         //3 direction
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[3][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[3][i].s1);
         //_vector_i_add_assign(qout[i].s2, plqcd_g.phim[3][i].s0);
         //_vector_i_sub_assign(qout[i].s3, plqcd_g.phim[3][i].s1);     
         copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[3][i].s0, &plqcd_g.phim[3][i+1].s0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[3][i].s1, &plqcd_g.phim[3][i+1].s1);
         intrin_vector_load_256(map0,&v0_256);
         intrin_vector_load_256(map1,&v1_256);
         intrin_vector_add_256(mq0,mq0,map0);
         intrin_vector_add_256(mq1,mq1,map1);
         intrin_vector_i_add_256(mq2,mq2,map0);
         intrin_vector_i_add_256(mq3,mq3,map1);

         //store the result
         intrin_vector_store_256(&qout[k].s0, mq0);
         intrin_vector_store_256(&qout[k].s1, mq1);
         intrin_vector_store_256(&qout[k].s2, mq2);
         intrin_vector_store_256(&qout[k].s3, mq3);
   }
#ifdef _OPENMP
}
#endif  //end of the openmp parallel reigon

   
   return stop_watch(ts);
}


//===================================OE===================================================
double plqcd_hopping_matrix_oe_intrin_256(spinor_256 *qin, spinor_256 *qout, su3_256 *u)
{

   int snd_err[8],rcv_err[8];
   MPI_Status rcv_mpi_stat[8],snd_mpi_stat[8];
   if(itags==0)
      get_tags(); //get the permanent tags for communications

   double ts; //timer

   ts=stop_watch(0.0);
   
   #ifdef _OPENMP
   #pragma omp parallel
   {
   #endif
      int i,j,k,mu,V,face[4];
      V = plqcd_g.VOLUME;
      for(i=0; i<4; i++)
        face[i] = plqcd_g.face[i];
   
      __m256d qins0[3],qins1[3],qins2[3],qins3[3], U[3][3];
      __m256d in1[3],out[3];
      __m256d map0[3],map1[3],mq0[3],mq1[3],mq2[3],mq3[3];

      su3_vector_256 v0_256  __attribute__ ((aligned (64))) ;
      su3_vector_256 v1_256  __attribute__ ((aligned (64))) ;

      //check if the dimensions allows for using compact data representation
      if((V%4) != 0 ){ //V/2 must be a factor of 2
          fprintf(stderr,"Volume must be a factor of 4\n");
          exit(1);
      }

    
      int iup0[2],iup1[2],iup2[2],iup3[2],idn0[2],idn1[2],idn2[2],idn3[2];    
  
      su3_256 *ub0;


      //------------------------------------------------
      // compute (1-gamma_mu)qin terms and store in phip
      //------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for 
      #endif  
      //for(i=V/2; i < V; i++)
      for(i=0; i < V/2; i +=2)
      {
         k=i/2;
         intrin_prefetch_spinor_256(&qin[k]);
         idn0[0]=plqcd_g.idn[i][0]-V/2;
         idn1[0]=plqcd_g.idn[i][1]-V/2;
         idn2[0]=plqcd_g.idn[i][2]-V/2;
         idn3[0]=plqcd_g.idn[i][3]-V/2;
         idn0[1]=plqcd_g.idn[i+1][0]-V/2;
         idn1[1]=plqcd_g.idn[i+1][1]-V/2;
         idn2[1]=plqcd_g.idn[i+1][2]-V/2;
         idn3[1]=plqcd_g.idn[i+1][3]-V/2;


         intrin_vector_load_256(qins0,&qin[k].s0);
         intrin_vector_load_256(qins1,&qin[k].s1);
         intrin_vector_load_256(qins2,&qin[k].s2);
         intrin_vector_load_256(qins3,&qin[k].s3);

         //-- 0 direction ---
         _prefetch_halfspinor(&plqcd_g.phip[1][idn1[0]]);
         _prefetch_halfspinor(&plqcd_g.phip[1][idn1[1]]);
         //_vector_add(plqcd_g.phip[0][idn0].s0, qin[i].s0, qin[i].s2);
         //_vector_add(plqcd_g.phip[0][idn0].s1, qin[i].s1, qin[i].s3);
         intrin_vector_add_256(out,qins0,qins2);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[0][idn0[0]].s0, &plqcd_g.phip[0][idn0[1]].s0, &v1_256);
         intrin_vector_add_256(out,qins1,qins3);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[0][idn0[0]].s1, &plqcd_g.phip[0][idn0[1]].s1, &v1_256);

        
         //-- 1 direction --
         _prefetch_halfspinor(&plqcd_g.phip[2][idn2[0]]);
         _prefetch_halfspinor(&plqcd_g.phip[2][idn2[1]]);
         //_vector_i_add(plqcd_g.phip[1][idn1].s0, qin[i].s0, qin[i].s3);
         //_vector_i_add(plqcd_g.phip[1][idn1].s1, qin[i].s1, qin[i].s2);
         intrin_vector_i_add_256(out,qins0,qins3);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[1][idn1[0]].s0, &plqcd_g.phip[1][idn1[1]].s0, &v1_256);
         intrin_vector_i_add_256(out,qins1,qins2);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[1][idn1[0]].s1, &plqcd_g.phip[1][idn1[1]].s1, &v1_256);

         //-- 2 direction --
         _prefetch_halfspinor(&plqcd_g.phip[3][idn3[0]]);
         _prefetch_halfspinor(&plqcd_g.phip[3][idn3[1]]);
         //_vector_add(plqcd_g.phip[2][idn2].s0, qin[i].s0, qin[i].s3);
         //_vector_sub(plqcd_g.phip[2][idn2].s1, qin[i].s1, qin[i].s2);
         intrin_vector_add_256(out,qins0,qins3);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[2][idn2[0]].s0, &plqcd_g.phip[2][idn2[1]].s0, &v1_256);
         intrin_vector_sub_256(out,qins1,qins2);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[2][idn2[0]].s1, &plqcd_g.phip[2][idn2[1]].s1, &v1_256);

         //-- 3 direction --
         //_vector_i_add(plqcd_g.phip[3][idn3].s0, qin[i].s0, qin[i].s2);
         //_vector_i_sub(plqcd_g.phip[3][idn3].s1, qin[i].s1, qin[i].s3);
         intrin_vector_i_add_256(out,qins0,qins2);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[3][idn3[0]].s0, &plqcd_g.phip[3][idn3[1]].s0, &v1_256);
         intrin_vector_i_sub_256(out,qins1,qins3);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[3][idn3[0]].s1, &plqcd_g.phip[3][idn3[1]].s1, &v1_256);
      }


      //start sending the buffers to the nearest neighbours in the -ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
            snd_err[mu] = MPI_Isend(&plqcd_g.phip[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu], MPI_COMM_WORLD, &snd_req[mu]);
            rcv_err[mu] = MPI_Irecv(&plqcd_g.phip[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_req[mu]);
            if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif


      //--------------------------------
      // U^dagger*(1+gamma_mu)qin terms
      //--------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=0; i < V/2; i +=2)
      {
         k=i/2;
         intrin_prefetch_spinor_256(&qin[k]);
         iup0[0]=plqcd_g.iup[i][0]-V/2;
         iup1[0]=plqcd_g.iup[i][1]-V/2;
         iup2[0]=plqcd_g.iup[i][2]-V/2;
         iup3[0]=plqcd_g.iup[i][3]-V/2;
         iup0[1]=plqcd_g.iup[i+1][0]-V/2;
         iup1[1]=plqcd_g.iup[i+1][1]-V/2;
         iup2[1]=plqcd_g.iup[i+1][2]-V/2;
         iup3[1]=plqcd_g.iup[i+1][3]-V/2;

         _prefetch_halfspinor(&plqcd_g.phim[0][iup0[0]]);
         _prefetch_halfspinor(&plqcd_g.phim[0][iup0[1]]);

         ub0 = &u[4*k];
      
         intrin_vector_load_256(qins0,&qin[k].s0);
         intrin_vector_load_256(qins1,&qin[k].s1);
         intrin_vector_load_256(qins2,&qin[k].s2);
         intrin_vector_load_256(qins3,&qin[k].s3);

         //-- 0 direction ---
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[1][iup1[0]]);
         _prefetch_halfspinor(&plqcd_g.phim[1][iup1[1]]);
         //_vector_sub(p,qin[i].s0,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s0,*ub0,p);
         //_vector_sub(p,qin[i].s1,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s1,*ub0,p);
         intrin_su3_load_256(U,ub0);
         intrin_vector_sub_256(in1,qins0,qins2);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[0][iup0[0]].s0, &plqcd_g.phim[0][iup0[1]].s0, &v1_256);
         intrin_vector_sub_256(in1,qins1,qins3);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[0][iup0[0]].s1, &plqcd_g.phim[0][iup0[1]].s1, &v1_256);
         ub0++;
  
         //-- 1 direction --
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[2][iup2[0]]);
         _prefetch_halfspinor(&plqcd_g.phim[2][iup2[1]]);
         //_vector_i_sub(p,qin[i].s0,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s0,*ub0,p);
         //_vector_i_sub(p,qin[i].s1,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s1,*ub0,p);
         intrin_su3_load_256(U,ub0);
         intrin_vector_i_sub_256(in1,qins0,qins3);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[1][iup1[0]].s0, &plqcd_g.phim[1][iup1[1]].s0, &v1_256);
         intrin_vector_i_sub_256(in1,qins1,qins2);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[1][iup1[0]].s1, &plqcd_g.phim[1][iup1[1]].s1, &v1_256);
         ub0++;

         //-- 2 direction --
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phim[3][iup3[0]]);
         _prefetch_halfspinor(&plqcd_g.phim[3][iup3[1]]);
         //_vector_sub(p,qin[i].s0,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s0,*ub0,p);
         //_vector_add(p,qin[i].s1,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s1,*ub0,p);
         intrin_su3_load_256(U,ub0);
         intrin_vector_sub_256(in1,qins0,qins3);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup2[0]].s0, &plqcd_g.phim[2][iup2[1]].s0, &v1_256);
         intrin_vector_add_256(in1,qins1,qins2);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup2[0]].s1, &plqcd_g.phim[2][iup2[1]].s1, &v1_256);
         ub0++;

         //-- 3 direction --
         //_vector_i_sub(p,qin[i].s0,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s0,*ub0,p);
         //_vector_i_add(p,qin[i].s1,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s1,*ub0,p);
         intrin_su3_load_256(U,ub0);
         intrin_vector_i_sub_256(in1,qins0,qins2);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup3[0]].s0, &plqcd_g.phim[2][iup3[1]].s0, &v1_256);
         intrin_vector_i_add_256(in1,qins1,qins3);
         intrin_su3_inverse_multiply_256(out,U,in1);
         intrin_vector_store_256(&v1_256,out);
         copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[3][iup3[0]].s1, &plqcd_g.phim[3][iup3[1]].s1, &v1_256);
      }

      //start sending the buffers to the nearest neighbours in the +ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
            snd_err[mu+4] = MPI_Isend(&plqcd_g.phim[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD, &snd_req[mu+4]);
            rcv_err[mu+4] = MPI_Irecv(&plqcd_g.phim[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_req[mu+4]);
            if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
               exit(1);}
         }
      }
      
      //wait for the communications of phip to finish
      for(mu=0; mu<4; mu++)
      {
         if(plqcd_g.nprocs[mu]>1)
         {   
            rcv_err[mu]=MPI_Wait(&rcv_req[mu],&rcv_mpi_stat[mu]);
            snd_err[mu]=MPI_Wait(&snd_req[mu],&snd_mpi_stat[mu]);
            if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Wait\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif
     

      //-------------------------------------------------------
      //complete computation of the U_mu*(1-gamma_m)*qin terms
      //-------------------------------------------------------
      //copy the exchanged boundaries to the correspoding locations on the local phip fields
      for(mu=0; mu<4; mu++)
      { 
         if(plqcd_g.nprocs[mu] > 1)
         {  
            #ifdef _OPENMP 
            #pragma omp for 
            #endif  
            for(i=0; i< face[mu]/2; i++)
            {
                //can we prefetch here
                j=V/2+face[mu]/2+i;
                k=plqcd_g.nn_bnde[2*mu][i]-V/2;
                _vector_assign(plqcd_g.phip[mu][k].s0 , plqcd_g.phip[mu][j].s0);
                _vector_assign(plqcd_g.phip[mu][k].s1 , plqcd_g.phip[mu][j].s1);
            }
         }
      } 



      #ifdef _OPENMP
      #pragma omp for 
      #endif 
      for(i=0; i< V/2; i +=2)
      {

         k=i/2+V/4;
         ub0 = &u[4*k];
   
         // +0
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[1][i]);
         _prefetch_halfspinor(&plqcd_g.phip[1][i+1]);
         //_su3_multiply(q0, *ub0, plqcd_g.phip[0][i].s0);
         //_su3_multiply(q1, *ub0, plqcd_g.phip[0][i].s1);
         //_vector_assign(q2, q0);
         //_vector_assign(q3, q1);
         intrin_su3_load_256(U,ub0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[0][i].s0, &plqcd_g.phip[0][i+1].s0);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(mq0,U,in1);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[0][i].s1, &plqcd_g.phip[0][i+1].s1);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(mq1,U,in1);
         for(j=0; j<3; j++){
            mq2[j]= mq0[j];
            mq3[j]= mq1[j];}
         ub0++;

         // +1
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[2][i]);
         _prefetch_halfspinor(&plqcd_g.phip[2][i+1]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[1][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[1][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2, -1.0, ap1);
         //_vector_add_i_mul(q3, -1.0, ap0);
         intrin_su3_load_256(U,ub0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[1][i].s0, &plqcd_g.phip[1][i+1].s0);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map0,U,in1);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[1][i].s1, &plqcd_g.phip[1][i+1].s1);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm256_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm256_add_pd(mq1[j],map1[j]);
            mq2[j] = complex_i_sub_regs_256(mq2[j],map1[j]);
            mq3[j] = complex_i_sub_regs_256(mq3[j],map0[j]);}
         ub0++;        

         // +2
         intrin_prefetch_su3_256(ub0+1);
         _prefetch_halfspinor(&plqcd_g.phip[3][i]);
         _prefetch_halfspinor(&plqcd_g.phip[3][i+1]);
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[2][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[2][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_sub_assign(q2,ap1);
         //_vector_add_assign(q3,ap0);
         intrin_su3_load_256(U,ub0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[2][i].s0, &plqcd_g.phip[2][i+1].s0);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map0,U,in1);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[2][i].s1, &plqcd_g.phip[2][i+1].s1);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm256_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm256_add_pd(mq1[j],map1[j]);
            mq2[j] = _mm256_sub_pd(mq2[j],map1[j]);
            mq3[j] = _mm256_add_pd(mq3[j],map0[j]);}
         ub0++;        


         intrin_prefetch_spinor_256(&qout[i]);
         // +3
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[3][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[3][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2,-1.0, ap0);
         //_vector_add_i_mul(q3, 1.0, ap1);
         intrin_su3_load_256(U,ub0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[3][i].s0, &plqcd_g.phip[3][i+1].s0);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map0,U,in1);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[3][i].s1, &plqcd_g.phip[3][i+1].s1);
         intrin_vector_load_256(in1,&v1_256);
         intrin_su3_multiply_256(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm256_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm256_add_pd(mq1[j],map1[j]);
            mq2[j] = complex_i_sub_regs_256(mq2[j],map0[j]);
            mq3[j] = complex_i_add_regs_256(mq3[j],map1[j]);}

 
         //store the result
         //_vector_assign(qout[i].s0,q0);
         //_vector_assign(qout[i].s1,q1);
         //_vector_assign(qout[i].s2,q2);
         //_vector_assign(qout[i].s3,q3);
         intrin_vector_store_256(&qout[k].s0,mq0);         
         intrin_vector_store_256(&qout[k].s1,mq1);         
         intrin_vector_store_256(&qout[k].s2,mq1);         
         intrin_vector_store_256(&qout[k].s3,mq1);         

      } 

      //wait for the communications of phim to finish
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu<4; mu++)
      {
         if(plqcd_g.nprocs[mu]>1)
         {   
            rcv_err[mu+4]=MPI_Wait(&rcv_req[mu+4],&rcv_mpi_stat[mu+4]);
            snd_err[mu+4]=MPI_Wait(&snd_req[mu+4],&snd_mpi_stat[mu+4]);
            if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Wait\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif

      //copy the exchanged boundaries to the correspoding locations on the local phim fields
      for(mu=0; mu<4; mu++)
      { 
         if(plqcd_g.nprocs[mu] > 1)
         {
            #ifdef _OPENMP   
            #pragma omp for
            #endif
            for(i=0; i< face[mu]/2; i++)
            {
               j=V/2+face[mu]/2+i;
               k=plqcd_g.nn_bnde[2*mu+1][i]-V/2;
               _vector_assign(plqcd_g.phim[mu][k].s0, plqcd_g.phim[mu][j].s0);
               _vector_assign(plqcd_g.phim[mu][k].s1, plqcd_g.phim[mu][j].s1);
            }
         }
      }


      //---------------------------------------------------------------------
      //finish computation of the U^dagger*(1+gamma_mu)*qin
      //---------------------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif
      for(i=0; i< V/2; i +=2)
      {

         k=i/2+V/4;
         intrin_vector_load_256(mq0,&qout[k].s0);
         intrin_vector_load_256(mq1,&qout[k].s1);
         intrin_vector_load_256(mq2,&qout[k].s2);
         intrin_vector_load_256(mq3,&qout[k].s3);

         // 0 direction
         _prefetch_halfspinor(&plqcd_g.phim[1][i]);
         _prefetch_halfspinor(&plqcd_g.phim[1][i+1]);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[0][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[0][i].s1);
         //_vector_minus_assign(qout[i].s2, plqcd_g.phim[0][i].s0);
         //_vector_minus_assign(qout[i].s3, plqcd_g.phim[0][i].s1);
         copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[0][i].s0, &plqcd_g.phim[0][i+1].s0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[0][i].s1, &plqcd_g.phim[0][i+1].s1);
         intrin_vector_load_256(map0,&v0_256);
         intrin_vector_load_256(map1,&v1_256);
         intrin_vector_add_256(mq0,mq0,map0);
         intrin_vector_add_256(mq1,mq1,map1);
         intrin_vector_sub_256(mq2,mq2,map0);
         intrin_vector_sub_256(mq3,mq3,map1);

         // 1 direction
         _prefetch_halfspinor(&plqcd_g.phim[2][i]);
         _prefetch_halfspinor(&plqcd_g.phim[2][i+1]);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[1][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s2, 1.0,plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s3, 1.0, plqcd_g.phim[1][i].s0);
         copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[1][i].s0, &plqcd_g.phim[1][i+1].s0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[1][i].s1, &plqcd_g.phim[1][i+1].s1);
         intrin_vector_load_256(map0,&v0_256);
         intrin_vector_load_256(map1,&v1_256);
         intrin_vector_add_256(mq0,mq0,map0);
         intrin_vector_add_256(mq1,mq1,map1);
         intrin_vector_i_add_256(mq2,mq2,map1);
         intrin_vector_i_add_256(mq3,mq3,map0);
        

         // 2 direction
         _prefetch_halfspinor(&plqcd_g.phim[3][i]);
         _prefetch_halfspinor(&plqcd_g.phim[3][i+1]);
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[2][i].s0);
         //_vector_add_assign(qout[i].s1,  plqcd_g.phim[2][i].s1 );
         //_vector_add_assign(qout[i].s2,  plqcd_g.phim[2][i].s1);
         //_vector_sub_assign(qout[i].s3,   plqcd_g.phim[2][i].s0);
         copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[2][i].s0, &plqcd_g.phim[2][i+1].s0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[2][i].s1, &plqcd_g.phim[2][i+1].s1);
         intrin_vector_load_256(map0,&v0_256);
         intrin_vector_load_256(map1,&v1_256);
         intrin_vector_add_256(mq0,mq0,map0);
         intrin_vector_add_256(mq1,mq1,map1);
         intrin_vector_add_256(mq2,mq2,map1);
         intrin_vector_sub_256(mq3,mq3,map0);
         

         //3 direction
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[3][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[3][i].s1);
         //_vector_i_add_assign(qout[i].s2, plqcd_g.phim[3][i].s0);
         //_vector_i_sub_assign(qout[i].s3, plqcd_g.phim[3][i].s1);     
         copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[3][i].s0, &plqcd_g.phim[3][i+1].s0);
         copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[3][i].s1, &plqcd_g.phim[3][i+1].s1);
         intrin_vector_load_256(map0,&v0_256);
         intrin_vector_load_256(map1,&v1_256);
         intrin_vector_add_256(mq0,mq0,map0);
         intrin_vector_add_256(mq1,mq1,map1);
         intrin_vector_i_add_256(mq2,mq2,map0);
         intrin_vector_i_add_256(mq3,mq3,map1);

         //store the result
         intrin_vector_store_256(&qout[k].s0, mq0);
         intrin_vector_store_256(&qout[k].s1, mq1);
         intrin_vector_store_256(&qout[k].s2, mq2);
         intrin_vector_store_256(&qout[k].s3, mq3);
   }
#ifdef _OPENMP
}
#endif  //end of the openmp parallel reigon

   
   return stop_watch(ts);
}

#ifdef AVX_SPLIT
//======================================================
//avx with openmp only, split real and imaginary, double
//No halfspinor, loop over the output spinor 
//in exactly the same way the operaotr is written 
//================Even-Odd==============================
double plqcd_hopping_matrix_eo_avx_nohalfspinor(
                                double *qout_re, 
                                double *qout_im, 
                                double *u_re, 
                                double *u_im,
                                double *qin_re, 
                                double *qin_im) 
{
   double ts;          //timer
   ts=stop_watch(0.0); //start
   #ifdef _OPENMP
   #pragma omp parallel 
   {
   #endif
      int V,Vavx_split,Vyzt,lx;
      V = plqcd_g.VOLUME;
      Vavx_split = plqcd_g.Vavx_split;
      Vyzt = plqcd_g.latdims[1]*plqcd_g.latdims[2]*plqcd_g.latdims[3];
      lx=plqcd_g.latdims[0]/4;
      if((plqcd_g.latdims[0]%8) != 0 ){ 
        fprintf(stderr,"lattice size in the 0 direction must be a factor of 8\n");
        exit(1);
      }
      
      __m256d  mqin_re[12],mqin_im[12], mqout_re[12],mqout_im[12];
      __m256d  U_re[3][3], U_im[3][3];
      __m256d  out1_re[3],out2_re[3],out3_re[3],out4_re[3];
      __m256d  out1_im[3],out2_im[3],out3_im[3],out4_im[3];
      __m256d  map0_re[3],map0_im[3],map1_re[3],map1_im[3];
      __m256d  register m1,m2;

      double perm_in[4],perm_out[4];

      int ix,iy,it,iz,is,xcor[4],ipt;

      int L[4];
      for(int i=0; i<4; i++)
         L[i] = plqcd_g.latdims[i];

      //needed pointers
      //input spinor, links and output spinor
      double *sx_re,*sx_im,*ux_re,*ux_im,*gx_re,*gx_im;


      //for permuting the 4 doubles from 0,1,2,3 order into 1,2,3,0 order
      //note each double correspond to two ints
      //int  __attribute__ ((aligned(32)))  permback[8]={2,3,4,5,6,7,0,1};
      //__m256i mpermback = _mm256_load_si32((__m256i) permback);

      //for permuting the 4 doubles from 0,1,2,3 order into 3,0,1,2 order
      //note each double correspond to two ints
      //int __attribute__ ((aligned(32)))   permfor[8]={6,7,0,1,2,3,4,5};
      //__m256i mpermfor = _mm256_load_si32((__m256i) permfor);


      //-----------------------------------------------------
      // compute the result
      //----------------------------------------------------
      #ifdef _OPENMP
      //#pragma omp for schedule(static,20) //use this if you want to play with how work is shared among threads 
      #pragma omp for 
      #endif   
      for(int is=0; is < Vyzt; is++) 
      {
         iy = is%L[1];
         iz = (is/L[1]) %L[2];
         it = (is/L[1]/L[2])%L[3];

         for(ix = (iy+iz+it)%2; ix < lx; ix +=2)  //result is on even sites
         {
            xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
            ipt=plqcd_g.ipt_eo_avx_split[cart2lex(xcor)];

            gx_re = qout_re + ipt*48;
            gx_im = qout_im + ipt*48;

            //================
            //====  +0  ======
            //================

            ux_re = u_re + ipt*144;
            ux_im = u_im + ipt*144;

            sx_re = qin_re + plqcd_g.iup_avx_split[ipt][0]*48;
            sx_im = qin_im + plqcd_g.iup_avx_split[ipt][0]*48;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm256_load_pd(sx_re);
            mqin_re[1]  = _mm256_load_pd(sx_re+4);
            mqin_re[2]  = _mm256_load_pd(sx_re+8);
            mqin_im[0]  = _mm256_load_pd(sx_im);
            mqin_im[1]  = _mm256_load_pd(sx_im+4);
            mqin_im[2]  = _mm256_load_pd(sx_im+8);


            mqin_re[9]   = _mm256_load_pd(sx_re+36);
            mqin_re[10]  = _mm256_load_pd(sx_re+40);
            mqin_re[11]  = _mm256_load_pd(sx_re+44);
            mqin_im[9]   = _mm256_load_pd(sx_im+36);
            mqin_im[10]  = _mm256_load_pd(sx_im+40);
            mqin_im[11]  = _mm256_load_pd(sx_im+44);


            out1_re[0] = _mm256_sub_pd(mqin_re[0],mqin_im[9]);            
            out1_re[1] = _mm256_sub_pd(mqin_re[1],mqin_im[10]);            
            out1_re[2] = _mm256_sub_pd(mqin_re[2],mqin_im[11]);            
            out1_im[0] = _mm256_add_pd(mqin_im[0],mqin_re[9]);            
            out1_im[1] = _mm256_add_pd(mqin_im[1],mqin_re[10]);            
            out1_im[2] = _mm256_add_pd(mqin_im[2],mqin_re[11]);


            mqin_re[3]  = _mm256_load_pd(sx_re+12);
            mqin_re[4]  = _mm256_load_pd(sx_re+16);
            mqin_re[5]  = _mm256_load_pd(sx_re+20);
            mqin_im[3]  = _mm256_load_pd(sx_im+12);
            mqin_im[4]  = _mm256_load_pd(sx_im+16);
            mqin_im[5]  = _mm256_load_pd(sx_im+20);


            mqin_re[6]  = _mm256_load_pd(sx_re+24);
            mqin_re[7]  = _mm256_load_pd(sx_re+28);
            mqin_re[8]  = _mm256_load_pd(sx_re+32);
            mqin_im[6]  = _mm256_load_pd(sx_im+24);
            mqin_im[7]  = _mm256_load_pd(sx_im+28);
            mqin_im[8]  = _mm256_load_pd(sx_im+32);


            out2_re[0] = _mm256_sub_pd(mqin_re[3],mqin_im[6]);            
            out2_re[1] = _mm256_sub_pd(mqin_re[4],mqin_im[7]);            
            out2_re[2] = _mm256_sub_pd(mqin_re[5],mqin_im[8]);            
            out2_im[0] = _mm256_add_pd(mqin_im[3],mqin_re[6]);            
            out2_im[1] = _mm256_add_pd(mqin_im[4],mqin_re[7]);            
            out2_im[2] = _mm256_add_pd(mqin_im[5],mqin_re[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 2 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm256_load_pd(ux_re+jc*4+ic*12);
                  U_im[ic][jc] = _mm256_load_pd(ux_im+jc*4+ic*12);
               }

            su3_multiply_splitlayout_256(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_256(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            //store the result, shuffle if on the boundary
            if(ix == 0)
            {
               //for(int i=0; i<3; i++)
               //{  
               //   out3_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) map0_re[i]);
               //   out3_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) map0_im[i]);
               //   out4_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) map1_re[i]);
               //   out4_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) map1_im[i]);
               //}
              
               //this is the best I can do at the moment. Couldn't find a swizzle function to make this permutation
               for(int i=0; i<3; i++)
               {
                  _mm256_store_pd(perm_in,map0_re[i]);
                  perm_out[0] = perm_in[1]; perm_out[1]= perm_in[2]; perm_out[2]=perm_in[3]; perm_out[3]=perm_in[0];
                  out3_re[i] = _mm256_load_pd(perm_out);


                  _mm256_store_pd(perm_in,map0_im[i]);
                  perm_out[0] = perm_in[1]; perm_out[1]= perm_in[2]; perm_out[2]=perm_in[3]; perm_out[3]=perm_in[0];
                  out3_im[i] = _mm256_load_pd(perm_out);


                  _mm256_store_pd(perm_in,map1_re[i]);
                  perm_out[0] = perm_in[1]; perm_out[1]= perm_in[2]; perm_out[2]=perm_in[3]; perm_out[3]=perm_in[0];
                  out4_re[i] = _mm256_load_pd(perm_out);


                  _mm256_store_pd(perm_in,map1_im[i]);
                  perm_out[0] = perm_in[1]; perm_out[1]= perm_in[2]; perm_out[2]=perm_in[3]; perm_out[3]=perm_in[0];
                  out4_im[i] = _mm256_load_pd(perm_out);


               }




 
               mqout_re[0] = _mm256_add_pd(mqout_re[0] ,out3_re[0]);
               mqout_re[1] = _mm256_add_pd(mqout_re[1] ,out3_re[1]);
               mqout_re[2] = _mm256_add_pd(mqout_re[2] ,out3_re[2]);
               mqout_re[3] = _mm256_add_pd(mqout_re[3] ,out4_re[0]);
               mqout_re[4] = _mm256_add_pd(mqout_re[4] ,out4_re[1]);
               mqout_re[5] = _mm256_add_pd(mqout_re[5] ,out4_re[2]);

               mqout_re[6] = _mm256_add_pd(mqout_re[6] ,out4_im[0]);
               mqout_re[7] = _mm256_add_pd(mqout_re[7] ,out4_im[1]);
               mqout_re[8] = _mm256_add_pd(mqout_re[8] ,out4_im[2]);
               mqout_re[9] = _mm256_add_pd(mqout_re[9] ,out3_im[0]);
               mqout_re[10]= _mm256_add_pd(mqout_re[10],out3_im[1]);
               mqout_re[11]= _mm256_add_pd(mqout_re[11],out3_im[2]);

               mqout_im[0] = _mm256_add_pd(mqout_im[0] ,out3_im[0]);
               mqout_im[1] = _mm256_add_pd(mqout_im[1] ,out3_im[1]);
               mqout_im[2] = _mm256_add_pd(mqout_im[2] ,out3_im[2]);
               mqout_im[3] = _mm256_add_pd(mqout_im[3] ,out4_im[0]);
               mqout_im[4] = _mm256_add_pd(mqout_im[4] ,out4_im[1]);
               mqout_im[5] = _mm256_add_pd(mqout_im[5] ,out4_im[2]);

               mqout_im[6] = _mm256_sub_pd(mqout_im[6] ,out4_re[0]);
               mqout_im[7] = _mm256_sub_pd(mqout_im[7] ,out4_re[1]);
               mqout_im[8] = _mm256_sub_pd(mqout_im[8] ,out4_re[2]);
               mqout_im[9] = _mm256_sub_pd(mqout_im[9] ,out3_re[0]);
               mqout_im[10]= _mm256_sub_pd(mqout_im[10],out3_re[1]);
               mqout_im[11]= _mm256_sub_pd(mqout_im[11],out3_re[2]);

            }
            else
            {
               mqout_re[0] = _mm256_add_pd(mqout_re[0] ,map0_re[0]);
               mqout_re[1] = _mm256_add_pd(mqout_re[1] ,map0_re[1]);
               mqout_re[2] = _mm256_add_pd(mqout_re[2] ,map0_re[2]);
               mqout_re[3] = _mm256_add_pd(mqout_re[3] ,map1_re[0]);
               mqout_re[4] = _mm256_add_pd(mqout_re[4] ,map1_re[1]);
               mqout_re[5] = _mm256_add_pd(mqout_re[5] ,map1_re[2]);

               mqout_re[6] = _mm256_add_pd(mqout_re[6] ,map1_im[0]);
               mqout_re[7] = _mm256_add_pd(mqout_re[7] ,map1_im[1]);
               mqout_re[8] = _mm256_add_pd(mqout_re[8] ,map1_im[2]);
               mqout_re[9] = _mm256_add_pd(mqout_re[9] ,map0_im[0]);
               mqout_re[10]= _mm256_add_pd(mqout_re[10],map0_im[1]);
               mqout_re[11]= _mm256_add_pd(mqout_re[11],map0_im[2]);

               mqout_im[0] = _mm256_add_pd(mqout_im[0] ,map0_im[0]);
               mqout_im[1] = _mm256_add_pd(mqout_im[1] ,map0_im[1]);
               mqout_im[2] = _mm256_add_pd(mqout_im[2] ,map0_im[2]);
               mqout_im[3] = _mm256_add_pd(mqout_im[3] ,map1_im[0]);
               mqout_im[4] = _mm256_add_pd(mqout_im[4] ,map1_im[1]);
               mqout_im[5] = _mm256_add_pd(mqout_im[5] ,map1_im[2]);

               mqout_im[6] = _mm256_sub_pd(mqout_im[6] ,map1_re[0]);
               mqout_im[7] = _mm256_sub_pd(mqout_im[7] ,map1_re[1]);
               mqout_im[8] = _mm256_sub_pd(mqout_im[8] ,map1_re[2]);
               mqout_im[9] = _mm256_sub_pd(mqout_im[9] ,map0_re[0]);
               mqout_im[10]= _mm256_sub_pd(mqout_im[10],map0_re[1]);
               mqout_im[11]= _mm256_sub_pd(mqout_im[11],map0_re[2]);
            } 

            
            //===================
            //===== +1 ==========
            //===================
      
            ux_re += 36;
            ux_im += 36;

            sx_re = qin_re + plqcd_g.iup_avx_split[ipt][1]*48;
            sx_im = qin_im + plqcd_g.iup_avx_split[ipt][1]*48;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm256_load_pd(sx_re);
            mqin_re[1]  = _mm256_load_pd(sx_re+4);
            mqin_re[2]  = _mm256_load_pd(sx_re+8);
            mqin_im[0]  = _mm256_load_pd(sx_im);
            mqin_im[1]  = _mm256_load_pd(sx_im+4);
            mqin_im[2]  = _mm256_load_pd(sx_im+8);


            mqin_re[9]   = _mm256_load_pd(sx_re+36);
            mqin_re[10]  = _mm256_load_pd(sx_re+40);
            mqin_re[11]  = _mm256_load_pd(sx_re+44);
            mqin_im[9]   = _mm256_load_pd(sx_im+36);
            mqin_im[10]  = _mm256_load_pd(sx_im+40);
            mqin_im[11]  = _mm256_load_pd(sx_im+44);


            out1_re[0] = _mm256_add_pd(mqin_re[0],mqin_re[9]);            
            out1_re[1] = _mm256_add_pd(mqin_re[1],mqin_re[10]);            
            out1_re[2] = _mm256_add_pd(mqin_re[2],mqin_re[11]);            
            out1_im[0] = _mm256_add_pd(mqin_im[0],mqin_im[9]);            
            out1_im[1] = _mm256_add_pd(mqin_im[1],mqin_im[10]);            
            out1_im[2] = _mm256_add_pd(mqin_im[2],mqin_im[11]);

            mqin_re[3]  = _mm256_load_pd(sx_re+12);
            mqin_re[4]  = _mm256_load_pd(sx_re+16);
            mqin_re[5]  = _mm256_load_pd(sx_re+20);
            mqin_im[3]  = _mm256_load_pd(sx_im+12);
            mqin_im[4]  = _mm256_load_pd(sx_im+16);
            mqin_im[5]  = _mm256_load_pd(sx_im+20);


            mqin_re[6]  = _mm256_load_pd(sx_re+24);
            mqin_re[7]  = _mm256_load_pd(sx_re+28);
            mqin_re[8]  = _mm256_load_pd(sx_re+32);
            mqin_im[6]  = _mm256_load_pd(sx_im+24);
            mqin_im[7]  = _mm256_load_pd(sx_im+28);
            mqin_im[8]  = _mm256_load_pd(sx_im+32);


            out2_re[0] = _mm256_sub_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm256_sub_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm256_sub_pd(mqin_re[5],mqin_re[8]);            
            out2_im[0] = _mm256_sub_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm256_sub_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm256_sub_pd(mqin_im[5],mqin_im[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm256_load_pd(ux_re+jc*4+ic*12);
                  U_im[ic][jc] = _mm256_load_pd(ux_im+jc*4+ic*12);
               }
            
            su3_multiply_splitlayout_256(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_256(map1_re,map1_im,U_re,U_im,out2_re,out2_im);

            mqout_re[0] = _mm256_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm256_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm256_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm256_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm256_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm256_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm256_sub_pd(mqout_re[6] ,map1_re[0]);
            mqout_re[7] = _mm256_sub_pd(mqout_re[7] ,map1_re[1]);
            mqout_re[8] = _mm256_sub_pd(mqout_re[8] ,map1_re[2]);
            mqout_re[9] = _mm256_add_pd(mqout_re[9] ,map0_re[0]);
            mqout_re[10]= _mm256_add_pd(mqout_re[10],map0_re[1]);
            mqout_re[11]= _mm256_add_pd(mqout_re[11],map0_re[2]);

            mqout_im[0] = _mm256_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm256_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm256_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm256_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm256_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm256_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm256_sub_pd(mqout_im[6] ,map1_im[0]);
            mqout_im[7] = _mm256_sub_pd(mqout_im[7] ,map1_im[1]);
            mqout_im[8] = _mm256_sub_pd(mqout_im[8] ,map1_im[2]);
            mqout_im[9] = _mm256_add_pd(mqout_im[9] ,map0_im[0]);
            mqout_im[10]= _mm256_add_pd(mqout_im[10],map0_im[1]);
            mqout_im[11]= _mm256_add_pd(mqout_im[11],map0_im[2]);




            //===================
            //===== +2 ==========
            //===================
      
            ux_re += 36;
            ux_im += 36;

            sx_re = qin_re + plqcd_g.iup_avx_split[ipt][2]*48;
            sx_im = qin_im + plqcd_g.iup_avx_split[ipt][2]*48;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm256_load_pd(sx_re);
            mqin_re[1]  = _mm256_load_pd(sx_re+4);
            mqin_re[2]  = _mm256_load_pd(sx_re+8);
            mqin_im[0]  = _mm256_load_pd(sx_im);
            mqin_im[1]  = _mm256_load_pd(sx_im+4);
            mqin_im[2]  = _mm256_load_pd(sx_im+8);


            mqin_re[6]  = _mm256_load_pd(sx_re+24);
            mqin_re[7]  = _mm256_load_pd(sx_re+28);
            mqin_re[8]  = _mm256_load_pd(sx_re+32);
            mqin_im[6]  = _mm256_load_pd(sx_im+24);
            mqin_im[7]  = _mm256_load_pd(sx_im+28);
            mqin_im[8]  = _mm256_load_pd(sx_im+32);


            out1_re[0] = _mm256_sub_pd(mqin_re[0],mqin_im[6]);            
            out1_re[1] = _mm256_sub_pd(mqin_re[1],mqin_im[7]);            
            out1_re[2] = _mm256_sub_pd(mqin_re[2],mqin_im[8]);            
            out1_im[0] = _mm256_add_pd(mqin_im[0],mqin_re[6]);            
            out1_im[1] = _mm256_add_pd(mqin_im[1],mqin_re[7]);            
            out1_im[2] = _mm256_add_pd(mqin_im[2],mqin_re[8]);

            mqin_re[3]  = _mm256_load_pd(sx_re+12);
            mqin_re[4]  = _mm256_load_pd(sx_re+16);
            mqin_re[5]  = _mm256_load_pd(sx_re+20);
            mqin_im[3]  = _mm256_load_pd(sx_im+12);
            mqin_im[4]  = _mm256_load_pd(sx_im+16);
            mqin_im[5]  = _mm256_load_pd(sx_im+20);


            mqin_re[9]   = _mm256_load_pd(sx_re+36);
            mqin_re[10]  = _mm256_load_pd(sx_re+40);
            mqin_re[11]  = _mm256_load_pd(sx_re+44);
            mqin_im[9]   = _mm256_load_pd(sx_im+36);
            mqin_im[10]  = _mm256_load_pd(sx_im+40);
            mqin_im[11]  = _mm256_load_pd(sx_im+44);


            out2_re[0] = _mm256_add_pd(mqin_re[3],mqin_im[9]);            
            out2_re[1] = _mm256_add_pd(mqin_re[4],mqin_im[10]);            
            out2_re[2] = _mm256_add_pd(mqin_re[5],mqin_im[11]);            
            out2_im[0] = _mm256_sub_pd(mqin_im[3],mqin_re[9]);            
            out2_im[1] = _mm256_sub_pd(mqin_im[4],mqin_re[10]);            
            out2_im[2] = _mm256_sub_pd(mqin_im[5],mqin_re[11]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm256_load_pd(ux_re+jc*4+ic*12);
                  U_im[ic][jc] = _mm256_load_pd(ux_im+jc*4+ic*12);
               }
            
            su3_multiply_splitlayout_256(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_256(map1_re,map1_im,U_re,U_im,out2_re,out2_im);

            mqout_re[0] = _mm256_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm256_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm256_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm256_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm256_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm256_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm256_add_pd(mqout_re[6] ,map0_im[0]);
            mqout_re[7] = _mm256_add_pd(mqout_re[7] ,map0_im[1]);
            mqout_re[8] = _mm256_add_pd(mqout_re[8] ,map0_im[2]);
            mqout_re[9] = _mm256_sub_pd(mqout_re[9] ,map1_im[0]);
            mqout_re[10]= _mm256_sub_pd(mqout_re[10],map1_im[1]);
            mqout_re[11]= _mm256_sub_pd(mqout_re[11],map1_im[2]);

            mqout_im[0] = _mm256_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm256_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm256_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm256_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm256_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm256_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm256_sub_pd(mqout_im[6] ,map0_re[0]);
            mqout_im[7] = _mm256_sub_pd(mqout_im[7] ,map0_re[1]);
            mqout_im[8] = _mm256_sub_pd(mqout_im[8] ,map0_re[2]);
            mqout_im[9] = _mm256_add_pd(mqout_im[9] ,map1_re[0]);
            mqout_im[10]= _mm256_add_pd(mqout_im[10],map1_re[1]);
            mqout_im[11]= _mm256_add_pd(mqout_im[11],map1_re[2]);



            //===================
            //===== +3 ==========
            //===================
      
            ux_re += 36;
            ux_im += 36;

            sx_re = qin_re + plqcd_g.iup_avx_split[ipt][3]*48;
            sx_im = qin_im + plqcd_g.iup_avx_split[ipt][3]*48;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm256_load_pd(sx_re);
            mqin_re[1]  = _mm256_load_pd(sx_re+4);
            mqin_re[2]  = _mm256_load_pd(sx_re+8);
            mqin_im[0]  = _mm256_load_pd(sx_im);
            mqin_im[1]  = _mm256_load_pd(sx_im+4);
            mqin_im[2]  = _mm256_load_pd(sx_im+8);


            mqin_re[6]  = _mm256_load_pd(sx_re+24);
            mqin_re[7]  = _mm256_load_pd(sx_re+28);
            mqin_re[8]  = _mm256_load_pd(sx_re+32);
            mqin_im[6]  = _mm256_load_pd(sx_im+24);
            mqin_im[7]  = _mm256_load_pd(sx_im+28);
            mqin_im[8]  = _mm256_load_pd(sx_im+32);


            out1_re[0] = _mm256_add_pd(mqin_re[0],mqin_re[6]);            
            out1_re[1] = _mm256_add_pd(mqin_re[1],mqin_re[7]);            
            out1_re[2] = _mm256_add_pd(mqin_re[2],mqin_re[8]);            
            out1_im[0] = _mm256_add_pd(mqin_im[0],mqin_im[6]);            
            out1_im[1] = _mm256_add_pd(mqin_im[1],mqin_im[7]);            
            out1_im[2] = _mm256_add_pd(mqin_im[2],mqin_im[8]);

            mqin_re[3]  = _mm256_load_pd(sx_re+12);
            mqin_re[4]  = _mm256_load_pd(sx_re+16);
            mqin_re[5]  = _mm256_load_pd(sx_re+20);
            mqin_im[3]  = _mm256_load_pd(sx_im+12);
            mqin_im[4]  = _mm256_load_pd(sx_im+16);
            mqin_im[5]  = _mm256_load_pd(sx_im+20);


            mqin_re[9]   = _mm256_load_pd(sx_re+36);
            mqin_re[10]  = _mm256_load_pd(sx_re+40);
            mqin_re[11]  = _mm256_load_pd(sx_re+44);
            mqin_im[9]   = _mm256_load_pd(sx_im+36);
            mqin_im[10]  = _mm256_load_pd(sx_im+40);
            mqin_im[11]  = _mm256_load_pd(sx_im+44);


            out2_re[0] = _mm256_add_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm256_add_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm256_add_pd(mqin_re[5],mqin_re[8]);            
            out2_im[0] = _mm256_add_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm256_add_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm256_add_pd(mqin_im[5],mqin_im[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm256_load_pd(ux_re+jc*4+ic*12);
                  U_im[ic][jc] = _mm256_load_pd(ux_im+jc*4+ic*12);
               }
            
            su3_multiply_splitlayout_256(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_256(map1_re,map1_im,U_re,U_im,out2_re,out2_im);

            mqout_re[0] = _mm256_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm256_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm256_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm256_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm256_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm256_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm256_add_pd(mqout_re[6] ,map0_re[0]);
            mqout_re[7] = _mm256_add_pd(mqout_re[7] ,map0_re[1]);
            mqout_re[8] = _mm256_add_pd(mqout_re[8] ,map0_re[2]);
            mqout_re[9] = _mm256_add_pd(mqout_re[9] ,map1_re[0]);
            mqout_re[10]= _mm256_add_pd(mqout_re[10],map1_re[1]);
            mqout_re[11]= _mm256_add_pd(mqout_re[11],map1_re[2]);

            mqout_im[0] = _mm256_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm256_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm256_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm256_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm256_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm256_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm256_add_pd(mqout_im[6] ,map0_im[0]);
            mqout_im[7] = _mm256_add_pd(mqout_im[7] ,map0_im[1]);
            mqout_im[8] = _mm256_add_pd(mqout_im[8] ,map0_im[2]);
            mqout_im[9] = _mm256_add_pd(mqout_im[9] ,map1_im[0]);
            mqout_im[10]= _mm256_add_pd(mqout_im[10],map1_im[1]);
            mqout_im[11]= _mm256_add_pd(mqout_im[11],map1_im[2]);



            //=======================
            //===== -0  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_avx_split[ipt][0]*144;
            ux_im = u_im + plqcd_g.idn_avx_split[ipt][0]*144;

            sx_re = qin_re + plqcd_g.idn_avx_split[ipt][0]*48;
            sx_im = qin_im + plqcd_g.idn_avx_split[ipt][0]*48;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm256_load_pd(sx_re);
            mqin_re[1]  = _mm256_load_pd(sx_re+4);
            mqin_re[2]  = _mm256_load_pd(sx_re+8);
            mqin_im[0]  = _mm256_load_pd(sx_im);
            mqin_im[1]  = _mm256_load_pd(sx_im+4);
            mqin_im[2]  = _mm256_load_pd(sx_im+8);


            mqin_re[9]   = _mm256_load_pd(sx_re+36);
            mqin_re[10]  = _mm256_load_pd(sx_re+40);
            mqin_re[11]  = _mm256_load_pd(sx_re+44);
            mqin_im[9]   = _mm256_load_pd(sx_im+36);
            mqin_im[10]  = _mm256_load_pd(sx_im+40);
            mqin_im[11]  = _mm256_load_pd(sx_im+44);


            out1_re[0] = _mm256_add_pd(mqin_re[0],mqin_im[9]);            
            out1_re[1] = _mm256_add_pd(mqin_re[1],mqin_im[10]);            
            out1_re[2] = _mm256_add_pd(mqin_re[2],mqin_im[11]);            
            out1_im[0] = _mm256_sub_pd(mqin_im[0],mqin_re[9]);            
            out1_im[1] = _mm256_sub_pd(mqin_im[1],mqin_re[10]);            
            out1_im[2] = _mm256_sub_pd(mqin_im[2],mqin_re[11]);


            mqin_re[3]  = _mm256_load_pd(sx_re+12);
            mqin_re[4]  = _mm256_load_pd(sx_re+16);
            mqin_re[5]  = _mm256_load_pd(sx_re+20);
            mqin_im[3]  = _mm256_load_pd(sx_im+12);
            mqin_im[4]  = _mm256_load_pd(sx_im+16);
            mqin_im[5]  = _mm256_load_pd(sx_im+20);


            mqin_re[6]  = _mm256_load_pd(sx_re+24);
            mqin_re[7]  = _mm256_load_pd(sx_re+28);
            mqin_re[8]  = _mm256_load_pd(sx_re+32);
            mqin_im[6]  = _mm256_load_pd(sx_im+24);
            mqin_im[7]  = _mm256_load_pd(sx_im+28);
            mqin_im[8]  = _mm256_load_pd(sx_im+32);


            out2_re[0] = _mm256_add_pd(mqin_re[3],mqin_im[6]);            
            out2_re[1] = _mm256_add_pd(mqin_re[4],mqin_im[7]);            
            out2_re[2] = _mm256_add_pd(mqin_re[5],mqin_im[8]);            
            out2_im[0] = _mm256_sub_pd(mqin_im[3],mqin_re[6]);            
            out2_im[1] = _mm256_sub_pd(mqin_im[4],mqin_re[7]);            
            out2_im[2] = _mm256_sub_pd(mqin_im[5],mqin_re[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm256_load_pd(ux_re+jc*4+ic*12);
                  U_im[ic][jc] = _mm256_load_pd(ux_im+jc*4+ic*12);
               }
            
            su3_inverse_multiply_splitlayout_256(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_256(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            //store the result, shuffle if on the boundary
            if(ix == lx-1)
            {
               //need to figure out how to do this. At the moment I am turning it off which should give higher performance
               //for(int i=0; i<3; i++)
               //{  
               //   out3_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map0_re[i]);
               //   out3_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map0_im[i]);
               //   out4_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map1_re[i]);
               //   out4_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map1_im[i]);
               //}
              
               //this is the best I can do at the moment. Couldn't find a reasonable instruction for this
               for(int i=0; i<3;  i++)
               {

                  _mm256_store_pd(perm_in,map0_re[i]);
                  perm_out[0] = perm_in[3]; perm_out[1]= perm_in[0]; perm_out[2]=perm_in[1]; perm_out[3]=perm_in[2];
                  out3_re[i] = _mm256_load_pd(perm_out);


                  _mm256_store_pd(perm_in,map0_im[i]);
                  perm_out[0] = perm_in[3]; perm_out[1]= perm_in[0]; perm_out[2]=perm_in[1]; perm_out[3]=perm_in[2];
                  out3_im[i] = _mm256_load_pd(perm_out);


                  _mm256_store_pd(perm_in,map1_re[i]);
                  perm_out[0] = perm_in[3]; perm_out[1]= perm_in[0]; perm_out[2]=perm_in[1]; perm_out[3]=perm_in[2];
                  out4_re[i] = _mm256_load_pd(perm_out);

                  _mm256_store_pd(perm_in,map1_im[i]);
                  perm_out[0] = perm_in[3]; perm_out[1]= perm_in[0]; perm_out[2]=perm_in[1]; perm_out[3]=perm_in[2];
                  out4_im[i] = _mm256_load_pd(perm_out);

               }



 
               mqout_re[0] = _mm256_add_pd(mqout_re[0] ,out3_re[0]);
               mqout_re[1] = _mm256_add_pd(mqout_re[1] ,out3_re[1]);
               mqout_re[2] = _mm256_add_pd(mqout_re[2] ,out3_re[2]);
               mqout_re[3] = _mm256_add_pd(mqout_re[3] ,out4_re[0]);
               mqout_re[4] = _mm256_add_pd(mqout_re[4] ,out4_re[1]);
               mqout_re[5] = _mm256_add_pd(mqout_re[5] ,out4_re[2]);

               mqout_re[6] = _mm256_sub_pd(mqout_re[6] ,out4_im[0]);
               mqout_re[7] = _mm256_sub_pd(mqout_re[7] ,out4_im[1]);
               mqout_re[8] = _mm256_sub_pd(mqout_re[8] ,out4_im[2]);
               mqout_re[9] = _mm256_sub_pd(mqout_re[9] ,out3_im[0]);
               mqout_re[10]= _mm256_sub_pd(mqout_re[10],out3_im[1]);
               mqout_re[11]= _mm256_sub_pd(mqout_re[11],out3_im[2]);

               mqout_im[0] = _mm256_add_pd(mqout_im[0] ,out3_im[0]);
               mqout_im[1] = _mm256_add_pd(mqout_im[1] ,out3_im[1]);
               mqout_im[2] = _mm256_add_pd(mqout_im[2] ,out3_im[2]);
               mqout_im[3] = _mm256_add_pd(mqout_im[3] ,out4_im[0]);
               mqout_im[4] = _mm256_add_pd(mqout_im[4] ,out4_im[1]);
               mqout_im[5] = _mm256_add_pd(mqout_im[5] ,out4_im[2]);

               mqout_im[6] = _mm256_add_pd(mqout_im[6] ,out4_re[0]);
               mqout_im[7] = _mm256_add_pd(mqout_im[7] ,out4_re[1]);
               mqout_im[8] = _mm256_add_pd(mqout_im[8] ,out4_re[2]);
               mqout_im[9] = _mm256_add_pd(mqout_im[9] ,out3_re[0]);
               mqout_im[10]= _mm256_add_pd(mqout_im[10],out3_re[1]);
               mqout_im[11]= _mm256_add_pd(mqout_im[11],out3_re[2]);

            }
            else
            {
               mqout_re[0] = _mm256_add_pd(mqout_re[0] ,map0_re[0]);
               mqout_re[1] = _mm256_add_pd(mqout_re[1] ,map0_re[1]);
               mqout_re[2] = _mm256_add_pd(mqout_re[2] ,map0_re[2]);
               mqout_re[3] = _mm256_add_pd(mqout_re[3] ,map1_re[0]);
               mqout_re[4] = _mm256_add_pd(mqout_re[4] ,map1_re[1]);
               mqout_re[5] = _mm256_add_pd(mqout_re[5] ,map1_re[2]);

               mqout_re[6] = _mm256_sub_pd(mqout_re[6] ,map1_im[0]);
               mqout_re[7] = _mm256_sub_pd(mqout_re[7] ,map1_im[1]);
               mqout_re[8] = _mm256_sub_pd(mqout_re[8] ,map1_im[2]);
               mqout_re[9] = _mm256_sub_pd(mqout_re[9] ,map0_im[0]);
               mqout_re[10]= _mm256_sub_pd(mqout_re[10],map0_im[1]);
               mqout_re[11]= _mm256_sub_pd(mqout_re[11],map0_im[2]);

               mqout_im[0] = _mm256_add_pd(mqout_im[0] ,map0_im[0]);
               mqout_im[1] = _mm256_add_pd(mqout_im[1] ,map0_im[1]);
               mqout_im[2] = _mm256_add_pd(mqout_im[2] ,map0_im[2]);
               mqout_im[3] = _mm256_add_pd(mqout_im[3] ,map1_im[0]);
               mqout_im[4] = _mm256_add_pd(mqout_im[4] ,map1_im[1]);
               mqout_im[5] = _mm256_add_pd(mqout_im[5] ,map1_im[2]);

               mqout_im[6] = _mm256_add_pd(mqout_im[6] ,map1_re[0]);
               mqout_im[7] = _mm256_add_pd(mqout_im[7] ,map1_re[1]);
               mqout_im[8] = _mm256_add_pd(mqout_im[8] ,map1_re[2]);
               mqout_im[9] = _mm256_add_pd(mqout_im[9] ,map0_re[0]);
               mqout_im[10]= _mm256_add_pd(mqout_im[10],map0_re[1]);
               mqout_im[11]= _mm256_add_pd(mqout_im[11],map0_re[2]);
            } 

            //=======================
            //===== -1  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_avx_split[ipt][1]*144;
            ux_im = u_im + plqcd_g.idn_avx_split[ipt][1]*144;

            sx_re = qin_re + plqcd_g.idn_avx_split[ipt][1]*48;
            sx_im = qin_im + plqcd_g.idn_avx_split[ipt][1]*48;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm256_load_pd(sx_re);
            mqin_re[1]  = _mm256_load_pd(sx_re+4);
            mqin_re[2]  = _mm256_load_pd(sx_re+8);
            mqin_im[0]  = _mm256_load_pd(sx_im);
            mqin_im[1]  = _mm256_load_pd(sx_im+4);
            mqin_im[2]  = _mm256_load_pd(sx_im+8);


            mqin_re[9]   = _mm256_load_pd(sx_re+36);
            mqin_re[10]  = _mm256_load_pd(sx_re+40);
            mqin_re[11]  = _mm256_load_pd(sx_re+44);
            mqin_im[9]   = _mm256_load_pd(sx_im+36);
            mqin_im[10]  = _mm256_load_pd(sx_im+40);
            mqin_im[11]  = _mm256_load_pd(sx_im+44);


            out1_re[0] = _mm256_sub_pd(mqin_re[0],mqin_re[9]);            
            out1_re[1] = _mm256_sub_pd(mqin_re[1],mqin_re[10]);            
            out1_re[2] = _mm256_sub_pd(mqin_re[2],mqin_re[11]);            
            out1_im[0] = _mm256_sub_pd(mqin_im[0],mqin_im[9]);            
            out1_im[1] = _mm256_sub_pd(mqin_im[1],mqin_im[10]);            
            out1_im[2] = _mm256_sub_pd(mqin_im[2],mqin_im[11]);


            mqin_re[3]  = _mm256_load_pd(sx_re+12);
            mqin_re[4]  = _mm256_load_pd(sx_re+16);
            mqin_re[5]  = _mm256_load_pd(sx_re+20);
            mqin_im[3]  = _mm256_load_pd(sx_im+12);
            mqin_im[4]  = _mm256_load_pd(sx_im+16);
            mqin_im[5]  = _mm256_load_pd(sx_im+20);


            mqin_re[6]  = _mm256_load_pd(sx_re+24);
            mqin_re[7]  = _mm256_load_pd(sx_re+28);
            mqin_re[8]  = _mm256_load_pd(sx_re+32);
            mqin_im[6]  = _mm256_load_pd(sx_im+24);
            mqin_im[7]  = _mm256_load_pd(sx_im+28);
            mqin_im[8]  = _mm256_load_pd(sx_im+32);


            out2_re[0] = _mm256_add_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm256_add_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm256_add_pd(mqin_re[5],mqin_re[8]);            
            out2_im[0] = _mm256_add_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm256_add_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm256_add_pd(mqin_im[5],mqin_im[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm256_load_pd(ux_re+jc*4+ic*12);
                  U_im[ic][jc] = _mm256_load_pd(ux_im+jc*4+ic*12);
               }
            
            su3_inverse_multiply_splitlayout_256(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_256(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            mqout_re[0] = _mm256_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm256_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm256_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm256_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm256_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm256_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm256_add_pd(mqout_re[6] ,map1_re[0]);
            mqout_re[7] = _mm256_add_pd(mqout_re[7] ,map1_re[1]);
            mqout_re[8] = _mm256_add_pd(mqout_re[8] ,map1_re[2]);
            mqout_re[9] = _mm256_sub_pd(mqout_re[9] ,map0_re[0]);
            mqout_re[10]= _mm256_sub_pd(mqout_re[10],map0_re[1]);
            mqout_re[11]= _mm256_sub_pd(mqout_re[11],map0_re[2]);

            mqout_im[0] = _mm256_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm256_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm256_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm256_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm256_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm256_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm256_add_pd(mqout_im[6] ,map1_im[0]);
            mqout_im[7] = _mm256_add_pd(mqout_im[7] ,map1_im[1]);
            mqout_im[8] = _mm256_add_pd(mqout_im[8] ,map1_im[2]);
            mqout_im[9] = _mm256_sub_pd(mqout_im[9] ,map0_im[0]);
            mqout_im[10]= _mm256_sub_pd(mqout_im[10],map0_im[1]);
            mqout_im[11]= _mm256_sub_pd(mqout_im[11],map0_im[2]);
            
            //=======================
            //===== -2  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_avx_split[ipt][2]*144;
            ux_im = u_im + plqcd_g.idn_avx_split[ipt][2]*144;

            sx_re = qin_re + plqcd_g.idn_avx_split[ipt][2]*48;
            sx_im = qin_im + plqcd_g.idn_avx_split[ipt][2]*48;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm256_load_pd(sx_re);
            mqin_re[1]  = _mm256_load_pd(sx_re+4);
            mqin_re[2]  = _mm256_load_pd(sx_re+8);
            mqin_im[0]  = _mm256_load_pd(sx_im);
            mqin_im[1]  = _mm256_load_pd(sx_im+4);
            mqin_im[2]  = _mm256_load_pd(sx_im+8);


            mqin_re[6]   = _mm256_load_pd(sx_re+24);
            mqin_re[7]   = _mm256_load_pd(sx_re+28);
            mqin_re[8]   = _mm256_load_pd(sx_re+32);
            mqin_im[6]   = _mm256_load_pd(sx_im+24);
            mqin_im[7]   = _mm256_load_pd(sx_im+28);
            mqin_im[8]   = _mm256_load_pd(sx_im+32);


            out1_re[0] = _mm256_add_pd(mqin_re[0],mqin_im[6]);            
            out1_re[1] = _mm256_add_pd(mqin_re[1],mqin_im[7]);            
            out1_re[2] = _mm256_add_pd(mqin_re[2],mqin_im[8]);            
            out1_im[0] = _mm256_sub_pd(mqin_im[0],mqin_re[6]);            
            out1_im[1] = _mm256_sub_pd(mqin_im[1],mqin_re[7]);            
            out1_im[2] = _mm256_sub_pd(mqin_im[2],mqin_re[8]);


            mqin_re[3]  = _mm256_load_pd(sx_re+12);
            mqin_re[4]  = _mm256_load_pd(sx_re+16);
            mqin_re[5]  = _mm256_load_pd(sx_re+20);
            mqin_im[3]  = _mm256_load_pd(sx_im+12);
            mqin_im[4]  = _mm256_load_pd(sx_im+16);
            mqin_im[5]  = _mm256_load_pd(sx_im+20);


            mqin_re[9]   = _mm256_load_pd(sx_re+36);
            mqin_re[10]  = _mm256_load_pd(sx_re+40);
            mqin_re[11]  = _mm256_load_pd(sx_re+44);
            mqin_im[9]   = _mm256_load_pd(sx_im+36);
            mqin_im[10]  = _mm256_load_pd(sx_im+40);
            mqin_im[11]  = _mm256_load_pd(sx_im+44);


            out2_re[0] = _mm256_sub_pd(mqin_re[3],mqin_im[9]);            
            out2_re[1] = _mm256_sub_pd(mqin_re[4],mqin_im[10]);            
            out2_re[2] = _mm256_sub_pd(mqin_re[5],mqin_im[11]);            
            out2_im[0] = _mm256_add_pd(mqin_im[3],mqin_re[9]);            
            out2_im[1] = _mm256_add_pd(mqin_im[4],mqin_re[10]);            
            out2_im[2] = _mm256_add_pd(mqin_im[5],mqin_re[11]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm256_load_pd(ux_re+jc*4+ic*12);
                  U_im[ic][jc] = _mm256_load_pd(ux_im+jc*4+ic*12);
               }
            
            su3_inverse_multiply_splitlayout_256(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_256(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            mqout_re[0] = _mm256_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm256_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm256_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm256_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm256_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm256_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm256_sub_pd(mqout_re[6] ,map0_im[0]);
            mqout_re[7] = _mm256_sub_pd(mqout_re[7] ,map0_im[1]);
            mqout_re[8] = _mm256_sub_pd(mqout_re[8] ,map0_im[2]);
            mqout_re[9] = _mm256_add_pd(mqout_re[9] ,map1_im[0]);
            mqout_re[10]= _mm256_add_pd(mqout_re[10],map1_im[1]);
            mqout_re[11]= _mm256_add_pd(mqout_re[11],map1_im[2]);

            mqout_im[0] = _mm256_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm256_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm256_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm256_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm256_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm256_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm256_add_pd(mqout_im[6] ,map0_re[0]);
            mqout_im[7] = _mm256_add_pd(mqout_im[7] ,map0_re[1]);
            mqout_im[8] = _mm256_add_pd(mqout_im[8] ,map0_re[2]);
            mqout_im[9] = _mm256_sub_pd(mqout_im[9] ,map1_re[0]);
            mqout_im[10]= _mm256_sub_pd(mqout_im[10],map1_re[1]);
            mqout_im[11]= _mm256_sub_pd(mqout_im[11],map1_re[2]);


            //=======================
            //===== -3  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_avx_split[ipt][3]*144;
            ux_im = u_im + plqcd_g.idn_avx_split[ipt][3]*144;

            sx_re = qin_re + plqcd_g.idn_avx_split[ipt][3]*48;
            sx_im = qin_im + plqcd_g.idn_avx_split[ipt][3]*48;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm256_load_pd(sx_re);
            mqin_re[1]  = _mm256_load_pd(sx_re+4);
            mqin_re[2]  = _mm256_load_pd(sx_re+8);
            mqin_im[0]  = _mm256_load_pd(sx_im);
            mqin_im[1]  = _mm256_load_pd(sx_im+4);
            mqin_im[2]  = _mm256_load_pd(sx_im+8);


            mqin_re[6]   = _mm256_load_pd(sx_re+24);
            mqin_re[7]   = _mm256_load_pd(sx_re+28);
            mqin_re[8]   = _mm256_load_pd(sx_re+32);
            mqin_im[6]   = _mm256_load_pd(sx_im+24);
            mqin_im[7]   = _mm256_load_pd(sx_im+28);
            mqin_im[8]   = _mm256_load_pd(sx_im+32);


            out1_re[0] = _mm256_sub_pd(mqin_re[0],mqin_re[6]);            
            out1_re[1] = _mm256_sub_pd(mqin_re[1],mqin_re[7]);            
            out1_re[2] = _mm256_sub_pd(mqin_re[2],mqin_re[8]);            
            out1_im[0] = _mm256_sub_pd(mqin_im[0],mqin_im[6]);            
            out1_im[1] = _mm256_sub_pd(mqin_im[1],mqin_im[7]);            
            out1_im[2] = _mm256_sub_pd(mqin_im[2],mqin_im[8]);


            mqin_re[3]  = _mm256_load_pd(sx_re+12);
            mqin_re[4]  = _mm256_load_pd(sx_re+16);
            mqin_re[5]  = _mm256_load_pd(sx_re+20);
            mqin_im[3]  = _mm256_load_pd(sx_im+12);
            mqin_im[4]  = _mm256_load_pd(sx_im+16);
            mqin_im[5]  = _mm256_load_pd(sx_im+20);


            mqin_re[9]   = _mm256_load_pd(sx_re+36);
            mqin_re[10]  = _mm256_load_pd(sx_re+40);
            mqin_re[11]  = _mm256_load_pd(sx_re+44);
            mqin_im[9]   = _mm256_load_pd(sx_im+36);
            mqin_im[10]  = _mm256_load_pd(sx_im+40);
            mqin_im[11]  = _mm256_load_pd(sx_im+44);


            out2_re[0] = _mm256_sub_pd(mqin_re[3],mqin_re[9]);            
            out2_re[1] = _mm256_sub_pd(mqin_re[4],mqin_re[10]);            
            out2_re[2] = _mm256_sub_pd(mqin_re[5],mqin_re[11]);            
            out2_im[0] = _mm256_sub_pd(mqin_im[3],mqin_im[9]);            
            out2_im[1] = _mm256_sub_pd(mqin_im[4],mqin_im[10]);            
            out2_im[2] = _mm256_sub_pd(mqin_im[5],mqin_im[11]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm256_load_pd(ux_re+jc*4+ic*12);
                  U_im[ic][jc] = _mm256_load_pd(ux_im+jc*4+ic*12);
               }
            
            su3_inverse_multiply_splitlayout_256(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_256(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            mqout_re[0] = _mm256_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm256_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm256_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm256_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm256_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm256_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm256_sub_pd(mqout_re[6] ,map0_re[0]);
            mqout_re[7] = _mm256_sub_pd(mqout_re[7] ,map0_re[1]);
            mqout_re[8] = _mm256_sub_pd(mqout_re[8] ,map0_re[2]);
            mqout_re[9] = _mm256_sub_pd(mqout_re[9] ,map1_re[0]);
            mqout_re[10]= _mm256_sub_pd(mqout_re[10],map1_re[1]);
            mqout_re[11]= _mm256_sub_pd(mqout_re[11],map1_re[2]);

            mqout_im[0] = _mm256_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm256_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm256_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm256_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm256_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm256_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm256_sub_pd(mqout_im[6] ,map0_im[0]);
            mqout_im[7] = _mm256_sub_pd(mqout_im[7] ,map0_im[1]);
            mqout_im[8] = _mm256_sub_pd(mqout_im[8] ,map0_im[2]);
            mqout_im[9] = _mm256_sub_pd(mqout_im[9] ,map1_im[0]);
            mqout_im[10]= _mm256_sub_pd(mqout_im[10],map1_im[1]);
            mqout_im[11]= _mm256_sub_pd(mqout_im[11],map1_im[2]);



            //_mm512_stream_pd(gx_re       , mqout_re[0]);
            _mm256_store_pd(gx_re       , mqout_re[0]);
            _mm256_store_pd(gx_re+4     , mqout_re[1]);
            _mm256_store_pd(gx_re+8    , mqout_re[2]);
            _mm256_store_pd(gx_re+12    , mqout_re[3]);
            _mm256_store_pd(gx_re+16    , mqout_re[4]);
            _mm256_store_pd(gx_re+20    , mqout_re[5]);
            _mm256_store_pd(gx_re+24    , mqout_re[6]);
            _mm256_store_pd(gx_re+28    , mqout_re[7]);
            _mm256_store_pd(gx_re+32    , mqout_re[8]);
            _mm256_store_pd(gx_re+36    , mqout_re[9]);
            _mm256_store_pd(gx_re+40    , mqout_re[10]);
            _mm256_store_pd(gx_re+44    , mqout_re[11]);

            _mm256_store_pd(gx_im       , mqout_im[0]);
            _mm256_store_pd(gx_im+4     , mqout_im[1]);
            _mm256_store_pd(gx_im+8    , mqout_im[2]);
            _mm256_store_pd(gx_im+12    , mqout_im[3]);
            _mm256_store_pd(gx_im+16    , mqout_im[4]);
            _mm256_store_pd(gx_im+20    , mqout_im[5]);
            _mm256_store_pd(gx_im+24    , mqout_im[6]);
            _mm256_store_pd(gx_im+28    , mqout_im[7]);
            _mm256_store_pd(gx_im+32    , mqout_im[8]);
            _mm256_store_pd(gx_im+36    , mqout_im[9]);
            _mm256_store_pd(gx_im+40    , mqout_im[10]);
            _mm256_store_pd(gx_im+44    , mqout_im[11]);

      }

   }
#ifdef _OPENMP
}
#endif  //end of the openmp parallel reigon

   
   return stop_watch(ts);
}

#endif  //AVX_SPLIT

#endif //AVX



#ifdef MIC
//========================================================================================
//=======================================  2   ===========================================
//=======================           MIC with intrinsics       ============================
//========================================================================================

//===================================EO===================================================
double plqcd_hopping_matrix_eo_intrin_512(spinor_512 *qin, spinor_512 *qout, su3_512 *u)
{

   int snd_err[8],rcv_err[8];
   MPI_Status rcv_mpi_stat[8],snd_mpi_stat[8];
   if(itags==0)
      get_tags(); //get the permanent tags for communications

   double ts; //timer

   ts=stop_watch(0.0); //start
   
   #ifdef _OPENMP
   #pragma omp parallel private(sign)
   {
   #endif
      sign=_mm512_load_pd(dsign);
      int alpha,k1,k2,j1,j2;
      int i,j,k,mu,V,face[4];
      V = plqcd_g.VOLUME;
      for(i=0; i<4; i++)
        face[i] = plqcd_g.face[i];
   
      __m512d qins0[3],qins1[3],qins2[3],qins3[3], U[3][3];
      __m512d in1[3],out[3];
      __m512d map0[3],map1[3],mq0[3],mq1[3],mq2[3],mq3[3];

      su3_vector_512 v0_512  __attribute__ ((aligned (64))) ;
      su3_vector_512 v1_512  __attribute__ ((aligned (64))) ;

      //check if the dimensions allows for using compact data representation
      if((V%8) != 0 ){ //V/2 must be a factor of 4
          fprintf(stderr,"Volume must be a factor of 8\n");
          exit(1);
      }

    
      int iup0[4],iup1[4],iup2[4],iup3[4],idn0[4],idn1[4],idn2[4],idn3[4];
      int m0[4],m1[4],m2[4],m3[4],l0[4],l1[4],l2[4],l3[4];    
  
      su3_512 *ub0;


      //------------------------------------------------
      // compute (1-gamma_mu)qin terms and store in phip
      //------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for 
      #endif  
      for(i=V/8; i<V/4; i ++)
      {
         k=i*4;
         #ifdef MIC_PREFETCH
         intrin_prefetch_spinor_512(&qin[i]);
         #endif

         //idnx[y] means nn in the -ve x direction for the component y of the 512 structure
         //nn of the first component of the quartet in the -ve direction (for the 4 directions)
         idn0[0]=plqcd_g.idn[k][0];
         idn1[0]=plqcd_g.idn[k][1];
         idn2[0]=plqcd_g.idn[k][2];
         idn3[0]=plqcd_g.idn[k][3];
         //nn of the second component of the quartet in the -ve direction(for the four directions)
         idn0[1]=plqcd_g.idn[k+1][0];
         idn1[1]=plqcd_g.idn[k+1][1];
         idn2[1]=plqcd_g.idn[k+1][2];
         idn3[1]=plqcd_g.idn[k+1][3];
         //nn of the third component of the quartet in the -ve direction(for the four directions)
         idn0[2]=plqcd_g.idn[k+2][0];
         idn1[2]=plqcd_g.idn[k+2][1];
         idn2[2]=plqcd_g.idn[k+2][2];
         idn3[2]=plqcd_g.idn[k+2][3];
         //nn of the fourth component of the quartet in the -ve direction(for the four directions)
         idn0[3]=plqcd_g.idn[k+3][0];
         idn1[3]=plqcd_g.idn[k+3][1];
         idn2[3]=plqcd_g.idn[k+3][2];
         idn3[3]=plqcd_g.idn[k+3][3];

         //mapping indices from 128 representation to 512 representation 
         for(j=0; j<4; j++){
            //which 512 element?
            m0[j] = idn0[j]/4; 
            m1[j] = idn1[j]/4; 
            m2[j] = idn2[j]/4;
            m3[j] = idn3[j]/4;
            //which component
            l0[j] = idn0[j]%4;
            l1[j] = idn1[j]%4; 
            l2[j] = idn2[j]%4;
            l3[j] = idn3[j]%4;
            
         }


         intrin_vector_load_512(qins0,&qin[i].s0);
         intrin_vector_load_512(qins1,&qin[i].s1);
         intrin_vector_load_512(qins2,&qin[i].s2);
         intrin_vector_load_512(qins3,&qin[i].s3);

         //-- 0 direction ---
         //_prefetch_halfspinor(&plqcd_g.phip[1][idn1[0]]);
         //_prefetch_halfspinor(&plqcd_g.phip[1][idn1[1]]);
         //_vector_add(plqcd_g.phip[0][idn0].s0, qin[i].s0, qin[i].s2);
         //_vector_add(plqcd_g.phip[0][idn0].s1, qin[i].s1, qin[i].s3);
         intrin_vector_add_512(out,qins0,qins2);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[0][idn0[0]].s0, &plqcd_g.phip[0][idn0[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[0][m0[j]].s0.c0.z[l0[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[0][m0[j]].s0.c1.z[l0[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[0][m0[j]].s0.c2.z[l0[j]] = v1_512.c2.z[j];
         }
         intrin_vector_add_512(out,qins1,qins3);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[0][idn0[0]].s1, &plqcd_g.phip[0][idn0[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[0][m0[j]].s1.c0.z[l0[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[0][m0[j]].s1.c1.z[l0[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[0][m0[j]].s1.c2.z[l0[j]] = v1_512.c2.z[j];
         }


        
         //-- 1 direction --
         //_prefetch_halfspinor(&plqcd_g.phip[2][idn2[0]]);
         //_prefetch_halfspinor(&plqcd_g.phip[2][idn2[1]]);
         //_vector_i_add(plqcd_g.phip[1][idn1].s0, qin[i].s0, qin[i].s3);
         //_vector_i_add(plqcd_g.phip[1][idn1].s1, qin[i].s1, qin[i].s2);
         intrin_vector_i_add_512(out,qins0,qins3);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[1][idn1[0]].s0, &plqcd_g.phip[1][idn1[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[1][m1[j]].s0.c0.z[l1[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[1][m1[j]].s0.c1.z[l1[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[1][m1[j]].s0.c2.z[l1[j]] = v1_512.c2.z[j];
         }

         intrin_vector_i_add_512(out,qins1,qins2);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[1][idn1[0]].s1, &plqcd_g.phip[1][idn1[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[1][m1[j]].s1.c0.z[l1[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[1][m1[j]].s1.c1.z[l1[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[1][m1[j]].s1.c2.z[l1[j]] = v1_512.c2.z[j];
         }


         //-- 2 direction --
         //_prefetch_halfspinor(&plqcd_g.phip[3][idn3[0]]);
         //_prefetch_halfspinor(&plqcd_g.phip[3][idn3[1]]);
         //_vector_add(plqcd_g.phip[2][idn2].s0, qin[i].s0, qin[i].s3);
         //_vector_sub(plqcd_g.phip[2][idn2].s1, qin[i].s1, qin[i].s2);
         intrin_vector_add_512(out,qins0,qins3);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[2][idn2[0]].s0, &plqcd_g.phip[2][idn2[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[2][m2[j]].s0.c0.z[l2[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[2][m2[j]].s0.c1.z[l2[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[2][m2[j]].s0.c2.z[l2[j]] = v1_512.c2.z[j];
         }

         intrin_vector_sub_512(out,qins1,qins2);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[2][idn2[0]].s1, &plqcd_g.phip[2][idn2[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[2][m2[j]].s1.c0.z[l2[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[2][m2[j]].s1.c1.z[l2[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[2][m2[j]].s1.c2.z[l2[j]] = v1_512.c2.z[j];
         }


         //-- 3 direction --
         //_vector_i_add(plqcd_g.phip[3][idn3].s0, qin[i].s0, qin[i].s2);
         //_vector_i_sub(plqcd_g.phip[3][idn3].s1, qin[i].s1, qin[i].s3);
         intrin_vector_i_add_512(out,qins0,qins2);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[3][idn3[0]].s0, &plqcd_g.phip[3][idn3[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[3][m3[j]].s0.c0.z[l3[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[3][m3[j]].s0.c1.z[l3[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[3][m3[j]].s0.c2.z[l3[j]] = v1_512.c2.z[j];
         }

         intrin_vector_i_sub_512(out,qins1,qins3);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[3][idn3[0]].s1, &plqcd_g.phip[3][idn3[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[3][m3[j]].s1.c0.z[l3[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[3][m3[j]].s1.c1.z[l3[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[3][m3[j]].s1.c2.z[l3[j]] = v1_512.c2.z[j];
         }

      }


      //start sending the buffers to the nearest neighbours in the -ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
            //snd_err[mu] = MPI_Isend(&plqcd_g.phip[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu], MPI_COMM_WORLD, &snd_req[mu]);
            //rcv_err[mu] = MPI_Irecv(&plqcd_g.phip[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_req[mu]);
            snd_err[mu] = MPI_Isend(&plqcd_g.phip512[mu][V/8],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu], MPI_COMM_WORLD, &snd_req[mu]);
            rcv_err[mu] = MPI_Irecv(&plqcd_g.phip512[mu][V/8+face[mu]/8],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_req[mu]);
            if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif


      //----------------------------------------------------------
      // compute U^dagger*(1+gamma_mu)qin terms and store in phim
      //----------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=V/8; i < V/4; i ++)
      {
         k=i*4;
         #ifdef MIC_PREFETCH
         intrin_prefetch_spinor_512(&qin[i]);
         #endif

         //iupx[y] means nn in the +ve x direction for the component y of the 512 structure
         iup0[0]=plqcd_g.iup[k][0];
         iup1[0]=plqcd_g.iup[k][1];
         iup2[0]=plqcd_g.iup[k][2];
         iup3[0]=plqcd_g.iup[k][3];

         iup0[1]=plqcd_g.iup[k+1][0];
         iup1[1]=plqcd_g.iup[k+1][1];
         iup2[1]=plqcd_g.iup[k+1][2];
         iup3[1]=plqcd_g.iup[k+1][3];

         iup0[2]=plqcd_g.iup[k+2][0];
         iup1[2]=plqcd_g.iup[k+2][1];
         iup2[2]=plqcd_g.iup[k+2][2];
         iup3[2]=plqcd_g.iup[k+2][3];

         iup0[3]=plqcd_g.iup[k+3][0];
         iup1[3]=plqcd_g.iup[k+3][1];
         iup2[3]=plqcd_g.iup[k+3][2];
         iup3[3]=plqcd_g.iup[k+3][3];

         //mapping indices from 128 representation to 512 representation 
         for(j=0; j<4; j++){
            //which 512 element?
            m0[j] = iup0[j]/4;
            m1[j] = iup1[j]/4; 
            m2[j] = iup2[j]/4;
            m3[j] = iup3[j]/4;
            //which component of the 512 structure?
            l0[j] = iup0[j]%4;
            l1[j] = iup1[j]%4; 
            l2[j] = iup2[j]%4;
            l3[j] = iup3[j]%4;
         }


         //_prefetch_halfspinor(&plqcd_g.phim[0][iup0[0]]);
         //_prefetch_halfspinor(&plqcd_g.phim[0][iup0[1]]);

         ub0 = &u[4*i];
         intrin_vector_load_512(qins0,&qin[i].s0);
         intrin_vector_load_512(qins1,&qin[i].s1);
         intrin_vector_load_512(qins2,&qin[i].s2);
         intrin_vector_load_512(qins3,&qin[i].s3);

         //-- 0 direction ---
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         #endif
         //_prefetch_halfspinor(&plqcd_g.phim[1][iup1[0]]);
         //_prefetch_halfspinor(&plqcd_g.phim[1][iup1[1]]);
         //_vector_sub(p,qin[i].s0,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s0,*ub0,p);
         //_vector_sub(p,qin[i].s1,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s1,*ub0,p);
         intrin_su3_load_512(U,ub0);
         intrin_vector_sub_512(in1,qins0,qins2);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[0][iup0[0]].s0, &plqcd_g.phim[0][iup0[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[0][m0[j]].s0.c0.z[l0[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[0][m0[j]].s0.c1.z[l0[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[0][m0[j]].s0.c2.z[l0[j]] = v1_512.c2.z[j];
         }

         intrin_vector_sub_512(in1,qins1,qins3);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[0][iup0[0]].s1, &plqcd_g.phim[0][iup0[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[0][m0[j]].s1.c0.z[l0[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[0][m0[j]].s1.c1.z[l0[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[0][m0[j]].s1.c2.z[l0[j]] = v1_512.c2.z[j];
         }
         ub0++;
  
         //-- 1 direction --
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         #endif
         //_prefetch_halfspinor(&plqcd_g.phim[2][iup2[0]]);
         //_prefetch_halfspinor(&plqcd_g.phim[2][iup2[1]]);
         //_vector_i_sub(p,qin[i].s0,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s0,*ub0,p);
         //_vector_i_sub(p,qin[i].s1,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s1,*ub0,p);
         intrin_su3_load_512(U,ub0);
         intrin_vector_i_sub_512(in1,qins0,qins3);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[1][iup1[0]].s0, &plqcd_g.phim[1][iup1[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[1][m1[j]].s0.c0.z[l1[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[1][m1[j]].s0.c1.z[l1[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[1][m1[j]].s0.c2.z[l1[j]] = v1_512.c2.z[j];
         }

         intrin_vector_i_sub_512(in1,qins1,qins2);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[1][iup1[0]].s1, &plqcd_g.phim[1][iup1[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[1][m0[j]].s1.c0.z[l0[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[1][m0[j]].s1.c1.z[l0[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[1][m0[j]].s1.c2.z[l0[j]] = v1_512.c2.z[j];
         }
         ub0++;

         //-- 2 direction --
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         #endif
         //_prefetch_halfspinor(&plqcd_g.phim[3][iup3[0]]);
         //_prefetch_halfspinor(&plqcd_g.phim[3][iup3[1]]);
         //_vector_sub(p,qin[i].s0,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s0,*ub0,p);
         //_vector_add(p,qin[i].s1,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s1,*ub0,p);
         intrin_su3_load_512(U,ub0);
         intrin_vector_sub_512(in1,qins0,qins3);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup2[0]].s0, &plqcd_g.phim[2][iup2[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[2][m2[j]].s0.c0.z[l2[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[2][m2[j]].s0.c1.z[l2[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[2][m2[j]].s0.c2.z[l2[j]] = v1_512.c2.z[j];
         }

         intrin_vector_add_512(in1,qins1,qins2);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup2[0]].s1, &plqcd_g.phim[2][iup2[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[2][m2[j]].s1.c0.z[l2[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[2][m2[j]].s1.c1.z[l2[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[2][m2[j]].s1.c2.z[l2[j]] = v1_512.c2.z[j];
         }
         ub0++;

         //-- 3 direction --
         //_vector_i_sub(p,qin[i].s0,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s0,*ub0,p);
         //_vector_i_add(p,qin[i].s1,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s1,*ub0,p);
         intrin_su3_load_512(U,ub0);
         intrin_vector_i_sub_512(in1,qins0,qins2);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup3[0]].s0, &plqcd_g.phim[2][iup3[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[3][m3[j]].s0.c0.z[l3[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[3][m3[j]].s0.c1.z[l3[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[3][m3[j]].s0.c2.z[l3[j]] = v1_512.c2.z[j];
         }

         intrin_vector_i_add_512(in1,qins1,qins3);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[3][iup3[0]].s1, &plqcd_g.phim[3][iup3[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[3][m3[j]].s1.c0.z[l3[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[3][m3[j]].s1.c1.z[l3[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[3][m3[j]].s1.c2.z[l3[j]] = v1_512.c2.z[j];
         }

      }

      //start sending the buffers to the nearest neighbours in the +ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
            //snd_err[mu+4] = MPI_Isend(&plqcd_g.phim[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD, &snd_req[mu+4]);
            //rcv_err[mu+4] = MPI_Irecv(&plqcd_g.phim[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_req[mu+4]);
            snd_err[mu+4] = MPI_Isend(&plqcd_g.phim512[mu][V/8],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD, &snd_req[mu+4]);
            rcv_err[mu+4] = MPI_Irecv(&plqcd_g.phim512[mu][V/8+face[mu]/8],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_req[mu+4]);
            if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
               exit(1);}
         }
      }
      
      //wait for the communications of phip to finish
      for(mu=0; mu<4; mu++)
      {
         if(plqcd_g.nprocs[mu]>1)
         {   
            rcv_err[mu]=MPI_Wait(&rcv_req[mu],&rcv_mpi_stat[mu]);
            snd_err[mu]=MPI_Wait(&snd_req[mu],&snd_mpi_stat[mu]);
            if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Wait\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif
     

      //-------------------------------------------------------
      //complete computation of the U_mu*(1-gamma_m)*qin terms
      //-------------------------------------------------------
      //copy the exchanged boundaries to the correspoding locations on the local phip fields
      for(mu=0; mu<4; mu++)
      { 
         if(plqcd_g.nprocs[mu] > 1)
         {  
            #ifdef _OPENMP 
            #pragma omp for 
            #endif  
            //for(i=0; i< face[mu]/2; i++)
            //{
                //can we prefetch here
            //    j=V/2+face[mu]/2+i;
            //    k=plqcd_g.nn_bndo[2*mu][i];
            //    _vector_assign(plqcd_g.phip[mu][k].s0 , plqcd_g.phip[mu][j].s0);
            //    _vector_assign(plqcd_g.phip[mu][k].s1 , plqcd_g.phip[mu][j].s1);
            //}
            for(i=0; i< face[mu]/2; i +=4)
            {
                for(alpha=0; alpha<4; alpha++)
                {
                   j=V/2+face[mu]/2+i+alpha;
                   k=plqcd_g.nn_bndo[2*mu][i+alpha];
                   j1=j/4; 
                   j2=j%4;
                   k1=k/4;
                   k2=k%4;
                   plqcd_g.phip512[mu][k1].s0.c0.z[k2] = plqcd_g.phip512[mu][j1].s0.c0.z[j2];
                   plqcd_g.phip512[mu][k1].s0.c1.z[k2] = plqcd_g.phip512[mu][j1].s0.c1.z[j2];
                   plqcd_g.phip512[mu][k1].s0.c2.z[k2] = plqcd_g.phip512[mu][j1].s0.c2.z[j2];
                   plqcd_g.phip512[mu][k1].s1.c0.z[k2] = plqcd_g.phip512[mu][j1].s1.c0.z[j2];
                   plqcd_g.phip512[mu][k1].s1.c1.z[k2] = plqcd_g.phip512[mu][j1].s1.c1.z[j2];
                   plqcd_g.phip512[mu][k1].s1.c2.z[k2] = plqcd_g.phip512[mu][j1].s1.c2.z[j2];
                }
            }
         }
      } 



      #ifdef _OPENMP
      #pragma omp for 
      #endif 
      for(i=0; i< V/8; i ++)
      {

         ub0 = &u[4*i];
   
         // +0
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         intrin_prefetch_halfspinor_512(&plqcd_g.phip512[1][i]);
         #endif
         //_su3_multiply(q0, *ub0, plqcd_g.phip[0][i].s0);
         //_su3_multiply(q1, *ub0, plqcd_g.phip[0][i].s1);
         //_vector_assign(q2, q0);
         //_vector_assign(q3, q1);
         intrin_su3_load_512(U,ub0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[0][i].s0, &plqcd_g.phip[0][i+1].s0);
         intrin_vector_load_512(in1,&plqcd_g.phip512[0][i].s0);
         intrin_su3_multiply_512(mq0,U,in1);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[0][i].s1, &plqcd_g.phip[0][i+1].s1);
         intrin_vector_load_512(in1,&plqcd_g.phip512[0][i].s1);
         intrin_su3_multiply_512(mq1,U,in1);
         for(j=0; j<3; j++){
            mq2[j]= mq0[j];
            mq3[j]= mq1[j];}
         ub0++;

         // +1
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         intrin_prefetch_halfspinor_512(&plqcd_g.phip512[2][i]);
         #endif
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[1][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[1][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2, -1.0, ap1);
         //_vector_add_i_mul(q3, -1.0, ap0);
         intrin_su3_load_512(U,ub0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[1][i].s0, &plqcd_g.phip[1][i+1].s0);
         intrin_vector_load_512(in1,&plqcd_g.phip512[1][i].s0);
         intrin_su3_multiply_512(map0,U,in1);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[1][i].s1, &plqcd_g.phip[1][i+1].s1);
         intrin_vector_load_512(in1,&plqcd_g.phip512[1][i].s1);
         intrin_su3_multiply_512(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm512_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm512_add_pd(mq1[j],map1[j]);
            mq2[j] = complex_i_sub_regs_512(mq2[j],map1[j]);
            mq3[j] = complex_i_sub_regs_512(mq3[j],map0[j]);}
         ub0++;        

         // +2
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         intrin_prefetch_halfspinor_512(&plqcd_g.phip512[3][i]);
         #endif
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[2][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[2][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_sub_assign(q2,ap1);
         //_vector_add_assign(q3,ap0);
         intrin_su3_load_512(U,ub0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[2][i].s0, &plqcd_g.phip[2][i+1].s0);
         intrin_vector_load_512(in1,&plqcd_g.phip512[2][i].s0);
         intrin_su3_multiply_512(map0,U,in1);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[2][i].s1, &plqcd_g.phip[2][i+1].s1);
         intrin_vector_load_512(in1,&plqcd_g.phip512[2][i].s1);
         intrin_su3_multiply_512(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm512_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm512_add_pd(mq1[j],map1[j]);
            mq2[j] = _mm512_sub_pd(mq2[j],map1[j]);
            mq3[j] = _mm512_add_pd(mq3[j],map0[j]);}
         ub0++;        

         #ifdef MIC_PREFETCH
         intrin_prefetch_spinor_512(&qout[i]);
         #endif
         // +3
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[3][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[3][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2,-1.0, ap0);
         //_vector_add_i_mul(q3, 1.0, ap1);
         intrin_su3_load_512(U,ub0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[3][i].s0, &plqcd_g.phip[3][i+1].s0);
         intrin_vector_load_512(in1,&plqcd_g.phip512[3][i].s0);
         intrin_su3_multiply_512(map0,U,in1);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[3][i].s1, &plqcd_g.phip[3][i+1].s1);
         intrin_vector_load_512(in1,&plqcd_g.phip512[3][i].s1);
         intrin_su3_multiply_512(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm512_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm512_add_pd(mq1[j],map1[j]);
            mq2[j] = complex_i_sub_regs_512(mq2[j],map0[j]);
            mq3[j] = complex_i_add_regs_512(mq3[j],map1[j]);}

 
         //store the result
         //_vector_assign(qout[i].s0,q0);
         //_vector_assign(qout[i].s1,q1);
         //_vector_assign(qout[i].s2,q2);
         //_vector_assign(qout[i].s3,q3);
         intrin_vector_store_512(&qout[i].s0,mq0);         
         intrin_vector_store_512(&qout[i].s1,mq1);         
         intrin_vector_store_512(&qout[i].s2,mq1);         
         intrin_vector_store_512(&qout[i].s3,mq1);         

      } 

      //wait for the communications of phim to finish
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu<4; mu++)
      {
         if(plqcd_g.nprocs[mu]>1)
         {   
            rcv_err[mu+4]=MPI_Wait(&rcv_req[mu+4],&rcv_mpi_stat[mu+4]);
            snd_err[mu+4]=MPI_Wait(&snd_req[mu+4],&snd_mpi_stat[mu+4]);
            if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Wait\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif

      //copy the exchanged boundaries to the correspoding locations on the local phim fields
      for(mu=0; mu<4; mu++)
      { 
         if(plqcd_g.nprocs[mu] > 1)
         {
            #ifdef _OPENMP   
            #pragma omp for
            #endif
            for(i=0; i< face[mu]/2; i +=4)
            {
               //j=V/2+face[mu]/2+i;
               //k=plqcd_g.nn_bndo[2*mu+1][i];
               //_vector_assign(plqcd_g.phim[mu][k].s0, plqcd_g.phim[mu][j].s0);
               //_vector_assign(plqcd_g.phim[mu][k].s1, plqcd_g.phim[mu][j].s1);
               for(alpha=0; alpha<4; alpha++)
               {
                   j=V/2+face[mu]/2+i+alpha;
                   k=plqcd_g.nn_bndo[2*mu+1][i+alpha];
                   j1=j/4;
                   j2=j%4;
                   k1=k/4;
                   k2=k%4;
                   plqcd_g.phim512[mu][k1].s0.c0.z[k2] = plqcd_g.phim512[mu][j1].s0.c0.z[j2];
                   plqcd_g.phim512[mu][k1].s0.c1.z[k2] = plqcd_g.phim512[mu][j1].s0.c1.z[j2];
                   plqcd_g.phim512[mu][k1].s0.c2.z[k2] = plqcd_g.phim512[mu][j1].s0.c2.z[j2];
                   plqcd_g.phim512[mu][k1].s1.c0.z[k2] = plqcd_g.phim512[mu][j1].s1.c0.z[j2];
                   plqcd_g.phim512[mu][k1].s1.c1.z[k2] = plqcd_g.phim512[mu][j1].s1.c1.z[j2];
                   plqcd_g.phim512[mu][k1].s1.c2.z[k2] = plqcd_g.phim512[mu][j1].s1.c2.z[j2];
                }
            }
         }
      }


      //---------------------------------------------------------------------
      //finish computation of the U^dagger*(1+gamma_mu)*qin
      //---------------------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif
      for(i=0; i< V/8; i ++)
      {

         intrin_vector_load_512(mq0,&qout[i].s0);
         intrin_vector_load_512(mq1,&qout[i].s1);
         intrin_vector_load_512(mq2,&qout[i].s2);
         intrin_vector_load_512(mq3,&qout[i].s3);

         // 0 direction
         #ifdef MIC_PREFETCH
         intrin_prefetch_halfspinor_512(&plqcd_g.phim512[1][i]);
         #endif
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[0][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[0][i].s1);
         //_vector_minus_assign(qout[i].s2, plqcd_g.phim[0][i].s0);
         //_vector_minus_assign(qout[i].s3, plqcd_g.phim[0][i].s1);
         //copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[0][i].s0, &plqcd_g.phim[0][i+1].s0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[0][i].s1, &plqcd_g.phim[0][i+1].s1);
         intrin_vector_load_512(map0,&plqcd_g.phim512[0][i].s0);
         intrin_vector_load_512(map1,&plqcd_g.phim512[0][i].s1);
         intrin_vector_add_512(mq0,mq0,map0);
         intrin_vector_add_512(mq1,mq1,map1);
         intrin_vector_sub_512(mq2,mq2,map0);
         intrin_vector_sub_512(mq3,mq3,map1);

         // 1 direction
         #ifdef MIC_PREFETCH
         intrin_prefetch_halfspinor_512(&plqcd_g.phim512[2][i]);
         #endif
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[1][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s2, 1.0,plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s3, 1.0, plqcd_g.phim[1][i].s0);
         //copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[1][i].s0, &plqcd_g.phim[1][i+1].s0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[1][i].s1, &plqcd_g.phim[1][i+1].s1);
         intrin_vector_load_512(map0,&plqcd_g.phim512[1][i].s0);
         intrin_vector_load_512(map1,&plqcd_g.phim512[1][i].s1);
         intrin_vector_add_512(mq0,mq0,map0);
         intrin_vector_add_512(mq1,mq1,map1);
         intrin_vector_i_add_512(mq2,mq2,map1);
         intrin_vector_i_add_512(mq3,mq3,map0);
        

         // 2 direction
         #ifdef MIC_PREFETCH
         intrin_prefetch_halfspinor_512(&plqcd_g.phim512[3][i]);
         #endif
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[2][i].s0);
         //_vector_add_assign(qout[i].s1,  plqcd_g.phim[2][i].s1 );
         //_vector_add_assign(qout[i].s2,  plqcd_g.phim[2][i].s1);
         //_vector_sub_assign(qout[i].s3,   plqcd_g.phim[2][i].s0);
         //copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[2][i].s0, &plqcd_g.phim[2][i+1].s0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[2][i].s1, &plqcd_g.phim[2][i+1].s1);
         intrin_vector_load_512(map0,&plqcd_g.phim512[2][i].s0);
         intrin_vector_load_512(map1,&plqcd_g.phim512[2][i].s1);
         intrin_vector_add_512(mq0,mq0,map0);
         intrin_vector_add_512(mq1,mq1,map1);
         intrin_vector_add_512(mq2,mq2,map1);
         intrin_vector_sub_512(mq3,mq3,map0);
         

         //3 direction
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[3][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[3][i].s1);
         //_vector_i_add_assign(qout[i].s2, plqcd_g.phim[3][i].s0);
         //_vector_i_sub_assign(qout[i].s3, plqcd_g.phim[3][i].s1);     
         //copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[3][i].s0, &plqcd_g.phim[3][i+1].s0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[3][i].s1, &plqcd_g.phim[3][i+1].s1);
         intrin_vector_load_512(map0,&plqcd_g.phim512[3][i].s0);
         intrin_vector_load_512(map1,&plqcd_g.phim512[3][i].s1);
         intrin_vector_add_512(mq0,mq0,map0);
         intrin_vector_add_512(mq1,mq1,map1);
         intrin_vector_i_add_512(mq2,mq2,map0);
         intrin_vector_i_add_512(mq3,mq3,map1);

         //store the result
         intrin_vector_store_512(&qout[i].s0, mq0);
         intrin_vector_store_512(&qout[i].s1, mq1);
         intrin_vector_store_512(&qout[i].s2, mq2);
         intrin_vector_store_512(&qout[i].s3, mq3);
   }
#ifdef _OPENMP
}
#endif  //end of the openmp parallel reigon

   
   return stop_watch(ts);
}


//===================================OE===================================================
double plqcd_hopping_matrix_oe_intrin_512(spinor_512 *qin, spinor_512 *qout, su3_512 *u)
{

   int snd_err[8],rcv_err[8];
   MPI_Status rcv_mpi_stat[8],snd_mpi_stat[8];
   if(itags==0)
      get_tags(); //get the permanent tags for communications

   double ts; //timer

   ts=stop_watch(0.0); //start
   
   #ifdef _OPENMP
   #pragma omp parallel
   {
   #endif
      sign=_mm512_load_pd(dsign);
      int alpha,k1,k2,j1,j2;
      int i,j,k,mu,V,face[4];
      V = plqcd_g.VOLUME;
      for(i=0; i<4; i++)
        face[i] = plqcd_g.face[i];
   
      __m512d qins0[3],qins1[3],qins2[3],qins3[3], U[3][3];
      __m512d in1[3],out[3];
      __m512d map0[3],map1[3],mq0[3],mq1[3],mq2[3],mq3[3];

      su3_vector_512 v0_512  __attribute__ ((aligned (64))) ;
      su3_vector_512 v1_512  __attribute__ ((aligned (64))) ;

      //check if the dimensions allows for using compact data representation
      if((V%8) != 0 ){ //V/2 must be a factor of 4
          fprintf(stderr,"Volume must be a factor of 8\n");
          exit(1);
      }

    
      int iup0[4],iup1[4],iup2[4],iup3[4],idn0[4],idn1[4],idn2[4],idn3[4];
      int m0[4],m1[4],m2[4],m3[4],l0[4],l1[4],l2[4],l3[4];    
  
      su3_512 *ub0;


      //------------------------------------------------
      // compute (1-gamma_mu)qin terms and store in phip
      //------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for 
      #endif  
      for(i=0; i<V/8; i ++)
      {
         k=i*4;
         #ifdef MIC_PREFETCH
         intrin_prefetch_spinor_512(&qin[i]);
         #endif

         //idnx[y] means nn in the -ve x direction for the component y of the 512 structure
         //nn of the first component of the quartet in the -ve direction (for the 4 directions)
         idn0[0]=plqcd_g.idn[k][0]-V/2;
         idn1[0]=plqcd_g.idn[k][1]-V/2;
         idn2[0]=plqcd_g.idn[k][2]-V/2;
         idn3[0]=plqcd_g.idn[k][3]-V/2;
         //nn of the second component of the quartet in the -ve direction(for the four directions)
         idn0[1]=plqcd_g.idn[k+1][0]-V/2;
         idn1[1]=plqcd_g.idn[k+1][1]-V/2;
         idn2[1]=plqcd_g.idn[k+1][2]-V/2;
         idn3[1]=plqcd_g.idn[k+1][3]-V/2;
         //nn of the third component of the quartet in the -ve direction(for the four directions)
         idn0[2]=plqcd_g.idn[k+2][0]-V/2;
         idn1[2]=plqcd_g.idn[k+2][1]-V/2;
         idn2[2]=plqcd_g.idn[k+2][2]-V/2;
         idn3[2]=plqcd_g.idn[k+2][3]-V/2;
         //nn of the fourth component of the quartet in the -ve direction(for the four directions)
         idn0[3]=plqcd_g.idn[k+3][0]-V/2;
         idn1[3]=plqcd_g.idn[k+3][1]-V/2;
         idn2[3]=plqcd_g.idn[k+3][2]-V/2;
         idn3[3]=plqcd_g.idn[k+3][3]-V/2;

         //mapping indices from 128 representation to 512 representation 
         for(j=0; j<4; j++){
            //which 512 element?
            m0[j] = idn0[j]/4; 
            m1[j] = idn1[j]/4; 
            m2[j] = idn2[j]/4;
            m3[j] = idn3[j]/4;
            //which component
            l0[j] = idn0[j]%4;
            l1[j] = idn1[j]%4; 
            l2[j] = idn2[j]%4;
            l3[j] = idn3[j]%4;
            
         }


         intrin_vector_load_512(qins0,&qin[i].s0);
         intrin_vector_load_512(qins1,&qin[i].s1);
         intrin_vector_load_512(qins2,&qin[i].s2);
         intrin_vector_load_512(qins3,&qin[i].s3);

         //-- 0 direction ---
         //_prefetch_halfspinor(&plqcd_g.phip[1][idn1[0]]);
         //_prefetch_halfspinor(&plqcd_g.phip[1][idn1[1]]);
         //_vector_add(plqcd_g.phip[0][idn0].s0, qin[i].s0, qin[i].s2);
         //_vector_add(plqcd_g.phip[0][idn0].s1, qin[i].s1, qin[i].s3);
         intrin_vector_add_512(out,qins0,qins2);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[0][idn0[0]].s0, &plqcd_g.phip[0][idn0[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[0][m0[j]].s0.c0.z[l0[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[0][m0[j]].s0.c1.z[l0[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[0][m0[j]].s0.c2.z[l0[j]] = v1_512.c2.z[j];
         }
         intrin_vector_add_512(out,qins1,qins3);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[0][idn0[0]].s1, &plqcd_g.phip[0][idn0[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[0][m0[j]].s1.c0.z[l0[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[0][m0[j]].s1.c1.z[l0[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[0][m0[j]].s1.c2.z[l0[j]] = v1_512.c2.z[j];
         }


        
         //-- 1 direction --
         //_prefetch_halfspinor(&plqcd_g.phip[2][idn2[0]]);
         //_prefetch_halfspinor(&plqcd_g.phip[2][idn2[1]]);
         //_vector_i_add(plqcd_g.phip[1][idn1].s0, qin[i].s0, qin[i].s3);
         //_vector_i_add(plqcd_g.phip[1][idn1].s1, qin[i].s1, qin[i].s2);
         intrin_vector_i_add_512(out,qins0,qins3);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[1][idn1[0]].s0, &plqcd_g.phip[1][idn1[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[1][m1[j]].s0.c0.z[l1[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[1][m1[j]].s0.c1.z[l1[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[1][m1[j]].s0.c2.z[l1[j]] = v1_512.c2.z[j];
         }

         intrin_vector_i_add_512(out,qins1,qins2);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[1][idn1[0]].s1, &plqcd_g.phip[1][idn1[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[1][m1[j]].s1.c0.z[l1[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[1][m1[j]].s1.c1.z[l1[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[1][m1[j]].s1.c2.z[l1[j]] = v1_512.c2.z[j];
         }


         //-- 2 direction --
         //_prefetch_halfspinor(&plqcd_g.phip[3][idn3[0]]);
         //_prefetch_halfspinor(&plqcd_g.phip[3][idn3[1]]);
         //_vector_add(plqcd_g.phip[2][idn2].s0, qin[i].s0, qin[i].s3);
         //_vector_sub(plqcd_g.phip[2][idn2].s1, qin[i].s1, qin[i].s2);
         intrin_vector_add_512(out,qins0,qins3);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[2][idn2[0]].s0, &plqcd_g.phip[2][idn2[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[2][m2[j]].s0.c0.z[l2[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[2][m2[j]].s0.c1.z[l2[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[2][m2[j]].s0.c2.z[l2[j]] = v1_512.c2.z[j];
         }

         intrin_vector_sub_512(out,qins1,qins2);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[2][idn2[0]].s1, &plqcd_g.phip[2][idn2[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[2][m2[j]].s1.c0.z[l2[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[2][m2[j]].s1.c1.z[l2[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[2][m2[j]].s1.c2.z[l2[j]] = v1_512.c2.z[j];
         }


         //-- 3 direction --
         //_vector_i_add(plqcd_g.phip[3][idn3].s0, qin[i].s0, qin[i].s2);
         //_vector_i_sub(plqcd_g.phip[3][idn3].s1, qin[i].s1, qin[i].s3);
         intrin_vector_i_add_512(out,qins0,qins2);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[3][idn3[0]].s0, &plqcd_g.phip[3][idn3[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[3][m3[j]].s0.c0.z[l3[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[3][m3[j]].s0.c1.z[l3[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[3][m3[j]].s0.c2.z[l3[j]] = v1_512.c2.z[j];
         }

         intrin_vector_i_sub_512(out,qins1,qins3);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phip[3][idn3[0]].s1, &plqcd_g.phip[3][idn3[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phip512[3][m3[j]].s1.c0.z[l3[j]] = v1_512.c0.z[j];
            plqcd_g.phip512[3][m3[j]].s1.c1.z[l3[j]] = v1_512.c1.z[j];
            plqcd_g.phip512[3][m3[j]].s1.c2.z[l3[j]] = v1_512.c2.z[j];
         }

      }


      //start sending the buffers to the nearest neighbours in the -ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
            //snd_err[mu] = MPI_Isend(&plqcd_g.phip[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu], MPI_COMM_WORLD, &snd_req[mu]);
            //rcv_err[mu] = MPI_Irecv(&plqcd_g.phip[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_req[mu]);
            snd_err[mu] = MPI_Isend(&plqcd_g.phip512[mu][V/8],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]  , tags[mu], MPI_COMM_WORLD, &snd_req[mu]);
            rcv_err[mu] = MPI_Irecv(&plqcd_g.phip512[mu][V/8+face[mu]/8],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1], tags[mu], MPI_COMM_WORLD, &rcv_req[mu]);
            if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif


      //----------------------------------------------------------
      // compute U^dagger*(1+gamma_mu)qin terms and store in phim
      //----------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif  
      for(i=0; i < V/8; i ++)
      {
         k=i*4;
         #ifdef MIC_PREFETCH
         intrin_prefetch_spinor_512(&qin[i]);
         #endif

         //iupx[y] means nn in the +ve x direction for the component y of the 512 structure
         iup0[0]=plqcd_g.iup[k][0]-V/2;
         iup1[0]=plqcd_g.iup[k][1]-V/2;
         iup2[0]=plqcd_g.iup[k][2]-V/2;
         iup3[0]=plqcd_g.iup[k][3]-V/2;

         iup0[1]=plqcd_g.iup[k+1][0]-V/2;
         iup1[1]=plqcd_g.iup[k+1][1]-V/2;
         iup2[1]=plqcd_g.iup[k+1][2]-V/2;
         iup3[1]=plqcd_g.iup[k+1][3]-V/2;

         iup0[2]=plqcd_g.iup[k+2][0]-V/2;
         iup1[2]=plqcd_g.iup[k+2][1]-V/2;
         iup2[2]=plqcd_g.iup[k+2][2]-V/2;
         iup3[2]=plqcd_g.iup[k+2][3]-V/2;

         iup0[3]=plqcd_g.iup[k+3][0]-V/2;
         iup1[3]=plqcd_g.iup[k+3][1]-V/2;
         iup2[3]=plqcd_g.iup[k+3][2]-V/2;
         iup3[3]=plqcd_g.iup[k+3][3]-V/2;

         //mapping indices from 128 representation to 512 representation 
         for(j=0; j<4; j++){
            //which 512 element?
            m0[j] = iup0[j]/4;
            m1[j] = iup1[j]/4; 
            m2[j] = iup2[j]/4;
            m3[j] = iup3[j]/4;
            //which component of the 512 structure?
            l0[j] = iup0[j]%4;
            l1[j] = iup1[j]%4; 
            l2[j] = iup2[j]%4;
            l3[j] = iup3[j]%4;
         }


         //_prefetch_halfspinor(&plqcd_g.phim[0][iup0[0]]);
         //_prefetch_halfspinor(&plqcd_g.phim[0][iup0[1]]);

         ub0 = &u[4*i];
         intrin_vector_load_512(qins0,&qin[i].s0);
         intrin_vector_load_512(qins1,&qin[i].s1);
         intrin_vector_load_512(qins2,&qin[i].s2);
         intrin_vector_load_512(qins3,&qin[i].s3);

         //-- 0 direction ---
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         #endif
         //_prefetch_halfspinor(&plqcd_g.phim[1][iup1[0]]);
         //_prefetch_halfspinor(&plqcd_g.phim[1][iup1[1]]);
         //_vector_sub(p,qin[i].s0,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s0,*ub0,p);
         //_vector_sub(p,qin[i].s1,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[0][iup0].s1,*ub0,p);
         intrin_su3_load_512(U,ub0);
         intrin_vector_sub_512(in1,qins0,qins2);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[0][iup0[0]].s0, &plqcd_g.phim[0][iup0[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[0][m0[j]].s0.c0.z[l0[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[0][m0[j]].s0.c1.z[l0[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[0][m0[j]].s0.c2.z[l0[j]] = v1_512.c2.z[j];
         }

         intrin_vector_sub_512(in1,qins1,qins3);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[0][iup0[0]].s1, &plqcd_g.phim[0][iup0[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[0][m0[j]].s1.c0.z[l0[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[0][m0[j]].s1.c1.z[l0[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[0][m0[j]].s1.c2.z[l0[j]] = v1_512.c2.z[j];
         }
         ub0++;
  
         //-- 1 direction --
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         #endif
         //_prefetch_halfspinor(&plqcd_g.phim[2][iup2[0]]);
         //_prefetch_halfspinor(&plqcd_g.phim[2][iup2[1]]);
         //_vector_i_sub(p,qin[i].s0,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s0,*ub0,p);
         //_vector_i_sub(p,qin[i].s1,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[1][iup1].s1,*ub0,p);
         intrin_su3_load_512(U,ub0);
         intrin_vector_i_sub_512(in1,qins0,qins3);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[1][iup1[0]].s0, &plqcd_g.phim[1][iup1[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[1][m1[j]].s0.c0.z[l1[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[1][m1[j]].s0.c1.z[l1[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[1][m1[j]].s0.c2.z[l1[j]] = v1_512.c2.z[j];
         }

         intrin_vector_i_sub_512(in1,qins1,qins2);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[1][iup1[0]].s1, &plqcd_g.phim[1][iup1[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[1][m0[j]].s1.c0.z[l0[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[1][m0[j]].s1.c1.z[l0[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[1][m0[j]].s1.c2.z[l0[j]] = v1_512.c2.z[j];
         }
         ub0++;

         //-- 2 direction --
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         #endif
         //_prefetch_halfspinor(&plqcd_g.phim[3][iup3[0]]);
         //_prefetch_halfspinor(&plqcd_g.phim[3][iup3[1]]);
         //_vector_sub(p,qin[i].s0,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s0,*ub0,p);
         //_vector_add(p,qin[i].s1,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[2][iup2].s1,*ub0,p);
         intrin_su3_load_512(U,ub0);
         intrin_vector_sub_512(in1,qins0,qins3);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup2[0]].s0, &plqcd_g.phim[2][iup2[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[2][m2[j]].s0.c0.z[l2[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[2][m2[j]].s0.c1.z[l2[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[2][m2[j]].s0.c2.z[l2[j]] = v1_512.c2.z[j];
         }

         intrin_vector_add_512(in1,qins1,qins2);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup2[0]].s1, &plqcd_g.phim[2][iup2[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[2][m2[j]].s1.c0.z[l2[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[2][m2[j]].s1.c1.z[l2[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[2][m2[j]].s1.c2.z[l2[j]] = v1_512.c2.z[j];
         }
         ub0++;

         //-- 3 direction --
         //_vector_i_sub(p,qin[i].s0,qin[i].s2);
         //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s0,*ub0,p);
         //_vector_i_add(p,qin[i].s1,qin[i].s3);
         //_su3_inverse_multiply(plqcd_g.phim[3][iup3].s1,*ub0,p);
         intrin_su3_load_512(U,ub0);
         intrin_vector_i_sub_512(in1,qins0,qins2);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[2][iup3[0]].s0, &plqcd_g.phim[2][iup3[1]].s0, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[3][m3[j]].s0.c0.z[l3[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[3][m3[j]].s0.c1.z[l3[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[3][m3[j]].s0.c2.z[l3[j]] = v1_512.c2.z[j];
         }

         intrin_vector_i_add_512(in1,qins1,qins3);
         intrin_su3_inverse_multiply_512(out,U,in1);
         intrin_vector_store_512(&v1_512,out);
         //copy_su3_vector_256_to_su3_vector(&plqcd_g.phim[3][iup3[0]].s1, &plqcd_g.phim[3][iup3[1]].s1, &v1_256);
         for(j=0; j<4; j++){
            plqcd_g.phim512[3][m3[j]].s1.c0.z[l3[j]] = v1_512.c0.z[j];
            plqcd_g.phim512[3][m3[j]].s1.c1.z[l3[j]] = v1_512.c1.z[j];
            plqcd_g.phim512[3][m3[j]].s1.c2.z[l3[j]] = v1_512.c2.z[j];
         }

      }

      //start sending the buffers to the nearest neighbours in the +ve directions
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu < 4; mu++){  
         if(plqcd_g.nprocs[mu] > 1){  
            //snd_err[mu+4] = MPI_Isend(&plqcd_g.phim[mu][V/2],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD, &snd_req[mu+4]);
            //rcv_err[mu+4] = MPI_Irecv(&plqcd_g.phim[mu][V/2+face[mu]/2],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_req[mu+4]);
            snd_err[mu+4] = MPI_Isend(&plqcd_g.phim512[mu][V/8],                   6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu+1]  , tags[mu+4], MPI_COMM_WORLD, &snd_req[mu+4]);
            rcv_err[mu+4] = MPI_Irecv(&plqcd_g.phim512[mu][V/8+face[mu]/8],        6*plqcd_g.face[mu], MPI_DOUBLE, plqcd_g.npr[2*mu]    , tags[mu+4], MPI_COMM_WORLD, &rcv_req[mu+4]);
            if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Isend or MPI_Ircv\n");
               exit(1);}
         }
      }
      
      //wait for the communications of phip to finish
      for(mu=0; mu<4; mu++)
      {
         if(plqcd_g.nprocs[mu]>1)
         {   
            rcv_err[mu]=MPI_Wait(&rcv_req[mu],&rcv_mpi_stat[mu]);
            snd_err[mu]=MPI_Wait(&snd_req[mu],&snd_mpi_stat[mu]);
            if( (snd_err[mu] != MPI_SUCCESS) || (rcv_err[mu] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Wait\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif
     

      //-------------------------------------------------------
      //complete computation of the U_mu*(1-gamma_m)*qin terms
      //-------------------------------------------------------
      //copy the exchanged boundaries to the correspoding locations on the local phip fields
      for(mu=0; mu<4; mu++)
      { 
         if(plqcd_g.nprocs[mu] > 1)
         {  
            #ifdef _OPENMP 
            #pragma omp for 
            #endif  
            //for(i=0; i< face[mu]/2; i++)
            //{
                //can we prefetch here
            //    j=V/2+face[mu]/2+i;
            //    k=plqcd_g.nn_bndo[2*mu][i];
            //    _vector_assign(plqcd_g.phip[mu][k].s0 , plqcd_g.phip[mu][j].s0);
            //    _vector_assign(plqcd_g.phip[mu][k].s1 , plqcd_g.phip[mu][j].s1);
            //}
            for(i=0; i< face[mu]/2; i +=4)
            {
                for(alpha=0; alpha<4; alpha++)
                {
                   j=V/2+face[mu]/2+i+alpha;
                   k=plqcd_g.nn_bnde[2*mu][i+alpha]-V/2;
                   j1=j/4; 
                   j2=j%4;
                   k1=k/4;
                   k2=k%4;
                   plqcd_g.phip512[mu][k1].s0.c0.z[k2] = plqcd_g.phip512[mu][j1].s0.c0.z[j2];
                   plqcd_g.phip512[mu][k1].s0.c1.z[k2] = plqcd_g.phip512[mu][j1].s0.c1.z[j2];
                   plqcd_g.phip512[mu][k1].s0.c2.z[k2] = plqcd_g.phip512[mu][j1].s0.c2.z[j2];
                   plqcd_g.phip512[mu][k1].s1.c0.z[k2] = plqcd_g.phip512[mu][j1].s1.c0.z[j2];
                   plqcd_g.phip512[mu][k1].s1.c1.z[k2] = plqcd_g.phip512[mu][j1].s1.c1.z[j2];
                   plqcd_g.phip512[mu][k1].s1.c2.z[k2] = plqcd_g.phip512[mu][j1].s1.c2.z[j2];
                }
            }
         }
      } 



      #ifdef _OPENMP
      #pragma omp for 
      #endif 
      for(i=V/8; i< V/4; i ++)
      {

         ub0 = &u[4*i];
         k=i-V/8; 
         // +0
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         intrin_prefetch_halfspinor_512(&plqcd_g.phip512[1][k]);
         #endif
         //_su3_multiply(q0, *ub0, plqcd_g.phip[0][i].s0);
         //_su3_multiply(q1, *ub0, plqcd_g.phip[0][i].s1);
         //_vector_assign(q2, q0);
         //_vector_assign(q3, q1);
         intrin_su3_load_512(U,ub0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[0][i].s0, &plqcd_g.phip[0][i+1].s0);
         intrin_vector_load_512(in1,&plqcd_g.phip512[0][k].s0);
         intrin_su3_multiply_512(mq0,U,in1);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[0][i].s1, &plqcd_g.phip[0][i+1].s1);
         intrin_vector_load_512(in1,&plqcd_g.phip512[0][k].s1);
         intrin_su3_multiply_512(mq1,U,in1);
         for(j=0; j<3; j++){
            mq2[j]= mq0[j];
            mq3[j]= mq1[j];}
         ub0++;

         // +1
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         intrin_prefetch_halfspinor_512(&plqcd_g.phip512[2][k]);
         #endif
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[1][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[1][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2, -1.0, ap1);
         //_vector_add_i_mul(q3, -1.0, ap0);
         intrin_su3_load_512(U,ub0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[1][i].s0, &plqcd_g.phip[1][i+1].s0);
         intrin_vector_load_512(in1,&plqcd_g.phip512[1][k].s0);
         intrin_su3_multiply_512(map0,U,in1);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[1][i].s1, &plqcd_g.phip[1][i+1].s1);
         intrin_vector_load_512(in1,&plqcd_g.phip512[1][k].s1);
         intrin_su3_multiply_512(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm512_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm512_add_pd(mq1[j],map1[j]);
            mq2[j] = complex_i_sub_regs_512(mq2[j],map1[j]);
            mq3[j] = complex_i_sub_regs_512(mq3[j],map0[j]);}
         ub0++;        

         // +2
         #ifdef MIC_PREFETCH
         intrin_prefetch_su3_512(ub0+1);
         intrin_prefetch_halfspinor_512(&plqcd_g.phip512[3][k]);
         #endif
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[2][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[2][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_sub_assign(q2,ap1);
         //_vector_add_assign(q3,ap0);
         intrin_su3_load_512(U,ub0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[2][i].s0, &plqcd_g.phip[2][i+1].s0);
         intrin_vector_load_512(in1,&plqcd_g.phip512[2][k].s0);
         intrin_su3_multiply_512(map0,U,in1);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[2][i].s1, &plqcd_g.phip[2][i+1].s1);
         intrin_vector_load_512(in1,&plqcd_g.phip512[2][k].s1);
         intrin_su3_multiply_512(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm512_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm512_add_pd(mq1[j],map1[j]);
            mq2[j] = _mm512_sub_pd(mq2[j],map1[j]);
            mq3[j] = _mm512_add_pd(mq3[j],map0[j]);}
         ub0++;        

         #ifdef MIC_PREFETCH
         intrin_prefetch_spinor_512(&qout[i]);
         #endif
         // +3
         //_su3_multiply(ap0, *ub0, plqcd_g.phip[3][i].s0);
         //_su3_multiply(ap1, *ub0, plqcd_g.phip[3][i].s1);
         //_vector_add_assign(q0,ap0);
         //_vector_add_assign(q1,ap1);
         //_vector_add_i_mul(q2,-1.0, ap0);
         //_vector_add_i_mul(q3, 1.0, ap1);
         intrin_su3_load_512(U,ub0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[3][i].s0, &plqcd_g.phip[3][i+1].s0);
         intrin_vector_load_512(in1,&plqcd_g.phip512[3][k].s0);
         intrin_su3_multiply_512(map0,U,in1);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phip[3][i].s1, &plqcd_g.phip[3][i+1].s1);
         intrin_vector_load_512(in1,&plqcd_g.phip512[3][k].s1);
         intrin_su3_multiply_512(map1,U,in1);
         for(j=0; j<3; j++){
            mq0[j] = _mm512_add_pd(mq0[j],map0[j]);
            mq1[j] = _mm512_add_pd(mq1[j],map1[j]);
            mq2[j] = complex_i_sub_regs_512(mq2[j],map0[j]);
            mq3[j] = complex_i_add_regs_512(mq3[j],map1[j]);}

 
         //store the result
         //_vector_assign(qout[i].s0,q0);
         //_vector_assign(qout[i].s1,q1);
         //_vector_assign(qout[i].s2,q2);
         //_vector_assign(qout[i].s3,q3);
         intrin_vector_store_512(&qout[i].s0,mq0);         
         intrin_vector_store_512(&qout[i].s1,mq1);         
         intrin_vector_store_512(&qout[i].s2,mq1);         
         intrin_vector_store_512(&qout[i].s3,mq1);         

      } 

      //wait for the communications of phim to finish
      #ifdef _OPENMP
      #pragma omp master
      {
      #endif
      for(mu=0; mu<4; mu++)
      {
         if(plqcd_g.nprocs[mu]>1)
         {   
            rcv_err[mu+4]=MPI_Wait(&rcv_req[mu+4],&rcv_mpi_stat[mu+4]);
            snd_err[mu+4]=MPI_Wait(&snd_req[mu+4],&snd_mpi_stat[mu+4]);
            if( (snd_err[mu+4] != MPI_SUCCESS) || (rcv_err[mu+4] != MPI_SUCCESS) ){
               fprintf(stderr,"Error in MPI_Wait\n");
               exit(1);}
         }
      }
      #ifdef _OPENMP
      }
      #endif

      //copy the exchanged boundaries to the correspoding locations on the local phim fields
      for(mu=0; mu<4; mu++)
      { 
         if(plqcd_g.nprocs[mu] > 1)
         {
            #ifdef _OPENMP   
            #pragma omp for
            #endif
            for(i=0; i< face[mu]/2; i +=4)
            {
               //j=V/2+face[mu]/2+i;
               //k=plqcd_g.nn_bndo[2*mu+1][i];
               //_vector_assign(plqcd_g.phim[mu][k].s0, plqcd_g.phim[mu][j].s0);
               //_vector_assign(plqcd_g.phim[mu][k].s1, plqcd_g.phim[mu][j].s1);
               for(alpha=0; alpha<4; alpha++)
               {
                   j=V/2+face[mu]/2+i+alpha;
                   k=plqcd_g.nn_bnde[2*mu+1][i+alpha]-V/2;
                   j1=j/4;
                   j2=j%4;
                   k1=k/4;
                   k2=k%4;
                   plqcd_g.phim512[mu][k1].s0.c0.z[k2] = plqcd_g.phim512[mu][j1].s0.c0.z[j2];
                   plqcd_g.phim512[mu][k1].s0.c1.z[k2] = plqcd_g.phim512[mu][j1].s0.c1.z[j2];
                   plqcd_g.phim512[mu][k1].s0.c2.z[k2] = plqcd_g.phim512[mu][j1].s0.c2.z[j2];
                   plqcd_g.phim512[mu][k1].s1.c0.z[k2] = plqcd_g.phim512[mu][j1].s1.c0.z[j2];
                   plqcd_g.phim512[mu][k1].s1.c1.z[k2] = plqcd_g.phim512[mu][j1].s1.c1.z[j2];
                   plqcd_g.phim512[mu][k1].s1.c2.z[k2] = plqcd_g.phim512[mu][j1].s1.c2.z[j2];
                }
            }
         }
      }


      //---------------------------------------------------------------------
      //finish computation of the U^dagger*(1+gamma_mu)*qin
      //---------------------------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for
      #endif
      for(i=0; i< V/8; i ++)
      {

         intrin_vector_load_512(mq0,&qout[i+V/8].s0);
         intrin_vector_load_512(mq1,&qout[i+V/8].s1);
         intrin_vector_load_512(mq2,&qout[i+V/8].s2);
         intrin_vector_load_512(mq3,&qout[i+V/8].s3);

         // 0 direction
         #ifdef MIC_PREFETCH
         intrin_prefetch_halfspinor_512(&plqcd_g.phim512[1][i]);
         #endif
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[0][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[0][i].s1);
         //_vector_minus_assign(qout[i].s2, plqcd_g.phim[0][i].s0);
         //_vector_minus_assign(qout[i].s3, plqcd_g.phim[0][i].s1);
         //copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[0][i].s0, &plqcd_g.phim[0][i+1].s0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[0][i].s1, &plqcd_g.phim[0][i+1].s1);
         intrin_vector_load_512(map0,&plqcd_g.phim512[0][i].s0);
         intrin_vector_load_512(map1,&plqcd_g.phim512[0][i].s1);
         intrin_vector_add_512(mq0,mq0,map0);
         intrin_vector_add_512(mq1,mq1,map1);
         intrin_vector_sub_512(mq2,mq2,map0);
         intrin_vector_sub_512(mq3,mq3,map1);

         // 1 direction
         #ifdef MIC_PREFETCH
         intrin_prefetch_halfspinor_512(&plqcd_g.phim512[2][i]);
         #endif
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[1][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s2, 1.0,plqcd_g.phim[1][i].s1);
         //_vector_add_i_mul(qout[i].s3, 1.0, plqcd_g.phim[1][i].s0);
         //copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[1][i].s0, &plqcd_g.phim[1][i+1].s0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[1][i].s1, &plqcd_g.phim[1][i+1].s1);
         intrin_vector_load_512(map0,&plqcd_g.phim512[1][i].s0);
         intrin_vector_load_512(map1,&plqcd_g.phim512[1][i].s1);
         intrin_vector_add_512(mq0,mq0,map0);
         intrin_vector_add_512(mq1,mq1,map1);
         intrin_vector_i_add_512(mq2,mq2,map1);
         intrin_vector_i_add_512(mq3,mq3,map0);
        

         // 2 direction
         #ifdef MIC_PREFETCH
         intrin_prefetch_halfspinor_512(&plqcd_g.phim512[3][i]);
         #endif
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[2][i].s0);
         //_vector_add_assign(qout[i].s1,  plqcd_g.phim[2][i].s1 );
         //_vector_add_assign(qout[i].s2,  plqcd_g.phim[2][i].s1);
         //_vector_sub_assign(qout[i].s3,   plqcd_g.phim[2][i].s0);
         //copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[2][i].s0, &plqcd_g.phim[2][i+1].s0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[2][i].s1, &plqcd_g.phim[2][i+1].s1);
         intrin_vector_load_512(map0,&plqcd_g.phim512[2][i].s0);
         intrin_vector_load_512(map1,&plqcd_g.phim512[2][i].s1);
         intrin_vector_add_512(mq0,mq0,map0);
         intrin_vector_add_512(mq1,mq1,map1);
         intrin_vector_add_512(mq2,mq2,map1);
         intrin_vector_sub_512(mq3,mq3,map0);
         

         //3 direction
         //_vector_add_assign(qout[i].s0, plqcd_g.phim[3][i].s0);
         //_vector_add_assign(qout[i].s1, plqcd_g.phim[3][i].s1);
         //_vector_i_add_assign(qout[i].s2, plqcd_g.phim[3][i].s0);
         //_vector_i_sub_assign(qout[i].s3, plqcd_g.phim[3][i].s1);     
         //copy_su3_vector_to_su3_vector_256(&v0_256, &plqcd_g.phim[3][i].s0, &plqcd_g.phim[3][i+1].s0);
         //copy_su3_vector_to_su3_vector_256(&v1_256, &plqcd_g.phim[3][i].s1, &plqcd_g.phim[3][i+1].s1);
         intrin_vector_load_512(map0,&plqcd_g.phim512[3][i].s0);
         intrin_vector_load_512(map1,&plqcd_g.phim512[3][i].s1);
         intrin_vector_add_512(mq0,mq0,map0);
         intrin_vector_add_512(mq1,mq1,map1);
         intrin_vector_i_add_512(mq2,mq2,map0);
         intrin_vector_i_add_512(mq3,mq3,map1);

         //store the result
         intrin_vector_store_512(&qout[i+V/8].s0, mq0);
         intrin_vector_store_512(&qout[i+V/8].s1, mq1);
         intrin_vector_store_512(&qout[i+V/8].s2, mq2);
         intrin_vector_store_512(&qout[i+V/8].s3, mq3);
   }
#ifdef _OPENMP
}
#endif  //end of the openmp parallel reigon

   
   return stop_watch(ts);
}


//========================================================================================
//Single MIC version using auxilary fields and openMP
//===================================EO===================================================
double plqcd_hopping_matrix_eo_single_mic(spinor_512 *qin, spinor_512 *qout, su3_512 *u)
{
   double ts; //timer
   ts=stop_watch(0.0); //start
   
   #ifdef _OPENMP
   #pragma omp parallel private(sign) 
   {
   #endif
      sign = _mm512_load_pd(dsign);
      int V;
      V = plqcd_g.VOLUME;
      __m512d qins0[3],qins1[3],qins2[3],qins3[3], U[3][3];
      __m512d out1[3],out2[3],out3[3],out4[3];
      __m512d map0[3],map1[3];

      //check if the dimensions allows for using compact data representation
      int lx=plqcd_g.LX/4;
      if((plqcd_g.LX%8) != 0 ){ 
          fprintf(stderr,"LX must be a factor of 8\n");
          exit(1);
      }
    
      int iup[4],idn[4];    
      su3_512 *ub0;

      int LX,LY,LZ,LT;
      LX = plqcd_g.latdims[0];
      LY = plqcd_g.latdims[1];
      LZ = plqcd_g.latdims[2];
      LT = plqcd_g.latdims[3];

      int ix,iy,it,iz,is,xcor[4],ipt;

      halfspinor_512 *phip0,*phip1,*phip2,*phip3;
      halfspinor_512 *phim0,*phim1,*phim2,*phim3;

      //-----------------------------------------------------
      // loop over input spinor on odd sites
      // compute (1-gamma_mu)qin terms and store in phip
      // compute U^dagger*(1+gamma_mu)qin and store in phim
      //----------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for 
      #endif   
      for(int is=0; is < LT*LZ*LY; is++) 
      {
         iy = is%LY;
         iz = (is/LY) %LZ;
         it = (is/LY/LZ)%LT;

         for(ix = (iy+iz+it+1)%2; ix < lx; ix +=2)
         {
            xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;

            ipt=plqcd_g.ipt_eo[cart2lex(xcor)];

            ub0 = &u[ipt];
         
            phip0 = &plqcd_g.phip512[0][plqcd_g.idn[ipt][0]];
            phip1 = &plqcd_g.phip512[1][plqcd_g.idn[ipt][1]];
            phip2 = &plqcd_g.phip512[2][plqcd_g.idn[ipt][2]];
            phip3 = &plqcd_g.phip512[3][plqcd_g.idn[ipt][3]];


            #ifdef MIC_PREFETCH
            intrin_prefetch_spinor_512(&qin[ipt]);
            intrin_prefetch_halfspinor_512(phip0);
            #endif


            intrin_vector_load_512(qins0,&qin[ipt].s0);
            intrin_vector_load_512(qins1,&qin[ipt].s1);
            intrin_vector_load_512(qins2,&qin[ipt].s2);
            intrin_vector_load_512(qins3,&qin[ipt].s3);



            //=====   +0  ========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phip1);
            #endif
            intrin_vector_i_add_512(out1,qins0,qins3);
            intrin_vector_i_add_512(out2,qins1,qins2);
            if(ix == 0)
            {
               for(int i=0; i<3; i++)
               {
                  out3[i]= (__m512d)  _mm512_permute4f128_epi32 ( (__m512i) out1[i], 0b00111001);
                  out4[i]= (__m512d)  _mm512_permute4f128_epi32 ( (__m512i) out2[i], 0b00111001);
               }
               intrin_vector_store_512(&(phip0->s0),out3);
               intrin_vector_store_512(&(phip0->s1),out4);
            }
            else
            {
               intrin_vector_store_512(&(phip0->s0),out1);
               intrin_vector_store_512(&(phip0->s1),out2);
            }

            //===== +1 ==========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phip2);  
            #endif
            intrin_vector_add_512(out1,qins0,qins3);
            intrin_vector_sub_512(out2,qins1,qins2);
            intrin_vector_store_512(&(phip1->s0),out1);
            intrin_vector_store_512(&(phip1->s1),out2);
        

            //===== +2 ==========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phip3);
            #endif
            intrin_vector_i_add_512(out1,qins0,qins2);
            intrin_vector_i_sub_512(out2,qins1,qins3);
            intrin_vector_store_512(&(phip2->s0),out1);
            intrin_vector_store_512(&(phip2->s1),out2);
        

            phim0 = &plqcd_g.phim512[0][plqcd_g.iup[ipt][0]];
            phim1 = &plqcd_g.phim512[1][plqcd_g.iup[ipt][1]];
            phim2 = &plqcd_g.phim512[2][plqcd_g.iup[ipt][2]];
            phim3 = &plqcd_g.phim512[3][plqcd_g.iup[ipt][3]];




            //===== +3 ==========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim0);
            intrin_prefetch_su3_512(ub0);
            #endif
            intrin_vector_add_512(out1,qins0,qins2);
            intrin_vector_add_512(out2,qins1,qins3);
            intrin_vector_store_512(&(phip3->s0),out1);
            intrin_vector_store_512(&(phip3->s1),out2);
        

            //===== -0  ========== 
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim1);
            intrin_prefetch_su3_512(ub0+1);
            #endif
            intrin_su3_load_512(U,ub0);
            intrin_vector_i_sub_512(out1,qins0,qins3);
            intrin_su3_inverse_multiply_512(map0,U,out1);
            intrin_vector_i_sub_512(out2,qins1,qins2);
            intrin_su3_inverse_multiply_512(map1,U,out2);
            if(xcor[0] == (lx-1))
            {
               for(int i=0; i<3; i++)
               {
                  out3[i] = (__m512d)  _mm512_permute4f128_epi32 ( (__m512i) map0[i], 0b10010011);
                  out4[i] = (__m512d)  _mm512_permute4f128_epi32 ( (__m512i) map1[i], 0b10010011);
               }
               intrin_vector_store_512(&(phim0->s0),out3);
               intrin_vector_store_512(&(phim0->s1),out4);
            }
            else
            {
               intrin_vector_store_512(&(phim0->s0),map0);
               intrin_vector_store_512(&(phim0->s1),map1);
            }
            ub0++;


            
            //===== -1  ========== 
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim2);
            intrin_prefetch_su3_512(ub0+1);
            #endif
            intrin_su3_load_512(U,ub0);
            intrin_vector_sub_512(out1,qins0,qins3);
            intrin_su3_inverse_multiply_512(map0,U,out1);
            intrin_vector_add_512(out2,qins1,qins2);
            intrin_su3_inverse_multiply_512(map1,U,out2);
            intrin_vector_store_512(&(phim0->s0),map0);
            intrin_vector_store_512(&(phim0->s1),map1);
            ub0++;


            //===== -2  ========== 
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim3);
            intrin_prefetch_su3_512(ub0+1);
            #endif
            intrin_su3_load_512(U,ub0);
            intrin_vector_i_sub_512(out1,qins0,qins2);
            intrin_su3_inverse_multiply_512(map0,U,out1);
            intrin_vector_i_add_512(out2,qins1,qins3);
            intrin_su3_inverse_multiply_512(map1,U,out2);
            intrin_vector_store_512(&(phim0->s0),map0);
            intrin_vector_store_512(&(phim0->s1),map1);
            ub0++;


            //===== -3  ========== 
            intrin_su3_load_512(U,ub0);
            intrin_vector_sub_512(out1,qins0,qins2);
            intrin_su3_inverse_multiply_512(map0,U,out1);
            intrin_vector_sub_512(out2,qins1,qins3);
            intrin_su3_inverse_multiply_512(map1,U,out2);
            intrin_vector_store_512(&(phim0->s0),map0);
            intrin_vector_store_512(&(phim0->s1),map1);
         }
      }


      //-------------------------------------------------------
      // build the result
      // loop over the output spinor on even sites
      // compute U*phip terms and store in qout
      // store phim in qout
      // later on we will build the full twisted mass operator
      //--------------------------------------------------------
      __m512d mq0[3],mq1[3],mq2[3],mq3[3]; 
      #ifdef _OPENMP
      #pragma omp for 
      #endif   
      for(int is=0; is < LT*LZ*LY; is++) 
      {
         iy = is%LY;
         iz = (is/LY) %LZ;
         it = (is/LY/LZ)%LT;

         for(ix = (iy+iz+it)%2; ix < lx; ix +=2)
         {
            xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
            ipt=plqcd_g.ipt_eo[cart2lex(xcor)];
            ub0 = &u[ipt];
         
            phip0 = &plqcd_g.phip512[0][ipt];
            phip1 = &plqcd_g.phip512[1][ipt];
            phip2 = &plqcd_g.phip512[2][ipt];
            phip3 = &plqcd_g.phip512[3][ipt];


            //=====   +0  ========
            #ifdef MIC_PREFETCH
            intrin_prefetch_su3_512(ub0+1);
            intrin_prefetch_halfspinor_512(phip1);
            #endif
            intrin_su3_load_512(U,ub0);
            intrin_vector_load_512(out1,&(phip0->s0));
            intrin_vector_load_512(out2,&(phip0->s1));
            intrin_su3_multiply_512(mq0,U,out1);
            intrin_su3_multiply_512(mq1,U,out2);
            intrin_vector_i_sub_512(mq2,mq2,mq1);
            intrin_vector_i_sub_512(mq3,mq3,mq0);
            ub0++;
        
            //=====   +1  ========
            #ifdef MIC_PREFETCH
            intrin_prefetch_su3_512(ub0+1);
            intrin_prefetch_halfspinor_512(phip2);
            #endif
            intrin_su3_load_512(U,ub0);
            intrin_vector_load_512(out1,&(phip1->s0));
            intrin_vector_load_512(out2,&(phip1->s1));
            intrin_su3_multiply_512(map0,U,out1);
            intrin_su3_multiply_512(map1,U,out2);
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_sub_512(mq2,mq2,map1);
            intrin_vector_add_512(mq3,mq3,map0);
            ub0++;

        
            //=====   +2  ========
            #ifdef MIC_PREFETCH
            intrin_prefetch_su3_512(ub0+1);
            intrin_prefetch_halfspinor_512(phip3);
            #endif
            intrin_su3_load_512(U,ub0);
            intrin_vector_load_512(out1,&(phip2->s0));
            intrin_vector_load_512(out2,&(phip2->s1));
            intrin_su3_multiply_512(map0,U,out1);
            intrin_su3_multiply_512(map1,U,out2);
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_i_sub_512(mq2,mq2,map0);
            intrin_vector_i_add_512(mq3,mq3,map1);
            ub0++;

        

            phim0 = &plqcd_g.phim512[0][ipt];
            phim1 = &plqcd_g.phim512[1][ipt];
            phim2 = &plqcd_g.phim512[2][ipt];
            phim3 = &plqcd_g.phim512[3][ipt];


            //=====   +3  ========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim0);
            #endif
            intrin_su3_load_512(U,ub0);
            intrin_vector_load_512(out1,&(phip3->s0));
            intrin_vector_load_512(out2,&(phip3->s1));
            intrin_su3_multiply_512(map0,U,out1);
            intrin_su3_multiply_512(map1,U,out2);
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_add_512(mq2,mq2,map0);
            intrin_vector_add_512(mq3,mq3,map1);

            //now add the -ve pieces
            //======= -0 =========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim1);
            #endif
            intrin_vector_load_512(map0,&(phim0->s0));
            intrin_vector_load_512(map1,&(phim0->s1));
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_i_add_512(mq2,mq2,map1);
            intrin_vector_i_add_512(mq3,mq3,map0);


            //======= -1 =========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim2);
            #endif
            intrin_vector_load_512(map0,&(phim1->s0));
            intrin_vector_load_512(map1,&(phim1->s1));
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_add_512(mq2,mq2,map1);
            intrin_vector_sub_512(mq3,mq3,map0);

            //======= -2 =========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim3);
            #endif
            intrin_vector_load_512(map0,&(phim2->s0));
            intrin_vector_load_512(map1,&(phim2->s1));
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_i_add_512(mq2,mq2,map0);
            intrin_vector_i_sub_512(mq3,mq3,map1);


            //======= -3 =========
            #ifdef MIC_PREFETCH
            intrin_prefetch_spinor_512(&qout[ipt]);
            #endif
            intrin_vector_load_512(map0,&(phim3->s0));
            intrin_vector_load_512(map1,&(phim3->s1));
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_sub_512(mq2,mq2,map0);
            intrin_vector_sub_512(mq3,mq3,map1);


            //store the result
            intrin_vector_store_512(&qout[ipt].s0,mq0);
            intrin_vector_store_512(&qout[ipt].s1,mq1);
            intrin_vector_store_512(&qout[ipt].s2,mq2);
            intrin_vector_store_512(&qout[ipt].s3,mq3);

      }

   }

#ifdef _OPENMP
}
#endif  //end of the openmp parallel reigon

   
   return stop_watch(ts);
}



//=============================================================================================
//Single MIC version using auxilary fields and openMP using short loads of the gauge links
//i.e. only the first two rows of the links are loaded and the third row is computed
//using unitarity
//===================================EO========================================================
double plqcd_hopping_matrix_eo_single_mic_short(spinor_512 *qin, spinor_512 *qout, su3_512 *u)
{
   double ts; //timer
   ts=stop_watch(0.0); //start
   
   #ifdef _OPENMP
   #pragma omp parallel private(sign) 
   {
   #endif
      sign = _mm512_load_pd(dsign);
      int V;
      V = plqcd_g.VOLUME;
      __m512d qins0[3],qins1[3],qins2[3],qins3[3], U[3][3];
      __m512d out1[3],out2[3],out3[3],out4[3];
      __m512d map0[3],map1[3];

      //check if the dimensions allows for using compact data representation
      int lx=plqcd_g.LX/4;
      if((plqcd_g.LX%8) != 0 ){ 
          fprintf(stderr,"LX must be a factor of 8\n");
          exit(1);
      }
    
      int iup[4],idn[4];    
      su3_512 *ub0;

      int LX,LY,LZ,LT;
      LX = plqcd_g.latdims[0];
      LY = plqcd_g.latdims[1];
      LZ = plqcd_g.latdims[2];
      LT = plqcd_g.latdims[3];

      int ix,iy,it,iz,is,xcor[4],ipt;

      halfspinor_512 *phip0,*phip1,*phip2,*phip3;
      halfspinor_512 *phim0,*phim1,*phim2,*phim3;

      //-----------------------------------------------------
      // loop over input spinor on odd sites
      // compute (1-gamma_mu)qin terms and store in phip
      // compute U^dagger*(1+gamma_mu)qin and store in phim
      //----------------------------------------------------
      #ifdef _OPENMP
      #pragma omp for 
      #endif   
      for(int is=0; is < LT*LZ*LY; is++) 
      {
         iy = is%LY;
         iz = (is/LY) %LZ;
         it = (is/LY/LZ)%LT;

         for(ix = (iy+iz+it+1)%2; ix < lx; ix +=2)
         {
            xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;

            ipt=plqcd_g.ipt_eo[cart2lex(xcor)];

            ub0 = &u[4*ipt];
         
            phip0 = &plqcd_g.phip512[0][plqcd_g.idn[ipt][0]];
            phip1 = &plqcd_g.phip512[1][plqcd_g.idn[ipt][1]];
            phip2 = &plqcd_g.phip512[2][plqcd_g.idn[ipt][2]];
            phip3 = &plqcd_g.phip512[3][plqcd_g.idn[ipt][3]];


            #ifdef MIC_PREFETCH
            intrin_prefetch_spinor_512(&qin[ipt]);
            intrin_prefetch_halfspinor_512(phip0);
            #endif


            intrin_vector_load_512(qins0,&qin[ipt].s0);
            intrin_vector_load_512(qins1,&qin[ipt].s1);
            intrin_vector_load_512(qins2,&qin[ipt].s2);
            intrin_vector_load_512(qins3,&qin[ipt].s3);



            //=====   +0  ========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phip1);
            #endif
            intrin_vector_i_add_512(out1,qins0,qins3);
            intrin_vector_i_add_512(out2,qins1,qins2);
            if(ix == 0)
            {
               for(int i=0; i<3; i++)
               {
                  out3[i]= (__m512d)  _mm512_permute4f128_epi32 ( (__m512i) out1[i], 0b00111001);
                  out4[i]= (__m512d)  _mm512_permute4f128_epi32 ( (__m512i) out2[i], 0b00111001);
               }
               intrin_vector_store_512(&(phip0->s0),out3);
               intrin_vector_store_512(&(phip0->s1),out4);
            }
            else
            {
               intrin_vector_store_512(&(phip0->s0),out1);
               intrin_vector_store_512(&(phip0->s1),out2);
            }

            //===== +1 ==========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phip2);  
            #endif
            intrin_vector_add_512(out1,qins0,qins3);
            intrin_vector_sub_512(out2,qins1,qins2);
            intrin_vector_store_512(&(phip1->s0),out1);
            intrin_vector_store_512(&(phip1->s1),out2);
        

            //===== +2 ==========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phip3);
            #endif
            intrin_vector_i_add_512(out1,qins0,qins2);
            intrin_vector_i_sub_512(out2,qins1,qins3);
            intrin_vector_store_512(&(phip2->s0),out1);
            intrin_vector_store_512(&(phip2->s1),out2);
        

            phim0 = &plqcd_g.phim512[0][plqcd_g.iup[ipt][0]];
            phim1 = &plqcd_g.phim512[1][plqcd_g.iup[ipt][1]];
            phim2 = &plqcd_g.phim512[2][plqcd_g.iup[ipt][2]];
            phim3 = &plqcd_g.phim512[3][plqcd_g.iup[ipt][3]];




            //===== +3 ==========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim0);
            intrin_prefetch_su3_512(ub0);
            #endif
            intrin_vector_add_512(out1,qins0,qins2);
            intrin_vector_add_512(out2,qins1,qins3);
            intrin_vector_store_512(&(phip3->s0),out1);
            intrin_vector_store_512(&(phip3->s1),out2);
        

            //===== -0  ========== 
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim1);
            intrin_prefetch_su3_512(ub0+1);
            #endif
            intrin_su3_load_short_512(U,ub0);
            intrin_vector_i_sub_512(out1,qins0,qins3);
            intrin_su3_inverse_multiply_512(map0,U,out1);
            intrin_vector_i_sub_512(out2,qins1,qins2);
            intrin_su3_inverse_multiply_512(map1,U,out2);
            if(xcor[0] == (lx-1))
            {
               for(int i=0; i<3; i++)
               {
                  out3[i] = (__m512d)  _mm512_permute4f128_epi32 ( (__m512i) map0[i], 0b10010011);
                  out4[i] = (__m512d)  _mm512_permute4f128_epi32 ( (__m512i) map1[i], 0b10010011);
               }
               intrin_vector_store_512(&(phim0->s0),out3);
               intrin_vector_store_512(&(phim0->s1),out4);
            }
            else
            {
               intrin_vector_store_512(&(phim0->s0),map0);
               intrin_vector_store_512(&(phim0->s1),map1);
            }
            ub0++;


            
            //===== -1  ========== 
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim2);
            intrin_prefetch_su3_512(ub0+1);
            #endif
            intrin_su3_load_short_512(U,ub0);
            intrin_vector_sub_512(out1,qins0,qins3);
            intrin_su3_inverse_multiply_512(map0,U,out1);
            intrin_vector_add_512(out2,qins1,qins2);
            intrin_su3_inverse_multiply_512(map1,U,out2);
            intrin_vector_store_512(&(phim0->s0),map0);
            intrin_vector_store_512(&(phim0->s1),map1);
            ub0++;


            //===== -2  ========== 
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim3);
            intrin_prefetch_su3_512(ub0+1);
            #endif
            intrin_su3_load_short_512(U,ub0);
            intrin_vector_i_sub_512(out1,qins0,qins2);
            intrin_su3_inverse_multiply_512(map0,U,out1);
            intrin_vector_i_add_512(out2,qins1,qins3);
            intrin_su3_inverse_multiply_512(map1,U,out2);
            intrin_vector_store_512(&(phim0->s0),map0);
            intrin_vector_store_512(&(phim0->s1),map1);
            ub0++;


            //===== -3  ========== 
            intrin_su3_load_short_512(U,ub0);
            intrin_vector_sub_512(out1,qins0,qins2);
            intrin_su3_inverse_multiply_512(map0,U,out1);
            intrin_vector_sub_512(out2,qins1,qins3);
            intrin_su3_inverse_multiply_512(map1,U,out2);
            intrin_vector_store_512(&(phim0->s0),map0);
            intrin_vector_store_512(&(phim0->s1),map1);
         }
      }


      //-------------------------------------------------------
      // build the result
      // loop over the output spinor on even sites
      // compute U*phip terms and store in qout
      // store phim in qout
      // later on we will build the full twisted mass operator
      //--------------------------------------------------------
      __m512d mq0[3],mq1[3],mq2[3],mq3[3]; 
      #ifdef _OPENMP
      #pragma omp for 
      #endif   
      for(int is=0; is < LT*LZ*LY; is++) 
      {
         iy = is%LY;
         iz = (is/LY) %LZ;
         it = (is/LY/LZ)%LT;

         for(ix = (iy+iz+it)%2; ix < lx; ix +=2)
         {
            xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
            ipt=plqcd_g.ipt_eo[cart2lex(xcor)];
            ub0 = &u[4*ipt];
         
            phip0 = &plqcd_g.phip512[0][ipt];
            phip1 = &plqcd_g.phip512[1][ipt];
            phip2 = &plqcd_g.phip512[2][ipt];
            phip3 = &plqcd_g.phip512[3][ipt];


            //=====   +0  ========
            #ifdef MIC_PREFETCH
            intrin_prefetch_su3_512(ub0+1);
            intrin_prefetch_halfspinor_512(phip1);
            #endif
            intrin_su3_load_short_512(U,ub0);
            intrin_vector_load_512(out1,&(phip0->s0));
            intrin_vector_load_512(out2,&(phip0->s1));
            intrin_su3_multiply_512(mq0,U,out1);
            intrin_su3_multiply_512(mq1,U,out2);
            intrin_vector_i_sub_512(mq2,mq2,mq1);
            intrin_vector_i_sub_512(mq3,mq3,mq0);
            ub0++;
        
            //=====   +1  ========
            #ifdef MIC_PREFETCH
            intrin_prefetch_su3_512(ub0+1);
            intrin_prefetch_halfspinor_512(phip2);
            #endif
            intrin_su3_load_short_512(U,ub0);
            intrin_vector_load_512(out1,&(phip1->s0));
            intrin_vector_load_512(out2,&(phip1->s1));
            intrin_su3_multiply_512(map0,U,out1);
            intrin_su3_multiply_512(map1,U,out2);
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_sub_512(mq2,mq2,map1);
            intrin_vector_add_512(mq3,mq3,map0);
            ub0++;

        
            //=====   +2  ========
            #ifdef MIC_PREFETCH
            intrin_prefetch_su3_512(ub0+1);
            intrin_prefetch_halfspinor_512(phip3);
            #endif
            intrin_su3_load_short_512(U,ub0);
            intrin_vector_load_512(out1,&(phip2->s0));
            intrin_vector_load_512(out2,&(phip2->s1));
            intrin_su3_multiply_512(map0,U,out1);
            intrin_su3_multiply_512(map1,U,out2);
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_i_sub_512(mq2,mq2,map0);
            intrin_vector_i_add_512(mq3,mq3,map1);
            ub0++;

        

            phim0 = &plqcd_g.phim512[0][ipt];
            phim1 = &plqcd_g.phim512[1][ipt];
            phim2 = &plqcd_g.phim512[2][ipt];
            phim3 = &plqcd_g.phim512[3][ipt];


            //=====   +3  ========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim0);
            #endif
            intrin_su3_load_short_512(U,ub0);
            intrin_vector_load_512(out1,&(phip3->s0));
            intrin_vector_load_512(out2,&(phip3->s1));
            intrin_su3_multiply_512(map0,U,out1);
            intrin_su3_multiply_512(map1,U,out2);
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_add_512(mq2,mq2,map0);
            intrin_vector_add_512(mq3,mq3,map1);

            //now add the -ve pieces
            //======= -0 =========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim1);
            #endif
            intrin_vector_load_512(map0,&(phim0->s0));
            intrin_vector_load_512(map1,&(phim0->s1));
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_i_add_512(mq2,mq2,map1);
            intrin_vector_i_add_512(mq3,mq3,map0);


            //======= -1 =========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim2);
            #endif
            intrin_vector_load_512(map0,&(phim1->s0));
            intrin_vector_load_512(map1,&(phim1->s1));
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_add_512(mq2,mq2,map1);
            intrin_vector_sub_512(mq3,mq3,map0);

            //======= -2 =========
            #ifdef MIC_PREFETCH
            intrin_prefetch_halfspinor_512(phim3);
            #endif
            intrin_vector_load_512(map0,&(phim2->s0));
            intrin_vector_load_512(map1,&(phim2->s1));
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_i_add_512(mq2,mq2,map0);
            intrin_vector_i_sub_512(mq3,mq3,map1);


            //======= -3 =========
            #ifdef MIC_PREFETCH
            intrin_prefetch_spinor_512(&qout[ipt]);
            #endif
            intrin_vector_load_512(map0,&(phim3->s0));
            intrin_vector_load_512(map1,&(phim3->s1));
            intrin_vector_add_512(mq0,mq0,map0);
            intrin_vector_add_512(mq1,mq1,map1);
            intrin_vector_sub_512(mq2,mq2,map0);
            intrin_vector_sub_512(mq3,mq3,map1);


            //store the result
            intrin_vector_store_512(&qout[ipt].s0,mq0);
            intrin_vector_store_512(&qout[ipt].s1,mq1);
            intrin_vector_store_512(&qout[ipt].s2,mq2);
            intrin_vector_store_512(&qout[ipt].s3,mq3);

      }

   }

#ifdef _OPENMP
}
#endif  //end of the openmp parallel reigon

   
   return stop_watch(ts);
}




#ifdef MIC_SPLIT
//======================================================
//Single MIC, split real and imaginary, double 
//================Even-Odd==============================
double plqcd_hopping_matrix_eo_single_mic_split(double *qout_re, 
                                                double *qout_im, 
                                                double *u_re, 
                                                double *u_im,
                                                double *qin_re, 
                                                double *qin_im) 
{
   double ts;          //timer
   ts=stop_watch(0.0); //start
   
   #ifdef _OPENMP
   #pragma omp parallel 
   {
   #endif
      int V,Vmic_split,Vyzt,lx;
      V = plqcd_g.VOLUME;
      Vmic_split = plqcd_g.Vmic_split;
      Vyzt = plqcd_g.latdims[1]*plqcd_g.latdims[2]*plqcd_g.latdims[3];
      lx=plqcd_g.latdims[0]/8;
      if((plqcd_g.latdims[0]%16) != 0 ){ 
        fprintf(stderr,"lattice size in the 0 direction must be a factor of 16\n");
        exit(1);
      }
      
      __m512d  mqin_re[12],mqin_im[12];
      __m512d  U_re[3][3], U_im[3][3];
      __m512d  out1_re[3],out2_re[3],out3_re[3],out4_re[3];
      __m512d  out1_im[3],out2_im[3],out3_im[3],out4_im[3];
      __m512d  map0_re[3],map0_im[3],map1_re[3],map1_im[3];

      int iup[4],idn[4];    
      double *ub0_re,*ub0_im;
      double *sin_re,*sin_im;
      double *sout_re,*sout_im;
      int ix,iy,it,iz,is,xcor[4],ipt;

      int L[4];
      for(int i=0; i<4; i++)
         L[i] = plqcd_g.latdims[i];


      double *phip0_re,*phip1_re,*phip2_re,*phip3_re,*phip0_im,*phip1_im,*phip2_im,*phip3_im;
      double *phim0_re,*phim1_re,*phim2_re,*phim3_re,*phim0_im,*phim1_im,*phim2_im,*phim3_im;

      //for permuting the eight doubles from 0,1,2,3,4,5,6,7 order into 1,2,3,4,5,6,7,0 order
      //note each double correspond to two ints
      int  __attribute__ ((aligned(64)))  permback[16]={2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1};
      __m512i mpermback = _mm512_load_epi32(permback);

      //for permuting the eight doubles from 0,1,2,3,4,5,6,7 order into 7,0,1,2,3,4,5,6 order
      //note each double correspond to two ints
      int __attribute__ ((aligned(64)))   permfor[16]={14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13};
      __m512i mpermfor = _mm512_load_epi32(permfor);


      //-----------------------------------------------------
      // loop over input spinor on odd sites
      // compute (1-gamma_mu)qin terms and store in phip
      // compute U^dagger*(1+gamma_mu)qin and store in phim
      //----------------------------------------------------
      #ifdef _OPENMP
      //#pragma omp for schedule(static,20) 
      #pragma omp for 
      #endif   
      for(int is=0; is < Vyzt; is++) 
      {
         iy = is%L[1];
         iz = (is/L[1]) %L[2];
         it = (is/L[1]/L[2])%L[3];

         for(ix = (iy+iz+it+1)%2; ix < lx; ix +=2)
         {
            xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
            ipt=plqcd_g.ipt_eo_mic_split[cart2lex(xcor)]; //odd site
            //printf("A\n");
            //printf("ix iy iz it ipt idn iup : %d %d %d %d %d %d %d %d %d %d %d %d %d\n",ix,iy,iz,it,ipt,
            //       plqcd_g.idn_mic_split[ipt][0],plqcd_g.idn_mic_split[ipt][1],plqcd_g.idn_mic_split[ipt][2],plqcd_g.idn_mic_split[ipt][3],
            //       plqcd_g.iup_mic_split[ipt][0],plqcd_g.iup_mic_split[ipt][1],plqcd_g.iup_mic_split[ipt][2],plqcd_g.iup_mic_split[ipt][3]);

            ub0_re = u_re+ipt*8*36;
            ub0_im = u_im+ipt*8*36;

            sin_re = qin_re + ipt*8*12;
            sin_im = qin_im + ipt*8*12;
         
            phip0_re = &plqcd_g.phip512_re[0][plqcd_g.idn_mic_split[ipt][0]*8*6]; 
            phip1_re = &plqcd_g.phip512_re[1][plqcd_g.idn_mic_split[ipt][1]*8*6];
            phip2_re = &plqcd_g.phip512_re[2][plqcd_g.idn_mic_split[ipt][2]*8*6];
            phip3_re = &plqcd_g.phip512_re[3][plqcd_g.idn_mic_split[ipt][3]*8*6];

            phip0_im = &plqcd_g.phip512_im[0][plqcd_g.idn_mic_split[ipt][0]*8*6];
            phip1_im = &plqcd_g.phip512_im[1][plqcd_g.idn_mic_split[ipt][1]*8*6];
            phip2_im = &plqcd_g.phip512_im[2][plqcd_g.idn_mic_split[ipt][2]*8*6];
            phip3_im = &plqcd_g.phip512_im[3][plqcd_g.idn_mic_split[ipt][3]*8*6];


            phim0_re = &plqcd_g.phim512_re[0][plqcd_g.iup_mic_split[ipt][0]*8*6]; 
            phim1_re = &plqcd_g.phim512_re[1][plqcd_g.iup_mic_split[ipt][1]*8*6];
            phim2_re = &plqcd_g.phim512_re[2][plqcd_g.iup_mic_split[ipt][2]*8*6];
            phim3_re = &plqcd_g.phim512_re[3][plqcd_g.iup_mic_split[ipt][3]*8*6];

            phim0_im = &plqcd_g.phim512_im[0][plqcd_g.iup_mic_split[ipt][0]*8*6];
            phim1_im = &plqcd_g.phim512_im[1][plqcd_g.iup_mic_split[ipt][1]*8*6];
            phim2_im = &plqcd_g.phim512_im[2][plqcd_g.iup_mic_split[ipt][2]*8*6];
            phim3_im = &plqcd_g.phim512_im[3][plqcd_g.iup_mic_split[ipt][3]*8*6];




            //#ifdef MIC_PREFETCH
            //intrin_prefetch_spinor_512(&qin[ipt]);
            //intrin_prefetch_halfspinor_512(phip0);
            //#endif


            //intrin_vector_load_512(qins0,&qin[ipt].s0);
            //intrin_vector_load_512(qins1,&qin[ipt].s1);
            //intrin_vector_load_512(qins2,&qin[ipt].s2);
            //intrin_vector_load_512(qins3,&qin[ipt].s3);

            mqin_re[0]  = _mm512_load_pd(sin_re);
            mqin_re[1]  = _mm512_load_pd(sin_re+8);
            mqin_re[2]  = _mm512_load_pd(sin_re+16);
            mqin_re[3]  = _mm512_load_pd(sin_re+24);
            mqin_re[4]  = _mm512_load_pd(sin_re+32);
            mqin_re[5]  = _mm512_load_pd(sin_re+40);
            mqin_re[6]  = _mm512_load_pd(sin_re+48);
            mqin_re[7]  = _mm512_load_pd(sin_re+56);
            mqin_re[8]  = _mm512_load_pd(sin_re+64);
            mqin_re[9]  = _mm512_load_pd(sin_re+72);
            mqin_re[10] = _mm512_load_pd(sin_re+80);
            mqin_re[11] = _mm512_load_pd(sin_re+88);

            mqin_im[0]  = _mm512_load_pd(sin_im);
            mqin_im[1]  = _mm512_load_pd(sin_im+8);
            mqin_im[2]  = _mm512_load_pd(sin_im+16);
            mqin_im[3]  = _mm512_load_pd(sin_im+24);
            mqin_im[4]  = _mm512_load_pd(sin_im+32);
            mqin_im[5]  = _mm512_load_pd(sin_im+40);
            mqin_im[6]  = _mm512_load_pd(sin_im+48);
            mqin_im[7]  = _mm512_load_pd(sin_im+56);
            mqin_im[8]  = _mm512_load_pd(sin_im+64);
            mqin_im[9]  = _mm512_load_pd(sin_im+72);
            mqin_im[10] = _mm512_load_pd(sin_im+80);
            mqin_im[11] = _mm512_load_pd(sin_im+88);




            //=====   +0  ========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phip1);
            //#endif
            //intrin_vector_i_add_512(out1,qins0,qins3);
            //intrin_vector_i_add_512(out2,qins1,qins2);
            
            out1_re[0] = _mm512_sub_pd(mqin_re[0],mqin_im[9]);            
            out1_re[1] = _mm512_sub_pd(mqin_re[1],mqin_im[10]);            
            out1_re[2] = _mm512_sub_pd(mqin_re[2],mqin_im[11]);            

            out1_im[0] = _mm512_add_pd(mqin_im[0],mqin_re[9]);            
            out1_im[1] = _mm512_add_pd(mqin_im[1],mqin_re[10]);            
            out1_im[2] = _mm512_add_pd(mqin_im[2],mqin_re[11]);            

            out2_re[0] = _mm512_sub_pd(mqin_re[3],mqin_im[6]);            
            out2_re[1] = _mm512_sub_pd(mqin_re[4],mqin_im[7]);            
            out2_re[2] = _mm512_sub_pd(mqin_re[5],mqin_im[8]);            

            out2_im[0] = _mm512_add_pd(mqin_im[3],mqin_re[6]);            
            out2_im[1] = _mm512_add_pd(mqin_im[4],mqin_re[7]);            
            out2_im[2] = _mm512_add_pd(mqin_im[5],mqin_re[8]);            

            
            if(ix == 0)
            {
               //printf("ix iy iz it ipt Hello: %d %d %d %d %d\n",ix,iy,iz,it,ipt);
               for(int i=0; i<3; i++)
               {  
                  out3_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) out1_re[i]);
                  out3_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) out1_im[i]);
                  out4_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) out2_re[i]);
                  out4_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) out2_im[i]);
               }

               //intrin_vector_store_512(&(phip0->s0),out3);
               //intrin_vector_store_512(&(phip0->s1),out4);

               _mm512_store_pd(phip0_re   ,out3_re[0]);
               _mm512_store_pd(phip0_re+8 ,out3_re[1]);
               _mm512_store_pd(phip0_re+16,out3_re[2]);
               _mm512_store_pd(phip0_re+24,out4_re[0]);
               _mm512_store_pd(phip0_re+32,out4_re[1]);
               _mm512_store_pd(phip0_re+40,out4_re[2]);

               _mm512_store_pd(phip0_im   ,out3_im[0]);
               _mm512_store_pd(phip0_im+8 ,out3_im[1]);
               _mm512_store_pd(phip0_im+16,out3_im[2]);
               _mm512_store_pd(phip0_im+24,out4_im[0]);
               _mm512_store_pd(phip0_im+32,out4_im[1]);
               _mm512_store_pd(phip0_im+40,out4_im[2]);
            }
            else
            {
            
               //printf("ix iy iz it ipt Hi: %d %d %d %d %d\n",ix,iy,iz,it,ipt);
               _mm512_store_pd(phip0_re   ,out1_re[0]);
               _mm512_store_pd(phip0_re+8 ,out1_re[1]);
               _mm512_store_pd(phip0_re+16,out1_re[2]);
               _mm512_store_pd(phip0_re+24,out2_re[0]);
               _mm512_store_pd(phip0_re+32,out2_re[1]);
               _mm512_store_pd(phip0_re+40,out2_re[2]);

               _mm512_store_pd(phip0_im   ,out1_im[0]);
               _mm512_store_pd(phip0_im+8 ,out1_im[1]);
               _mm512_store_pd(phip0_im+16,out1_im[2]);
               _mm512_store_pd(phip0_im+24,out2_im[0]);
               _mm512_store_pd(phip0_im+32,out2_im[1]);
               _mm512_store_pd(phip0_im+40,out2_im[2]);
            }
           
            //printf("N\n");


            //===== +1 ==========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phip2);  
            //#endif
            //intrin_vector_add_512(out1,qins0,qins3);
            //intrin_vector_sub_512(out2,qins1,qins2);
            //intrin_vector_store_512(&(phip1->s0),out1);
            //intrin_vector_store_512(&(phip1->s1),out2);
      

            out1_re[0] = _mm512_add_pd(mqin_re[0],mqin_re[9]);            
            out1_re[1] = _mm512_add_pd(mqin_re[1],mqin_re[10]);            
            out1_re[2] = _mm512_add_pd(mqin_re[2],mqin_re[11]);            

            out1_im[0] = _mm512_add_pd(mqin_im[0],mqin_im[9]);            
            out1_im[1] = _mm512_add_pd(mqin_im[1],mqin_im[10]);            
            out1_im[2] = _mm512_add_pd(mqin_im[2],mqin_im[11]);            

            out2_re[0] = _mm512_sub_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm512_sub_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm512_sub_pd(mqin_re[5],mqin_re[8]);            

            out2_im[0] = _mm512_sub_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm512_sub_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm512_sub_pd(mqin_im[5],mqin_im[8]);            


            _mm512_store_pd(phip1_re   ,out1_re[0]);
            _mm512_store_pd(phip1_re+8 ,out1_re[1]);
            _mm512_store_pd(phip1_re+16,out1_re[2]);
            _mm512_store_pd(phip1_re+24,out2_re[0]);
            _mm512_store_pd(phip1_re+32,out2_re[1]);
            _mm512_store_pd(phip1_re+40,out2_re[2]);

            _mm512_store_pd(phip1_im   ,out1_im[0]);
            _mm512_store_pd(phip1_im+8 ,out1_im[1]);
            _mm512_store_pd(phip1_im+16,out1_im[2]);
            _mm512_store_pd(phip1_im+24,out2_im[0]);
            _mm512_store_pd(phip1_im+32,out2_im[1]);
            _mm512_store_pd(phip1_im+40,out2_im[2]);


            //===== +2 ==========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phip3);
            //#endif
            //intrin_vector_i_add_512(out1,qins0,qins2);
            //intrin_vector_i_sub_512(out2,qins1,qins3);
            //intrin_vector_store_512(&(phip2->s0),out1);
            //intrin_vector_store_512(&(phip2->s1),out2);
        

            out1_re[0] = _mm512_sub_pd(mqin_re[0],mqin_im[6]);            
            out1_re[1] = _mm512_sub_pd(mqin_re[1],mqin_im[7]);            
            out1_re[2] = _mm512_sub_pd(mqin_re[2],mqin_im[8]);            

            out1_im[0] = _mm512_add_pd(mqin_im[0],mqin_re[6]);            
            out1_im[1] = _mm512_add_pd(mqin_im[1],mqin_re[7]);            
            out1_im[2] = _mm512_add_pd(mqin_im[2],mqin_re[8]);            

            out2_re[0] = _mm512_add_pd(mqin_re[3],mqin_im[9]);            
            out2_re[1] = _mm512_add_pd(mqin_re[4],mqin_im[10]);            
            out2_re[2] = _mm512_add_pd(mqin_re[5],mqin_im[11]);            

            out2_im[0] = _mm512_sub_pd(mqin_im[3],mqin_re[9]);            
            out2_im[1] = _mm512_sub_pd(mqin_im[4],mqin_re[10]);            
            out2_im[2] = _mm512_sub_pd(mqin_im[5],mqin_re[11]);            


            _mm512_store_pd(phip2_re   ,out1_re[0]);
            _mm512_store_pd(phip2_re+8 ,out1_re[1]);
            _mm512_store_pd(phip2_re+16,out1_re[2]);
            _mm512_store_pd(phip2_re+24,out2_re[0]);
            _mm512_store_pd(phip2_re+32,out2_re[1]);
            _mm512_store_pd(phip2_re+40,out2_re[2]);

            _mm512_store_pd(phip2_im   ,out1_im[0]);
            _mm512_store_pd(phip2_im+8 ,out1_im[1]);
            _mm512_store_pd(phip2_im+16,out1_im[2]);
            _mm512_store_pd(phip2_im+24,out2_im[0]);
            _mm512_store_pd(phip2_im+32,out2_im[1]);
            _mm512_store_pd(phip2_im+40,out2_im[2]);


            //===== +3 ==========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phim0);
            //intrin_prefetch_su3_512(ub0);
            //#endif
            //intrin_vector_add_512(out1,qins0,qins2);
            //intrin_vector_add_512(out2,qins1,qins3);
            //intrin_vector_store_512(&(phip3->s0),out1);
            //intrin_vector_store_512(&(phip3->s1),out2);
        

            out1_re[0] = _mm512_add_pd(mqin_re[0],mqin_re[6]);            
            out1_re[1] = _mm512_add_pd(mqin_re[1],mqin_re[7]);            
            out1_re[2] = _mm512_add_pd(mqin_re[2],mqin_re[8]);            

            out1_im[0] = _mm512_add_pd(mqin_im[0],mqin_im[6]);            
            out1_im[1] = _mm512_add_pd(mqin_im[1],mqin_im[7]);            
            out1_im[2] = _mm512_add_pd(mqin_im[2],mqin_im[8]);            

            out2_re[0] = _mm512_add_pd(mqin_re[3],mqin_re[9]);            
            out2_re[1] = _mm512_add_pd(mqin_re[4],mqin_re[10]);            
            out2_re[2] = _mm512_add_pd(mqin_re[5],mqin_re[11]);            

            out2_im[0] = _mm512_add_pd(mqin_im[3],mqin_im[9]);            
            out2_im[1] = _mm512_add_pd(mqin_im[4],mqin_im[10]);            
            out2_im[2] = _mm512_add_pd(mqin_im[5],mqin_im[11]);            


            _mm512_store_pd(phip3_re   ,out1_re[0]);
            _mm512_store_pd(phip3_re+8 ,out1_re[1]);
            _mm512_store_pd(phip3_re+16,out1_re[2]);
            _mm512_store_pd(phip3_re+24,out2_re[0]);
            _mm512_store_pd(phip3_re+32,out2_re[1]);
            _mm512_store_pd(phip3_re+40,out2_re[2]);

            _mm512_store_pd(phip3_im   ,out1_im[0]);
            _mm512_store_pd(phip3_im+8 ,out1_im[1]);
            _mm512_store_pd(phip3_im+16,out1_im[2]);
            _mm512_store_pd(phip3_im+24,out2_im[0]);
            _mm512_store_pd(phip3_im+32,out2_im[1]);
            _mm512_store_pd(phip3_im+40,out2_im[2]);


            //===== -0  ============================= 
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phim1);
            //intrin_prefetch_su3_512(ub0+1);
            //#endif
            //=======================================

            //intrin_su3_load_512(U,ub0);
            
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ub0_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ub0_im+jc*8+ic*24);
               }
            

            //intrin_vector_i_sub_512(out1,qins0,qins3);
            //intrin_vector_i_sub_512(out2,qins1,qins2);

            out1_re[0] = _mm512_add_pd(mqin_re[0],mqin_im[9]);            
            out1_re[1] = _mm512_add_pd(mqin_re[1],mqin_im[10]);            
            out1_re[2] = _mm512_add_pd(mqin_re[2],mqin_im[11]);            

            out1_im[0] = _mm512_sub_pd(mqin_im[0],mqin_re[9]);            
            out1_im[1] = _mm512_sub_pd(mqin_im[1],mqin_re[10]);            
            out1_im[2] = _mm512_sub_pd(mqin_im[2],mqin_re[11]);            

            out2_re[0] = _mm512_add_pd(mqin_re[3],mqin_im[6]);            
            out2_re[1] = _mm512_add_pd(mqin_re[4],mqin_im[7]);            
            out2_re[2] = _mm512_add_pd(mqin_re[5],mqin_im[8]);            

            out2_im[0] = _mm512_sub_pd(mqin_im[3],mqin_re[6]);            
            out2_im[1] = _mm512_sub_pd(mqin_im[4],mqin_re[7]);            
            out2_im[2] = _mm512_sub_pd(mqin_im[5],mqin_re[8]);            

            su3_inverse_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
             
            if(xcor[0] == (lx-1))
            {
               for(int i=0; i<3; i++)
               {
                  out3_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map0_re[i]);
                  out3_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map0_im[i]);
                  out4_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map1_re[i]);
                  out4_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map1_im[i]);
               }

               //intrin_vector_store_512(&(phim0->s0),out3);
               //intrin_vector_store_512(&(phim0->s1),out4);
               _mm512_store_pd(phim0_re   ,out3_re[0]);
               _mm512_store_pd(phim0_re+8 ,out3_re[1]);
               _mm512_store_pd(phim0_re+16,out3_re[2]);
               _mm512_store_pd(phim0_re+24,out4_re[0]);
               _mm512_store_pd(phim0_re+32,out4_re[1]);
               _mm512_store_pd(phim0_re+40,out4_re[2]);

               _mm512_store_pd(phim0_im   ,out3_im[0]);
               _mm512_store_pd(phim0_im+8 ,out3_im[1]);
               _mm512_store_pd(phim0_im+16,out3_im[2]);
               _mm512_store_pd(phim0_im+24,out4_im[0]);
               _mm512_store_pd(phim0_im+32,out4_im[1]);
               _mm512_store_pd(phim0_im+40,out4_im[2]);
            }
            else
            {
            
               _mm512_store_pd(phim0_re   ,map0_re[0]);
               _mm512_store_pd(phim0_re+8 ,map0_re[1]);
               _mm512_store_pd(phim0_re+16,map0_re[2]);
               _mm512_store_pd(phim0_re+24,map1_re[0]);
               _mm512_store_pd(phim0_re+32,map1_re[1]);
               _mm512_store_pd(phim0_re+40,map1_re[2]);

               _mm512_store_pd(phim0_im   ,map0_im[0]);
               _mm512_store_pd(phim0_im+8 ,map0_im[1]);
               _mm512_store_pd(phim0_im+16,map0_im[2]);
               _mm512_store_pd(phim0_im+24,map1_im[0]);
               _mm512_store_pd(phim0_im+32,map1_im[1]);
               _mm512_store_pd(phim0_im+40,map1_im[2]);
            }

            ub0_re += 72 ;
            ub0_im += 72 ;


            
            //===== -1  ========== 
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phim2);
            //intrin_prefetch_su3_512(ub0+1);
            //#endif
            //intrin_su3_load__512(U,ub0);
            
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ub0_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ub0_im+jc*8+ic*24);
               }
            
            //intrin_vector_sub_512(out1,qins0,qins3);
            //intrin_vector_add_512(out2,qins1,qins2);
            //intrin_su3_inverse_multiply_512(map0,U,out1);
            //intrin_su3_inverse_multiply_512(map1,U,out2);
            //intrin_vector_store_512(&(phim0->s0),map0);
            //intrin_vector_store_512(&(phim0->s1),map1);
            //ub0++;
            //
            out1_re[0] = _mm512_sub_pd(mqin_re[0],mqin_re[9]);            
            out1_re[1] = _mm512_sub_pd(mqin_re[1],mqin_re[10]);            
            out1_re[2] = _mm512_sub_pd(mqin_re[2],mqin_re[11]);            

            out1_im[0] = _mm512_sub_pd(mqin_im[0],mqin_im[9]);            
            out1_im[1] = _mm512_sub_pd(mqin_im[1],mqin_im[10]);            
            out1_im[2] = _mm512_sub_pd(mqin_im[2],mqin_im[11]);            


            out2_re[0] = _mm512_add_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm512_add_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm512_add_pd(mqin_re[5],mqin_re[8]);            


            out2_im[0] = _mm512_add_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm512_add_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm512_add_pd(mqin_im[5],mqin_im[8]);            

            su3_inverse_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            
            _mm512_store_pd(phim1_re   ,map0_re[0]);
            _mm512_store_pd(phim1_re+8 ,map0_re[1]);
            _mm512_store_pd(phim1_re+16,map0_re[2]);
            _mm512_store_pd(phim1_re+24,map1_re[0]);
            _mm512_store_pd(phim1_re+32,map1_re[1]);
            _mm512_store_pd(phim1_re+40,map1_re[2]);

            _mm512_store_pd(phim1_im   ,map0_im[0]);
            _mm512_store_pd(phim1_im+8 ,map0_im[1]);
            _mm512_store_pd(phim1_im+16,map0_im[2]);
            _mm512_store_pd(phim1_im+24,map1_im[0]);
            _mm512_store_pd(phim1_im+32,map1_im[1]);
            _mm512_store_pd(phim1_im+40,map1_im[2]);

            ub0_re += 72 ;
            ub0_im += 72 ;


            //===== -2  ========== 
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phim3);
            //intrin_prefetch_su3_512(ub0+1);
            //#endif
            //intrin_su3_load_short_512(U,ub0);
            //intrin_vector_i_sub_512(out1,qins0,qins2);
            //intrin_su3_inverse_multiply_512(map0,U,out1);
            //intrin_vector_i_add_512(out2,qins1,qins3);
            //intrin_su3_inverse_multiply_512(map1,U,out2);
            //intrin_vector_store_512(&(phim0->s0),map0);
            //intrin_vector_store_512(&(phim0->s1),map1);
            //ub0++;
            //
            //
           
            //intrin_su3_load__512(U,ub0);
            
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ub0_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ub0_im+jc*8+ic*24);
               }
            
            out1_re[0] = _mm512_add_pd(mqin_re[0],mqin_im[6]);            
            out1_re[1] = _mm512_add_pd(mqin_re[1],mqin_im[7]);            
            out1_re[2] = _mm512_add_pd(mqin_re[2],mqin_im[8]);            

            out1_im[0] = _mm512_sub_pd(mqin_im[0],mqin_re[6]);            
            out1_im[1] = _mm512_sub_pd(mqin_im[1],mqin_re[7]);            
            out1_im[2] = _mm512_sub_pd(mqin_im[2],mqin_re[8]);            


            out2_re[0] = _mm512_sub_pd(mqin_re[3],mqin_im[9]);            
            out2_re[1] = _mm512_sub_pd(mqin_re[4],mqin_im[10]);            
            out2_re[2] = _mm512_sub_pd(mqin_re[5],mqin_im[11]);            


            out2_im[0] = _mm512_add_pd(mqin_im[3],mqin_re[9]);            
            out2_im[1] = _mm512_add_pd(mqin_im[4],mqin_re[10]);            
            out2_im[2] = _mm512_add_pd(mqin_im[5],mqin_re[11]);            

            su3_inverse_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            
            _mm512_store_pd(phim2_re   ,map0_re[0]);
            _mm512_store_pd(phim2_re+8 ,map0_re[1]);
            _mm512_store_pd(phim2_re+16,map0_re[2]);
            _mm512_store_pd(phim2_re+24,map1_re[0]);
            _mm512_store_pd(phim2_re+32,map1_re[1]);
            _mm512_store_pd(phim2_re+40,map1_re[2]);

            _mm512_store_pd(phim2_im   ,map0_im[0]);
            _mm512_store_pd(phim2_im+8 ,map0_im[1]);
            _mm512_store_pd(phim2_im+16,map0_im[2]);
            _mm512_store_pd(phim2_im+24,map1_im[0]);
            _mm512_store_pd(phim2_im+32,map1_im[1]);
            _mm512_store_pd(phim2_im+40,map1_im[2]);

            ub0_re += 72 ;
            ub0_im += 72 ;



            //===== -3  ========== 
            //intrin_su3_load_short_512(U,ub0);
            //intrin_vector_sub_512(out1,qins0,qins2);
            //intrin_su3_inverse_multiply_512(map0,U,out1);
            //intrin_vector_sub_512(out2,qins1,qins3);
            //intrin_su3_inverse_multiply_512(map1,U,out2);
            //intrin_vector_store_512(&(phim0->s0),map0);
            //intrin_vector_store_512(&(phim0->s1),map1);
            //
           
            //intrin_su3_load__512(U,ub0);
            
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ub0_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ub0_im+jc*8+ic*24);
               }
            
            out1_re[0] = _mm512_sub_pd(mqin_re[0],mqin_re[6]);            
            out1_re[1] = _mm512_sub_pd(mqin_re[1],mqin_re[7]);            
            out1_re[2] = _mm512_sub_pd(mqin_re[2],mqin_re[8]);            

            out1_im[0] = _mm512_sub_pd(mqin_im[0],mqin_im[6]);            
            out1_im[1] = _mm512_sub_pd(mqin_im[1],mqin_im[7]);            
            out1_im[2] = _mm512_sub_pd(mqin_im[2],mqin_im[8]);            


            out2_re[0] = _mm512_sub_pd(mqin_re[3],mqin_re[9]);            
            out2_re[1] = _mm512_sub_pd(mqin_re[4],mqin_re[10]);            
            out2_re[2] = _mm512_sub_pd(mqin_re[5],mqin_re[11]);            


            out2_im[0] = _mm512_sub_pd(mqin_im[3],mqin_im[9]);            
            out2_im[1] = _mm512_sub_pd(mqin_im[4],mqin_im[10]);            
            out2_im[2] = _mm512_sub_pd(mqin_im[5],mqin_im[11]);            

            su3_inverse_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
           
            _mm512_store_pd(phim3_re   ,map0_re[0]);
            _mm512_store_pd(phim3_re+8 ,map0_re[1]);
            _mm512_store_pd(phim3_re+16,map0_re[2]);
            _mm512_store_pd(phim3_re+24,map1_re[0]);
            _mm512_store_pd(phim3_re+32,map1_re[1]);
            _mm512_store_pd(phim3_re+40,map1_re[2]);

            _mm512_store_pd(phim3_im   ,map0_im[0]);
            _mm512_store_pd(phim3_im+8 ,map0_im[1]);
            _mm512_store_pd(phim3_im+16,map0_im[2]);
            _mm512_store_pd(phim3_im+24,map1_im[0]);
            _mm512_store_pd(phim3_im+32,map1_im[1]);
            _mm512_store_pd(phim3_im+40,map1_im[2]);

           
         }
      }


      //-------------------------------------------------------
      // build the result
      // loop over the output spinor on even sites
      // compute U*phip terms and store in qout
      // store phim in qout
      // later on we will build the full twisted mass operator
      //--------------------------------------------------------


      __m512d mq0_re[3],mq1_re[3],mq2_re[3],mq3_re[3]; 
      __m512d mq0_im[3],mq1_im[3],mq2_im[3],mq3_im[3]; 

      #ifdef _OPENMP
      //#pragma omp for schedule(static,20) 
      #pragma omp for  
      #endif   
      for(int is=0; is < Vyzt; is++) 
      {
         iy = is%L[1];
         iz = (is/L[1]) %L[2];
         it = (is/L[1]/L[2])%L[3];

         for(ix = (iy+iz+it)%2; ix < lx; ix +=2)
         {
            xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
            ipt=plqcd_g.ipt_eo_mic_split[cart2lex(xcor)];


            ub0_re = &u_re[ipt*8*36];
            ub0_im = &u_im[ipt*8*36];
         
            phip0_re = &plqcd_g.phip512_re[0][ipt*8*6]; 
            phip1_re = &plqcd_g.phip512_re[1][ipt*8*6];
            phip2_re = &plqcd_g.phip512_re[2][ipt*8*6];
            phip3_re = &plqcd_g.phip512_re[3][ipt*8*6];

            phip0_im = &plqcd_g.phip512_im[0][ipt*8*6];
            phip1_im = &plqcd_g.phip512_im[1][ipt*8*6];
            phip2_im = &plqcd_g.phip512_im[2][ipt*8*6];
            phip3_im = &plqcd_g.phip512_im[3][ipt*8*6];


            phim0_re = &plqcd_g.phim512_re[0][ipt*8*6]; 
            phim1_re = &plqcd_g.phim512_re[1][ipt*8*6];
            phim2_re = &plqcd_g.phim512_re[2][ipt*8*6];
            phim3_re = &plqcd_g.phim512_re[3][ipt*8*6];

            phim0_im = &plqcd_g.phim512_im[0][ipt*8*6];
            phim1_im = &plqcd_g.phim512_im[1][ipt*8*6];
            phim2_im = &plqcd_g.phim512_im[2][ipt*8*6];
            phim3_im = &plqcd_g.phim512_im[3][ipt*8*6];


            //=====   +0  ========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_su3_512(ub0+1);
            //intrin_prefetch_halfspinor_512(phip1);
            //#endif
            //intrin_su3_load_512(U,ub0);
            //intrin_vector_load_512(out1,&(phip0->s0));
            //intrin_vector_load_512(out2,&(phip0->s1));
            //intrin_su3_multiply_512(mq0,U,out1);
            //intrin_su3_multiply_512(mq1,U,out2);
            //intrin_vector_i_sub_512(mq2,mq2,mq1);
            //intrin_vector_i_sub_512(mq3,mq3,mq0);
            //ub0++;
            //
            
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ub0_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ub0_im+jc*8+ic*24);
               }
            
            out1_re[0]  = _mm512_load_pd(phip0_re);
            out1_re[1]  = _mm512_load_pd(phip0_re+8);
            out1_re[2]  = _mm512_load_pd(phip0_re+16);
            out2_re[0]  = _mm512_load_pd(phip0_re+24);
            out2_re[1]  = _mm512_load_pd(phip0_re+32);
            out2_re[2]  = _mm512_load_pd(phip0_re+40);


            out1_im[0]  = _mm512_load_pd(phip0_im);
            out1_im[1]  = _mm512_load_pd(phip0_im+8);
            out1_im[2]  = _mm512_load_pd(phip0_im+16);
            out2_im[0]  = _mm512_load_pd(phip0_im+24);
            out2_im[1]  = _mm512_load_pd(phip0_im+32);
            out2_im[2]  = _mm512_load_pd(phip0_im+40);

            su3_multiply_splitlayout_512(mq0_re,mq0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_512(mq1_re,mq1_im,U_re,U_im,out2_re,out2_im);

            mq2_re[0] = _mm512_add_pd(mq2_re[0],mq1_im[0]);            
            mq2_re[1] = _mm512_add_pd(mq2_re[1],mq1_im[1]);            
            mq2_re[2] = _mm512_add_pd(mq2_re[2],mq1_im[2]);            

            mq2_im[0] = _mm512_sub_pd(mq2_im[0],mq1_re[0]);            
            mq2_im[1] = _mm512_sub_pd(mq2_im[1],mq1_re[1]);            
            mq2_im[2] = _mm512_sub_pd(mq2_im[2],mq1_re[2]);            

            mq3_re[0] = _mm512_add_pd(mq3_re[0],mq0_im[0]);            
            mq3_re[1] = _mm512_add_pd(mq3_re[1],mq0_im[1]);            
            mq3_re[2] = _mm512_add_pd(mq3_re[2],mq0_im[2]);            

            mq3_im[0] = _mm512_sub_pd(mq3_im[0],mq0_re[0]);            
            mq3_im[1] = _mm512_sub_pd(mq3_im[1],mq0_re[1]);            
            mq3_im[2] = _mm512_sub_pd(mq3_im[2],mq0_re[2]);            

            ub0_re += 72;
            ub0_im += 72;



            //=====   +1  ========
           // #ifdef MIC_PREFETCH
            //intrin_prefetch_su3_512(ub0+1);
            //intrin_prefetch_halfspinor_512(phip2);
            //#endif
            //intrin_su3_load_short_512(U,ub0);
            //intrin_vector_load_512(out1,&(phip1->s0));
            //intrin_vector_load_512(out2,&(phip1->s1));
            //intrin_su3_multiply_512(map0,U,out1);
            //intrin_su3_multiply_512(map1,U,out2);
            //intrin_vector_add_512(mq0,mq0,map0);
            //intrin_vector_add_512(mq1,mq1,map1);
            //intrin_vector_sub_512(mq2,mq2,map1);
            //intrin_vector_add_512(mq3,mq3,map0);
            //ub0++;

            
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ub0_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ub0_im+jc*8+ic*24);
               }
            
        
            out1_re[0]  = _mm512_load_pd(phip1_re);
            out1_re[1]  = _mm512_load_pd(phip1_re+8);
            out1_re[2]  = _mm512_load_pd(phip1_re+16);
            out2_re[0]  = _mm512_load_pd(phip1_re+24);
            out2_re[1]  = _mm512_load_pd(phip1_re+32);
            out2_re[2]  = _mm512_load_pd(phip1_re+40);


            out1_im[0]  = _mm512_load_pd(phip1_im);
            out1_im[1]  = _mm512_load_pd(phip1_im+8);
            out1_im[2]  = _mm512_load_pd(phip1_im+16);
            out2_im[0]  = _mm512_load_pd(phip1_im+24);
            out2_im[1]  = _mm512_load_pd(phip1_im+32);
            out2_im[2]  = _mm512_load_pd(phip1_im+40);

            su3_multiply_splitlayout_512(map0_re,mq0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_512(map1_re,mq1_im,U_re,U_im,out2_re,out2_im);

            mq0_re[0] = _mm512_add_pd(mq0_re[0],map0_re[0]);            
            mq0_re[1] = _mm512_add_pd(mq0_re[1],map0_re[1]);            
            mq0_re[2] = _mm512_add_pd(mq0_re[2],map0_re[2]);            

            mq0_im[0] = _mm512_add_pd(mq0_im[0],map0_im[0]);            
            mq0_im[1] = _mm512_add_pd(mq0_im[1],map0_im[1]);            
            mq0_im[2] = _mm512_add_pd(mq0_im[2],map0_im[2]);            

            mq1_re[0] = _mm512_add_pd(mq1_re[0],map1_re[0]);            
            mq1_re[1] = _mm512_add_pd(mq1_re[1],map1_re[1]);            
            mq1_re[2] = _mm512_add_pd(mq1_re[2],map1_re[2]);            

            mq1_im[0] = _mm512_add_pd(mq1_im[0],map1_im[0]);            
            mq1_im[1] = _mm512_add_pd(mq1_im[1],map1_im[1]);            
            mq1_im[2] = _mm512_add_pd(mq1_im[2],map1_im[2]);            


            mq2_re[0] = _mm512_sub_pd(mq2_re[0],map1_re[0]);            
            mq2_re[1] = _mm512_sub_pd(mq2_re[1],map1_re[1]);            
            mq2_re[2] = _mm512_sub_pd(mq2_re[2],map1_re[2]);            

            mq2_im[0] = _mm512_sub_pd(mq2_im[0],map1_im[0]);            
            mq2_im[1] = _mm512_sub_pd(mq2_im[1],map1_im[1]);            
            mq2_im[2] = _mm512_sub_pd(mq2_im[2],map1_im[2]);            

            mq3_re[0] = _mm512_add_pd(mq3_re[0],map0_re[0]);            
            mq3_re[1] = _mm512_add_pd(mq3_re[1],map0_re[1]);            
            mq3_re[2] = _mm512_add_pd(mq3_re[2],map0_re[2]);            

            mq3_im[0] = _mm512_add_pd(mq3_im[0],map0_im[0]);            
            mq3_im[1] = _mm512_add_pd(mq3_im[1],map0_im[1]);            
            mq3_im[2] = _mm512_add_pd(mq3_im[2],map0_im[2]);            

            ub0_re += 72;
            ub0_im += 72;

        
            //=====   +2  ========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_su3_512(ub0+1);
            //intrin_prefetch_halfspinor_512(phip3);
            //#endif
            
            //intrin_su3_load_512(U,ub0);
            //intrin_vector_load_512(out1,&(phip2->s0));
            //intrin_vector_load_512(out2,&(phip2->s1));
            //intrin_su3_multiply_512(map0,U,out1);
            //intrin_su3_multiply_512(map1,U,out2);
            //intrin_vector_add_512(mq0,mq0,map0);
            //intrin_vector_add_512(mq1,mq1,map1);
            //intrin_vector_i_sub_512(mq2,mq2,map0);
            //intrin_vector_i_add_512(mq3,mq3,map1);
            //ub0++;
            
            
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ub0_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ub0_im+jc*8+ic*24);
               }
            
        
            out1_re[0]  = _mm512_load_pd(phip2_re);
            out1_re[1]  = _mm512_load_pd(phip2_re+8);
            out1_re[2]  = _mm512_load_pd(phip2_re+16);
            out2_re[0]  = _mm512_load_pd(phip2_re+24);
            out2_re[1]  = _mm512_load_pd(phip2_re+32);
            out2_re[2]  = _mm512_load_pd(phip2_re+40);


            out1_im[0]  = _mm512_load_pd(phip2_im);
            out1_im[1]  = _mm512_load_pd(phip2_im+8);
            out1_im[2]  = _mm512_load_pd(phip2_im+16);
            out2_im[0]  = _mm512_load_pd(phip2_im+24);
            out2_im[1]  = _mm512_load_pd(phip2_im+32);
            out2_im[2]  = _mm512_load_pd(phip2_im+40);

            su3_multiply_splitlayout_512(map0_re,mq0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_512(map1_re,mq1_im,U_re,U_im,out2_re,out2_im);

            mq0_re[0] = _mm512_add_pd(mq0_re[0],map0_re[0]);            
            mq0_re[1] = _mm512_add_pd(mq0_re[1],map0_re[1]);            
            mq0_re[2] = _mm512_add_pd(mq0_re[2],map0_re[2]);            

            mq0_im[0] = _mm512_add_pd(mq0_im[0],map0_im[0]);            
            mq0_im[1] = _mm512_add_pd(mq0_im[1],map0_im[1]);            
            mq0_im[2] = _mm512_add_pd(mq0_im[2],map0_im[2]);            

            mq1_re[0] = _mm512_add_pd(mq1_re[0],map1_re[0]);            
            mq1_re[1] = _mm512_add_pd(mq1_re[1],map1_re[1]);            
            mq1_re[2] = _mm512_add_pd(mq1_re[2],map1_re[2]);            

            mq1_im[0] = _mm512_add_pd(mq1_im[0],map1_im[0]);            
            mq1_im[1] = _mm512_add_pd(mq1_im[1],map1_im[1]);            
            mq1_im[2] = _mm512_add_pd(mq1_im[2],map1_im[2]);            


            mq2_re[0] = _mm512_add_pd(mq2_re[0],map0_im[0]);            
            mq2_re[1] = _mm512_add_pd(mq2_re[1],map0_im[1]);            
            mq2_re[2] = _mm512_add_pd(mq2_re[2],map0_im[2]);            

            mq2_im[0] = _mm512_sub_pd(mq2_im[0],map0_re[0]);            
            mq2_im[1] = _mm512_sub_pd(mq2_im[1],map0_re[1]);            
            mq2_im[2] = _mm512_sub_pd(mq2_im[2],map0_re[2]);            

            mq3_re[0] = _mm512_sub_pd(mq3_re[0],map1_im[0]);            
            mq3_re[1] = _mm512_sub_pd(mq3_re[1],map1_im[1]);            
            mq3_re[2] = _mm512_sub_pd(mq3_re[2],map1_im[2]);            

            mq3_im[0] = _mm512_add_pd(mq3_im[0],map1_re[0]);            
            mq3_im[1] = _mm512_add_pd(mq3_im[1],map1_re[1]);            
            mq3_im[2] = _mm512_add_pd(mq3_im[2],map1_re[2]);            

            ub0_re += 72;
            ub0_im += 72;

            //=====   +3  ========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phim0);
            //#endif
            
            //intrin_su3_load_512(U,ub0);
            //intrin_vector_load_512(out1,&(phip3->s0));
            //intrin_vector_load_512(out2,&(phip3->s1));
            //intrin_su3_multiply_512(map0,U,out1);
            //intrin_su3_multiply_512(map1,U,out2);
            //intrin_vector_add_512(mq0,mq0,map0);
            //intrin_vector_add_512(mq1,mq1,map1);
            //intrin_vector_add_512(mq2,mq2,map0);
            //intrin_vector_add_512(mq3,mq3,map1);
            
            
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ub0_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ub0_im+jc*8+ic*24);
               }
            
        
            out1_re[0]  = _mm512_load_pd(phip3_re);
            out1_re[1]  = _mm512_load_pd(phip3_re+8);
            out1_re[2]  = _mm512_load_pd(phip3_re+16);
            out2_re[0]  = _mm512_load_pd(phip3_re+24);
            out2_re[1]  = _mm512_load_pd(phip3_re+32);
            out2_re[2]  = _mm512_load_pd(phip3_re+40);


            out1_im[0]  = _mm512_load_pd(phip3_im);
            out1_im[1]  = _mm512_load_pd(phip3_im+8);
            out1_im[2]  = _mm512_load_pd(phip3_im+16);
            out2_im[0]  = _mm512_load_pd(phip3_im+24);
            out2_im[1]  = _mm512_load_pd(phip3_im+32);
            out2_im[2]  = _mm512_load_pd(phip3_im+40);

            su3_multiply_splitlayout_512(map0_re,mq0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_512(map1_re,mq1_im,U_re,U_im,out2_re,out2_im);

            mq0_re[0] = _mm512_add_pd(mq0_re[0],map0_re[0]);            
            mq0_re[1] = _mm512_add_pd(mq0_re[1],map0_re[1]);            
            mq0_re[2] = _mm512_add_pd(mq0_re[2],map0_re[2]);            

            mq0_im[0] = _mm512_add_pd(mq0_im[0],map0_im[0]);            
            mq0_im[1] = _mm512_add_pd(mq0_im[1],map0_im[1]);            
            mq0_im[2] = _mm512_add_pd(mq0_im[2],map0_im[2]);            

            mq1_re[0] = _mm512_add_pd(mq1_re[0],map1_re[0]);            
            mq1_re[1] = _mm512_add_pd(mq1_re[1],map1_re[1]);            
            mq1_re[2] = _mm512_add_pd(mq1_re[2],map1_re[2]);            

            mq1_im[0] = _mm512_add_pd(mq1_im[0],map1_im[0]);            
            mq1_im[1] = _mm512_add_pd(mq1_im[1],map1_im[1]);            
            mq1_im[2] = _mm512_add_pd(mq1_im[2],map1_im[2]);            


            mq2_re[0] = _mm512_add_pd(mq2_re[0],map0_re[0]);            
            mq2_re[1] = _mm512_add_pd(mq2_re[1],map0_re[1]);            
            mq2_re[2] = _mm512_add_pd(mq2_re[2],map0_re[2]);            

            mq2_im[0] = _mm512_add_pd(mq2_im[0],map0_im[0]);            
            mq2_im[1] = _mm512_add_pd(mq2_im[1],map0_im[1]);            
            mq2_im[2] = _mm512_add_pd(mq2_im[2],map0_im[2]);            

            mq3_re[0] = _mm512_add_pd(mq3_re[0],map1_re[0]);            
            mq3_re[1] = _mm512_add_pd(mq3_re[1],map1_re[1]);            
            mq3_re[2] = _mm512_add_pd(mq3_re[2],map1_re[2]);            

            mq3_im[0] = _mm512_add_pd(mq3_im[0],map1_im[0]);            
            mq3_im[1] = _mm512_add_pd(mq3_im[1],map1_im[1]);            
            mq3_im[2] = _mm512_add_pd(mq3_im[2],map1_im[2]);            



            //now add the -ve pieces
            //======= -0 =========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phim1);
            //#endif
            
            //intrin_vector_load_512(map0,&(phim0->s0));
            //intrin_vector_load_512(map1,&(phim0->s1));
            //intrin_vector_add_512(mq0,mq0,map0);
            //intrin_vector_add_512(mq1,mq1,map1);
            //intrin_vector_i_add_512(mq2,mq2,map1);
            //intrin_vector_i_add_512(mq3,mq3,map0);
            

            map0_re[0]  = _mm512_load_pd(phim0_re);
            map0_re[1]  = _mm512_load_pd(phim0_re+8);
            map0_re[2]  = _mm512_load_pd(phim0_re+16);
            map1_re[0]  = _mm512_load_pd(phim0_re+24);
            map1_re[1]  = _mm512_load_pd(phim0_re+32);
            map1_re[2]  = _mm512_load_pd(phim0_re+40);


            map0_im[0]  = _mm512_load_pd(phim0_im);
            map0_im[1]  = _mm512_load_pd(phim0_im+8);
            map0_im[2]  = _mm512_load_pd(phim0_im+16);
            map1_im[0]  = _mm512_load_pd(phim0_im+24);
            map1_im[1]  = _mm512_load_pd(phim0_im+32);
            map1_im[2]  = _mm512_load_pd(phim0_im+40);

            mq0_re[0] = _mm512_add_pd(mq0_re[0],map0_re[0]);            
            mq0_re[1] = _mm512_add_pd(mq0_re[1],map0_re[1]);            
            mq0_re[2] = _mm512_add_pd(mq0_re[2],map0_re[2]);            

            mq0_im[0] = _mm512_add_pd(mq0_im[0],map0_im[0]);            
            mq0_im[1] = _mm512_add_pd(mq0_im[1],map0_im[1]);            
            mq0_im[2] = _mm512_add_pd(mq0_im[2],map0_im[2]);            

            mq1_re[0] = _mm512_add_pd(mq1_re[0],map1_re[0]);            
            mq1_re[1] = _mm512_add_pd(mq1_re[1],map1_re[1]);            
            mq1_re[2] = _mm512_add_pd(mq1_re[2],map1_re[2]);            

            mq1_im[0] = _mm512_add_pd(mq1_im[0],map1_im[0]);            
            mq1_im[1] = _mm512_add_pd(mq1_im[1],map1_im[1]);            
            mq1_im[2] = _mm512_add_pd(mq1_im[2],map1_im[2]);            


            mq2_re[0] = _mm512_sub_pd(mq2_re[0],map1_im[0]);            
            mq2_re[1] = _mm512_sub_pd(mq2_re[1],map1_im[1]);            
            mq2_re[2] = _mm512_sub_pd(mq2_re[2],map1_im[2]);            

            mq2_im[0] = _mm512_add_pd(mq2_im[0],map1_re[0]);            
            mq2_im[1] = _mm512_add_pd(mq2_im[1],map1_re[1]);            
            mq2_im[2] = _mm512_add_pd(mq2_im[2],map1_re[2]);            

            mq3_re[0] = _mm512_sub_pd(mq3_re[0],map0_im[0]);            
            mq3_re[1] = _mm512_sub_pd(mq3_re[1],map0_im[1]);            
            mq3_re[2] = _mm512_sub_pd(mq3_re[2],map0_im[2]);            

            mq3_im[0] = _mm512_add_pd(mq3_im[0],map0_re[0]);            
            mq3_im[1] = _mm512_add_pd(mq3_im[1],map0_re[1]);            
            mq3_im[2] = _mm512_add_pd(mq3_im[2],map0_re[2]);            

            //======= -1 =========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phim2);
            //#endif
            
            //intrin_vector_load_512(map0,&(phim1->s0));
            //intrin_vector_load_512(map1,&(phim1->s1));
            //intrin_vector_add_512(mq0,mq0,map0);
            //intrin_vector_add_512(mq1,mq1,map1);
            //intrin_vector_add_512(mq2,mq2,map1);
            //intrin_vector_sub_512(mq3,mq3,map0);
            

            map0_re[0]  = _mm512_load_pd(phim1_re);
            map0_re[1]  = _mm512_load_pd(phim1_re+8);
            map0_re[2]  = _mm512_load_pd(phim1_re+16);
            map1_re[0]  = _mm512_load_pd(phim1_re+24);
            map1_re[1]  = _mm512_load_pd(phim1_re+32);
            map1_re[2]  = _mm512_load_pd(phim1_re+40);


            map0_im[0]  = _mm512_load_pd(phim1_im);
            map0_im[1]  = _mm512_load_pd(phim1_im+8);
            map0_im[2]  = _mm512_load_pd(phim1_im+16);
            map1_im[0]  = _mm512_load_pd(phim1_im+24);
            map1_im[1]  = _mm512_load_pd(phim1_im+32);
            map1_im[2]  = _mm512_load_pd(phim1_im+40);

            mq0_re[0] = _mm512_add_pd(mq0_re[0],map0_re[0]);            
            mq0_re[1] = _mm512_add_pd(mq0_re[1],map0_re[1]);            
            mq0_re[2] = _mm512_add_pd(mq0_re[2],map0_re[2]);            

            mq0_im[0] = _mm512_add_pd(mq0_im[0],map0_im[0]);            
            mq0_im[1] = _mm512_add_pd(mq0_im[1],map0_im[1]);            
            mq0_im[2] = _mm512_add_pd(mq0_im[2],map0_im[2]);            

            mq1_re[0] = _mm512_add_pd(mq1_re[0],map1_re[0]);            
            mq1_re[1] = _mm512_add_pd(mq1_re[1],map1_re[1]);            
            mq1_re[2] = _mm512_add_pd(mq1_re[2],map1_re[2]);            

            mq1_im[0] = _mm512_add_pd(mq1_im[0],map1_im[0]);            
            mq1_im[1] = _mm512_add_pd(mq1_im[1],map1_im[1]);            
            mq1_im[2] = _mm512_add_pd(mq1_im[2],map1_im[2]);            


            mq2_re[0] = _mm512_add_pd(mq2_re[0],map1_re[0]);            
            mq2_re[1] = _mm512_add_pd(mq2_re[1],map1_re[1]);            
            mq2_re[2] = _mm512_add_pd(mq2_re[2],map1_re[2]);            

            mq2_im[0] = _mm512_add_pd(mq2_im[0],map1_im[0]);            
            mq2_im[1] = _mm512_add_pd(mq2_im[1],map1_im[1]);            
            mq2_im[2] = _mm512_add_pd(mq2_im[2],map1_im[2]);            

            mq3_re[0] = _mm512_sub_pd(mq3_re[0],map0_re[0]);            
            mq3_re[1] = _mm512_sub_pd(mq3_re[1],map0_re[1]);            
            mq3_re[2] = _mm512_sub_pd(mq3_re[2],map0_re[2]);            

            mq3_im[0] = _mm512_sub_pd(mq3_im[0],map0_im[0]);            
            mq3_im[1] = _mm512_sub_pd(mq3_im[1],map0_im[1]);            
            mq3_im[2] = _mm512_sub_pd(mq3_im[2],map0_im[2]);            


            //======= -2 =========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_halfspinor_512(phim3);
            //#endif
            
            //intrin_vector_load_512(map0,&(phim2->s0));
            //intrin_vector_load_512(map1,&(phim2->s1));
            //intrin_vector_add_512(mq0,mq0,map0);
            //intrin_vector_add_512(mq1,mq1,map1);
            //intrin_vector_i_add_512(mq2,mq2,map0);
            //intrin_vector_i_sub_512(mq3,mq3,map1);
            

            map0_re[0]  = _mm512_load_pd(phim2_re);
            map0_re[1]  = _mm512_load_pd(phim2_re+8);
            map0_re[2]  = _mm512_load_pd(phim2_re+16);
            map1_re[0]  = _mm512_load_pd(phim2_re+24);
            map1_re[1]  = _mm512_load_pd(phim2_re+32);
            map1_re[2]  = _mm512_load_pd(phim2_re+40);


            map0_im[0]  = _mm512_load_pd(phim2_im);
            map0_im[1]  = _mm512_load_pd(phim2_im+8);
            map0_im[2]  = _mm512_load_pd(phim2_im+16);
            map1_im[0]  = _mm512_load_pd(phim2_im+24);
            map1_im[1]  = _mm512_load_pd(phim2_im+32);
            map1_im[2]  = _mm512_load_pd(phim2_im+40);

            mq0_re[0] = _mm512_add_pd(mq0_re[0],map0_re[0]);            
            mq0_re[1] = _mm512_add_pd(mq0_re[1],map0_re[1]);            
            mq0_re[2] = _mm512_add_pd(mq0_re[2],map0_re[2]);            

            mq0_im[0] = _mm512_add_pd(mq0_im[0],map0_im[0]);            
            mq0_im[1] = _mm512_add_pd(mq0_im[1],map0_im[1]);            
            mq0_im[2] = _mm512_add_pd(mq0_im[2],map0_im[2]);            

            mq1_re[0] = _mm512_add_pd(mq1_re[0],map1_re[0]);            
            mq1_re[1] = _mm512_add_pd(mq1_re[1],map1_re[1]);            
            mq1_re[2] = _mm512_add_pd(mq1_re[2],map1_re[2]);            

            mq1_im[0] = _mm512_add_pd(mq1_im[0],map1_im[0]);            
            mq1_im[1] = _mm512_add_pd(mq1_im[1],map1_im[1]);            
            mq1_im[2] = _mm512_add_pd(mq1_im[2],map1_im[2]);            


            mq2_re[0] = _mm512_sub_pd(mq2_re[0],map0_im[0]);            
            mq2_re[1] = _mm512_sub_pd(mq2_re[1],map0_im[1]);            
            mq2_re[2] = _mm512_sub_pd(mq2_re[2],map0_im[2]);            

            mq2_im[0] = _mm512_add_pd(mq2_im[0],map0_re[0]);            
            mq2_im[1] = _mm512_add_pd(mq2_im[1],map0_re[1]);            
            mq2_im[2] = _mm512_add_pd(mq2_im[2],map0_re[2]);            

            mq3_re[0] = _mm512_add_pd(mq3_re[0],map1_im[0]);            
            mq3_re[1] = _mm512_add_pd(mq3_re[1],map1_im[1]);            
            mq3_re[2] = _mm512_add_pd(mq3_re[2],map1_im[2]);            

            mq3_im[0] = _mm512_sub_pd(mq3_im[0],map1_re[0]);            
            mq3_im[1] = _mm512_sub_pd(mq3_im[1],map1_re[1]);            
            mq3_im[2] = _mm512_sub_pd(mq3_im[2],map1_re[2]);            



            //======= -3 =========
            //#ifdef MIC_PREFETCH
            //intrin_prefetch_spinor_512(&qout[ipt]);
            //#endif
            
            //intrin_vector_load_512(map0,&(phim3->s0));
            //intrin_vector_load_512(map1,&(phim3->s1));
            //intrin_vector_add_512(mq0,mq0,map0);
            //intrin_vector_add_512(mq1,mq1,map1);
            //intrin_vector_sub_512(mq2,mq2,map0);
            //intrin_vector_sub_512(mq3,mq3,map1);
            

            map0_re[0]  = _mm512_load_pd(phim3_re);
            map0_re[1]  = _mm512_load_pd(phim3_re+8);
            map0_re[2]  = _mm512_load_pd(phim3_re+16);
            map1_re[0]  = _mm512_load_pd(phim3_re+24);
            map1_re[1]  = _mm512_load_pd(phim3_re+32);
            map1_re[2]  = _mm512_load_pd(phim3_re+40);


            map0_im[0]  = _mm512_load_pd(phim3_im);
            map0_im[1]  = _mm512_load_pd(phim3_im+8);
            map0_im[2]  = _mm512_load_pd(phim3_im+16);
            map1_im[0]  = _mm512_load_pd(phim3_im+24);
            map1_im[1]  = _mm512_load_pd(phim3_im+32);
            map1_im[2]  = _mm512_load_pd(phim3_im+40);

            mq0_re[0] = _mm512_add_pd(mq0_re[0],map0_re[0]);            
            mq0_re[1] = _mm512_add_pd(mq0_re[1],map0_re[1]);            
            mq0_re[2] = _mm512_add_pd(mq0_re[2],map0_re[2]);            

            mq0_im[0] = _mm512_add_pd(mq0_im[0],map0_im[0]);            
            mq0_im[1] = _mm512_add_pd(mq0_im[1],map0_im[1]);            
            mq0_im[2] = _mm512_add_pd(mq0_im[2],map0_im[2]);            

            mq1_re[0] = _mm512_add_pd(mq1_re[0],map1_re[0]);            
            mq1_re[1] = _mm512_add_pd(mq1_re[1],map1_re[1]);            
            mq1_re[2] = _mm512_add_pd(mq1_re[2],map1_re[2]);            

            mq1_im[0] = _mm512_add_pd(mq1_im[0],map1_im[0]);            
            mq1_im[1] = _mm512_add_pd(mq1_im[1],map1_im[1]);            
            mq1_im[2] = _mm512_add_pd(mq1_im[2],map1_im[2]);            


            mq2_re[0] = _mm512_sub_pd(mq2_re[0],map0_re[0]);            
            mq2_re[1] = _mm512_sub_pd(mq2_re[1],map0_re[1]);            
            mq2_re[2] = _mm512_sub_pd(mq2_re[2],map0_re[2]);            

            mq2_im[0] = _mm512_sub_pd(mq2_im[0],map0_im[0]);            
            mq2_im[1] = _mm512_sub_pd(mq2_im[1],map0_im[1]);            
            mq2_im[2] = _mm512_sub_pd(mq2_im[2],map0_im[2]);            

            mq3_re[0] = _mm512_sub_pd(mq3_re[0],map1_re[0]);            
            mq3_re[1] = _mm512_sub_pd(mq3_re[1],map1_re[1]);            
            mq3_re[2] = _mm512_sub_pd(mq3_re[2],map1_re[2]);            

            mq3_im[0] = _mm512_sub_pd(mq3_im[0],map1_im[0]);            
            mq3_im[1] = _mm512_sub_pd(mq3_im[1],map1_im[1]);            
            mq3_im[2] = _mm512_sub_pd(mq3_im[2],map1_im[2]);            



            //store the result
            
            //intrin_vector_store_512(&qout[ipt].s0,mq0);
            //intrin_vector_store_512(&qout[ipt].s1,mq1);
            //intrin_vector_store_512(&qout[ipt].s2,mq2);
            //intrin_vector_store_512(&qout[ipt].s3,mq3);
            


            _mm512_store_pd(qout_re+ipt*8*12    , mq0_re[0]);
            _mm512_store_pd(qout_re+ipt*8*12+8  , mq0_re[1]);
            _mm512_store_pd(qout_re+ipt*8*12+16 , mq0_re[2]);
            _mm512_store_pd(qout_re+ipt*8*12+24 , mq1_re[0]);
            _mm512_store_pd(qout_re+ipt*8*12+32 , mq1_re[1]);
            _mm512_store_pd(qout_re+ipt*8*12+40 , mq1_re[2]);
            _mm512_store_pd(qout_re+ipt*8*12+48 , mq2_re[0]);
            _mm512_store_pd(qout_re+ipt*8*12+56 , mq2_re[1]);
            _mm512_store_pd(qout_re+ipt*8*12+64 , mq2_re[2]);
            _mm512_store_pd(qout_re+ipt*8*12+72 , mq3_re[0]);
            _mm512_store_pd(qout_re+ipt*8*12+80 , mq3_re[1]);
            _mm512_store_pd(qout_re+ipt*8*12+88 , mq3_re[2]);


      }

   }
#ifdef _OPENMP
}
#endif  //end of the openmp parallel reigon

   
   return stop_watch(ts);
}



//======================================================
//Single MIC, split real and imaginary, double
//No halfspinor, loop over the output spinor 
//in exactly the same way the operaotr is written 
//================Even-Odd==============================
double plqcd_hopping_matrix_eo_single_mic_split_nohalfspinor(
                                                double *qout_re, 
                                                double *qout_im, 
                                                double *u_re, 
                                                double *u_im,
                                                double *qin_re, 
                                                double *qin_im) 
{
   double ts;          //timer
   ts=stop_watch(0.0); //start
   #ifdef _OPENMP
   #pragma omp parallel 
   {
   #endif
      int V,Vmic_split,Vyzt,lx;
      V = plqcd_g.VOLUME;
      Vmic_split = plqcd_g.Vmic_split;
      Vyzt = plqcd_g.latdims[1]*plqcd_g.latdims[2]*plqcd_g.latdims[3];
      lx=plqcd_g.latdims[0]/8;
      if((plqcd_g.latdims[0]%16) != 0 ){ 
        fprintf(stderr,"lattice size in the 0 direction must be a factor of 16\n");
        exit(1);
      }
      
      __m512d  mqin_re[12],mqin_im[12], mqout_re[12],mqout_im[12];
      __m512d  U_re[3][3], U_im[3][3];
      __m512d  out1_re[3],out2_re[3],out3_re[3],out4_re[3];
      __m512d  out1_im[3],out2_im[3],out3_im[3],out4_im[3];
      __m512d  map0_re[3],map0_im[3],map1_re[3],map1_im[3];

      int ix,iy,it,iz,is,xcor[4],ipt;

      int L[4];
      for(int i=0; i<4; i++)
         L[i] = plqcd_g.latdims[i];

      //needed pointers
      //input spinor, links and output spinor
      double *sx_re,*sx_im,*ux_re,*ux_im,*gx_re,*gx_im;


      //for permuting the eight doubles from 0,1,2,3,4,5,6,7 order into 1,2,3,4,5,6,7,0 order
      //note each double correspond to two ints
      int  __attribute__ ((aligned(64)))  permback[16]={2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1};
      __m512i mpermback = _mm512_load_epi32(permback);

      //for permuting the eight doubles from 0,1,2,3,4,5,6,7 order into 7,0,1,2,3,4,5,6 order
      //note each double correspond to two ints
      int __attribute__ ((aligned(64)))   permfor[16]={14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13};
      __m512i mpermfor = _mm512_load_epi32(permfor);


      //-----------------------------------------------------
      // compute the result
      //----------------------------------------------------
      #ifdef _OPENMP
      //#pragma omp for schedule(static,20) //use this if you want to play with how work is shared among threads 
      #pragma omp for 
      #endif   
      for(int is=0; is < Vyzt; is++) 
      {
         iy = is%L[1];
         iz = (is/L[1]) %L[2];
         it = (is/L[1]/L[2])%L[3];

         for(ix = (iy+iz+it)%2; ix < lx; ix +=2)  //result is on even sites
         {
            xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
            ipt=plqcd_g.ipt_eo_mic_split[cart2lex(xcor)];

            gx_re = qout_re + ipt*96;
            gx_im = qout_im + ipt*96;

            //================
            //====  +0  ======
            //================

            ux_re = u_re + ipt*288;
            ux_im = u_im + ipt*288;

            sx_re = qin_re + plqcd_g.iup_mic_split[ipt][0]*96;
            sx_im = qin_im + plqcd_g.iup_mic_split[ipt][0]*96;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm512_load_pd(sx_re);
            mqin_re[1]  = _mm512_load_pd(sx_re+8);
            mqin_re[2]  = _mm512_load_pd(sx_re+16);
            mqin_im[0]  = _mm512_load_pd(sx_im);
            mqin_im[1]  = _mm512_load_pd(sx_im+8);
            mqin_im[2]  = _mm512_load_pd(sx_im+16);


            mqin_re[9]  = _mm512_load_pd(sx_re+72);
            mqin_re[10]  = _mm512_load_pd(sx_re+80);
            mqin_re[11]  = _mm512_load_pd(sx_re+88);
            mqin_im[9]  = _mm512_load_pd(sx_im+72);
            mqin_im[10]  = _mm512_load_pd(sx_im+80);
            mqin_im[11]  = _mm512_load_pd(sx_im+88);


            out1_re[0] = _mm512_sub_pd(mqin_re[0],mqin_im[9]);            
            out1_re[1] = _mm512_sub_pd(mqin_re[1],mqin_im[10]);            
            out1_re[2] = _mm512_sub_pd(mqin_re[2],mqin_im[11]);            
            out1_im[0] = _mm512_add_pd(mqin_im[0],mqin_re[9]);            
            out1_im[1] = _mm512_add_pd(mqin_im[1],mqin_re[10]);            
            out1_im[2] = _mm512_add_pd(mqin_im[2],mqin_re[11]);


            mqin_re[3]  = _mm512_load_pd(sx_re+24);
            mqin_re[4]  = _mm512_load_pd(sx_re+32);
            mqin_re[5]  = _mm512_load_pd(sx_re+40);
            mqin_im[3]  = _mm512_load_pd(sx_im+24);
            mqin_im[4]  = _mm512_load_pd(sx_im+32);
            mqin_im[5]  = _mm512_load_pd(sx_im+40);


            mqin_re[6]  = _mm512_load_pd(sx_re+48);
            mqin_re[7]  = _mm512_load_pd(sx_re+56);
            mqin_re[8]  = _mm512_load_pd(sx_re+64);
            mqin_im[6]  = _mm512_load_pd(sx_im+48);
            mqin_im[7]  = _mm512_load_pd(sx_im+56);
            mqin_im[8]  = _mm512_load_pd(sx_im+64);


            out2_re[0] = _mm512_sub_pd(mqin_re[3],mqin_im[6]);            
            out2_re[1] = _mm512_sub_pd(mqin_re[4],mqin_im[7]);            
            out2_re[2] = _mm512_sub_pd(mqin_re[5],mqin_im[8]);            
            out2_im[0] = _mm512_add_pd(mqin_im[3],mqin_re[6]);            
            out2_im[1] = _mm512_add_pd(mqin_im[4],mqin_re[7]);            
            out2_im[2] = _mm512_add_pd(mqin_im[5],mqin_re[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ux_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ux_im+jc*8+ic*24);
               }
            
            su3_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            //store the result, shuffle if on the boundary
            if(ix == 0)
            {
               for(int i=0; i<3; i++)
               {  
                  out3_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) map0_re[i]);
                  out3_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) map0_im[i]);
                  out4_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) map1_re[i]);
                  out4_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermback, (__m512i) map1_im[i]);
               }
               
               mqout_re[0] = _mm512_add_pd(mqout_re[0] ,out3_re[0]);
               mqout_re[1] = _mm512_add_pd(mqout_re[1] ,out3_re[1]);
               mqout_re[2] = _mm512_add_pd(mqout_re[2] ,out3_re[2]);
               mqout_re[3] = _mm512_add_pd(mqout_re[3] ,out4_re[0]);
               mqout_re[4] = _mm512_add_pd(mqout_re[4] ,out4_re[1]);
               mqout_re[5] = _mm512_add_pd(mqout_re[5] ,out4_re[2]);

               mqout_re[6] = _mm512_add_pd(mqout_re[6] ,out4_im[0]);
               mqout_re[7] = _mm512_add_pd(mqout_re[7] ,out4_im[1]);
               mqout_re[8] = _mm512_add_pd(mqout_re[8] ,out4_im[2]);
               mqout_re[9] = _mm512_add_pd(mqout_re[9] ,out3_im[0]);
               mqout_re[10]= _mm512_add_pd(mqout_re[10],out3_im[1]);
               mqout_re[11]= _mm512_add_pd(mqout_re[11],out3_im[2]);

               mqout_im[0] = _mm512_add_pd(mqout_im[0] ,out3_im[0]);
               mqout_im[1] = _mm512_add_pd(mqout_im[1] ,out3_im[1]);
               mqout_im[2] = _mm512_add_pd(mqout_im[2] ,out3_im[2]);
               mqout_im[3] = _mm512_add_pd(mqout_im[3] ,out4_im[0]);
               mqout_im[4] = _mm512_add_pd(mqout_im[4] ,out4_im[1]);
               mqout_im[5] = _mm512_add_pd(mqout_im[5] ,out4_im[2]);

               mqout_im[6] = _mm512_sub_pd(mqout_im[6] ,out4_re[0]);
               mqout_im[7] = _mm512_sub_pd(mqout_im[7] ,out4_re[1]);
               mqout_im[8] = _mm512_sub_pd(mqout_im[8] ,out4_re[2]);
               mqout_im[9] = _mm512_sub_pd(mqout_im[9] ,out3_re[0]);
               mqout_im[10]= _mm512_sub_pd(mqout_im[10],out3_re[1]);
               mqout_im[11]= _mm512_sub_pd(mqout_im[11],out3_re[2]);

            }
            else
            {
               mqout_re[0] = _mm512_add_pd(mqout_re[0] ,map0_re[0]);
               mqout_re[1] = _mm512_add_pd(mqout_re[1] ,map0_re[1]);
               mqout_re[2] = _mm512_add_pd(mqout_re[2] ,map0_re[2]);
               mqout_re[3] = _mm512_add_pd(mqout_re[3] ,map1_re[0]);
               mqout_re[4] = _mm512_add_pd(mqout_re[4] ,map1_re[1]);
               mqout_re[5] = _mm512_add_pd(mqout_re[5] ,map1_re[2]);

               mqout_re[6] = _mm512_add_pd(mqout_re[6] ,map1_im[0]);
               mqout_re[7] = _mm512_add_pd(mqout_re[7] ,map1_im[1]);
               mqout_re[8] = _mm512_add_pd(mqout_re[8] ,map1_im[2]);
               mqout_re[9] = _mm512_add_pd(mqout_re[9] ,map0_im[0]);
               mqout_re[10]= _mm512_add_pd(mqout_re[10],map0_im[1]);
               mqout_re[11]= _mm512_add_pd(mqout_re[11],map0_im[2]);

               mqout_im[0] = _mm512_add_pd(mqout_im[0] ,map0_im[0]);
               mqout_im[1] = _mm512_add_pd(mqout_im[1] ,map0_im[1]);
               mqout_im[2] = _mm512_add_pd(mqout_im[2] ,map0_im[2]);
               mqout_im[3] = _mm512_add_pd(mqout_im[3] ,map1_im[0]);
               mqout_im[4] = _mm512_add_pd(mqout_im[4] ,map1_im[1]);
               mqout_im[5] = _mm512_add_pd(mqout_im[5] ,map1_im[2]);

               mqout_im[6] = _mm512_sub_pd(mqout_im[6] ,map1_re[0]);
               mqout_im[7] = _mm512_sub_pd(mqout_im[7] ,map1_re[1]);
               mqout_im[8] = _mm512_sub_pd(mqout_im[8] ,map1_re[2]);
               mqout_im[9] = _mm512_sub_pd(mqout_im[9] ,map0_re[0]);
               mqout_im[10]= _mm512_sub_pd(mqout_im[10],map0_re[1]);
               mqout_im[11]= _mm512_sub_pd(mqout_im[11],map0_re[2]);
            } 

            
            //===================
            //===== +1 ==========
            //===================
      
            ux_re += 72;
            ux_im += 72;

            sx_re = qin_re + plqcd_g.iup_mic_split[ipt][1]*96;
            sx_im = qin_im + plqcd_g.iup_mic_split[ipt][1]*96;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm512_load_pd(sx_re);
            mqin_re[1]  = _mm512_load_pd(sx_re+8);
            mqin_re[2]  = _mm512_load_pd(sx_re+16);
            mqin_im[0]  = _mm512_load_pd(sx_im);
            mqin_im[1]  = _mm512_load_pd(sx_im+8);
            mqin_im[2]  = _mm512_load_pd(sx_im+16);


            mqin_re[9]   = _mm512_load_pd(sx_re+72);
            mqin_re[10]  = _mm512_load_pd(sx_re+80);
            mqin_re[11]  = _mm512_load_pd(sx_re+88);
            mqin_im[9]   = _mm512_load_pd(sx_im+72);
            mqin_im[10]  = _mm512_load_pd(sx_im+80);
            mqin_im[11]  = _mm512_load_pd(sx_im+88);


            out1_re[0] = _mm512_add_pd(mqin_re[0],mqin_re[9]);            
            out1_re[1] = _mm512_add_pd(mqin_re[1],mqin_re[10]);            
            out1_re[2] = _mm512_add_pd(mqin_re[2],mqin_re[11]);            
            out1_im[0] = _mm512_add_pd(mqin_im[0],mqin_im[9]);            
            out1_im[1] = _mm512_add_pd(mqin_im[1],mqin_im[10]);            
            out1_im[2] = _mm512_add_pd(mqin_im[2],mqin_im[11]);

            mqin_re[3]  = _mm512_load_pd(sx_re+24);
            mqin_re[4]  = _mm512_load_pd(sx_re+32);
            mqin_re[5]  = _mm512_load_pd(sx_re+40);
            mqin_im[3]  = _mm512_load_pd(sx_im+24);
            mqin_im[4]  = _mm512_load_pd(sx_im+32);
            mqin_im[5]  = _mm512_load_pd(sx_im+40);


            mqin_re[6]  = _mm512_load_pd(sx_re+48);
            mqin_re[7]  = _mm512_load_pd(sx_re+56);
            mqin_re[8]  = _mm512_load_pd(sx_re+64);
            mqin_im[6]  = _mm512_load_pd(sx_im+48);
            mqin_im[7]  = _mm512_load_pd(sx_im+56);
            mqin_im[8]  = _mm512_load_pd(sx_im+64);


            out2_re[0] = _mm512_sub_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm512_sub_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm512_sub_pd(mqin_re[5],mqin_re[8]);            
            out2_im[0] = _mm512_sub_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm512_sub_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm512_sub_pd(mqin_im[5],mqin_im[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ux_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ux_im+jc*8+ic*24);
               }
            
            su3_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);

            mqout_re[0] = _mm512_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm512_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm512_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm512_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm512_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm512_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm512_sub_pd(mqout_re[6] ,map1_re[0]);
            mqout_re[7] = _mm512_sub_pd(mqout_re[7] ,map1_re[1]);
            mqout_re[8] = _mm512_sub_pd(mqout_re[8] ,map1_re[2]);
            mqout_re[9] = _mm512_add_pd(mqout_re[9] ,map0_re[0]);
            mqout_re[10]= _mm512_add_pd(mqout_re[10],map0_re[1]);
            mqout_re[11]= _mm512_add_pd(mqout_re[11],map0_re[2]);

            mqout_im[0] = _mm512_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm512_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm512_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm512_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm512_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm512_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm512_sub_pd(mqout_im[6] ,map1_im[0]);
            mqout_im[7] = _mm512_sub_pd(mqout_im[7] ,map1_im[1]);
            mqout_im[8] = _mm512_sub_pd(mqout_im[8] ,map1_im[2]);
            mqout_im[9] = _mm512_add_pd(mqout_im[9] ,map0_im[0]);
            mqout_im[10]= _mm512_add_pd(mqout_im[10],map0_im[1]);
            mqout_im[11]= _mm512_add_pd(mqout_im[11],map0_im[2]);




            //===================
            //===== +2 ==========
            //===================
      
            ux_re += 72;
            ux_im += 72;

            sx_re = qin_re + plqcd_g.iup_mic_split[ipt][2]*96;
            sx_im = qin_im + plqcd_g.iup_mic_split[ipt][2]*96;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm512_load_pd(sx_re);
            mqin_re[1]  = _mm512_load_pd(sx_re+8);
            mqin_re[2]  = _mm512_load_pd(sx_re+16);
            mqin_im[0]  = _mm512_load_pd(sx_im);
            mqin_im[1]  = _mm512_load_pd(sx_im+8);
            mqin_im[2]  = _mm512_load_pd(sx_im+16);


            mqin_re[6]  = _mm512_load_pd(sx_re+48);
            mqin_re[7]  = _mm512_load_pd(sx_re+56);
            mqin_re[8]  = _mm512_load_pd(sx_re+64);
            mqin_im[6]  = _mm512_load_pd(sx_im+48);
            mqin_im[7]  = _mm512_load_pd(sx_im+56);
            mqin_im[8]  = _mm512_load_pd(sx_im+64);


            out1_re[0] = _mm512_sub_pd(mqin_re[0],mqin_im[6]);            
            out1_re[1] = _mm512_sub_pd(mqin_re[1],mqin_im[7]);            
            out1_re[2] = _mm512_sub_pd(mqin_re[2],mqin_im[8]);            
            out1_im[0] = _mm512_add_pd(mqin_im[0],mqin_re[6]);            
            out1_im[1] = _mm512_add_pd(mqin_im[1],mqin_re[7]);            
            out1_im[2] = _mm512_add_pd(mqin_im[2],mqin_re[8]);

            mqin_re[3]  = _mm512_load_pd(sx_re+24);
            mqin_re[4]  = _mm512_load_pd(sx_re+32);
            mqin_re[5]  = _mm512_load_pd(sx_re+40);
            mqin_im[3]  = _mm512_load_pd(sx_im+24);
            mqin_im[4]  = _mm512_load_pd(sx_im+32);
            mqin_im[5]  = _mm512_load_pd(sx_im+40);


            mqin_re[9]   = _mm512_load_pd(sx_re+72);
            mqin_re[10]  = _mm512_load_pd(sx_re+80);
            mqin_re[11]  = _mm512_load_pd(sx_re+88);
            mqin_im[9]   = _mm512_load_pd(sx_im+72);
            mqin_im[10]  = _mm512_load_pd(sx_im+80);
            mqin_im[11]  = _mm512_load_pd(sx_im+88);


            out2_re[0] = _mm512_add_pd(mqin_re[3],mqin_im[9]);            
            out2_re[1] = _mm512_add_pd(mqin_re[4],mqin_im[10]);            
            out2_re[2] = _mm512_add_pd(mqin_re[5],mqin_im[11]);            
            out2_im[0] = _mm512_sub_pd(mqin_im[3],mqin_re[9]);            
            out2_im[1] = _mm512_sub_pd(mqin_im[4],mqin_re[10]);            
            out2_im[2] = _mm512_sub_pd(mqin_im[5],mqin_re[11]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ux_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ux_im+jc*8+ic*24);
               }
            
            su3_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);

            mqout_re[0] = _mm512_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm512_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm512_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm512_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm512_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm512_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm512_add_pd(mqout_re[6] ,map0_im[0]);
            mqout_re[7] = _mm512_add_pd(mqout_re[7] ,map0_im[1]);
            mqout_re[8] = _mm512_add_pd(mqout_re[8] ,map0_im[2]);
            mqout_re[9] = _mm512_sub_pd(mqout_re[9] ,map1_im[0]);
            mqout_re[10]= _mm512_sub_pd(mqout_re[10],map1_im[1]);
            mqout_re[11]= _mm512_sub_pd(mqout_re[11],map1_im[2]);

            mqout_im[0] = _mm512_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm512_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm512_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm512_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm512_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm512_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm512_sub_pd(mqout_im[6] ,map0_re[0]);
            mqout_im[7] = _mm512_sub_pd(mqout_im[7] ,map0_re[1]);
            mqout_im[8] = _mm512_sub_pd(mqout_im[8] ,map0_re[2]);
            mqout_im[9] = _mm512_add_pd(mqout_im[9] ,map1_re[0]);
            mqout_im[10]= _mm512_add_pd(mqout_im[10],map1_re[1]);
            mqout_im[11]= _mm512_add_pd(mqout_im[11],map1_re[2]);



            //===================
            //===== +3 ==========
            //===================
      
            ux_re += 72;
            ux_im += 72;

            sx_re = qin_re + plqcd_g.iup_mic_split[ipt][3]*96;
            sx_im = qin_im + plqcd_g.iup_mic_split[ipt][3]*96;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm512_load_pd(sx_re);
            mqin_re[1]  = _mm512_load_pd(sx_re+8);
            mqin_re[2]  = _mm512_load_pd(sx_re+16);
            mqin_im[0]  = _mm512_load_pd(sx_im);
            mqin_im[1]  = _mm512_load_pd(sx_im+8);
            mqin_im[2]  = _mm512_load_pd(sx_im+16);


            mqin_re[6]  = _mm512_load_pd(sx_re+48);
            mqin_re[7]  = _mm512_load_pd(sx_re+56);
            mqin_re[8]  = _mm512_load_pd(sx_re+64);
            mqin_im[6]  = _mm512_load_pd(sx_im+48);
            mqin_im[7]  = _mm512_load_pd(sx_im+56);
            mqin_im[8]  = _mm512_load_pd(sx_im+64);


            out1_re[0] = _mm512_add_pd(mqin_re[0],mqin_re[6]);            
            out1_re[1] = _mm512_add_pd(mqin_re[1],mqin_re[7]);            
            out1_re[2] = _mm512_add_pd(mqin_re[2],mqin_re[8]);            
            out1_im[0] = _mm512_add_pd(mqin_im[0],mqin_im[6]);            
            out1_im[1] = _mm512_add_pd(mqin_im[1],mqin_im[7]);            
            out1_im[2] = _mm512_add_pd(mqin_im[2],mqin_im[8]);

            mqin_re[3]  = _mm512_load_pd(sx_re+24);
            mqin_re[4]  = _mm512_load_pd(sx_re+32);
            mqin_re[5]  = _mm512_load_pd(sx_re+40);
            mqin_im[3]  = _mm512_load_pd(sx_im+24);
            mqin_im[4]  = _mm512_load_pd(sx_im+32);
            mqin_im[5]  = _mm512_load_pd(sx_im+40);


            mqin_re[9]   = _mm512_load_pd(sx_re+72);
            mqin_re[10]  = _mm512_load_pd(sx_re+80);
            mqin_re[11]  = _mm512_load_pd(sx_re+88);
            mqin_im[9]   = _mm512_load_pd(sx_im+72);
            mqin_im[10]  = _mm512_load_pd(sx_im+80);
            mqin_im[11]  = _mm512_load_pd(sx_im+88);


            out2_re[0] = _mm512_add_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm512_add_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm512_add_pd(mqin_re[5],mqin_re[8]);            
            out2_im[0] = _mm512_add_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm512_add_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm512_add_pd(mqin_im[5],mqin_im[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ux_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ux_im+jc*8+ic*24);
               }
            
            su3_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);

            mqout_re[0] = _mm512_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm512_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm512_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm512_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm512_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm512_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm512_add_pd(mqout_re[6] ,map0_re[0]);
            mqout_re[7] = _mm512_add_pd(mqout_re[7] ,map0_re[1]);
            mqout_re[8] = _mm512_add_pd(mqout_re[8] ,map0_re[2]);
            mqout_re[9] = _mm512_add_pd(mqout_re[9] ,map1_re[0]);
            mqout_re[10]= _mm512_add_pd(mqout_re[10],map1_re[1]);
            mqout_re[11]= _mm512_add_pd(mqout_re[11],map1_re[2]);

            mqout_im[0] = _mm512_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm512_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm512_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm512_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm512_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm512_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm512_add_pd(mqout_im[6] ,map0_im[0]);
            mqout_im[7] = _mm512_add_pd(mqout_im[7] ,map0_im[1]);
            mqout_im[8] = _mm512_add_pd(mqout_im[8] ,map0_im[2]);
            mqout_im[9] = _mm512_add_pd(mqout_im[9] ,map1_im[0]);
            mqout_im[10]= _mm512_add_pd(mqout_im[10],map1_im[1]);
            mqout_im[11]= _mm512_add_pd(mqout_im[11],map1_im[2]);



            //=======================
            //===== -0  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_mic_split[ipt][0]*288;
            ux_im = u_im + plqcd_g.idn_mic_split[ipt][0]*288;

            sx_re = qin_re + plqcd_g.idn_mic_split[ipt][0]*96;
            sx_im = qin_im + plqcd_g.idn_mic_split[ipt][0]*96;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm512_load_pd(sx_re);
            mqin_re[1]  = _mm512_load_pd(sx_re+8);
            mqin_re[2]  = _mm512_load_pd(sx_re+16);
            mqin_im[0]  = _mm512_load_pd(sx_im);
            mqin_im[1]  = _mm512_load_pd(sx_im+8);
            mqin_im[2]  = _mm512_load_pd(sx_im+16);


            mqin_re[9]   = _mm512_load_pd(sx_re+72);
            mqin_re[10]  = _mm512_load_pd(sx_re+80);
            mqin_re[11]  = _mm512_load_pd(sx_re+88);
            mqin_im[9]   = _mm512_load_pd(sx_im+72);
            mqin_im[10]  = _mm512_load_pd(sx_im+80);
            mqin_im[11]  = _mm512_load_pd(sx_im+88);


            out1_re[0] = _mm512_add_pd(mqin_re[0],mqin_im[9]);            
            out1_re[1] = _mm512_add_pd(mqin_re[1],mqin_im[10]);            
            out1_re[2] = _mm512_add_pd(mqin_re[2],mqin_im[11]);            
            out1_im[0] = _mm512_sub_pd(mqin_im[0],mqin_re[9]);            
            out1_im[1] = _mm512_sub_pd(mqin_im[1],mqin_re[10]);            
            out1_im[2] = _mm512_sub_pd(mqin_im[2],mqin_re[11]);


            mqin_re[3]  = _mm512_load_pd(sx_re+24);
            mqin_re[4]  = _mm512_load_pd(sx_re+32);
            mqin_re[5]  = _mm512_load_pd(sx_re+40);
            mqin_im[3]  = _mm512_load_pd(sx_im+24);
            mqin_im[4]  = _mm512_load_pd(sx_im+32);
            mqin_im[5]  = _mm512_load_pd(sx_im+40);


            mqin_re[6]  = _mm512_load_pd(sx_re+48);
            mqin_re[7]  = _mm512_load_pd(sx_re+56);
            mqin_re[8]  = _mm512_load_pd(sx_re+64);
            mqin_im[6]  = _mm512_load_pd(sx_im+48);
            mqin_im[7]  = _mm512_load_pd(sx_im+56);
            mqin_im[8]  = _mm512_load_pd(sx_im+64);


            out2_re[0] = _mm512_add_pd(mqin_re[3],mqin_im[6]);            
            out2_re[1] = _mm512_add_pd(mqin_re[4],mqin_im[7]);            
            out2_re[2] = _mm512_add_pd(mqin_re[5],mqin_im[8]);            
            out2_im[0] = _mm512_sub_pd(mqin_im[3],mqin_re[6]);            
            out2_im[1] = _mm512_sub_pd(mqin_im[4],mqin_re[7]);            
            out2_im[2] = _mm512_sub_pd(mqin_im[5],mqin_re[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ux_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ux_im+jc*8+ic*24);
               }
            
            su3_inverse_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            //store the result, shuffle if on the boundary
            if(ix == lx-1)
            {
               for(int i=0; i<3; i++)
               {  
                  out3_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map0_re[i]);
                  out3_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map0_im[i]);
                  out4_re[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map1_re[i]);
                  out4_im[i]= (__m512d)  _mm512_permutevar_epi32 ( mpermfor, (__m512i) map1_im[i]);
               }
               
               mqout_re[0] = _mm512_add_pd(mqout_re[0] ,out3_re[0]);
               mqout_re[1] = _mm512_add_pd(mqout_re[1] ,out3_re[1]);
               mqout_re[2] = _mm512_add_pd(mqout_re[2] ,out3_re[2]);
               mqout_re[3] = _mm512_add_pd(mqout_re[3] ,out4_re[0]);
               mqout_re[4] = _mm512_add_pd(mqout_re[4] ,out4_re[1]);
               mqout_re[5] = _mm512_add_pd(mqout_re[5] ,out4_re[2]);

               mqout_re[6] = _mm512_sub_pd(mqout_re[6] ,out4_im[0]);
               mqout_re[7] = _mm512_sub_pd(mqout_re[7] ,out4_im[1]);
               mqout_re[8] = _mm512_sub_pd(mqout_re[8] ,out4_im[2]);
               mqout_re[9] = _mm512_sub_pd(mqout_re[9] ,out3_im[0]);
               mqout_re[10]= _mm512_sub_pd(mqout_re[10],out3_im[1]);
               mqout_re[11]= _mm512_sub_pd(mqout_re[11],out3_im[2]);

               mqout_im[0] = _mm512_add_pd(mqout_im[0] ,out3_im[0]);
               mqout_im[1] = _mm512_add_pd(mqout_im[1] ,out3_im[1]);
               mqout_im[2] = _mm512_add_pd(mqout_im[2] ,out3_im[2]);
               mqout_im[3] = _mm512_add_pd(mqout_im[3] ,out4_im[0]);
               mqout_im[4] = _mm512_add_pd(mqout_im[4] ,out4_im[1]);
               mqout_im[5] = _mm512_add_pd(mqout_im[5] ,out4_im[2]);

               mqout_im[6] = _mm512_add_pd(mqout_im[6] ,out4_re[0]);
               mqout_im[7] = _mm512_add_pd(mqout_im[7] ,out4_re[1]);
               mqout_im[8] = _mm512_add_pd(mqout_im[8] ,out4_re[2]);
               mqout_im[9] = _mm512_add_pd(mqout_im[9] ,out3_re[0]);
               mqout_im[10]= _mm512_add_pd(mqout_im[10],out3_re[1]);
               mqout_im[11]= _mm512_add_pd(mqout_im[11],out3_re[2]);

            }
            else
            {
               mqout_re[0] = _mm512_add_pd(mqout_re[0] ,map0_re[0]);
               mqout_re[1] = _mm512_add_pd(mqout_re[1] ,map0_re[1]);
               mqout_re[2] = _mm512_add_pd(mqout_re[2] ,map0_re[2]);
               mqout_re[3] = _mm512_add_pd(mqout_re[3] ,map1_re[0]);
               mqout_re[4] = _mm512_add_pd(mqout_re[4] ,map1_re[1]);
               mqout_re[5] = _mm512_add_pd(mqout_re[5] ,map1_re[2]);

               mqout_re[6] = _mm512_sub_pd(mqout_re[6] ,map1_im[0]);
               mqout_re[7] = _mm512_sub_pd(mqout_re[7] ,map1_im[1]);
               mqout_re[8] = _mm512_sub_pd(mqout_re[8] ,map1_im[2]);
               mqout_re[9] = _mm512_sub_pd(mqout_re[9] ,map0_im[0]);
               mqout_re[10]= _mm512_sub_pd(mqout_re[10],map0_im[1]);
               mqout_re[11]= _mm512_sub_pd(mqout_re[11],map0_im[2]);

               mqout_im[0] = _mm512_add_pd(mqout_im[0] ,map0_im[0]);
               mqout_im[1] = _mm512_add_pd(mqout_im[1] ,map0_im[1]);
               mqout_im[2] = _mm512_add_pd(mqout_im[2] ,map0_im[2]);
               mqout_im[3] = _mm512_add_pd(mqout_im[3] ,map1_im[0]);
               mqout_im[4] = _mm512_add_pd(mqout_im[4] ,map1_im[1]);
               mqout_im[5] = _mm512_add_pd(mqout_im[5] ,map1_im[2]);

               mqout_im[6] = _mm512_add_pd(mqout_im[6] ,map1_re[0]);
               mqout_im[7] = _mm512_add_pd(mqout_im[7] ,map1_re[1]);
               mqout_im[8] = _mm512_add_pd(mqout_im[8] ,map1_re[2]);
               mqout_im[9] = _mm512_add_pd(mqout_im[9] ,map0_re[0]);
               mqout_im[10]= _mm512_add_pd(mqout_im[10],map0_re[1]);
               mqout_im[11]= _mm512_add_pd(mqout_im[11],map0_re[2]);
            } 

            //=======================
            //===== -1  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_mic_split[ipt][1]*288;
            ux_im = u_im + plqcd_g.idn_mic_split[ipt][1]*288;

            sx_re = qin_re + plqcd_g.idn_mic_split[ipt][1]*96;
            sx_im = qin_im + plqcd_g.idn_mic_split[ipt][1]*96;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm512_load_pd(sx_re);
            mqin_re[1]  = _mm512_load_pd(sx_re+8);
            mqin_re[2]  = _mm512_load_pd(sx_re+16);
            mqin_im[0]  = _mm512_load_pd(sx_im);
            mqin_im[1]  = _mm512_load_pd(sx_im+8);
            mqin_im[2]  = _mm512_load_pd(sx_im+16);


            mqin_re[9]   = _mm512_load_pd(sx_re+72);
            mqin_re[10]  = _mm512_load_pd(sx_re+80);
            mqin_re[11]  = _mm512_load_pd(sx_re+88);
            mqin_im[9]   = _mm512_load_pd(sx_im+72);
            mqin_im[10]  = _mm512_load_pd(sx_im+80);
            mqin_im[11]  = _mm512_load_pd(sx_im+88);


            out1_re[0] = _mm512_sub_pd(mqin_re[0],mqin_re[9]);            
            out1_re[1] = _mm512_sub_pd(mqin_re[1],mqin_re[10]);            
            out1_re[2] = _mm512_sub_pd(mqin_re[2],mqin_re[11]);            
            out1_im[0] = _mm512_sub_pd(mqin_im[0],mqin_im[9]);            
            out1_im[1] = _mm512_sub_pd(mqin_im[1],mqin_im[10]);            
            out1_im[2] = _mm512_sub_pd(mqin_im[2],mqin_im[11]);


            mqin_re[3]  = _mm512_load_pd(sx_re+24);
            mqin_re[4]  = _mm512_load_pd(sx_re+32);
            mqin_re[5]  = _mm512_load_pd(sx_re+40);
            mqin_im[3]  = _mm512_load_pd(sx_im+24);
            mqin_im[4]  = _mm512_load_pd(sx_im+32);
            mqin_im[5]  = _mm512_load_pd(sx_im+40);


            mqin_re[6]  = _mm512_load_pd(sx_re+48);
            mqin_re[7]  = _mm512_load_pd(sx_re+56);
            mqin_re[8]  = _mm512_load_pd(sx_re+64);
            mqin_im[6]  = _mm512_load_pd(sx_im+48);
            mqin_im[7]  = _mm512_load_pd(sx_im+56);
            mqin_im[8]  = _mm512_load_pd(sx_im+64);


            out2_re[0] = _mm512_add_pd(mqin_re[3],mqin_re[6]);            
            out2_re[1] = _mm512_add_pd(mqin_re[4],mqin_re[7]);            
            out2_re[2] = _mm512_add_pd(mqin_re[5],mqin_re[8]);            
            out2_im[0] = _mm512_add_pd(mqin_im[3],mqin_im[6]);            
            out2_im[1] = _mm512_add_pd(mqin_im[4],mqin_im[7]);            
            out2_im[2] = _mm512_add_pd(mqin_im[5],mqin_im[8]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ux_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ux_im+jc*8+ic*24);
               }
            
            su3_inverse_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            mqout_re[0] = _mm512_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm512_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm512_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm512_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm512_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm512_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm512_add_pd(mqout_re[6] ,map1_re[0]);
            mqout_re[7] = _mm512_add_pd(mqout_re[7] ,map1_re[1]);
            mqout_re[8] = _mm512_add_pd(mqout_re[8] ,map1_re[2]);
            mqout_re[9] = _mm512_sub_pd(mqout_re[9] ,map0_re[0]);
            mqout_re[10]= _mm512_sub_pd(mqout_re[10],map0_re[1]);
            mqout_re[11]= _mm512_sub_pd(mqout_re[11],map0_re[2]);

            mqout_im[0] = _mm512_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm512_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm512_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm512_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm512_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm512_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm512_add_pd(mqout_im[6] ,map1_im[0]);
            mqout_im[7] = _mm512_add_pd(mqout_im[7] ,map1_im[1]);
            mqout_im[8] = _mm512_add_pd(mqout_im[8] ,map1_im[2]);
            mqout_im[9] = _mm512_sub_pd(mqout_im[9] ,map0_im[0]);
            mqout_im[10]= _mm512_sub_pd(mqout_im[10],map0_im[1]);
            mqout_im[11]= _mm512_sub_pd(mqout_im[11],map0_im[2]);
            
            //=======================
            //===== -2  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_mic_split[ipt][2]*288;
            ux_im = u_im + plqcd_g.idn_mic_split[ipt][2]*288;

            sx_re = qin_re + plqcd_g.idn_mic_split[ipt][2]*96;
            sx_im = qin_im + plqcd_g.idn_mic_split[ipt][2]*96;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm512_load_pd(sx_re);
            mqin_re[1]  = _mm512_load_pd(sx_re+8);
            mqin_re[2]  = _mm512_load_pd(sx_re+16);
            mqin_im[0]  = _mm512_load_pd(sx_im);
            mqin_im[1]  = _mm512_load_pd(sx_im+8);
            mqin_im[2]  = _mm512_load_pd(sx_im+16);


            mqin_re[6]   = _mm512_load_pd(sx_re+48);
            mqin_re[7]   = _mm512_load_pd(sx_re+56);
            mqin_re[8]   = _mm512_load_pd(sx_re+64);
            mqin_im[6]   = _mm512_load_pd(sx_im+48);
            mqin_im[7]   = _mm512_load_pd(sx_im+56);
            mqin_im[8]   = _mm512_load_pd(sx_im+64);


            out1_re[0] = _mm512_add_pd(mqin_re[0],mqin_im[6]);            
            out1_re[1] = _mm512_add_pd(mqin_re[1],mqin_im[7]);            
            out1_re[2] = _mm512_add_pd(mqin_re[2],mqin_im[8]);            
            out1_im[0] = _mm512_sub_pd(mqin_im[0],mqin_re[6]);            
            out1_im[1] = _mm512_sub_pd(mqin_im[1],mqin_re[7]);            
            out1_im[2] = _mm512_sub_pd(mqin_im[2],mqin_re[8]);


            mqin_re[3]  = _mm512_load_pd(sx_re+24);
            mqin_re[4]  = _mm512_load_pd(sx_re+32);
            mqin_re[5]  = _mm512_load_pd(sx_re+40);
            mqin_im[3]  = _mm512_load_pd(sx_im+24);
            mqin_im[4]  = _mm512_load_pd(sx_im+32);
            mqin_im[5]  = _mm512_load_pd(sx_im+40);


            mqin_re[9]   = _mm512_load_pd(sx_re+72);
            mqin_re[10]  = _mm512_load_pd(sx_re+80);
            mqin_re[11]  = _mm512_load_pd(sx_re+88);
            mqin_im[9]   = _mm512_load_pd(sx_im+72);
            mqin_im[10]  = _mm512_load_pd(sx_im+80);
            mqin_im[11]  = _mm512_load_pd(sx_im+88);


            out2_re[0] = _mm512_sub_pd(mqin_re[3],mqin_im[9]);            
            out2_re[1] = _mm512_sub_pd(mqin_re[4],mqin_im[10]);            
            out2_re[2] = _mm512_sub_pd(mqin_re[5],mqin_im[11]);            
            out2_im[0] = _mm512_add_pd(mqin_im[3],mqin_re[9]);            
            out2_im[1] = _mm512_add_pd(mqin_im[4],mqin_re[10]);            
            out2_im[2] = _mm512_add_pd(mqin_im[5],mqin_re[11]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ux_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ux_im+jc*8+ic*24);
               }
            
            su3_inverse_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            mqout_re[0] = _mm512_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm512_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm512_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm512_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm512_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm512_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm512_sub_pd(mqout_re[6] ,map0_im[0]);
            mqout_re[7] = _mm512_sub_pd(mqout_re[7] ,map0_im[1]);
            mqout_re[8] = _mm512_sub_pd(mqout_re[8] ,map0_im[2]);
            mqout_re[9] = _mm512_add_pd(mqout_re[9] ,map1_im[0]);
            mqout_re[10]= _mm512_add_pd(mqout_re[10],map1_im[1]);
            mqout_re[11]= _mm512_add_pd(mqout_re[11],map1_im[2]);

            mqout_im[0] = _mm512_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm512_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm512_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm512_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm512_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm512_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm512_add_pd(mqout_im[6] ,map0_re[0]);
            mqout_im[7] = _mm512_add_pd(mqout_im[7] ,map0_re[1]);
            mqout_im[8] = _mm512_add_pd(mqout_im[8] ,map0_re[2]);
            mqout_im[9] = _mm512_sub_pd(mqout_im[9] ,map1_re[0]);
            mqout_im[10]= _mm512_sub_pd(mqout_im[10],map1_re[1]);
            mqout_im[11]= _mm512_sub_pd(mqout_im[11],map1_re[2]);


            //=======================
            //===== -3  =============
            //======================= 

            ux_re = u_re + plqcd_g.idn_mic_split[ipt][3]*288;
            ux_im = u_im + plqcd_g.idn_mic_split[ipt][3]*288;

            sx_re = qin_re + plqcd_g.idn_mic_split[ipt][3]*96;
            sx_im = qin_im + plqcd_g.idn_mic_split[ipt][3]*96;

            //load the input spinor and build the halfspinor
            mqin_re[0]  = _mm512_load_pd(sx_re);
            mqin_re[1]  = _mm512_load_pd(sx_re+8);
            mqin_re[2]  = _mm512_load_pd(sx_re+16);
            mqin_im[0]  = _mm512_load_pd(sx_im);
            mqin_im[1]  = _mm512_load_pd(sx_im+8);
            mqin_im[2]  = _mm512_load_pd(sx_im+16);


            mqin_re[6]   = _mm512_load_pd(sx_re+48);
            mqin_re[7]   = _mm512_load_pd(sx_re+56);
            mqin_re[8]   = _mm512_load_pd(sx_re+64);
            mqin_im[6]   = _mm512_load_pd(sx_im+48);
            mqin_im[7]   = _mm512_load_pd(sx_im+56);
            mqin_im[8]   = _mm512_load_pd(sx_im+64);


            out1_re[0] = _mm512_sub_pd(mqin_re[0],mqin_re[6]);            
            out1_re[1] = _mm512_sub_pd(mqin_re[1],mqin_re[7]);            
            out1_re[2] = _mm512_sub_pd(mqin_re[2],mqin_re[8]);            
            out1_im[0] = _mm512_sub_pd(mqin_im[0],mqin_im[6]);            
            out1_im[1] = _mm512_sub_pd(mqin_im[1],mqin_im[7]);            
            out1_im[2] = _mm512_sub_pd(mqin_im[2],mqin_im[8]);


            mqin_re[3]  = _mm512_load_pd(sx_re+24);
            mqin_re[4]  = _mm512_load_pd(sx_re+32);
            mqin_re[5]  = _mm512_load_pd(sx_re+40);
            mqin_im[3]  = _mm512_load_pd(sx_im+24);
            mqin_im[4]  = _mm512_load_pd(sx_im+32);
            mqin_im[5]  = _mm512_load_pd(sx_im+40);


            mqin_re[9]   = _mm512_load_pd(sx_re+72);
            mqin_re[10]  = _mm512_load_pd(sx_re+80);
            mqin_re[11]  = _mm512_load_pd(sx_re+88);
            mqin_im[9]   = _mm512_load_pd(sx_im+72);
            mqin_im[10]  = _mm512_load_pd(sx_im+80);
            mqin_im[11]  = _mm512_load_pd(sx_im+88);


            out2_re[0] = _mm512_sub_pd(mqin_re[3],mqin_re[9]);            
            out2_re[1] = _mm512_sub_pd(mqin_re[4],mqin_re[10]);            
            out2_re[2] = _mm512_sub_pd(mqin_re[5],mqin_re[11]);            
            out2_im[0] = _mm512_sub_pd(mqin_im[3],mqin_im[9]);            
            out2_im[1] = _mm512_sub_pd(mqin_im[4],mqin_im[10]);            
            out2_im[2] = _mm512_sub_pd(mqin_im[5],mqin_im[11]);

            //load the gauge field and multiply
            for(int ic=0; ic < 3 ; ic++)
               for(int jc=0; jc < 3; jc++)
               {
                  U_re[ic][jc] = _mm512_load_pd(ux_re+jc*8+ic*24);
                  U_im[ic][jc] = _mm512_load_pd(ux_im+jc*8+ic*24);
               }
            
            su3_inverse_multiply_splitlayout_512(map0_re,map0_im,U_re,U_im,out1_re,out1_im);
            su3_inverse_multiply_splitlayout_512(map1_re,map1_im,U_re,U_im,out2_re,out2_im);
            
            mqout_re[0] = _mm512_add_pd(mqout_re[0] ,map0_re[0]);
            mqout_re[1] = _mm512_add_pd(mqout_re[1] ,map0_re[1]);
            mqout_re[2] = _mm512_add_pd(mqout_re[2] ,map0_re[2]);
            mqout_re[3] = _mm512_add_pd(mqout_re[3] ,map1_re[0]);
            mqout_re[4] = _mm512_add_pd(mqout_re[4] ,map1_re[1]);
            mqout_re[5] = _mm512_add_pd(mqout_re[5] ,map1_re[2]);

            mqout_re[6] = _mm512_sub_pd(mqout_re[6] ,map0_re[0]);
            mqout_re[7] = _mm512_sub_pd(mqout_re[7] ,map0_re[1]);
            mqout_re[8] = _mm512_sub_pd(mqout_re[8] ,map0_re[2]);
            mqout_re[9] = _mm512_sub_pd(mqout_re[9] ,map1_re[0]);
            mqout_re[10]= _mm512_sub_pd(mqout_re[10],map1_re[1]);
            mqout_re[11]= _mm512_sub_pd(mqout_re[11],map1_re[2]);

            mqout_im[0] = _mm512_add_pd(mqout_im[0] ,map0_im[0]);
            mqout_im[1] = _mm512_add_pd(mqout_im[1] ,map0_im[1]);
            mqout_im[2] = _mm512_add_pd(mqout_im[2] ,map0_im[2]);
            mqout_im[3] = _mm512_add_pd(mqout_im[3] ,map1_im[0]);
            mqout_im[4] = _mm512_add_pd(mqout_im[4] ,map1_im[1]);
            mqout_im[5] = _mm512_add_pd(mqout_im[5] ,map1_im[2]);

            mqout_im[6] = _mm512_sub_pd(mqout_im[6] ,map0_im[0]);
            mqout_im[7] = _mm512_sub_pd(mqout_im[7] ,map0_im[1]);
            mqout_im[8] = _mm512_sub_pd(mqout_im[8] ,map0_im[2]);
            mqout_im[9] = _mm512_sub_pd(mqout_im[9] ,map1_im[0]);
            mqout_im[10]= _mm512_sub_pd(mqout_im[10],map1_im[1]);
            mqout_im[11]= _mm512_sub_pd(mqout_im[11],map1_im[2]);






            //_mm512_stream_pd(gx_re       , mqout_re[0]);

            _mm512_store_pd(gx_re       , mqout_re[0]);
            _mm512_store_pd(gx_re+8     , mqout_re[1]);
            _mm512_store_pd(gx_re+16    , mqout_re[2]);
            _mm512_store_pd(gx_re+24    , mqout_re[3]);
            _mm512_store_pd(gx_re+32    , mqout_re[4]);
            _mm512_store_pd(gx_re+40    , mqout_re[5]);
            _mm512_store_pd(gx_re+48    , mqout_re[6]);
            _mm512_store_pd(gx_re+56    , mqout_re[7]);
            _mm512_store_pd(gx_re+64    , mqout_re[8]);
            _mm512_store_pd(gx_re+72    , mqout_re[9]);
            _mm512_store_pd(gx_re+80    , mqout_re[10]);
            _mm512_store_pd(gx_re+88    , mqout_re[11]);

            _mm512_store_pd(gx_im       , mqout_im[0]);
            _mm512_store_pd(gx_im+8     , mqout_im[1]);
            _mm512_store_pd(gx_im+16    , mqout_im[2]);
            _mm512_store_pd(gx_im+24    , mqout_im[3]);
            _mm512_store_pd(gx_im+32    , mqout_im[4]);
            _mm512_store_pd(gx_im+40    , mqout_im[5]);
            _mm512_store_pd(gx_im+48    , mqout_im[6]);
            _mm512_store_pd(gx_im+56    , mqout_im[7]);
            _mm512_store_pd(gx_im+64    , mqout_im[8]);
            _mm512_store_pd(gx_im+72    , mqout_im[9]);
            _mm512_store_pd(gx_im+80    , mqout_im[10]);
            _mm512_store_pd(gx_im+88    , mqout_im[11]);


      }

   }
#ifdef _OPENMP
}
#endif  //end of the openmp parallel reigon

   
   return stop_watch(ts);
}
//========================================================

#endif
#endif
