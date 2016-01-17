/*********************************************************************************
 * File test_comms.c
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
 * testing MPI communications in plqcd as will be used in the hopping matrix
 *********************************************************************************/

#include"plqcd.h"

/***************************************************
 *test code for MPI communications 
 **************************************************/

static void hop(spinor *chi, spinor *psi, int ieo, FILE *pf);    //chi[i] = sum_mu { psi[i+mu] + psi[i-mu] }

static void print_spinor(spinor *psi, FILE *pf);

static void print_halfspinor(halfspinor *psi, FILE *pf, int size);

static double norm_spinor(spinor *psi);

static double norm_halfspinor(halfspinor *psi);

int main(int argc, char **argv)
{
   //initialize plqcd
   int init_status;

   if(argc < 6){
     fprintf(stderr,"Error. Must pass the name of the input file and the location of the point source on the global lattice. \n");
     fprintf(stderr,"Usage: %s input_file_name xg[0] xg[1] xg[2] xg[3]\n",argv[0]);
     exit(1);
   }

   init_status = init_plqcd(argc,argv);   
   
   if(init_status != 0)
     printf("Error initializing plqcd\n");

   char ofname[128];

   char buff[128];

   int proc_id = ipr(plqcd_g.cpr);

   strcpy(ofname,"test_comm_output.procgrid.");

   sprintf(buff,"%d-%d-%d-%d.proc.%d",plqcd_g.nprocs[0],plqcd_g.nprocs[1],plqcd_g.nprocs[2],plqcd_g.nprocs[3],proc_id);

   strcat(ofname,buff);

   FILE *ofp=fopen(ofname,"w");

   print_params(ofp);

   //generate a spinor field
   //spinor *psi = (spinor *) alloc(plqcd_g.VOLUME*sizeof(spinor),plqcd_g.ALIGN); //input
   //spinor *chi = (spinor *) alloc(plqcd_g.VOLUME*sizeof(spinor),plqcd_g.ALIGN); //output
   spinor *psi = (spinor *) malloc(plqcd_g.VOLUME*sizeof(spinor),plqcd_g.ALIGN); //input
   spinor *chi = (spinor *) malloc(plqcd_g.VOLUME*sizeof(spinor),plqcd_g.ALIGN); //output

   //check allocation
   if(psi == NULL)
   {
       printf("not enough memory for psi\n");
       exit(0);
   }

   if(chi == NULL)
   {
       printf("not enough memory for chi\n");
       exit(0);
   }


   //setup values of psi as a point source at color 0, spin 0 and the input location on the global lattice
   
   //first set to zero
   int i,j,k;

   for(i=0; i< plqcd_g.VOLUME; i++)
   {
       psi[i].s0.c0=0.0;
       psi[i].s0.c1=0.0;
       psi[i].s0.c2=0.0;
       psi[i].s1.c0=0.0;
       psi[i].s1.c1=0.0;
       psi[i].s1.c2=0.0;
       psi[i].s2.c0=0.0;
       psi[i].s2.c1=0.0;
       psi[i].s2.c2=0.0;
       psi[i].s3.c0=0.0;
       psi[i].s3.c1=0.0;
       psi[i].s3.c2=0.0;


       chi[i].s0.c0=0.0;
       chi[i].s0.c1=0.0;
       chi[i].s0.c2=0.0;
       chi[i].s1.c0=0.0;
       chi[i].s1.c1=0.0;
       chi[i].s1.c2=0.0;
       chi[i].s2.c0=0.0;
       chi[i].s2.c1=0.0;
       chi[i].s2.c2=0.0;
       chi[i].s3.c0=0.0;
       chi[i].s3.c1=0.0;
       chi[i].s3.c2=0.0;
   }


   //now set the point source
   int xg[4]; //source location

   int source_pid, source_ix; //process id that contain the source and its index


   for(i=0; i<4; i++)
      xg[i]=atoi(argv[i+2]);

   int xsum=xg[0]+xg[1]+xg[2]+xg[3];

   //xg[0]=1; xg[1]=7; xg[2]=3; xg[3]=0;  //location of the source on the global lattice

   find_site(xg, &source_pid, &source_ix);
   
   if(proc_id == source_pid)
   {
      psi[source_ix].s0.c0=1.0;
   }
   
   

   //now we need to mimic the action of the hopping matrix
   if( (xsum%2) == 0) //apply H_oe 
      hop(chi,psi,1, ofp);
   else              //apply H_eo
      hop(chi,psi,0,ofp);

   fprintf(ofp,"#Input spinor\n");
   fprintf(ofp,"#--------------\n");
   print_spinor(psi,ofp);
   fprintf(ofp,"#Output spinor\n");
   fprintf(ofp,"#---------------\n");
   print_spinor(chi,ofp);

   afree(psi);
   afree(chi);
   finalize_plqcd();

   return 0;
}



//mimicing the action of the hopping matrix without the gauge field
//or the gamma matrices. Simply psi_out(x) = sum_mu{ psi_in(x-mu) + psi_in(x+mu) }
//---------------------------------------------------------------------------------
static void hop(spinor *chi, spinor *psi, int ieo, FILE *pf)
{

   int i,isub,cnt,mu;
   MPI_Status stp,stm;
   int tags[8];
   int V,face[4];

   V = plqcd_g.VOLUME;
   for(i=0; i<4; i++)
      face[i] = plqcd_g.face[i];

   //set the communication buffer to zero
   /*
   for(mu=0; mu<4; mu++)
   {
      for(i=0; i< V/2+face[mu]; i++)
      {
           plqcd_g.phip[mu][i].s0.c0 = 0.0;
           plqcd_g.phip[mu][i].s0.c1 = 0.0;
           plqcd_g.phip[mu][i].s0.c2 = 0.0;
           plqcd_g.phip[mu][i].s1.c0 = 0.0;
           plqcd_g.phip[mu][i].s1.c1 = 0.0;
           plqcd_g.phip[mu][i].s1.c2 = 0.0;

           plqcd_g.phim[mu][i].s0.c0 = 0.0;
           plqcd_g.phim[mu][i].s0.c1 = 0.0;
           plqcd_g.phim[mu][i].s0.c2 = 0.0;
           plqcd_g.phim[mu][i].s1.c0 = 0.0;
           plqcd_g.phim[mu][i].s1.c1 = 0.0;
           plqcd_g.phim[mu][i].s1.c2 = 0.0;
      }
   }
   */

   if(ieo==0)   //this is hop_eo
   {
       for(i=V/2; i< V; i++)
       {
           for(mu=0; mu < 4; mu++)
           {
              //phip[mu][x-mu] = psi[x]
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]].s0.c0 = psi[i].s0.c0;
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]].s0.c1 = psi[i].s0.c1;
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]].s0.c2 = psi[i].s0.c2;
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]].s1.c0 = psi[i].s1.c0;
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]].s1.c1 = psi[i].s1.c1;
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]].s1.c2 = psi[i].s1.c2;

              //phim[mu][x+mu] = psi[x] 
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]].s0.c0 = psi[i].s0.c0;
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]].s0.c1 = psi[i].s0.c1;
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]].s0.c2 = psi[i].s0.c2;
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]].s1.c0 = psi[i].s1.c0;
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]].s1.c1 = psi[i].s1.c1;
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]].s1.c2 = psi[i].s1.c2;
           }
       }

      //print auxilary fields before the exchange
      /*
      fprintf(pf,"\n#phip before exchange\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phip[mu],pf,V/2+face[mu]);
          fprintf(pf,"\n");
      } 

      fprintf(pf,"\n#phim before exchange\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phim[mu],pf,V/2+face[mu]);
          fprintf(pf,"\n");
      } 
      */

       //exchange
       for(mu=0; mu<4; mu++)
       {
          if(plqcd_g.nprocs[mu] > 1)
          {
             MPI_Send(&plqcd_g.phip[mu][V/2]  ,            face[mu]/2*2*3*2, MPI_DOUBLE, plqcd_g.npr[2*mu]  , 2*mu*10   , MPI_COMM_WORLD);
             MPI_Recv(&plqcd_g.phip[mu][V/2+face[mu]/2],            face[mu]/2*2*3*2, MPI_DOUBLE, plqcd_g.npr[2*mu+1], 2*mu*10   , MPI_COMM_WORLD, &stp);

             MPI_Send(&plqcd_g.phim[mu][V/2]  ,            face[mu]/2*2*3*2, MPI_DOUBLE, plqcd_g.npr[2*mu+1], (2*mu+7)*10, MPI_COMM_WORLD);
             MPI_Recv(&plqcd_g.phim[mu][V/2+face[mu]/2],            face[mu]/2*2*3*2, MPI_DOUBLE, plqcd_g.npr[2*mu]  , (2*mu+7)*10, MPI_COMM_WORLD, &stm);
          }
       }

      //print auxilary fields after the exchange
      /*
      fprintf(pf,"\n#phip after exchange\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phip[mu],pf,V/2+face[mu]);
          fprintf(pf,"\n");
      } 

      fprintf(pf,"\n#phim after exchange\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phim[mu],pf,V/2+face[mu]);
          fprintf(pf,"\n");
      } 
      */

      //copy the exchanged boundaries to the correspoding locations on the local phi fields
      for(mu=0; mu<4; mu++)
      { 
          if(plqcd_g.nprocs[mu] > 1)
          {   
             for(i=0; i< face[mu]/2; i++)
             {
                plqcd_g.phim[mu][plqcd_g.nn_bndo[2*mu+1][i]].s0.c0 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s0.c0;
                plqcd_g.phim[mu][plqcd_g.nn_bndo[2*mu+1][i]].s0.c1 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s0.c1;
                plqcd_g.phim[mu][plqcd_g.nn_bndo[2*mu+1][i]].s0.c2 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s0.c2;
                plqcd_g.phim[mu][plqcd_g.nn_bndo[2*mu+1][i]].s1.c0 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s1.c0;
                plqcd_g.phim[mu][plqcd_g.nn_bndo[2*mu+1][i]].s1.c1 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s1.c1;
                plqcd_g.phim[mu][plqcd_g.nn_bndo[2*mu+1][i]].s1.c2 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s1.c2;
             }
             for(i=0; i< face[mu]/2; i++)
             {
                plqcd_g.phip[mu][plqcd_g.nn_bndo[2*mu][i]].s0.c0 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s0.c0;
                plqcd_g.phip[mu][plqcd_g.nn_bndo[2*mu][i]].s0.c1 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s0.c1;
                plqcd_g.phip[mu][plqcd_g.nn_bndo[2*mu][i]].s0.c2 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s0.c2;
                plqcd_g.phip[mu][plqcd_g.nn_bndo[2*mu][i]].s1.c0 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s1.c0;
                plqcd_g.phip[mu][plqcd_g.nn_bndo[2*mu][i]].s1.c1 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s1.c1;
                plqcd_g.phip[mu][plqcd_g.nn_bndo[2*mu][i]].s1.c2 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s1.c2;
             }
          }
       }

      //print auxilary fields after the copying
      /*
      fprintf(pf,"\n#phip after copying\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phip[mu],pf,V/2+face[mu]/2);
          fprintf(pf,"\n");
      } 

      fprintf(pf,"\n#phim after copying\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phim[mu],pf,V/2+face[mu]/2);
          fprintf(pf,"\n");
      } 
      */




       //now build the solution
       for(i=0; i<V/2; i++)
       {
           chi[i].s0.c0 = 0.0;
           chi[i].s0.c1 = 0.0;
           chi[i].s0.c2 = 0.0;
           chi[i].s1.c0 = 0.0;
           chi[i].s1.c1 = 0.0;
           chi[i].s1.c2 = 0.0;
           
           for(mu=0; mu < 4; mu++)
           {
                chi[i].s0.c0 += plqcd_g.phip[mu][i].s0.c0 + plqcd_g.phim[mu][i].s0.c0; 
                chi[i].s0.c1 += plqcd_g.phip[mu][i].s0.c1 + plqcd_g.phim[mu][i].s0.c1; 
                chi[i].s0.c2 += plqcd_g.phip[mu][i].s0.c2 + plqcd_g.phim[mu][i].s0.c2; 

                chi[i].s1.c0 += plqcd_g.phip[mu][i].s1.c0 + plqcd_g.phim[mu][i].s1.c0; 
                chi[i].s1.c1 += plqcd_g.phip[mu][i].s1.c1 + plqcd_g.phim[mu][i].s1.c1; 
                chi[i].s1.c2 += plqcd_g.phip[mu][i].s1.c2 + plqcd_g.phim[mu][i].s1.c2;
           }
       } 
   //===============
   }
   else        //this is hop_oe
   {
       for(i=0; i< plqcd_g.VOLUME/2; i++)
       {
           for(mu=0; mu < 4; mu++)
           {
              //phip[mu][x-mu] = psi[x]
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]-V/2].s0.c0 = psi[i].s0.c0;
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]-V/2].s0.c1 = psi[i].s0.c1;
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]-V/2].s0.c2 = psi[i].s0.c2;
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]-V/2].s1.c0 = psi[i].s1.c0;
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]-V/2].s1.c1 = psi[i].s1.c1;
              plqcd_g.phip[mu][plqcd_g.idn[i][mu]-V/2].s1.c2 = psi[i].s1.c2;

              //phim[mu][x+mu] = psi[x] 
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]-V/2].s0.c0 = psi[i].s0.c0;
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]-V/2].s0.c1 = psi[i].s0.c1;
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]-V/2].s0.c2 = psi[i].s0.c2;
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]-V/2].s1.c0 = psi[i].s1.c0;
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]-V/2].s1.c1 = psi[i].s1.c1;
              plqcd_g.phim[mu][plqcd_g.iup[i][mu]-V/2].s1.c2 = psi[i].s1.c2;
           }
       }

      //print auxilary fields before the exchange
      /*
      fprintf(pf,"\n#phip before exchange\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phip[mu],pf,V/2+face[mu]);
          fprintf(pf,"\n");
      } 

      fprintf(pf,"\n#phim before exchange\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phim[mu],pf,V/2+face[mu]);
          fprintf(pf,"\n");
      } 
      */


       //exchange
       for(mu=0; mu<4; mu++)
       {
          if(plqcd_g.nprocs[mu] > 1)
          {
             MPI_Send(&plqcd_g.phip[mu][V/2]  ,            face[mu]/2*2*3*2, MPI_DOUBLE, plqcd_g.npr[2*mu]  , 2*mu*10   , MPI_COMM_WORLD);
             MPI_Recv(&plqcd_g.phip[mu][V/2+face[mu]/2],            face[mu]/2*2*3*2, MPI_DOUBLE, plqcd_g.npr[2*mu+1], 2*mu*10   , MPI_COMM_WORLD, &stp);

             MPI_Send(&plqcd_g.phim[mu][V/2]  ,            face[mu]/2*2*3*2, MPI_DOUBLE, plqcd_g.npr[2*mu+1], (2*mu+7)*10, MPI_COMM_WORLD);
             MPI_Recv(&plqcd_g.phim[mu][V/2+face[mu]/2],            face[mu]/2*2*3*2, MPI_DOUBLE, plqcd_g.npr[2*mu]  , (2*mu+7)*10, MPI_COMM_WORLD, &stm);
          }
       }

      //print auxilary fields after the exchange
      /*
      fprintf(pf,"\n#phip after exchange\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phip[mu],pf,V/2+face[mu]);
          fprintf(pf,"\n");
      } 

      fprintf(pf,"\n#phim after exchange\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phim[mu],pf,V/2+face[mu]);
          fprintf(pf,"\n");
      } 
      */

      //copy the exchanged boundaries to the correspoding locations on the phi fields
      for(mu=0; mu<4; mu++)
      { 
          if(plqcd_g.nprocs[mu] > 1)
          {
             
             for(i=0; i< face[mu]/2; i++)
             {
                plqcd_g.phim[mu][plqcd_g.nn_bnde[2*mu+1][i]-V/2].s0.c0 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s0.c0;
                plqcd_g.phim[mu][plqcd_g.nn_bnde[2*mu+1][i]-V/2].s0.c1 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s0.c1;
                plqcd_g.phim[mu][plqcd_g.nn_bnde[2*mu+1][i]-V/2].s0.c2 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s0.c2;
                plqcd_g.phim[mu][plqcd_g.nn_bnde[2*mu+1][i]-V/2].s1.c0 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s1.c0;
                plqcd_g.phim[mu][plqcd_g.nn_bnde[2*mu+1][i]-V/2].s1.c1 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s1.c1;
                plqcd_g.phim[mu][plqcd_g.nn_bnde[2*mu+1][i]-V/2].s1.c2 = plqcd_g.phim[mu][V/2+face[mu]/2+i].s1.c2;
             }

             
             for(i=0; i< face[mu]/2; i++)
             {
                plqcd_g.phip[mu][plqcd_g.nn_bnde[2*mu][i]-V/2].s0.c0 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s0.c0;
                plqcd_g.phip[mu][plqcd_g.nn_bnde[2*mu][i]-V/2].s0.c1 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s0.c1;
                plqcd_g.phip[mu][plqcd_g.nn_bnde[2*mu][i]-V/2].s0.c2 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s0.c2;
                plqcd_g.phip[mu][plqcd_g.nn_bnde[2*mu][i]-V/2].s1.c0 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s1.c0;
                plqcd_g.phip[mu][plqcd_g.nn_bnde[2*mu][i]-V/2].s1.c1 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s1.c1;
                plqcd_g.phip[mu][plqcd_g.nn_bnde[2*mu][i]-V/2].s1.c2 = plqcd_g.phip[mu][V/2+face[mu]/2+i].s1.c2;
             }
          }
       }

      //print auxilary fields after the copying
      /*
      fprintf(pf,"\n#phip after copying\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phip[mu],pf,V/2+face[mu]/2);
          fprintf(pf,"\n");
      } 

      fprintf(pf,"\n#phim after copying\n");
      for(mu=0; mu<4; mu++)
      {
          fprintf(pf,"#mu= %d\n",mu);
          print_halfspinor(plqcd_g.phim[mu],pf,V/2+face[mu]/2);
          fprintf(pf,"\n");
      } 
      */

       //now build the solution
       for(i=V/2; i<V; i++)
       {
           chi[i].s0.c0 = 0.0;
           chi[i].s0.c1 = 0.0;
           chi[i].s0.c2 = 0.0;
           chi[i].s1.c0 = 0.0;
           chi[i].s1.c1 = 0.0;
           chi[i].s1.c2 = 0.0;
           
           for(mu=0; mu < 4; mu++)
           {
                chi[i].s0.c0 += plqcd_g.phip[mu][i-V/2].s0.c0 + plqcd_g.phim[mu][i-V/2].s0.c0; 
                chi[i].s0.c1 += plqcd_g.phip[mu][i-V/2].s0.c1 + plqcd_g.phim[mu][i-V/2].s0.c1; 
                chi[i].s0.c2 += plqcd_g.phip[mu][i-V/2].s0.c2 + plqcd_g.phim[mu][i-V/2].s0.c2; 

                chi[i].s1.c0 += plqcd_g.phip[mu][i-V/2].s1.c0 + plqcd_g.phim[mu][i-V/2].s1.c0; 
                chi[i].s1.c1 += plqcd_g.phip[mu][i-V/2].s1.c1 + plqcd_g.phim[mu][i-V/2].s1.c1; 
                chi[i].s1.c2 += plqcd_g.phip[mu][i-V/2].s1.c2 + plqcd_g.phim[mu][i-V/2].s1.c2;
           }
       } 

   } //else 

   return;
}


static void print_spinor(spinor *psi, FILE *pf)
{
    int i,x[4],xg[4],mu;

    fprintf(pf,"# %8s %8s %8s %8s %8s %8s %8s %8s %8s\n\n","ipt_eo","x[0]","x[1]","x[2]","x[3]","xg[0]","xg[1]","xg[2]","xg[3]");

    for(i=0; i < plqcd_g.VOLUME; i++)
    {
        lex2cart(plqcd_g.ipt_lex[i],x);
        for(mu=0; mu<4; mu++)
           xg[mu] = x[mu]+plqcd_g.cpr[mu]*plqcd_g.latdims[mu];

        if(norm_spinor(&psi[i]) > 1e-16)
        {
      
            fprintf(pf,"# %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",i,x[0],x[1],x[2],x[3],xg[0],xg[1],xg[2],xg[3]);

            fprintf(pf,"# %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n\n",
                        creal(psi[i].s0.c0),cimag(psi[i].s0.c0),
                        creal(psi[i].s0.c1),cimag(psi[i].s0.c1),
                        creal(psi[i].s0.c2),cimag(psi[i].s0.c2),
                        creal(psi[i].s1.c0),cimag(psi[i].s1.c0),
                        creal(psi[i].s1.c1),cimag(psi[i].s1.c1),
                        creal(psi[i].s1.c2),cimag(psi[i].s1.c2));
         }
    } 
    
    return;
}


static void print_halfspinor(halfspinor *psi, FILE *pf, int size)
{
    int i,x[4],xg[4],mu;

    //fprintf(pf,"%8s %8s %8s %8s %8s %8s %8s %8s %8s\n\n","ipt_eo","x[0]","x[1]","x[2]","x[3]","xg[0]","xg[1]","xg[2]","xg[3]");

    for(i=0; i < size; i++)
    {
        //lex2cart(plqcd_g.ipt_lex[i],x);
        //for(mu=0; mu<4; mu++)
        //   xg[mu] = x[mu]+plqcd_g.cpr[mu]*plqcd_g.latdims[mu];

        if(norm_halfspinor(&psi[i]) > 1e-16)
        {
      
            //fprintf(pf,"%8d %8d %8d %8d %8d %8d %8d %8d %8d\n",i,x[0],x[1],x[2],x[3],xg[0],xg[1],xg[2],xg[3]);

            fprintf(pf,"#%8d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n",i,
                        creal(psi[i].s0.c0),cimag(psi[i].s0.c0),
                        creal(psi[i].s0.c1),cimag(psi[i].s0.c1),
                        creal(psi[i].s0.c2),cimag(psi[i].s0.c2),
                        creal(psi[i].s1.c0),cimag(psi[i].s1.c0),
                        creal(psi[i].s1.c1),cimag(psi[i].s1.c1),
                        creal(psi[i].s1.c2),cimag(psi[i].s1.c2));
         }
    } 
    
    return;
}

 
static double norm_spinor(spinor *psi)
{
    double val;

    val=cabs((*psi).s0.c0) + cabs((*psi).s0.c1) + cabs((*psi).s0.c2) + 
        cabs((*psi).s1.c0) + cabs((*psi).s1.c1) + cabs((*psi).s1.c2) + 
        cabs((*psi).s2.c0) + cabs((*psi).s2.c1) + cabs((*psi).s2.c2) + 
        cabs((*psi).s3.c0) + cabs((*psi).s3.c1) + cabs((*psi).s3.c2) ;

    return val; 

}


static double norm_halfspinor(halfspinor *psi)
{
    double val;

    val=cabs((*psi).s0.c0) + cabs((*psi).s0.c1) + cabs((*psi).s0.c2) + 
        cabs((*psi).s1.c0) + cabs((*psi).s1.c1) + cabs((*psi).s1.c2) ;

    return val; 

}


