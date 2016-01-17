/*********************************************************************************
 * File start.c
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
 *********************************************************************************/

#include"start.h"
#include<string.h>

int init_plqcd(int argc, char *argv[])
{

  int num_tasks=1,pid=0,rc,len;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int NPROCS,thread_support_avail;
  //Initialize MPI for multithreaded execuation
  //#ifdef MPI
  #ifdef _OPENMP
  rc=MPI_Init(&argc,&argv);
  if ( rc != MPI_SUCCESS ){
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  //rc=MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&thread_support_avail);
  //if ( (rc != MPI_SUCCESS) || (thread_support_avail != MPI_THREAD_SERIALIZED) ){
  //  printf ("Error starting MPI program. Terminating.\n");
  //  MPI_Abort(MPI_COMM_WORLD, rc);
  }
  #else
  rc=MPI_Init(&argc,&argv);
  if ( rc != MPI_SUCCESS ){
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  #endif

  MPI_Comm_size(MPI_COMM_WORLD,&num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&pid);
  MPI_Get_processor_name(hostname, &len);
  //#endif //MPI

  printf ("Number of tasks= %d My rank= %d Running on %s\n", num_tasks,pid,hostname);

  
  if(argc <2)
  {
      printf("Error: must pass input file name for the global paramters to be read from");
      return -1;
  }

  //initialize plqcd and read input prameters from file
  if(plqcd_g.init == 1) /*check if plqcd is already initialized*/
    return 0; //plqcd is already initialized, nothing to be done.

  
  /* Read input data from file */  
  FILE *fp = fopen(argv[1],"r");  
  if(fp == NULL)
  {
     fprintf(stderr,"Error opening input file. \n");
     exit(1); 
  }

  char var[128],val[128];
  int i;

  //set default values
  for(i=0; i<4; i++)
     plqcd_g.nprocs[i]=1;

  plqcd_g.nthread=1;
  plqcd_g.LX=plqcd_g.LY=plqcd_g.LZ=plqcd_g.LT=4;
  plqcd_g.ALIGN=16;

  //read input from the file
  while( fscanf(fp,"%s %s",var,val) != EOF )
  {
        /*assign value to the corresponding variables*/
        if(strcmp(var,"NPROC0")==0)
           plqcd_g.nprocs[0]=atoi(val);
        if ( strcmp(var,"NPROC1")==0)
           plqcd_g.nprocs[1]=atoi(val);
        if (strcmp(var,"NPROC2")==0)
           plqcd_g.nprocs[2]=atoi(val);
        if ( strcmp(var,"NPROC3")==0)
           plqcd_g.nprocs[3]=atoi(val);
        if ( strcmp(var,"NTHREAD")==0)
           plqcd_g.nthread=atoi(val);
        if ( strcmp(var,"L0")==0)
           plqcd_g.LX=atoi(val);
        if ( strcmp(var,"L1")==0)
           plqcd_g.LY=atoi(val);
        if ( strcmp(var,"L2")==0)
           plqcd_g.LZ=atoi(val);
        if (strcmp(var,"L3")==0)
           plqcd_g.LT=atoi(val);
        if (strcmp(var,"ALIGN")==0)
           plqcd_g.ALIGN=atoi(val);
  } /*while*/

  //set the number of openmp threads
  #ifdef _OPENMP
    omp_set_num_threads(plqcd_g.nthread);
  #else
    plqcd_g.nthread=1; //force this to be 1 regardless of the input value
  #endif

  /* Check the input */
  /* global lattice size in each direction has to be greater or equal to 4 and even*/
  if( (plqcd_g.LX<4) || ((plqcd_g.LX%2)!=0) || (plqcd_g.LY<4) || ((plqcd_g.LY%2) != 0) ||
      (plqcd_g.LZ<4) || ((plqcd_g.LZ%2)!=0) || (plqcd_g.LT<4) || ((plqcd_g.LT%2) != 0)    )
  {
          printf("ERROR: global lattice dimension in each direction must be >= 4 and even.\n");
          exit(1);
  }

  /*number of processes in each direction must be 1 or even number*/
  for(i=0; i<4; i++)
  {
     if( (plqcd_g.nprocs[i]<1) || ((plqcd_g.nprocs[i]>1) && ((plqcd_g.nprocs[i]%2)!=0)))
     {
           printf("ERROR: the number of processes in each direction must be 1 or even\n");
           exit(1);
     }
  }

  NPROCS = plqcd_g.nprocs[0]*plqcd_g.nprocs[1]*plqcd_g.nprocs[2]*plqcd_g.nprocs[3];
  //Check that this matches the number of requested proceeses passed to mpirun (or the quivalent)
  if( NPROCS != num_tasks ){
    printf("ERROR: number of requested processes by plqcd is different the runtime number passed to the excecutable.\n");
    exit(1);
  }

  //compute the cartesian coordinates of the process
  comp_cpr(pid,plqcd_g.cpr);


  //Compute the single index id of the nearest processes in the 4 directions
  set_npr();

  //set global parameters related to geometry
  set_geometry();



  /* Allocate aligned memory for half spinors needed by the hopping matrix*/
  #ifndef MIC
  for(i=0; i < 4; i++)
  {
     plqcd_g.phip[i] = (halfspinor *) amalloc( (plqcd_g.VOLUME/2+plqcd_g.face[i])*sizeof(halfspinor), plqcd_g.ALIGN);
     if(plqcd_g.phip[i] ==NULL)
     {
        printf("Error allocating memory for phip[%d]\n",i);
        exit(1);
     }
     plqcd_g.phim[i] = (halfspinor *) amalloc( (plqcd_g.VOLUME/2+plqcd_g.face[i])*sizeof(halfspinor), plqcd_g.ALIGN);
     if(plqcd_g.phim[i] ==NULL)
     {
        printf("Error allocating memory for phip[%d]\n",i);
        exit(1);
     }
  }
  #endif

  /* Allocate aligned memory for half spinors needed by the hopping matrix for the MIC*/

  #ifdef MIC & MIC_COMPLEX
  if( (plqcd_g.VOLUME%4) !=0 ){
    fprintf(stderr,"Error: Volume must be a multiple of 4 to use this version (MIC with complex variables)\n");
    exit(2);
  }

  for(i=0; i < 4; i++)
  {
     plqcd_g.phip512[i] = (halfspinor_512 *) amalloc((plqcd_g.VOLUME/8+plqcd_g.face[i]/4)*sizeof(halfspinor_512), plqcd_g.ALIGN);
     if(plqcd_g.phip512[i] ==NULL)
     {
        printf("Error allocating memory for phip512[%d]\n",i);
        exit(1);
     }
     plqcd_g.phim512[i] = (halfspinor_512 *) amalloc((plqcd_g.VOLUME/8+plqcd_g.face[i]/4)*sizeof(halfspinor_512), plqcd_g.ALIGN);
     if(plqcd_g.phim512[i] ==NULL)
     {
        printf("Error allocating memory for phip512[%d]\n",i);
        exit(1);
     }
  }
  #endif



  #ifdef MIC 
  #ifdef MIC_SPLIT

  int NC=3; //number of colors
  int NS=4; //number of spins
  if( (plqcd_g.VOLUME%8) !=0 ){
    fprintf(stderr,"Error: Volume must be a multiple of 8 to use this version (MIC with real and imaginary parts split)\n");
    exit(2);
  }

  for(i=0; i < 4; i++)
  {
     plqcd_g.phip512_re[i] = (double *) _mm_malloc( ( plqcd_g.VOLUME/2 + plqcd_g.face[i])*NC*NS/2 * sizeof(double), 64 );
     plqcd_g.phip512_im[i] = (double *) _mm_malloc( ( plqcd_g.VOLUME/2 + plqcd_g.face[i])*NC*NS/2 * sizeof(double), 64 );
     plqcd_g.phim512_re[i] = (double *) _mm_malloc( ( plqcd_g.VOLUME/2 + plqcd_g.face[i])*NC*NS/2 * sizeof(double), 64 );
     plqcd_g.phim512_im[i] = (double *) _mm_malloc( ( plqcd_g.VOLUME/2 + plqcd_g.face[i])*NC*NS/2 * sizeof(double), 64 );

     if( plqcd_g.phip512_re[i] == NULL || plqcd_g.phip512_im[i] ==NULL ||  plqcd_g.phim512_re[i] ==NULL || plqcd_g.phim512_im[i] ==NULL)
     {
        printf("Error allocating memory for phip/m512_re,im[%d]\n",i);
        exit(1);
     }
  }
  #endif
  #endif


  plqcd_g.init=1;  //now PLQCD is initialized


  return 0;

}


/*Finalize MPI communications and do other tasks that the program might need.
 */
void finalize_plqcd()
{
    //int i;
    //for(i=0; i<4; i++){
    //   afree(plqcd_g.phip[i]);
    //   afree(plqcd_g.phim[i]);
    //}
    //#ifdef MPI
    MPI_Finalize();
    //#endif  
}







void set_geometry()
{

   int i,k,jtmp;
   int nprocs[4],g_latsize[4], latsize[4], bnds[4];  //define these to save typing
   int x,y,z,t,xcor[4],xshift[4],xsave;
   int cnt_even,cnt_odd,cnt_bnd_even,cnt_bnd_odd,ilex,ilex_g;
   int idir;
   int mu;



   for(i=0; i<4; i++)
      nprocs[i] = plqcd_g.nprocs[i];

   g_latsize[0] = plqcd_g.LX; 
   g_latsize[1] = plqcd_g.LY; 
   g_latsize[2] = plqcd_g.LZ; 
   g_latsize[3] = plqcd_g.LT; 
   


   //set local lattice size and check
   for(i=0; i<4; i++){
      latsize[i] = g_latsize[i]/nprocs[i];
      k=latsize[i]*nprocs[i]-g_latsize[i];
      if( (k != 0) || (latsize[i] <4 ) || ((latsize[i]%2) != 0) ){
        printf("ERROR: local lattice size in the direction i must be a multiple of the number of MPI tasks in the i-th direction, >=4 and even number\n");
        exit(1);
      } 
      else{
        plqcd_g.latdims[i] = latsize[i];
      }
   }
       
   
   //local volume
   plqcd_g.VOLUME = latsize[0]*latsize[1]*latsize[2]*latsize[3];

   //size of the boundaries
   if(nprocs[0] > 1)
     bnds[0] = latsize[1]*latsize[2]*latsize[3];
   else
     bnds[0] = 0;

   if(nprocs[1] > 1)
     bnds[1] = latsize[0]*latsize[2]*latsize[3];
   else
     bnds[1] = 0;


   if(nprocs[2] > 1)
     bnds[2] = latsize[0]*latsize[1]*latsize[3];
   else
     bnds[2] = 0;

   if(nprocs[3] > 1)
     bnds[3] = latsize[0]*latsize[1]*latsize[2];
   else
     bnds[3] = 0;

   for(i=0; i<4; i++)
      plqcd_g.face[i] = bnds[i];


   //allocate memory for the geometry arrays of the lattice points and set them up
   if( (plqcd_g.ipt_eo=(int *) calloc(plqcd_g.VOLUME,sizeof(int))) == NULL)
   {
      printf("ERROR: insufficient memory for ipt_eo array.\n");
      exit(1);
   }
   
   if( (plqcd_g.ipt_lex=(int *) calloc(plqcd_g.VOLUME,sizeof(int))) == NULL)
   {
      printf("ERROR: insufficient memory for ipt_lex array.\n");
      exit(1);
   }
   
   for(mu=0; mu < 4; mu++)
   {
      if(plqcd_g.face[mu]> 0){

         if( (plqcd_g.nn_bnde[2*mu]=(int *) calloc(plqcd_g.face[mu]/2,sizeof(int))) == NULL)
         {
            printf("ERROR: insufficient memory for nn_bnde array.\n");
            exit(1);
         }

         if( (plqcd_g.nn_bndo[2*mu]=(int *) calloc(plqcd_g.face[mu]/2,sizeof(int))) == NULL)
         {
            printf("ERROR: insufficient memory for nn_bndo array.\n");
            exit(1);
         }


         if( (plqcd_g.nn_bnde[2*mu+1]=(int *) calloc(plqcd_g.face[mu]/2,sizeof(int))) == NULL)
         {
            printf("ERROR: insufficient memory for nn_bnde array.\n");
            exit(1);
         }

         if( (plqcd_g.nn_bndo[2*mu+1]=(int *) calloc(plqcd_g.face[mu]/2,sizeof(int))) == NULL)
         {
            printf("ERROR: insufficient memory for nn_bndo array.\n");
            exit(1);
         }

     }
     else{
         plqcd_g.nn_bnde[2*mu]=NULL;
         plqcd_g.nn_bndo[2*mu]=NULL;
         plqcd_g.nn_bnde[2*mu+1]=NULL;
         plqcd_g.nn_bndo[2*mu+1]=NULL;
     }
     
   }
   
   if( (plqcd_g.iup=(int **) calloc(plqcd_g.VOLUME,sizeof(int *))) == NULL)
   {
      printf("ERROR: insufficient memory for iup array.\n");
      exit(1);
   }
   
   if( (plqcd_g.idn=(int **) calloc(plqcd_g.VOLUME,sizeof(int *))) == NULL)
   {
      printf("ERROR: insufficient memory for idn array.\n");
      exit(1);
   }


   for(i=0; i<plqcd_g.VOLUME; i++)
   {

       if( (plqcd_g.iup[i]=(int *) calloc(4,sizeof(int))) == NULL)
       {
          printf("ERROR: insufficient memory for iup array.\n");
          exit(1);
       }
   
       if( (plqcd_g.idn[i]=(int *) calloc(4,sizeof(int))) == NULL)
       {
          printf("ERROR: insufficient memory for idn array.\n");
          exit(1);
       }
   }


   //set the lexiographic and even-odd indices of the sites
   //Note: 0,1,2,3 correspond to x,y,z,t directions and
   //x runs fastest (innermost loop) while t runs slowest(outermost loop)

   cnt_even=0;                 //first index of even sites
   cnt_odd =plqcd_g.VOLUME/2;  //first index of odd sites
   for(t=0; t < plqcd_g.latdims[3]; t++)
     for(z=0; z < plqcd_g.latdims[2]; z++)
       for(y=0; y < plqcd_g.latdims[1]; y++)
          for(x=0; x < plqcd_g.latdims[0]; x++)
          {           
              xcor[0] = x; xcor[1]=y; xcor[2]=z; xcor[3]=t;
              ilex    = cart2lex(xcor);
              ilex_g  = 0;  //lexiographic index w.r.t. the global lattice
              for(i=0; i<4; i++){
                 ilex_g += xcor[i] + plqcd_g.cpr[i]*plqcd_g.latdims[i];
              }

              //check even or odd
              if( (ilex_g%2) == 0 ) //even
              {     
                plqcd_g.ipt_eo[ilex]      = cnt_even;
                plqcd_g.ipt_lex[cnt_even] = ilex;
                cnt_even++;
              }
              else //odd
              {
                plqcd_g.ipt_eo[ilex]      = cnt_odd;
                plqcd_g.ipt_lex[cnt_odd]  = ilex;
                cnt_odd++;
              }
          }//end of the 4 for loops 



  //Nearest neighbours
  //------------------
  for(idir=0; idir < 4; idir++)
  {
     //neighbours in the -idir direction
     cnt_bnd_even = plqcd_g.VOLUME/2;
     cnt_bnd_odd  = plqcd_g.VOLUME;
     
     for(i=0; i< plqcd_g.VOLUME; i++) //i is the even-odd index of the sites 
     {
        ilex=plqcd_g.ipt_lex[i];
        lex2cart(ilex,xcor);
  
        xsave=xcor[idir];
 
        xcor[idir] -= 1;

        if( xcor[idir] >= 0)
        {
           plqcd_g.idn[i][idir] = plqcd_g.ipt_eo[cart2lex(xcor)];
        }

        if( (xcor[idir]==-1) && (plqcd_g.nprocs[idir]==1) )
        {
           xcor[idir] =(xcor[idir]+plqcd_g.latdims[idir])%plqcd_g.latdims[idir];
           plqcd_g.idn[i][idir] = plqcd_g.ipt_eo[cart2lex(xcor)];
        }

        if( (xcor[idir]==-1) && (plqcd_g.nprocs[idir] > 1) )
        {
           xcor[idir] =(xcor[idir]+plqcd_g.latdims[idir])%plqcd_g.latdims[idir];
           jtmp = plqcd_g.ipt_eo[cart2lex(xcor)];
           if(i<plqcd_g.VOLUME/2)
           {
              plqcd_g.idn[i][idir] = cnt_bnd_odd;
              plqcd_g.nn_bnde[2*idir][cnt_bnd_odd-plqcd_g.VOLUME]=jtmp;
              cnt_bnd_odd++;
           }else{
              plqcd_g.idn[i][idir]         = cnt_bnd_even;
              plqcd_g.nn_bndo[2*idir][cnt_bnd_even-plqcd_g.VOLUME/2] = jtmp;
              cnt_bnd_even++;
           }
        }
        
        xcor[idir]=xsave;
     }

     //neighbours in the +idir direction
     cnt_bnd_even = plqcd_g.VOLUME/2;
     cnt_bnd_odd  = plqcd_g.VOLUME;
     
     for(i=0; i< plqcd_g.VOLUME; i++) //i is the even-odd index of the sites 
     {
        ilex=plqcd_g.ipt_lex[i];
        lex2cart(ilex,xcor);
  
        xsave=xcor[idir]; 
        xcor[idir] += 1;
        if( xcor[idir] < plqcd_g.latdims[idir])
        {
           plqcd_g.iup[i][idir] = plqcd_g.ipt_eo[cart2lex(xcor)];
        }

        if( (xcor[idir]== plqcd_g.latdims[idir]) && (plqcd_g.nprocs[idir]==1) )
        {
           xcor[idir] = 0;
           plqcd_g.iup[i][idir] = plqcd_g.ipt_eo[cart2lex(xcor)];
        }

        if( (xcor[idir]== plqcd_g.latdims[idir]) && (plqcd_g.nprocs[idir] > 1) )
        {
           xcor[idir] = 0;
           jtmp = plqcd_g.ipt_eo[cart2lex(xcor)];
           if(i<plqcd_g.VOLUME/2)
           {
              plqcd_g.iup[i][idir] = cnt_bnd_odd;
              plqcd_g.nn_bnde[2*idir+1][cnt_bnd_odd-plqcd_g.VOLUME]=jtmp;
              cnt_bnd_odd++;
           }else{
              plqcd_g.iup[i][idir] = cnt_bnd_even;
              plqcd_g.nn_bndo[2*idir+1][cnt_bnd_even-plqcd_g.VOLUME/2]=jtmp;
              cnt_bnd_even++;
           }
        }
        
        xcor[idir]=xsave;
     }
    
      
  } //for idir

  int L[4];
  for(i=0; i<4; i++)
     L[i]=plqcd_g.latdims[i];

  int Vyzt=L[1]*L[2]*L[3];

  plqcd_g.ipt_eo_yzt    = (int *) malloc(Vyzt*sizeof(int));
  plqcd_g.ipt_lex_yzt   = (int *) malloc(Vyzt*sizeof(int));



  //set even-odd and lexiographic indices of the lattice in the yzt block (123 directions)
  cnt_even=0;
  cnt_odd =Vyzt/2;

  for(t=0; t < L[3]; t++)
     for( z=0; z < L[2]; z++)
        for(y=0; y < L[1]; y++)
        {
            ilex = y + z*L[1] + t*L[1]*L[2];
            if( ( (y+z+t)%2) == 0 )
            {
                plqcd_g.ipt_eo_yzt[ilex]      = cnt_even;
                plqcd_g.ipt_lex_yzt[cnt_even] = ilex;
                cnt_even++;
            }
            else
            {
                plqcd_g.ipt_eo_yzt[ilex]      = cnt_odd;
                plqcd_g.ipt_lex_yzt[cnt_odd]  = ilex;
                cnt_odd++;
            }
         }



  //sse splitlayout related geometry
  //================================
  if( (plqcd_g.latdims[0]%4) != 0)
  {
      fprintf(stderr,"Error: for the case of split layout with sse, the lattice dimension in the 0 direction must be multiple of 4\n");
      exit(1);
  }

  int lx = L[0]/2;

  plqcd_g.Vsse_split = lx*L[1]*L[2]*L[3];

  plqcd_g.ipt_eo_sse_split    = (int *) malloc(plqcd_g.VOLUME*sizeof(int));
  plqcd_g.ipt_lex_sse_split   = (int *) malloc(plqcd_g.Vsse_split*sizeof(int));
  plqcd_g.iup_sse_split       = (int **) malloc(plqcd_g.Vsse_split*sizeof(int *)); 
  plqcd_g.idn_sse_split       = (int **) malloc(plqcd_g.Vsse_split*sizeof(int *)); 

  for(int i=0; i< plqcd_g.Vsse_split; i++)
  {
     plqcd_g.iup_sse_split[i] = (int *) malloc(4*sizeof(int));
     plqcd_g.idn_sse_split[i] = (int *) malloc(4*sizeof(int));
  }
  
  //building indices for the sites in the case of sse3 with splitlayout
  //even sites
  cnt_even=0;
  for(int iseo=0; iseo < Vyzt; iseo++)
  {
       int islex = plqcd_g.ipt_lex_yzt[iseo];

       int iy,iz,it,xstart,xend;
      
       iy = islex%L[1];
       iz = (islex/L[1])%L[2];
       it = (islex/L[1]/L[2])%L[3];
       if( iseo < (Vyzt/2) )
       {
         xstart=0;
         xend  =lx-1;
       }
       else
       {
         xstart=1;
         xend  =lx;
       }
         
       for(int ix=xstart; ix < xend; ix +=2)
       {
          xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
          plqcd_g.ipt_lex_sse_split[cnt_even] = cart2lex(xcor);
          plqcd_g.ipt_eo_sse_split[cart2lex(xcor)]=cnt_even;
          cnt_even++;
       }
  }

  
  //odd sites
  cnt_odd=plqcd_g.Vsse_split/2;
  for(int iseo=0; iseo < Vyzt; iseo++)
  {
       int islex = plqcd_g.ipt_lex_yzt[iseo];

       int iy,iz,it,xstart,xend;
      
       iy = islex%L[1];
       iz = (islex/L[1]) %L[2];
       it = (islex/L[1]/L[2])%L[3];
       if( iseo < (Vyzt/2) )
       {
         xstart=1;
         xend  =lx;
       }
       else
       {
         xstart=0;
         xend  =lx-1;
       }
         
       for(int ix=xstart; ix < xend; ix +=2)
       {
          xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
          plqcd_g.ipt_lex_sse_split[cnt_even] = cart2lex(xcor);
          plqcd_g.ipt_eo_sse_split[cart2lex(xcor)]=cnt_odd;
          cnt_odd++;
       }
  }


  //nearest neighbours
  //even sites
  cnt_even=0;
  for(int iseo=0; iseo < Vyzt; iseo++)
  {
      int islex = plqcd_g.ipt_lex_yzt[iseo];
      int iy,iz,it,xstart,xend;
      iy = islex%L[1];
      iz = (islex/L[1]) %L[2];
      it = (islex/L[1]/L[2])%L[3];
      if( iseo < (L[1]*L[2]*L[3]/2) )
      {
         xstart=0;
         xend  =lx-1;
      }
      else
      {
         xstart=1;
        xend   =lx;
      }

      for(int ix=xstart; ix < xend; ix +=2)
      {
        xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
           
        for(mu=0; mu < 4; mu++)
        {
           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
              xshift[mu] = (xshift[mu] + 1 ) % lx;
           else
              xshift[mu] = (xshift[mu] + 1 ) % L[mu];

           plqcd_g.iup_sse_split[cnt_even][mu] = plqcd_g.ipt_eo_sse_split[cart2lex(xshift)];

           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
               xshift[mu] = (xshift[mu] -1 +lx) % lx;
           else
               xshift[mu] = (xshift[mu] -1 +L[mu]) % L[mu];

           plqcd_g.idn_sse_split[cnt_even][mu] = plqcd_g.ipt_eo_sse_split[cart2lex(xshift)];

        }
        
        cnt_even++;
      }
   }


   //odd sites
   cnt_odd=plqcd_g.Vsse_split/2;
   for(int iseo=0; iseo < Vyzt; iseo++)
   {
      int islex = plqcd_g.ipt_lex_yzt[iseo];
      int iy,iz,it,xstart,xend;
      iy = islex%L[1];
      iz = (islex/L[1]) %L[2];
      it = (islex/L[1]/L[2])%L[3];
      if( iseo < (Vyzt/2) )
      {
         xstart=1;
         xend  =lx;
      }
      else
      {
         xstart=0;
        xend   =lx-1;
      }

     for(int ix=xstart; ix < xend; ix +=2)
     {
        xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
           
        for(mu=0; mu < 4; mu++)
        {
           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
              xshift[mu] = (xshift[mu] + 1 ) % lx;
           else
              xshift[mu] = (xshift[mu] + 1 ) % L[mu];

           plqcd_g.iup_sse_split[cnt_odd][mu] = plqcd_g.ipt_eo_sse_split[cart2lex(xshift)];

           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
               xshift[mu] = (xshift[mu] -1 +lx) % lx;
           else
               xshift[mu] = (xshift[mu] -1 +L[mu]) % L[mu];

           plqcd_g.idn_sse_split[cnt_odd][mu] = plqcd_g.ipt_eo_sse_split[cart2lex(xshift)];

        }
        
        cnt_odd++;
      }
   }





  //avx splitlayout related geometry
  //================================
  #ifdef AVX_SPLIT
  if( (plqcd_g.latdims[0]%8) != 0)
  {
      fprintf(stderr,"Error: for the case of split layout with avx, the lattice dimension in the 0 direction must be multiple of 8\n");
      exit(1);
  }

  lx = L[0]/4;

  plqcd_g.Vavx_split = lx*L[1]*L[2]*L[3];


  plqcd_g.ipt_eo_avx_split    = (int *) malloc(plqcd_g.VOLUME*sizeof(int));
  plqcd_g.ipt_lex_avx_split   = (int *) malloc(plqcd_g.Vavx_split*sizeof(int));
  plqcd_g.iup_avx_split = (int **) malloc(plqcd_g.Vavx_split*sizeof(int *)); 
  plqcd_g.idn_avx_split = (int **) malloc(plqcd_g.Vavx_split*sizeof(int *)); 
  for(int i=0; i< plqcd_g.Vavx_split; i++)
  {
     plqcd_g.iup_avx_split[i] = (int *) malloc(4*sizeof(int));
     plqcd_g.idn_avx_split[i] = (int *) malloc(4*sizeof(int));
  }
  
   //building indices for the sites in the case of AVX with splitlayout
   //even sites
   cnt_even=0;
   for(int iseo=0; iseo < Vyzt; iseo++)
   {
       int islex = plqcd_g.ipt_lex_yzt[iseo];

       int iy,iz,it,xstart,xend;
      
       iy = islex%L[1];
       iz = (islex/L[1])%L[2];
       it = (islex/L[1]/L[2])%L[3];
       if( iseo < (Vyzt/2) )
       {
         xstart=0;
         xend  =lx-1;
       }
       else
       {
         xstart=1;
         xend  =lx;
       }
         
       for(int ix=xstart; ix < xend; ix +=2)
       {
          xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
          plqcd_g.ipt_lex_avx_split[cnt_even] = cart2lex(xcor);
          plqcd_g.ipt_eo_avx_split[cart2lex(xcor)]=cnt_even;
          cnt_even++;
       }
   }

  
   //odd sites
   cnt_odd=plqcd_g.Vavx_split/2;
   for(int iseo=0; iseo < Vyzt; iseo++)
   {
       int islex = plqcd_g.ipt_lex_yzt[iseo];

       int iy,iz,it,xstart,xend;
      
       iy = islex%L[1];
       iz = (islex/L[1]) %L[2];
       it = (islex/L[1]/L[2])%L[3];
       if( iseo < (Vyzt/2) )
       {
         xstart=1;
         xend  =lx;
       }
       else
       {
         xstart=0;
         xend  =lx-1;
       }
         
       for(int ix=xstart; ix < xend; ix +=2)
       {
          xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
          plqcd_g.ipt_lex_avx_split[cnt_even] = cart2lex(xcor);
          plqcd_g.ipt_eo_avx_split[cart2lex(xcor)]=cnt_odd;
          cnt_odd++;
       }
   }


   //nearest neighbours
   //even sites
   cnt_even=0;
   for(int iseo=0; iseo < Vyzt; iseo++)
   {
      int islex = plqcd_g.ipt_lex_yzt[iseo];
      int iy,iz,it,xstart,xend;
      iy = islex%L[1];
      iz = (islex/L[1]) %L[2];
      it = (islex/L[1]/L[2])%L[3];
      if( iseo < (L[1]*L[2]*L[3]/2) )
      {
         xstart=0;
         xend  =lx-1;
      }
      else
      {
         xstart=1;
        xend   =lx;
      }

      for(int ix=xstart; ix < xend; ix +=2)
      {
        xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
           
        for(mu=0; mu < 4; mu++)
        {
           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
              xshift[mu] = (xshift[mu] + 1 ) % lx;
           else
              xshift[mu] = (xshift[mu] + 1 ) % L[mu];

           plqcd_g.iup_avx_split[cnt_even][mu] = plqcd_g.ipt_eo_avx_split[cart2lex(xshift)];

           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
               xshift[mu] = (xshift[mu] -1 +lx) % lx;
           else
               xshift[mu] = (xshift[mu] -1 +L[mu]) % L[mu];

           plqcd_g.idn_avx_split[cnt_even][mu] = plqcd_g.ipt_eo_avx_split[cart2lex(xshift)];

        }
        
        cnt_even++;
      }
   }


   //odd sites
   cnt_odd=plqcd_g.Vavx_split/2;
   for(int iseo=0; iseo < Vyzt; iseo++)
   {
      int islex = plqcd_g.ipt_lex_yzt[iseo];
      int iy,iz,it,xstart,xend;
      iy = islex%L[1];
      iz = (islex/L[1]) %L[2];
      it = (islex/L[1]/L[2])%L[3];
      if( iseo < (Vyzt/2) )
      {
         xstart=1;
         xend  =lx;
      }
      else
      {
         xstart=0;
        xend   =lx-1;
      }

     for(int ix=xstart; ix < xend; ix +=2)
     {
        xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
           
        for(mu=0; mu < 4; mu++)
        {
           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
              xshift[mu] = (xshift[mu] + 1 ) % lx;
           else
              xshift[mu] = (xshift[mu] + 1 ) % L[mu];

           plqcd_g.iup_avx_split[cnt_odd][mu] = plqcd_g.ipt_eo_avx_split[cart2lex(xshift)];

           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
               xshift[mu] = (xshift[mu] -1 +lx) % lx;
           else
               xshift[mu] = (xshift[mu] -1 +L[mu]) % L[mu];

           plqcd_g.idn_avx_split[cnt_odd][mu] = plqcd_g.ipt_eo_avx_split[cart2lex(xshift)];

        }
        
        cnt_odd++;
      }
   }
   #endif //AVX_SPLIT



  //MIC splitlayout related geometry
  //================================

  #ifdef MIC
  #ifdef MIC_SPLIT
  if( (plqcd_g.latdims[0]%16) != 0)
  {
      fprintf(stderr,"Error: for the case of split layout with mic, the lattice dimension in the 0 direction must be multiple of 16\n");
      exit(1);
  }

  lx = L[0]/8;

  plqcd_g.Vmic_split = lx*L[1]*L[2]*L[3];


  plqcd_g.ipt_eo_mic_split    = (int *) malloc(plqcd_g.VOLUME*sizeof(int));
  plqcd_g.ipt_lex_mic_split   = (int *) malloc(plqcd_g.Vmic_split*sizeof(int));
  plqcd_g.iup_mic_split = (int **) malloc(plqcd_g.Vmic_split*sizeof(int *)); 
  plqcd_g.idn_mic_split = (int **) malloc(plqcd_g.Vmic_split*sizeof(int *)); 
  for(int i=0; i< plqcd_g.Vmic_split; i++)
  {
     plqcd_g.iup_mic_split[i] = (int *) malloc(4*sizeof(int));
     plqcd_g.idn_mic_split[i] = (int *) malloc(4*sizeof(int));
  }
  
   //building indices for the sites in the case of MIC with splitlayout
   //even sites
   cnt_even=0;
   for(int iseo=0; iseo < Vyzt; iseo++)
   {
       int islex = plqcd_g.ipt_lex_yzt[iseo];

       int iy,iz,it,xstart,xend;
      
       iy = islex%L[1];
       iz = (islex/L[1])%L[2];
       it = (islex/L[1]/L[2])%L[3];
       if( iseo < (Vyzt/2) )
       {
         xstart=0;
         xend  =lx-1;
       }
       else
       {
         xstart=1;
         xend  =lx;
       }
         
       for(int ix=xstart; ix < xend; ix +=2)
       {
          xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
          plqcd_g.ipt_lex_mic_split[cnt_even] = cart2lex(xcor);
          plqcd_g.ipt_eo_mic_split[cart2lex(xcor)]=cnt_even;
          cnt_even++;
       }
   }

  
   //odd sites
   cnt_odd=plqcd_g.Vmic_split/2;
   for(int iseo=0; iseo < Vyzt; iseo++)
   {
       int islex = plqcd_g.ipt_lex_yzt[iseo];

       int iy,iz,it,xstart,xend;
      
       iy = islex%L[1];
       iz = (islex/L[1]) %L[2];
       it = (islex/L[1]/L[2])%L[3];
       if( iseo < (Vyzt/2) )
       {
         xstart=1;
         xend  =lx;
       }
       else
       {
         xstart=0;
         xend  =lx-1;
       }
         
       for(int ix=xstart; ix < xend; ix +=2)
       {
          xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
          plqcd_g.ipt_lex_mic_split[cnt_even] = cart2lex(xcor);
          plqcd_g.ipt_eo_mic_split[cart2lex(xcor)]=cnt_odd;
          cnt_odd++;
       }
   }


   //nearest neighbours
   //even sites
   cnt_even=0;
   for(int iseo=0; iseo < Vyzt; iseo++)
   {
      int islex = plqcd_g.ipt_lex_yzt[iseo];
      int iy,iz,it,xstart,xend;
      iy = islex%L[1];
      iz = (islex/L[1]) %L[2];
      it = (islex/L[1]/L[2])%L[3];
      if( iseo < (L[1]*L[2]*L[3]/2) )
      {
         xstart=0;
         xend  =lx-1;
      }
      else
      {
         xstart=1;
        xend   =lx;
      }

      for(int ix=xstart; ix < xend; ix +=2)
      {
        xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
           
        for(mu=0; mu < 4; mu++)
        {
           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
              xshift[mu] = (xshift[mu] + 1 ) % lx;
           else
              xshift[mu] = (xshift[mu] + 1 ) % L[mu];

           plqcd_g.iup_mic_split[cnt_even][mu] = plqcd_g.ipt_eo_mic_split[cart2lex(xshift)];

           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
               xshift[mu] = (xshift[mu] -1 +lx) % lx;
           else
               xshift[mu] = (xshift[mu] -1 +L[mu]) % L[mu];

           plqcd_g.idn_mic_split[cnt_even][mu] = plqcd_g.ipt_eo_mic_split[cart2lex(xshift)];

        }
        
        cnt_even++;
      }
   }


   //odd sites
   cnt_odd=plqcd_g.Vmic_split/2;
   for(int iseo=0; iseo < Vyzt; iseo++)
   {
      int islex = plqcd_g.ipt_lex_yzt[iseo];
      int iy,iz,it,xstart,xend;
      iy = islex%L[1];
      iz = (islex/L[1]) %L[2];
      it = (islex/L[1]/L[2])%L[3];
      if( iseo < (Vyzt/2) )
      {
         xstart=1;
         xend  =lx;
      }
      else
      {
         xstart=0;
        xend   =lx-1;
      }

     for(int ix=xstart; ix < xend; ix +=2)
     {
        xcor[0]=ix; xcor[1]=iy; xcor[2]=iz; xcor[3]=it;
           
        for(mu=0; mu < 4; mu++)
        {
           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
              xshift[mu] = (xshift[mu] + 1 ) % lx;
           else
              xshift[mu] = (xshift[mu] + 1 ) % L[mu];

           plqcd_g.iup_mic_split[cnt_odd][mu] = plqcd_g.ipt_eo_mic_split[cart2lex(xshift)];

           for(int nu=0; nu<4; nu++)
              xshift[nu] = xcor[nu];

           if(mu==0)
               xshift[mu] = (xshift[mu] -1 +lx) % lx;
           else
               xshift[mu] = (xshift[mu] -1 +L[mu]) % L[mu];

           plqcd_g.idn_mic_split[cnt_odd][mu] = plqcd_g.ipt_eo_mic_split[cart2lex(xshift)];

        }
        
        cnt_odd++;
      }
   }
   #endif
   #endif //MIC & MIC_SPLIT



   return;
}


/**************************************************
 *single index for the process with co-ordinates 
 *n[0],n[1],n[2],n[3] on the grid of processes 
 **************************************************/
int ipr(int n[])
{
   int i,p[4],ip;
   //check the input values
   for(i=0; i<4; i++)
   {
       p[i] = plqcd_g.nprocs[i];
       if( (n[i] < 0) || (n[i] >= p[i]) ){   //out of allowed range
         printf("ERROR: co-ordinates of the process is out of the allowed range i %d n[i] %d\n",i,n[i]);
         exit(1);
       }
   } 
   ip = n[0] + p[0]*n[1] + p[0]*p[1]*n[2] + p[0]*p[1]*p[2]*n[3];
   return ip;
}


/***************************************************************
 *Computes Cartesian coordiantes n[] of a process on the 
 *grid of processes given its id value ip where ip=0,..,NPROCS-1 
 **************************************************************/
void comp_cpr(int ip, int n[])
{
    int i, p[4], NPROCS;

    for(i=0; i<4; i++)
       p[i] = plqcd_g.nprocs[i];

    NPROCS=p[0]*p[1]*p[2]*p[3];
    
    if((ip>=0) && (ip < NPROCS))
    {
        n[0] = ip%p[0];
        n[1] = (ip/p[0])%p[1];
        n[2] = (ip/p[0]/p[1])%p[2];
        n[3] = (ip/p[0]/p[1]/p[2])%p[3];
    }
    else
    {
      printf("Error in comp_cpr, the process id must be  0 =< ip < %d.\n",NPROCS);
    }
   
    return;
}




/***********************************************
 *Index of the nearest neghibour process in the 
 *-ve and +ve 4 directions.
 *
 * npr[2*mu] is the index of the neighbour process
 * in the -mu direction.
 * 
 * npr[2*mu+1] is the index of the neighbour process
 * in the +mu direction.
 ***********************************************/
void set_npr(void)
{
   int mu,n[4];

   for(mu=0; mu < 4; mu++)
     n[mu]=plqcd_g.cpr[mu];


   for(mu=0;mu<4;mu++)
   {
      n[mu] -= 1;
      if (n[mu] == -1)
         n[mu] += plqcd_g.nprocs[mu];  //periodic
      plqcd_g.npr[2*mu]=ipr(n);
      n[mu] = plqcd_g.cpr[mu];

      n[mu] += 1;
      n[mu] = n[mu]%plqcd_g.nprocs[mu]; //periodic
      plqcd_g.npr[2*mu+1]=ipr(n);
      n[mu] = plqcd_g.cpr[mu];
   }
}


/*********************************************
 *given the lexiographic index of the site
 *it computes the cartesian coordinates of the 
 *site w.r.t. the local lattice.
 ********************************************/
void lex2cart(int ix, int xcart[])
{
    if((ix>=0) && (ix < plqcd_g.VOLUME))
    {
        xcart[0] = ix%plqcd_g.latdims[0];
        xcart[1] = (ix/plqcd_g.latdims[0])%plqcd_g.latdims[1];
        xcart[2] = (ix/plqcd_g.latdims[0]/plqcd_g.latdims[1])% plqcd_g.latdims[2];
        xcart[3] = (ix/plqcd_g.latdims[0]/plqcd_g.latdims[1]/plqcd_g.latdims[2]) % plqcd_g.latdims[3];
    }
    else
    {
      printf("Error lex2cart expects the lexiographic index to be  0 =< ix < VOLUME.\n");
      exit(1);
    }
}



/*******************************************************
 *compute the lexiographic index of a site given the 
 *coordinates w.r.t. the local lattice.
 ******************************************************/
int cart2lex(int xcor[])
{
    int ix,i, L[4];
    
    //check the input
    for(i=0; i<4; i++)
    {
       if( (xcor[i] < 0) || (xcor[i] >= plqcd_g.latdims[i]) )
       {
          printf("Error in converting cartesian coordinates of a site to lexiographic index:coordiante in the %d th direction is out of bounds\n",i);
          exit(1);
       } 
    }

    for(i=0; i<4; i++)
       L[i] = plqcd_g.latdims[i];  //just to save in typing

    ix = xcor[0] + L[0]*xcor[1] + L[0]*L[1]*xcor[2] + L[0]*L[1]*L[2]*xcor[3];

    return ix;
}




/*given global coordinates of a site, compute a single index for 
  the process id that contain that site and the even-odd index of that
  site on that process.
 */

void find_site(int xg[], int *pid, int *xeo)
{
    int xcor[4],proc[4];

    int i, mu,ilex,ieo,myproc;

    //check the input
    for(i=0; i<4; i++)
    {
       if( (xg[i] < 0) || (xg[i] >= plqcd_g.nprocs[i]*plqcd_g.latdims[i]) )
       {
          printf("Error, global site coordinates are out of bounds in the %d -th direction\n",i);
          exit(1);
       } 
    }


    for(mu=0; mu<4; mu++)
    {
        proc[mu] = xg[mu]/plqcd_g.latdims[mu];
        
        xcor[mu] = xg[mu] - proc[mu]*plqcd_g.latdims[mu];
    }

    ilex = cart2lex(xcor);

    ieo  = plqcd_g.ipt_eo[ilex];

    myproc=ipr(proc);

    (*pid) = myproc;

    (*xeo) = ieo;

    return ;
} 






//print global parametrs stored in the plqcd_g struct to a file
void print_params(FILE* fp)
{

   int x[4];
   int i,ilex;
   int ix;
   int idir;



   //print the parameters related to parallelization
   fprintf(fp,"grid of processes: nprocs0 %d nprocs1 %d nprocs2 %d nprocs3 %d\n\n",
              plqcd_g.nprocs[0],plqcd_g.nprocs[1],plqcd_g.nprocs[2],plqcd_g.nprocs[3]);

   fprintf(fp,"Number of openmp threads %d\n\n",plqcd_g.nthread);
   
   fprintf(fp,"Cartesian co-ordinates of my process on the grid of processes: %d %d %d %d\n\n",
               plqcd_g.cpr[0], plqcd_g.cpr[1],plqcd_g.cpr[2],plqcd_g.cpr[3]);

   fprintf(fp,"Nearst neighbours of my process (-0 +0 -1 +1 -2 +2 -3 +3): %10d %10d %10d %10d %10d %10d %10d %10d \n\n", 
              plqcd_g.npr[0],plqcd_g.npr[1],plqcd_g.npr[2],plqcd_g.npr[3],plqcd_g.npr[4],plqcd_g.npr[5],plqcd_g.npr[6],plqcd_g.npr[7]);

   //parameters related to geometry
   fprintf(fp,"Global lattice size: LX %d LY %d LZ %d LT %d\n\n",plqcd_g.LX,plqcd_g.LY,plqcd_g.LZ,plqcd_g.LT);

   fprintf(fp,"Local lattice size: lx %d ly %d lz %d lt %d, local volume %d\n\n", 
               plqcd_g.latdims[0],plqcd_g.latdims[1],plqcd_g.latdims[2],plqcd_g.latdims[3],plqcd_g.VOLUME);

   fprintf(fp,"Sizes of the boundaries: 0-boundary %d 1-boundary %d 2-boundary %d 3-boundary %d\n\n",
               plqcd_g.face[0],plqcd_g.face[1],plqcd_g.face[2],plqcd_g.face[3]);

   fprintf(fp,"printing the lattice sites w.r.t. the local lattice:\n\n");

   fprintf(fp,"%8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n\n",
              "x0","x1","x2","x3","ipt_eo","ipt_lex","iup[0]","iup[1]","iup[2]","iup[3]","idn[0]","idn[1]","idn[2]","idn[3]");
   
   for(ix=0; ix < plqcd_g.VOLUME; ix++)   //even-odd index
   {

      ilex=plqcd_g.ipt_lex[ix];
      lex2cart(ilex,x);
      fprintf(fp,"%8d %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d\n\n",
                 x[0],x[1],x[2],x[3],ix,ilex,
                 plqcd_g.iup[ix][0],plqcd_g.iup[ix][1],plqcd_g.iup[ix][2],plqcd_g.iup[ix][3],
                 plqcd_g.idn[ix][0],plqcd_g.idn[ix][1],plqcd_g.idn[ix][2],plqcd_g.idn[ix][3]);
   }

   //printing the nn_bnd arrays
   fprintf(fp,"the nn_bnde and nn_bndo arrays\n");
   for(idir=0; idir<4; idir++)
   {
      fprintf(fp,"nn_bnde[%d] \n",2*idir);
      for(i=0; i < plqcd_g.face[idir]/2; i++){
         lex2cart(plqcd_g.ipt_lex[plqcd_g.nn_bnde[2*idir][i]],x);
         fprintf(fp,"%d %d %d %d %d %d\n",i,plqcd_g.nn_bnde[2*idir][i],x[0],x[1],x[2],x[3]);}

      fprintf(fp,"nn_bndo[%d]\n",2*idir);
      for(i=0; i < plqcd_g.face[idir]/2; i++){
         lex2cart(plqcd_g.ipt_lex[plqcd_g.nn_bndo[2*idir][i]],x);
         fprintf(fp,"%d %d %d %d %d %d\n",i,plqcd_g.nn_bndo[2*idir][i],x[0],x[1],x[2],x[3]);}

      fprintf(fp,"nn_bnde[%d] \n",2*idir+1);
      for(i=0; i < plqcd_g.face[idir]/2; i++){
         lex2cart(plqcd_g.ipt_lex[plqcd_g.nn_bnde[2*idir+1][i]],x);
         fprintf(fp,"%d %d %d %d %d %d\n",i,plqcd_g.nn_bnde[2*idir+1][i],x[0],x[1],x[2],x[3]);}

      fprintf(fp,"nn_bndo[%d]\n",2*idir+1);
      for(i=0; i < plqcd_g.face[idir]/2; i++){
         lex2cart(plqcd_g.ipt_lex[plqcd_g.nn_bndo[2*idir+1][i]],x);
         fprintf(fp,"%d %d %d %d %d %d\n",i,plqcd_g.nn_bndo[2*idir+1][i],x[0],x[1],x[2],x[3]);}
   }


   fprintf(fp,"memory alignment in bytes: ALIGN %d\n\n",plqcd_g.ALIGN);
   fprintf(fp,"initialization check parameter: init %d\n\n",plqcd_g.init);
     
   return;
}
 

