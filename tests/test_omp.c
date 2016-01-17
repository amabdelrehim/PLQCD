/*************************************
 *some elementary openmp tests
 ************************************/

#include<stdlib.h>
#include<stdio.h>
#ifdef _OPENMP
#include<omp.h>
#endif
#include"utils.h"
#include"su3.h"
#include"sse.h"
//#include"sse3.h"

int main(){

  int N=4000000;
  int i,j;
  spinor *pin= (spinor *) amalloc(N*sizeof(spinor), 4);
  if(pin==NULL)
  {
      fprintf(stderr,"ERROR: insufficient memory for spinor pin.\n");
      exit(2);
  }

  spinor *pout= (spinor *) amalloc(N*sizeof(spinor), 4);
  if(pout==NULL)
  {
       fprintf(stderr,"ERROR: insufficient memory for spinor pout.\n");
       exit(2);
  }

  su3 *ufield= (su3 *) amalloc(N*sizeof(su3), 4);
  if(ufield==NULL)
  {
       fprintf(stderr,"ERROR: insufficient memory for gauge field ufield.\n");
       exit(2);
  }

  double rs[24],ru[18];
  for(i=0; i<N; i++)
  {
       for(j=0; j<24; j++)
          rs[j]= rand() / (double)RAND_MAX;

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

       for(j=0; j<18; j++)
          ru[j]= rand() / (double)RAND_MAX;

       ufield[i].c00=ru[0]+I*ru[1];
       ufield[i].c01=ru[2]+I*ru[3];
       ufield[i].c02=ru[4]+I*ru[5];
       ufield[i].c10=ru[6]+I*ru[7];
       ufield[i].c11=ru[8]+I*ru[9];
       ufield[i].c12=ru[10]+I*ru[11];
       ufield[i].c20=ru[12]+I*ru[13];
       ufield[i].c21=ru[14]+I*ru[15];
       ufield[i].c22=ru[16]+I*ru[17];
   }



  double ts,tf;

  ts=stop_watch(0.0);

  #ifdef _OPENMP
  #pragma omp parallel 
  {
  #endif
    int k;
    #ifdef _OPENMP
    printf("openmp threads %d\n",omp_get_num_threads());
    #endif
    #ifdef _OPENMP
    #pragma omp for
    #endif
    for(k=0; k<N; k++)
    {
       _su3_multiply(pout[k].s0,ufield[k],pin[k].s0);
       _su3_multiply(pout[k].s1,ufield[k],pin[k].s1);
       _su3_multiply(pout[k].s2,ufield[k],pin[k].s2);
       _su3_multiply(pout[k].s3,ufield[k],pin[k].s3);
    }
  #ifdef _OPENMP
  }
  #endif


  tf=stop_watch(ts);

  //for(j=0; j< N; j++){
  //   sum = sum + d3[j];}

  printf("time= %f\n",tf);

  return 0;

}

