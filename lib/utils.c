#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "utils.h"

#define MAX_TAG 32767
#define MAX_PERMANENT_TAG MAX_TAG/2

static int pcmn_cnt=-1,cmn_cnt=MAX_TAG;

struct addr_t
{
   char *addr;
   char *true_addr;
   struct addr_t *next;
};

static struct addr_t *first=NULL;

/*
 * Allocate an aligned chunk of bytes and check if NULL returned
 */
void * alloc(size_t size, size_t alignment)
{
  void *ptr;
  int k=posix_memalign(&ptr, alignment, size);

  if(ptr == NULL || (k!=0)) {
    fprintf(stderr, " alloc() returned NULL. Out of memory?\n");
  }

  return ptr;
}

void *amalloc(size_t size,int p)
{
   int shift;
   char *true_addr,*addr;
   unsigned long mask;
   struct addr_t *new;

   if ((size<=0)||(p<0))
      return(NULL);

   shift=1<<p;
   mask=(unsigned long)(shift-1);

   true_addr=malloc(size+shift);
   new=malloc(sizeof(*first));

   if ((true_addr==NULL)||(new==NULL))
   {
      free(true_addr);
      free(new);
      return(NULL);
   }

   addr=(char*)(((unsigned long)(true_addr+shift))&(~mask));
   (*new).addr=addr;
   (*new).true_addr=true_addr;
   (*new).next=first;
   first=new;

   return (void*)(addr);
}


void afree(void *addr)
{
   struct addr_t *p,*q;

   q=NULL;

   for (p=first;p!=NULL;p=(*p).next)
   {
      if ((*p).addr==addr)
      {
         if (q!=NULL)
            (*q).next=(*p).next;
         else
            first=(*p).next;

         free((*p).true_addr);
         free(p);
         return;
      }

      q=p;
   }
}



int mpi_permanent_tag(void)
{
   if (pcmn_cnt<MAX_PERMANENT_TAG)
      pcmn_cnt+=1;
   else{
      fprintf(stderr,"mpi_permanent_tag [utils.c] Requested more than 16384 tags");
      exit(1);}

   return pcmn_cnt;
}


int mpi_tag(void)
{
   if (cmn_cnt==MAX_TAG)
      cmn_cnt=MAX_PERMANENT_TAG;

   cmn_cnt+=1;   

   return cmn_cnt;
}


/*
 * Returns the time in seconds since the argument "t"
 */
double stop_watch(double t)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec+ tv.tv_usec*0.000001) - t;
}

