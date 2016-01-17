#ifndef _UTILS_H
#define _UTILS_H 1

#include <limits.h>
#include <float.h>




//allocat a block of aligned memory
void *alloc(size_t size, size_t alignment);

/*
*   void *amalloc(size_t size,int p)
*     Allocates an aligned memory area of "size" bytes, with a starting
*     address (the return value) that is an integer multiple of 2^p. A
*     NULL pointer is returned if the allocation was not successful
*
*   void afree(void *addr)
*     Frees the aligned memory area at address "addr" that was previously
*     allocated using amalloc. If the memory space at this address was
*     already freed using afree, or if the address does not match an
*     address previously returned by amalloc, the program does not do
*     anything
*/

/* Using this allocation function gives a seg-fault when used with avx
   This was not the case with alloc function by Giannis
   One can also use __attribute__ ((aligned(n)))
   We may just use this explict allignment if we get problems
*/
void *amalloc(size_t size,int p);
void afree(void *addr);



int mpi_permanent_tag(void);
int mpi_tag(void);






//return time in micro seconds relative to an input time value
double stop_watch(double);

#endif /* _UTILS_H */

