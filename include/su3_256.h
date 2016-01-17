/*********************************************************************************
 * Copyright (C) 2012 Abdou Abdel-Rehim, Giannis Koutsou, Nikos Anastopolous
 * This file is part of the PLQCD library
 * 
 * function declarations and definitions for 256 related structs
 *********************************************************************************/

#ifndef _SU3_256_H
#define _SU3_256_H
#include "su3.h"
#include<immintrin.h>
#include<zmmintrin.h>

typedef struct {
    _Complex double lo,hi;
} complex_pair; 


typedef struct 
{
   complex_pair c00,c01,c02,c10,c11,c12,c20,c21,c22;
} su3_256;


typedef struct 
{
   complex_pair c00,c01,c02,c10,c11,c12;
} su3_compact_256;


typedef struct
{
   complex_pair c0,c1,c2;
} su3_vector_256;


typedef struct
{
   su3_vector_256 s0,s1,s2,s3;
} spinor_256;


typedef struct
{
  su3_vector_256 s0,s1;
} halfspinor_256;


//utility functions to copy data to and from 128 data types from/to 256 data types
static inline void copy_su3_to_su3_256(su3_256 *a, su3 *b1, su3 *b2)
{
    (a->c00).lo = (b1->c00);
    (a->c00).hi = (b2->c00);
    (a->c01).lo = (b1->c01);
    (a->c01).hi = (b2->c01);
    (a->c02).lo = (b1->c02);
    (a->c02).hi = (b2->c02);
    (a->c10).lo = (b1->c10);
    (a->c10).hi = (b2->c10);
    (a->c11).lo = (b1->c11);
    (a->c11).hi = (b2->c11);
    (a->c12).lo = (b1->c12);
    (a->c12).hi = (b2->c12);
    (a->c20).lo = (b1->c20);
    (a->c20).hi = (b2->c20);
    (a->c21).lo = (b1->c21);
    (a->c21).hi = (b2->c21);
    (a->c22).lo = (b1->c22);
    (a->c22).hi = (b2->c22);

} 

static inline void  copy_su3_256_to_su3(su3 *b1, su3 *b2, su3_256 *a)
{
    b1->c00 = (a->c00).lo;
    b2->c00 = (a->c00).hi;
    b1->c01 = (a->c01).lo;
    b2->c01 = (a->c01).hi;
    b1->c02 = (a->c02).lo;
    b2->c02 = (a->c02).hi;


    b1->c10 = (a->c10).lo;
    b2->c10 = (a->c10).hi;
    b1->c11 = (a->c11).lo;
    b2->c11 = (a->c11).hi;
    b1->c12 = (a->c12).lo;
    b2->c12 = (a->c12).hi;

    b1->c20 = (a->c20).lo;
    b2->c20 = (a->c20).hi;
    b1->c21 = (a->c21).lo;
    b2->c21 = (a->c21).hi;
    b1->c22 = (a->c22).lo;
    b2->c22 = (a->c22).hi;
}

static inline void copy_su3_vector_to_su3_vector_256(su3_vector_256 *a, su3_vector *b1, su3_vector *b2)
{
   /*
   (a->c0).lo = b1->c0;
   (a->c0).hi = b2->c0;

   (a->c1).lo = b1->c1;
   (a->c1).hi = b2->c1;

   (a->c2).lo = b1->c2;
   (a->c2).hi = b2->c2;
   */

   __m128d t1;

   t1 = _mm_load_pd((double *) &(b1->c0));
   _mm_store_pd((double *) &((a->c0).lo), t1);
   t1 = _mm_load_pd((double *) &(b2->c0));
   _mm_store_pd((double *) &((a->c0).hi), t1);
  
   t1 = _mm_load_pd((double *) &(b1->c1));
   _mm_store_pd((double *) &((a->c1).lo), t1);
   t1 = _mm_load_pd((double *) &(b2->c1));
   _mm_store_pd((double *) &((a->c1).hi), t1);

   t1 = _mm_load_pd((double *) &(b1->c2));
   _mm_store_pd((double *) &((a->c2).lo), t1);
   t1 = _mm_load_pd((double *) &(b2->c2));
   _mm_store_pd((double *) &((a->c2).hi), t1);

}



static inline void copy_su3_vector_256_to_su3_vector(su3_vector *b1, su3_vector *b2, su3_vector_256 *a)
{
   /*
   b1->c0 = (a->c0).lo;
   b2->c0 = (a->c0).hi;

   b1->c1 = (a->c1).lo;
   b2->c1 = (a->c1).hi;

   b1->c2 = (a->c2).lo;
   b2->c2 = (a->c2).hi;
   */


   __m128d t1;

   t1 = _mm_load_pd((double *) &((a->c0).lo));
   _mm_store_pd((double *) &(b1->c0), t1);
   t1 = _mm_load_pd((double *) &((a->c0).hi));
   _mm_store_pd((double *) &(b2->c0), t1);

   t1 = _mm_load_pd((double *) &((a->c1).lo));
   _mm_store_pd((double *) &(b1->c1), t1);
   t1 = _mm_load_pd((double *) &((a->c1).hi));
   _mm_store_pd((double *) &(b2->c1), t1);

   t1 = _mm_load_pd((double *) &((a->c2).lo));
   _mm_store_pd((double *) &(b1->c2), t1);
   t1 = _mm_load_pd((double *) &((a->c2).hi));
   _mm_store_pd((double *) &(b2->c2), t1);

}

static inline void copy_spinor_to_spinor_256(spinor_256 *a, spinor *b1, spinor *b2)
{
   copy_su3_vector_to_su3_vector_256(&(a->s0), &(b1->s0), &(b2->s0));
   copy_su3_vector_to_su3_vector_256(&(a->s1), &(b1->s1), &(b2->s1));
   copy_su3_vector_to_su3_vector_256(&(a->s2), &(b1->s2), &(b2->s2));
   copy_su3_vector_to_su3_vector_256(&(a->s3), &(b1->s3), &(b2->s3));
}

static inline void copy_spinor_256_to_spinor(spinor *b1, spinor *b2, spinor_256 *a)
{

   copy_su3_vector_256_to_su3_vector(&(b1->s0), &(b2->s0), &(a->s0));
   copy_su3_vector_256_to_su3_vector(&(b1->s1), &(b2->s0), &(a->s1));
   copy_su3_vector_256_to_su3_vector(&(b1->s2), &(b2->s0), &(a->s2));
   copy_su3_vector_256_to_su3_vector(&(b1->s3), &(b2->s0), &(a->s3));

}

static inline void copy_halfspinor_to_halfspinor_256(halfspinor_256 *a, halfspinor *b1, halfspinor *b2)
{

   copy_su3_vector_to_su3_vector_256(&(a->s0), &(b1->s0), &(b2->s0));
   copy_su3_vector_to_su3_vector_256(&(a->s1), &(b1->s1), &(b2->s1));
}


static inline void copy_halfspinor_256_to_halfspinor(halfspinor *b1, halfspinor *b2, halfspinor_256 *a)
{

   copy_su3_vector_256_to_su3_vector(&(b1->s0), &(b2->s0), &(a->s0));
   copy_su3_vector_256_to_su3_vector(&(b1->s1), &(b2->s0), &(a->s1));
}  

/*
void copy_su3_to_su3_256(su3_256 *a, su3 *b1, su3 *b2);
void copy_su3_256_to_su3(su3 *b1, su3 *b2, su3_256 *a);
void copy_su3_vector_to_su3_vector_256(su3_vector_256 *a, su3_vector *b1, su3_vector *b2);
void copy_su3_vector_256_to_su3_vector(su3_vector *b1, su3_vector *b2, su3_vector_256 *a);
void copy_spinor_to_spinor_256(spinor_256 *a, spinor *b1, spinor *b2);
void copy_spinor_256_to_spinor(spinor *b1, spinor *b2, spinor_256 *a);
void copy_halfspinor_to_halfspinor_256(halfspinor_256 *a, halfspinor *b1, halfspinor *b2);
void copy_halfspinor_256_to_halfspinor(halfspinor *b1, halfspinor *b2, halfspinor_256 *a);
*/

#endif
