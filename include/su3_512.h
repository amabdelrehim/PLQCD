/*********************************************************************************
 * Copyright (C) 2012 Abdou Abdel-Rehim, Giannis Koutsou, Nikos Anastopolous
 * This file is part of the PLQCD library
 * 
 * function declarations and definitions for 512 related structs
 *********************************************************************************/

#ifndef _SU3_512_H
#define _SU3_512_H


typedef struct {
    _Complex double z[4];
} complex4; 


typedef struct 
{
   complex4 c00,c01,c02,c10,c11,c12,c20,c21,c22;
} su3_512;


typedef struct 
{
   complex4 c00,c01,c02,c10,c11,c12;
} su3_compact_512;


typedef struct
{
   complex4 c0,c1,c2;
} su3_vector_512;


typedef struct
{
   su3_vector_512 s0,s1,s2,s3;
} spinor_512;


typedef struct
{
  su3_vector_512 s0,s1;
} halfspinor_512;


//utility functions to copy data to and from 128 data types from/to 256 data types
static inline void copy_su3_to_su3_512(su3_512 *a, su3 *b0, su3 *b1, su3 *b2, su3 *b3)
{
    (a->c00).z[0] = (b0->c00);
    (a->c00).z[1] = (b1->c00);
    (a->c00).z[2] = (b2->c00);
    (a->c00).z[3] = (b3->c00);

    (a->c01).z[0] = (b0->c01);
    (a->c01).z[1] = (b1->c01);
    (a->c01).z[2] = (b2->c01);
    (a->c01).z[3] = (b3->c01);

    (a->c02).z[0] = (b0->c02);
    (a->c02).z[1] = (b1->c02);
    (a->c02).z[2] = (b2->c02);
    (a->c02).z[3] = (b3->c02);


    (a->c10).z[0] = (b0->c10);
    (a->c10).z[1] = (b1->c10);
    (a->c10).z[2] = (b2->c10);
    (a->c10).z[3] = (b3->c10);

    (a->c11).z[0] = (b0->c11);
    (a->c11).z[1] = (b1->c11);
    (a->c11).z[2] = (b2->c11);
    (a->c11).z[3] = (b3->c11);

    (a->c12).z[0] = (b0->c12);
    (a->c12).z[1] = (b1->c12);
    (a->c12).z[2] = (b2->c12);
    (a->c12).z[3] = (b3->c12);

    (a->c20).z[0] = (b0->c20);
    (a->c20).z[1] = (b1->c20);
    (a->c20).z[2] = (b2->c20);
    (a->c20).z[3] = (b3->c20);

    (a->c21).z[0] = (b0->c21);
    (a->c21).z[1] = (b1->c21);
    (a->c21).z[2] = (b2->c21);
    (a->c21).z[3] = (b3->c21);

    (a->c22).z[0] = (b0->c22);
    (a->c22).z[1] = (b1->c22);
    (a->c22).z[2] = (b2->c22);
    (a->c22).z[3] = (b3->c22);

} 

static inline void  copy_su3_512_to_su3(su3 *b0, su3 *b1, su3 *b2, su3 *b3, su3_512 *a)
{
    b0->c00 = (a->c00).z[0];
    b1->c00 = (a->c00).z[1];
    b2->c00 = (a->c00).z[2];
    b3->c00 = (a->c00).z[3];

    b0->c01 = (a->c01).z[0];
    b1->c01 = (a->c01).z[1];
    b2->c01 = (a->c01).z[2];
    b3->c01 = (a->c01).z[3];

    b0->c02 = (a->c02).z[0];
    b1->c02 = (a->c02).z[1];
    b2->c02 = (a->c02).z[2];
    b3->c02 = (a->c02).z[3];

    b0->c10 = (a->c10).z[0];
    b1->c10 = (a->c10).z[1];
    b2->c10 = (a->c10).z[2];
    b3->c10 = (a->c10).z[3];

    b0->c11 = (a->c11).z[0];
    b1->c11 = (a->c11).z[1];
    b2->c11 = (a->c11).z[2];
    b3->c11 = (a->c11).z[3];

    b0->c12 = (a->c12).z[0];
    b1->c12 = (a->c12).z[1];
    b2->c12 = (a->c12).z[2];
    b3->c12 = (a->c12).z[3];


    b0->c20 = (a->c20).z[0];
    b1->c20 = (a->c20).z[1];
    b2->c20 = (a->c20).z[2];
    b3->c20 = (a->c20).z[3];

    b0->c21 = (a->c21).z[0];
    b1->c21 = (a->c21).z[1];
    b2->c21 = (a->c21).z[2];
    b3->c21 = (a->c21).z[3];

    b0->c22 = (a->c22).z[0];
    b1->c22 = (a->c22).z[1];
    b2->c22 = (a->c22).z[2];
    b3->c22 = (a->c22).z[3];


}

static inline void copy_su3_vector_to_su3_vector_512(su3_vector_512 *a, su3_vector *b0, su3_vector *b1, su3_vector *b2, su3_vector *b3)
{
   (a->c0).z[0] = b0->c0;
   (a->c0).z[1] = b1->c0;
   (a->c0).z[2] = b2->c0;
   (a->c0).z[3] = b3->c0;

   (a->c1).z[0] = b0->c1;
   (a->c1).z[1] = b1->c1;
   (a->c1).z[2] = b2->c1;
   (a->c1).z[3] = b3->c1;

   (a->c2).z[0] = b0->c2;
   (a->c2).z[1] = b1->c2;
   (a->c2).z[2] = b2->c2;
   (a->c2).z[3] = b3->c2;


}


static inline void copy_su3_vector_512_to_su3_vector(su3_vector *b0, su3_vector *b1, su3_vector *b2, su3_vector *b3, su3_vector_512 *a)
{
   b0->c0 = (a->c0).z[0];
   b1->c0 = (a->c0).z[1];
   b2->c0 = (a->c0).z[2];
   b3->c0 = (a->c0).z[3];

   b0->c1 = (a->c1).z[0];
   b1->c1 = (a->c1).z[1];
   b2->c1 = (a->c1).z[2];
   b3->c1 = (a->c1).z[3];

   b0->c2 = (a->c2).z[0];
   b1->c2 = (a->c2).z[1];
   b2->c2 = (a->c2).z[2];
   b3->c2 = (a->c2).z[3];
}

static inline void copy_spinor_to_spinor_512(spinor_512 *a, spinor *b0, spinor *b1, spinor *b2, spinor *b3)
{
   copy_su3_vector_to_su3_vector_512(&(a->s0),&(b0->s0), &(b1->s0), &(b2->s0), &(b3->s0));
   copy_su3_vector_to_su3_vector_512(&(a->s1),&(b0->s1), &(b1->s1), &(b2->s1), &(b3->s1));
   copy_su3_vector_to_su3_vector_512(&(a->s2),&(b0->s2), &(b1->s2), &(b2->s2), &(b3->s2));
   copy_su3_vector_to_su3_vector_512(&(a->s3),&(b0->s3), &(b1->s3), &(b2->s3), &(b3->s3));
}

static inline void copy_spinor_512_to_spinor(spinor *b0, spinor *b1, spinor *b2, spinor *b3, spinor_512 *a)
{
   copy_su3_vector_512_to_su3_vector(&(b0->s0), &(b1->s0), &(b2->s0), &(b3->s0), &(a->s0));
   copy_su3_vector_512_to_su3_vector(&(b0->s1), &(b1->s1), &(b2->s1), &(b3->s1), &(a->s1));
   copy_su3_vector_512_to_su3_vector(&(b0->s2), &(b1->s2), &(b2->s2), &(b3->s2), &(a->s2));
   copy_su3_vector_512_to_su3_vector(&(b0->s3), &(b1->s3), &(b2->s3), &(b3->s3), &(a->s3));
}

static inline void copy_halfspinor_to_halfspinor_512(halfspinor_512 *a, halfspinor *b0, halfspinor *b1, halfspinor *b2, halfspinor *b3)
{

   copy_su3_vector_to_su3_vector_512(&(a->s0), &(b0->s0), &(b1->s0), &(b2->s0), &(b3->s0));
   copy_su3_vector_to_su3_vector_512(&(a->s1), &(b0->s1), &(b1->s1), &(b2->s1), &(b3->s1));
}


static inline void copy_halfspinor_512_to_halfspinor(halfspinor *b0, halfspinor *b1, halfspinor *b2, halfspinor *b3, halfspinor_512 *a)
{
   copy_su3_vector_512_to_su3_vector(&(b0->s0), &(b1->s0), &(b2->s0), &(b3->s0), &(a->s0));
   copy_su3_vector_512_to_su3_vector(&(b0->s1), &(b1->s1), &(b2->s1), &(b3->s1), &(a->s1));
}  

/*
void copy_su3_to_su3_512(su3_512 *a, su3 *b0, su3 *b1, su3 *b2, su3 *b3);
void copy_su3_512_to_su3(su3 *b0, su3 *b1, su3 *b2, su3 *b3, su3_512 *a);
void copy_su3_vector_to_su3_vector_512(su3_vector_512 *a, su3_vector *b0, su3_vector *b1, su3_vector *b2, su3_vector *b3);
void copy_su3_vector_512_to_su3_vector(su3_vector *b0, su3_vector *b1, su3_vector *b2, su3_vector *b3, su3_vector_512 *a);
void copy_spinor_to_spinor_512(spinor_512 *a, spinor *b0, spinor *b1, spinor *b2, spinor *b3);
void copy_spinor_512_to_spinor(spinor *b0, spinor *b1, spinor *b2, spinor *b3, spinor_512 *a);
void copy_halfspinor_to_halfspinor_512(halfspinor_512 *a, halfspinor *b0, halfspinor *b1, halfspinor *b2, halfspinor *b3);
void copy_halfspinor_512_to_halfspinor(halfspinor *b0, halfspinor *b1, halfspinor *b2, halfspinor *b3, halfspinor_512 *a);
*/

#endif
