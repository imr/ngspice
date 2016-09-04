/*
 *  project.h 
 *
 *  Diagonalization by Successive Rotations Method
 *               (The Jacobi Method)
 *
 *  Date: October 4, 1991
 *
 *  Author: Shen Lin
 *
 *  Copyright (C) University of California, Berkeley
 *
 */

/************************************************************
 *
 *	Defines
 *
 ************************************************************/

#define MAX_DIM   16
#define Title     "Diagonalization of a Symmetric matrix A (A = S^-1 D S)\n"
#define Left_deg   7  /*  should be greater than or equal to 6  */
#define Right_deg  2


/************************************************************
 *
 *	Data Structure Definitions
 *
 ************************************************************/

typedef struct linked_list_of_max_entry{
   struct linked_list_of_max_entry  *next;
   int    row, col;
   double  value;
} MAXE, *MAXE_PTR;

typedef struct {
   double  *Poly[MAX_DIM];
   double  C_0[MAX_DIM];
} Mult_Out;

typedef struct {
   double  *Poly;
   double  C_0;
} Single_Out;
