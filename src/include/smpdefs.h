#ifndef SMP
#define SMP

typedef  void SMPmatrix;
typedef  struct MatrixElement  *SMPelement;

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000  AlansFixes
**********/

#include <stdio.h>
#include <math.h>
#include "complex.h"

int SMPaddElt( SMPmatrix *, int , int , double );
double * SMPmakeElt( SMPmatrix * , int , int );
void SMPcClear( SMPmatrix *);
void SMPclear( SMPmatrix *);
int SMPcLUfac( SMPmatrix *, double );
int SMPluFac( SMPmatrix *, double , double );
int SMPcReorder( SMPmatrix * , double , double , int *);
int SMPreorder( SMPmatrix * , double , double , double );
void SMPcaSolve(SMPmatrix *Matrix, double RHS[], double iRHS[],
		double Spare[], double iSpare[]);
void SMPcSolve( SMPmatrix *, double [], double [], double [], double []);
void SMPsolve( SMPmatrix *, double [], double []);
int SMPmatSize( SMPmatrix *);
int SMPnewMatrix( SMPmatrix ** );
void SMPdestroy( SMPmatrix *);
int SMPpreOrder( SMPmatrix *);
void SMPprint( SMPmatrix * , FILE *);
void SMPgetError( SMPmatrix *, int *, int *);
int SMPcProdDiag( SMPmatrix *, SPcomplex *, int *);
int SMPcDProd(SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent);
SMPelement * SMPfindElt( SMPmatrix *, int , int , int );
int SMPcZeroCol(SMPmatrix *eMatrix, int Col);
int SMPcAddCol(SMPmatrix *eMatrix, int Accum_Col, int Addend_Col);
int SMPzeroRow(SMPmatrix *eMatrix, int Row);
void spConstMult(SMPmatrix*, double);
#ifdef PARALLEL_ARCH
void SMPcombine(SMPmatrix *Matrix, double RHS[], double Spare[]);
void SMPcCombine(SMPmatrix *Matrix, double RHS[], double Spare[],
		 double iRHS[], double iSpare[]);
#endif

#endif /*SMP*/
