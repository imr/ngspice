#ifndef ngspice_SMPDEFS_H
#define ngspice_SMPDEFS_H

/* Typedef removed by Francesco Lannutti (2012-02) to create the new SMPmatrix structure */
/*
typedef  struct MatrixFrame     SMPmatrix;
*/
typedef  struct MatrixFrame     MatrixFrame;
typedef  struct MatrixElement  *SMPelement;

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000  AlansFixes
**********/

#include <stdio.h>
#include <math.h>
#include "ngspice/complex.h"

#ifdef KLU
#include "ngspice/klu.h"
#include "ngspice/spmatrix.h"
#endif

/* SMPmatrix structure - Francesco Lannutti (2012-02) */
typedef struct sSMPmatrix {
    MatrixFrame *SPmatrix ;               /* pointer to sparse matrix */

#ifdef KLU
    klu_common *CKTkluCommon ;            /* KLU common object */
    klu_symbolic *CKTkluSymbolic ;        /* KLU symbolic object */
    klu_numeric *CKTkluNumeric ;          /* KLU numeric object */
    int *CKTkluAp ;                       /* KLU column pointer */
    int *CKTkluAi ;                       /* KLU row pointer */
    double *CKTkluAx ;                    /* KLU Real Elements */
    double *CKTkluAx_Complex ;            /* KLU Complex Elements */
    int CKTkluMatrixIsComplex ;           /* KLU Matrix Is Complex Flag */
    #define CKTkluMatrixReal 0            /* KLU Matrix Real definition */
    #define CKTkluMatrixComplex 1         /* KLU Matrix Complex definition */
    double *CKTkluIntermediate ;          /* KLU RHS Intermediate for Solve Real Step */
    double *CKTkluIntermediate_Complex ;  /* KLU iRHS Intermediate for Solve Complex Step */
    BindElement *CKTbindStruct ;          /* KLU - Sparse Binding Structure */
    double **CKTdiag_CSC ;                /* KLU pointer to diagonal element to perform Gmin */
    int CKTkluN ;                         /* KLU N, copied */
    int CKTklunz ;                        /* KLU nz, copied for AC Analysis */
    unsigned int CKTkluMODE:1 ;           /* KLU MODE parameter to enable KLU or not from the heuristic */
    #define CKTkluON 1                    /* KLU MODE ON definition */
    #define CKTkluOFF 0                   /* KLU MODE OFF definition */
#endif

} SMPmatrix ;


#ifdef KLU
void SMPmatrix_CSC (SMPmatrix *) ;
void SMPnnz (SMPmatrix *) ;
#endif
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
int SMPnewMatrix( SMPmatrix *, int );
void SMPdestroy( SMPmatrix *);
int SMPpreOrder( SMPmatrix *);
void SMPprint( SMPmatrix * , char *);
void SMPprintRHS( SMPmatrix * , char *, double*, double*);
void SMPgetError( SMPmatrix *, int *, int *);
int SMPcProdDiag( SMPmatrix *, SPcomplex *, int *);
int SMPcDProd(SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent);
SMPelement * SMPfindElt( SMPmatrix *, int , int , int );
int SMPcZeroCol(SMPmatrix *Matrix, int Col);
int SMPcAddCol(SMPmatrix *Matrix, int Accum_Col, int Addend_Col);
int SMPzeroRow(SMPmatrix *Matrix, int Row);
void SMPconstMult(SMPmatrix *, double);
void SMPmultiply(SMPmatrix *, double *, double *, double *, double *);

#endif
