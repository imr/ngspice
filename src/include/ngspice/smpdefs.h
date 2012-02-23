#ifndef ngspice_SMPDEFS_H
#define ngspice_SMPDEFS_H

/* Typedef removed by Francesco Lannutti (2012-02) to create the new SMPmatrix structure */
/*
typedef  struct MatrixFrame     SMPmatrix;
*/
typedef  struct MatrixFrame     MatrixFrame;
typedef  struct MatrixElement   SMPelement;

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000  AlansFixes
**********/

#include <stdio.h>
#include <math.h>
#include "ngspice/complex.h"

#if defined(KLU)
#include "ngspice/klu.h"
#elif defined(SuperLU)
#include "ngspice/slu_ddefs.h"
#endif

struct SMPmatrix {
    MatrixFrame *SPmatrix ;               /* pointer to sparse matrix */

#if defined(KLU)
    klu_common *CKTkluCommon ;            /* KLU common object */
    klu_symbolic *CKTkluSymbolic ;        /* KLU symbolic object */
    klu_numeric *CKTkluNumeric ;          /* KLU numeric object */
    int *CKTkluAp ;                       /* KLU column pointer */
    int *CKTkluAi ;                       /* KLU row pointer */
    double *CKTkluAx ;                    /* KLU element */
    double *CKTkluIntermediate ;          /* KLU RHS Intermediate for Solve Real Step */
    double *CKTkluIntermediate_Complex ;  /* KLU iRHS Intermediate for Solve Complex Step */
    double **CKTbind_Sparse ;             /* KLU - Sparse original element position */
    double **CKTbind_CSC ;                /* KLU - KLU new element position */
    double **CKTbind_CSC_Complex ;        /* KLU - KLU new element position in Complex analysis */
    double **CKTdiag_CSC ;                 /* KLU pointer to diagonal element to perform Gmin */
    int CKTkluN ;                         /* KLU N, copied */
    int CKTklunz ;                        /* KLU nz, copied for AC Analysis */
    int CKTkluMODE ;                      /* KLU MODE parameter to enable KLU or not from the heuristic */
    #define CKTkluON  1                   /* KLU MODE ON  definition */
    #define CKTkluOFF 0                   /* KLU MODE OFF definition */
#elif defined(SuperLU)
    int *CKTsuperluAp ;
    int *CKTsuperluAi ;
    double *CKTsuperluAx ;
    SuperMatrix CKTsuperluA ;
    SuperMatrix CKTsuperluL ;
    SuperMatrix CKTsuperluU ;
    SuperMatrix CKTsuperluI ;
    SuperMatrix CKTsuperluAC ;
    int *CKTsuperluPerm_r ;
    int *CKTsuperluPerm_c ;
    int CKTsuperluInfo ;
    int *CKTsuperluEtree ;
    superlu_options_t CKTsuperluOptions ;
    SuperLUStat_t CKTsuperluStat ;
    double *CKTsuperluIntermediate ;
    double **CKTbind_Sparse ;
    double **CKTbind_CSC ;
    double **CKTbind_CSC_Complex ;
    double **CKTdiag_CSC ;
    int CKTsuperluN ;
    int CKTsuperlunz ;
    int CKTsuperluMODE ;
    #define CKTsuperluON 1		   /* SuperLU MODE ON definition */
    #define CKTsuperluOFF 0		   /* SuperLU MODE OFF definition */
#endif

};

/* SMPmatrix structure alias - Francesco Lannutti (2012-02) */
typedef struct SMPmatrix SMPmatrix ;


#if defined(KLU) || defined(SuperLU)
void SMPmatrix_CSC ( SMPmatrix * ) ;
void SMPnnz ( SMPmatrix * ) ;
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
int SMPnewMatrix( SMPmatrix * );
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

#endif
