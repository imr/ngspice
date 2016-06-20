/*
 *  Spice3 COMPATIBILITY MODULE
 *
 *  Author:                     Advising professor:
 *     Kenneth S. Kundert           Alberto Sangiovanni-Vincentelli
 *     UC Berkeley
 *
 *  This module contains routines that make Sparse1.3 a direct
 *  replacement for the SMP sparse matrix package in Spice3c1 or Spice3d1.
 *  Sparse1.3 is in general a faster and more robust package than SMP.
 *  These advantages become significant on large circuits.
 *
 *  >>> User accessible functions contained in this file:
 *  SMPaddElt
 *  SMPmakeElt
 *  SMPcClear
 *  SMPclear
 *  SMPcLUfac
 *  SMPluFac
 *  SMPcReorder
 *  SMPreorder
 *  SMPcaSolve
 *  SMPcSolve
 *  SMPsolve
 *  SMPmatSize
 *  SMPnewMatrix
 *  SMPdestroy
 *  SMPpreOrder
 *  SMPprint
 *  SMPgetError
 *  SMPcProdDiag
 *  LoadGmin
 *  SMPfindElt
 *  SMPcombine
 *  SMPcCombine
 */

/*
 *  To replace SMP with Sparse, rename the file spSpice3.h to
 *  spMatrix.h and place Sparse in a subdirectory of SPICE called
 *  `sparse'.  Then on UNIX compile Sparse by executing `make spice'.
 *  If not on UNIX, after compiling Sparse and creating the sparse.a
 *  archive, compile this file (spSMP.c) and spSMP.o to the archive,
 *  then copy sparse.a into the SPICE main directory and rename it
 *  SMP.a.  Finally link SPICE.
 *
 *  To be compatible with SPICE, the following Sparse compiler options
 *  (in spConfig.h) should be set as shown below:
 *
 *      EXPANDABLE                      YES
 *      TRANSLATE                       NO
 *      INITIALIZE                      NO or YES, YES for use with test prog.
 *      DIAGONAL_PIVOTING               YES
 *      MODIFIED_MARKOWITZ              NO
 *      DELETE                          NO
 *      STRIP                           NO
 *      MODIFIED_NODAL                  YES
 *      QUAD_ELEMENT                    NO
 *      TRANSPOSE                       YES
 *      SCALING                         NO
 *      DOCUMENTATION                   YES
 *      MULTIPLICATION                  NO
 *      DETERMINANT                     YES
 *      STABILITY                       NO
 *      CONDITION                       NO
 *      PSEUDOCONDITION                 NO
 *      DEBUG                           YES
 *
 *      spREAL  double
 */

/*
 *  Revision and copyright information.
 *
 *  Copyright (c) 1985,86,87,88,89,90
 *  by Kenneth S. Kundert and the University of California.
 *
 *  Permission to use, copy, modify, and distribute this software and its
 *  documentation for any purpose and without fee is hereby granted, provided
 *  that the above copyright notice appear in all copies and supporting
 *  documentation and that the authors and the University of California
 *  are properly credited.  The authors and the University of California
 *  make no representations as to the suitability of this software for
 *  any purpose.  It is provided `as is', without express or implied warranty.
 */

/*
 *  IMPORTS
 *
 *  >>> Import descriptions:
 *  spMatrix.h
 *     Sparse macros and declarations.
 *  SMPdefs.h
 *     Spice3's matrix macro definitions.
 */

#include "ngspice/config.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "ngspice/spmatrix.h"
#include "../sparse/spdefs.h"
#include "ngspice/smpdefs.h"

#if defined (_MSC_VER)
extern double scalbn(double, int);
#define logb _logb
extern double logb(double);
#endif

static void LoadGmin_CSC (double **diag, int n, double Gmin) ;
static void LoadGmin (SMPmatrix *eMatrix, double Gmin) ;

void
SMPmatrix_CSC (SMPmatrix *Matrix)
{
    spMatrix_CSC (Matrix->SPmatrix, Matrix->CKTkluAp, Matrix->CKTkluAi, Matrix->CKTkluAx,
                  Matrix->CKTkluAx_Complex, Matrix->CKTkluN, Matrix->CKTbindStruct, Matrix->CKTdiag_CSC) ;

//    spMatrix_CSC_dump (Matrix->SPmatrix, 1, NULL) ;

    return ;
}

void
SMPnnz (SMPmatrix *Matrix)
{
    Matrix->CKTklunz = Matrix->SPmatrix->Elements ;

    return ;
}

/*
 * SMPaddElt()
 */
int
SMPaddElt (SMPmatrix *Matrix, int Row, int Col, double Value)
{
    *spGetElement (Matrix->SPmatrix, Row, Col) = Value ;
    return spError (Matrix->SPmatrix) ;
}

/*
 * SMPmakeElt()
 */
double *
SMPmakeElt (SMPmatrix *Matrix, int Row, int Col)
{
    return spGetElement (Matrix->SPmatrix, Row, Col) ;
}

/*
 * SMPcClear()
 */

void
SMPcClear (SMPmatrix *Matrix)
{
    int i ;
    if (Matrix->CKTkluMODE)
    {
        spClear (Matrix->SPmatrix) ;
        if (Matrix->CKTkluAx_Complex != NULL)
        {
            for (i = 0 ; i < 2 * Matrix->CKTklunz ; i++)
                Matrix->CKTkluAx_Complex [i] = 0 ;
        }
    } else {
        spClear (Matrix->SPmatrix) ;
    }
}

/*
 * SMPclear()
 */

void
SMPclear (SMPmatrix *Matrix)
{
    int i ;
    if (Matrix->CKTkluMODE)
    {
        spClear (Matrix->SPmatrix) ;
        if (Matrix->CKTkluAx != NULL)
        {
            for (i = 0 ; i < Matrix->CKTklunz ; i++)
                Matrix->CKTkluAx [i] = 0 ;
        }
    } else {
        spClear (Matrix->SPmatrix) ;
    }
}

#define NG_IGNORE(x)  (void)x

/*
 * SMPcLUfac()
 */
/*ARGSUSED*/

int
SMPcLUfac (SMPmatrix *Matrix, double PivTol)
{
    int ret ;

    NG_IGNORE (PivTol) ;

    if (Matrix->CKTkluMODE)
    {
        spSetComplex (Matrix->SPmatrix) ;
        ret = klu_z_refactor (Matrix->CKTkluAp, Matrix->CKTkluAi, Matrix->CKTkluAx_Complex,
                              Matrix->CKTkluSymbolic, Matrix->CKTkluNumeric, Matrix->CKTkluCommon) ;

        if (Matrix->CKTkluCommon->status == KLU_EMPTY_MATRIX)
        {
            return 0 ;
        }
        return (!ret) ;
    } else {
        spSetComplex (Matrix->SPmatrix) ;
        return spFactor (Matrix->SPmatrix) ;
    }
}

/*
 * SMPluFac()
 */
/*ARGSUSED*/

int
SMPluFac (SMPmatrix *Matrix, double PivTol, double Gmin)
{
    int ret ;

    NG_IGNORE (PivTol) ;

    if (Matrix->CKTkluMODE)
    {
        spSetReal (Matrix->SPmatrix) ;
        LoadGmin_CSC (Matrix->CKTdiag_CSC, Matrix->CKTkluN, Gmin) ;
        ret = klu_refactor (Matrix->CKTkluAp, Matrix->CKTkluAi, Matrix->CKTkluAx,
                            Matrix->CKTkluSymbolic, Matrix->CKTkluNumeric, Matrix->CKTkluCommon) ;

        if (Matrix->CKTkluCommon->status == KLU_EMPTY_MATRIX)
        {
            return 0 ;
        }
        return (!ret) ;

//        if (ret == 1)
//            return 0 ;
//        else if (ret == 0)
//            return (E_SINGULAR) ;
//        else {
//            fprintf (stderr, "KLU Error in re-factor!") ;
//            return 1 ;
//        }
    } else {
        spSetReal (Matrix->SPmatrix) ;
        LoadGmin (Matrix, Gmin) ;
        return spFactor (Matrix->SPmatrix) ;
    }
}

/*
 * SMPcReorder()
 */

int
SMPcReorder (SMPmatrix *Matrix, double PivTol, double PivRel, int *NumSwaps)
{
    if (Matrix->CKTkluMODE)
    {
        *NumSwaps = 1 ;
        spSetComplex (Matrix->SPmatrix) ;
//        Matrix->CKTkluCommon->tol = PivTol ;

        if (Matrix->CKTkluNumeric != NULL)
        {
            klu_z_free_numeric (&(Matrix->CKTkluNumeric), Matrix->CKTkluCommon) ;
            Matrix->CKTkluNumeric = klu_z_factor (Matrix->CKTkluAp, Matrix->CKTkluAi, Matrix->CKTkluAx_Complex, Matrix->CKTkluSymbolic, Matrix->CKTkluCommon) ;
        } else
            Matrix->CKTkluNumeric = klu_z_factor (Matrix->CKTkluAp, Matrix->CKTkluAi, Matrix->CKTkluAx_Complex, Matrix->CKTkluSymbolic, Matrix->CKTkluCommon) ;

        if (Matrix->CKTkluNumeric == NULL)
        {
            if (Matrix->CKTkluCommon->status == KLU_EMPTY_MATRIX)
            {
                return 0 ;
            }
            return 1 ;
        }
        else
            return 0 ;
    } else {
        *NumSwaps = 1 ;
        spSetComplex (Matrix->SPmatrix) ;
        return spOrderAndFactor (Matrix->SPmatrix, NULL, (spREAL)PivRel, (spREAL)PivTol, YES) ;
    }
}

/*
 * SMPreorder()
 */

int
SMPreorder (SMPmatrix *Matrix, double PivTol, double PivRel, double Gmin)
{
    if (Matrix->CKTkluMODE)
    {
        spSetReal (Matrix->SPmatrix) ;
        LoadGmin_CSC (Matrix->CKTdiag_CSC, Matrix->CKTkluN, Gmin) ;
//        Matrix->CKTkluCommon->tol = PivTol ;

        if (Matrix->CKTkluNumeric != NULL)
        {
            klu_free_numeric (&(Matrix->CKTkluNumeric), Matrix->CKTkluCommon) ;
            Matrix->CKTkluNumeric = klu_factor (Matrix->CKTkluAp, Matrix->CKTkluAi, Matrix->CKTkluAx, Matrix->CKTkluSymbolic, Matrix->CKTkluCommon) ;
        } else
            Matrix->CKTkluNumeric = klu_factor (Matrix->CKTkluAp, Matrix->CKTkluAi, Matrix->CKTkluAx, Matrix->CKTkluSymbolic, Matrix->CKTkluCommon) ;

        if (Matrix->CKTkluNumeric == NULL)
        {
            if (Matrix->CKTkluCommon->status == KLU_EMPTY_MATRIX)
            {
                return 0 ;
            }
            return 1 ;
        }
        else
            return 0 ;
    } else {
        spSetReal (Matrix->SPmatrix) ;
        LoadGmin (Matrix, Gmin) ;
        return spOrderAndFactor (Matrix->SPmatrix, NULL, (spREAL)PivRel, (spREAL)PivTol, YES) ;
    }
}

/*
 * SMPcaSolve()
 */
void
SMPcaSolve (SMPmatrix *Matrix, double RHS[], double iRHS[], double Spare[], double iSpare[])
{
    int ret, i, *pExtOrder ;

    NG_IGNORE (iSpare) ;
    NG_IGNORE (Spare) ;

    if (Matrix->CKTkluMODE)
    {
        pExtOrder = &Matrix->SPmatrix->IntToExtRowMap [Matrix->CKTkluN] ;
        for (i = 2 * Matrix->CKTkluN - 1 ; i > 0 ; i -= 2)
        {
            Matrix->CKTkluIntermediate_Complex [i] = iRHS [*(pExtOrder)] ;
            Matrix->CKTkluIntermediate_Complex [i - 1] = RHS [*(pExtOrder--)] ;
        }

        ret = klu_z_tsolve (Matrix->CKTkluSymbolic, Matrix->CKTkluNumeric, Matrix->CKTkluN, 1, Matrix->CKTkluIntermediate_Complex, 0, Matrix->CKTkluCommon) ;

        pExtOrder = &Matrix->SPmatrix->IntToExtColMap [Matrix->CKTkluN] ;
        for (i = 2 * Matrix->CKTkluN - 1 ; i > 0 ; i -= 2)
        {
            iRHS [*(pExtOrder)] = Matrix->CKTkluIntermediate_Complex [i] ;
            RHS [*(pExtOrder--)] = Matrix->CKTkluIntermediate_Complex [i - 1] ;
        }
    } else {
        spSolveTransposed (Matrix->SPmatrix, RHS, RHS, iRHS, iRHS) ;
    }
}

/*
 * SMPcSolve()
 */

void
SMPcSolve (SMPmatrix *Matrix, double RHS[], double iRHS[], double Spare[], double iSpare[])
{
    int ret, i, *pExtOrder ;

    NG_IGNORE (iSpare) ;
    NG_IGNORE (Spare) ;

    if (Matrix->CKTkluMODE)
    {
        pExtOrder = &Matrix->SPmatrix->IntToExtRowMap [Matrix->CKTkluN] ;
        for (i = 2 * Matrix->CKTkluN - 1 ; i > 0 ; i -= 2)
        {
            Matrix->CKTkluIntermediate_Complex [i] = iRHS [*(pExtOrder)] ;
            Matrix->CKTkluIntermediate_Complex [i - 1] = RHS [*(pExtOrder--)] ;
        }

        ret = klu_z_solve (Matrix->CKTkluSymbolic, Matrix->CKTkluNumeric, Matrix->CKTkluN, 1, Matrix->CKTkluIntermediate_Complex, Matrix->CKTkluCommon) ;

        pExtOrder = &Matrix->SPmatrix->IntToExtColMap [Matrix->CKTkluN] ;
        for (i = 2 * Matrix->CKTkluN - 1 ; i > 0 ; i -= 2)
        {
            iRHS [*(pExtOrder)] = Matrix->CKTkluIntermediate_Complex [i] ;
            RHS [*(pExtOrder--)] = Matrix->CKTkluIntermediate_Complex [i - 1] ;
        }

    } else {

        spSolve (Matrix->SPmatrix, RHS, RHS, iRHS, iRHS) ;
    }
}

/*
 * SMPsolve()
 */

void
SMPsolve (SMPmatrix *Matrix, double RHS[], double Spare[])
{
    int ret, i, *pExtOrder ;

    NG_IGNORE (Spare) ;

    if (Matrix->CKTkluMODE) {

        pExtOrder = &Matrix->SPmatrix->IntToExtRowMap [Matrix->CKTkluN] ;
        for (i = Matrix->CKTkluN - 1 ; i >= 0 ; i--)
            Matrix->CKTkluIntermediate [i] = RHS [*(pExtOrder--)] ;

        ret = klu_solve (Matrix->CKTkluSymbolic, Matrix->CKTkluNumeric, Matrix->CKTkluN, 1, Matrix->CKTkluIntermediate, Matrix->CKTkluCommon) ;

        pExtOrder = &Matrix->SPmatrix->IntToExtColMap [Matrix->CKTkluN] ;
        for (i = Matrix->CKTkluN - 1 ; i >= 0 ; i--)
            RHS [*(pExtOrder--)] = Matrix->CKTkluIntermediate [i] ;
    } else {
        spSolve (Matrix->SPmatrix, RHS, RHS, NULL, NULL) ;
    }
}

/*
 * SMPmatSize()
 */
int
SMPmatSize (SMPmatrix *Matrix)
{
    return spGetSize (Matrix->SPmatrix, 1) ;
}

/*
 * SMPnewMatrix()
 */
int
SMPnewMatrix (SMPmatrix *Matrix, int size)
{
    int Error ;
    Matrix->SPmatrix = spCreate (size, 1, &Error) ;
    return Error ;
}

/*
 * SMPdestroy()
 */

void
SMPdestroy (SMPmatrix *Matrix)
{
    spDestroy (Matrix->SPmatrix) ;

    if (Matrix->CKTkluMODE)
    {
        klu_free_numeric (&(Matrix->CKTkluNumeric), Matrix->CKTkluCommon) ;
        klu_free_symbolic (&(Matrix->CKTkluSymbolic), Matrix->CKTkluCommon) ;
        free (Matrix->CKTkluAp) ;
        free (Matrix->CKTkluAi) ;
        free (Matrix->CKTkluAx) ;
        free (Matrix->CKTkluIntermediate) ;
        free (Matrix->CKTbindStruct) ;
        free (Matrix->CKTdiag_CSC) ;
        free (Matrix->CKTkluAx_Complex) ;
        free (Matrix->CKTkluIntermediate_Complex) ;
    }
}

/*
 * SMPpreOrder()
 */

int
SMPpreOrder (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE)
    {
        Matrix->CKTkluSymbolic = klu_analyze (Matrix->CKTkluN, Matrix->CKTkluAp, Matrix->CKTkluAi, Matrix->CKTkluCommon) ;

        if (Matrix->CKTkluSymbolic == NULL)
        {
            if (Matrix->CKTkluCommon->status == KLU_EMPTY_MATRIX)
            {
                return 0 ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        spMNA_Preorder (Matrix->SPmatrix) ;
        return spError (Matrix->SPmatrix) ;
    }
}

/*
 * SMPprintRHS()
 */

void
SMPprintRHS (SMPmatrix *Matrix, char *Filename, RealVector RHS, RealVector iRHS)
{
    if (!Matrix->CKTkluMODE)
        spFileVector (Matrix->SPmatrix, Filename, RHS, iRHS) ;
}

/*
 * SMPprint()
 */

void
SMPprint (SMPmatrix *Matrix, char *Filename)
{
    if (Matrix->CKTkluMODE)
    {
        klu_z_print (Matrix->CKTkluAp, Matrix->CKTkluAi, Matrix->CKTkluAx_Complex, Matrix->CKTkluN, Matrix->SPmatrix->IntToExtRowMap, Matrix->SPmatrix->IntToExtColMap) ;
    } else {
        if (Filename)
            spFileMatrix (Matrix->SPmatrix, Filename, "Circuit Matrix", 0, 1, 1) ;
        else
            spPrint (Matrix->SPmatrix, 0, 1, 1) ;
    }
}

/*
 * SMPgetError()
 */
void
SMPgetError (SMPmatrix *Matrix, int *Col, int *Row)
{
    if (Matrix->CKTkluMODE)
    {
        *Row = Matrix->SPmatrix->IntToExtRowMap [Matrix->CKTkluCommon->singular_col + 1] ;
        *Col = Matrix->SPmatrix->IntToExtColMap [Matrix->CKTkluCommon->singular_col + 1] ;
    } else {
        spWhereSingular (Matrix->SPmatrix, Row, Col) ;
    }
}

#ifdef KLU
void
spDeterminant_KLU (SMPmatrix *Matrix, int *pExponent, RealNumber *pDeterminant, RealNumber *piDeterminant)
{
    int I, Size ;
    RealNumber Norm, nr, ni ;
    ComplexNumber Pivot, cDeterminant, Udiag ;

    int *P, *Q ;
    double *Rs, *Ux, *Uz ;
    unsigned int nSwap, nSwapP, nSwapQ ;

#define  NORM(a)     (nr = ABS((a).Real), ni = ABS((a).Imag), MAX (nr,ni))

    *pExponent = 0 ;

    if (Matrix->CKTkluCommon->status == KLU_SINGULAR)
    {
	*pDeterminant = 0.0 ;
        if (Matrix->CKTkluMatrixIsComplex == CKTkluMatrixComplex)
        {
            *piDeterminant = 0.0 ;
        }
        return ;
    }

    Size = Matrix->CKTkluN ;
    I = 0 ;

    P = (int *) malloc ((size_t)Matrix->CKTkluN * sizeof (int)) ;
    Q = (int *) malloc ((size_t)Matrix->CKTkluN * sizeof (int)) ;

    Ux = (double *) malloc ((size_t)Matrix->CKTkluN * sizeof (double)) ;

    Rs = (double *) malloc ((size_t)Matrix->CKTkluN * sizeof (double)) ;

    if (Matrix->CKTkluMatrixIsComplex == CKTkluMatrixComplex)        /* Complex Case. */
    {
	cDeterminant.Real = 1.0 ;
        cDeterminant.Imag = 0.0 ;

        Uz = (double *) malloc ((size_t)Matrix->CKTkluN * sizeof (double)) ;
/*
        int *Lp, *Li, *Up, *Ui, *Fp, *Fi, *P, *Q ;
        double *Lx, *Lz, *Ux, *Uz, *Fx, *Fz, *Rs ;
        Lp = (int *) malloc (((size_t)Matrix->CKTkluN + 1) * sizeof (int)) ;
        Li = (int *) malloc ((size_t)Matrix->CKTkluNumeric->lnz * sizeof (int)) ;
        Lx = (double *) malloc ((size_t)Matrix->CKTkluNumeric->lnz * sizeof (double)) ;
        Lz = (double *) malloc ((size_t)Matrix->CKTkluNumeric->lnz * sizeof (double)) ;
        Up = (int *) malloc (((size_t)Matrix->CKTkluN + 1) * sizeof (int)) ;
        Ui = (int *) malloc ((size_t)Matrix->CKTkluNumeric->unz * sizeof (int)) ;
        Ux = (double *) malloc ((size_t)Matrix->CKTkluNumeric->unz * sizeof (double)) ;
        Uz = (double *) malloc ((size_t)Matrix->CKTkluNumeric->unz * sizeof (double)) ;
        Fp = (int *) malloc (((size_t)Matrix->CKTkluN + 1) * sizeof (int)) ;
        Fi = (int *) malloc ((size_t)Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] * sizeof (int)) ;
        Fx = (double *) malloc ((size_t)Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] * sizeof (double)) ;
        Fz = (double *) malloc ((size_t)Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] * sizeof (double)) ;
        klu_z_extract (Matrix->CKTkluNumeric, Matrix->CKTkluSymbolic,
                       Lp, Li, Lx, Lz,
                       Up, Ui, Ux, Uz,
                       Fp, Fi, Fx, Fz,
                       P, Q, Rs, NULL,
                       Matrix->CKTkluCommon) ;
*/
        klu_z_extract_Udiag (Matrix->CKTkluNumeric, Matrix->CKTkluSymbolic, Ux, Uz, P, Q, Rs, Matrix->CKTkluCommon) ;
/*
        for (I = 0 ; I < Matrix->CKTkluNumeric->lnz ; I++)
        {
            printf ("L - Value: %-.9g\t%-.9g\n", Lx [I], Lz [I]) ;
        }
        for (I = 0 ; I < Matrix->CKTkluNumeric->unz ; I++)
        {
            printf ("U - Value: %-.9g\t%-.9g\n", Ux [I], Uz [I]) ;
        }
        for (I = 0 ; I < Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] ; I++)
        {
            printf ("F - Value: %-.9g\t%-.9g\n", Fx [I], Fz [I]) ;
        }

        for (I = 0 ; I < Matrix->CKTkluN ; I++)
        {
            printf ("U - Value: %-.9g\t%-.9g\n", Ux [I], Uz [I]) ;
        }
*/
        nSwapP = 0 ;
        for (I = 0 ; I < Matrix->CKTkluN ; I++)
        {
            if (P [I] != I)
            {
                nSwapP++ ;
            }
        }
        nSwapP /= 2 ;

        nSwapQ = 0 ;
        for (I = 0 ; I < Matrix->CKTkluN ; I++)
        {
            if (Q [I] != I)
            {
                nSwapQ++ ;
            }
        }
        nSwapQ /= 2 ;

        nSwap = nSwapP + nSwapQ ;
/*
        free (Lp) ;
        free (Li) ;
        free (Lx) ;
        free (Lz) ;
        free (Up) ;
        free (Ui) ;
        free (Fp) ;
        free (Fi) ;
        free (Fx) ;
        free (Fz) ;
*/
        I = 0 ;
        while (I < Size)
        {
            Udiag.Real = 1 / (Ux [I] * Rs [I]) ;
            Udiag.Imag = Uz [I] * Rs [I] ;

//            printf ("Udiag.Real: %-.9g\tUdiag.Imag %-.9g\n", Udiag.Real, Udiag.Imag) ;

            CMPLX_RECIPROCAL (Pivot, Udiag) ;
            CMPLX_MULT_ASSIGN (cDeterminant, Pivot) ;

//            printf ("cDeterminant.Real: %-.9g\tcDeterminant.Imag %-.9g\n", cDeterminant.Real, cDeterminant.Imag) ;

	    /* Scale Determinant. */
            Norm = NORM (cDeterminant) ;
            if (Norm != 0.0)
            {
		while (Norm >= 1.0e12)
                {
		    cDeterminant.Real *= 1.0e-12 ;
                    cDeterminant.Imag *= 1.0e-12 ;
                    *pExponent += 12 ;
                    Norm = NORM (cDeterminant) ;
                }
                while (Norm < 1.0e-12)
                {
		    cDeterminant.Real *= 1.0e12 ;
                    cDeterminant.Imag *= 1.0e12 ;
                    *pExponent -= 12 ;
                    Norm = NORM (cDeterminant) ;
                }
            }
            I++ ;
        }

	/* Scale Determinant again, this time to be between 1.0 <= x < 10.0. */
        Norm = NORM (cDeterminant) ;
        if (Norm != 0.0)
        {
	    while (Norm >= 10.0)
            {
		cDeterminant.Real *= 0.1 ;
                cDeterminant.Imag *= 0.1 ;
                (*pExponent)++ ;
                Norm = NORM (cDeterminant) ;
            }
            while (Norm < 1.0)
            {
		cDeterminant.Real *= 10.0 ;
                cDeterminant.Imag *= 10.0 ;
                (*pExponent)-- ;
                Norm = NORM (cDeterminant) ;
            }
        }
        if (nSwap % 2 != 0)
        {
            CMPLX_NEGATE (cDeterminant) ;
        }

        *pDeterminant = cDeterminant.Real ;
        *piDeterminant = cDeterminant.Imag ;

        free (Uz) ;
    }
    else
    {
	/* Real Case. */
        *pDeterminant = 1.0 ;

        klu_extract_Udiag (Matrix->CKTkluNumeric, Matrix->CKTkluSymbolic, Ux, P, Q, Rs, Matrix->CKTkluCommon) ;

        nSwapP = 0 ;
        for (I = 0 ; I < Matrix->CKTkluN ; I++)
        {
            if (P [I] != I)
            {
                nSwapP++ ;
            }
        }
        nSwapP /= 2 ;

        nSwapQ = 0 ;
        for (I = 0 ; I < Matrix->CKTkluN ; I++)
        {
            if (Q [I] != I)
            {
                nSwapQ++ ;
            }
        }
        nSwapQ /= 2 ;

        nSwap = nSwapP + nSwapQ ;

        while (I < Size)
        {
            *pDeterminant /= (Ux [I] * Rs [I]) ;

	    /* Scale Determinant. */
            if (*pDeterminant != 0.0)
            {
		while (ABS(*pDeterminant) >= 1.0e12)
                {
		    *pDeterminant *= 1.0e-12 ;
                    *pExponent += 12 ;
                }
                while (ABS(*pDeterminant) < 1.0e-12)
                {
		    *pDeterminant *= 1.0e12 ;
                    *pExponent -= 12 ;
                }
            }
            I++ ;
        }

	/* Scale Determinant again, this time to be between 1.0 <= x <
           10.0. */
        if (*pDeterminant != 0.0)
        {
	    while (ABS(*pDeterminant) >= 10.0)
            {
		*pDeterminant *= 0.1 ;
                (*pExponent)++ ;
            }
            while (ABS(*pDeterminant) < 1.0)
            {
		*pDeterminant *= 10.0 ;
                (*pExponent)-- ;
            }
        }
        if (nSwap % 2 != 0)
        {
            *pDeterminant = -*pDeterminant ;
        }
    }

    free (P) ;
    free (Q) ;
    free (Ux) ;
    free (Rs) ;
}
#endif

/*
 * SMPcProdDiag()
 *    note: obsolete for Spice3d2 and later
 */
int
SMPcProdDiag (SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent)
{
    if (Matrix->CKTkluMODE)
    {
        spDeterminant_KLU (Matrix, pExponent, &(pMantissa->real), &(pMantissa->imag)) ;
    } else {
        spDeterminant (Matrix->SPmatrix, pExponent, &(pMantissa->real), &(pMantissa->imag)) ;
    }
    return spError (Matrix->SPmatrix) ;
}

/*
 * SMPcDProd()
 */
int
SMPcDProd (SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent)
{
    double	re, im, x, y, z;
    int		p;

    if (Matrix->CKTkluMODE)
    {
        spDeterminant_KLU (Matrix, &p, &re, &im) ;
    } else {
        spDeterminant (Matrix->SPmatrix, &p, &re, &im) ;
    }

#ifndef M_LN2
#define M_LN2   0.69314718055994530942
#endif
#ifndef M_LN10
#define M_LN10  2.30258509299404568402
#endif

#ifdef debug_print
    printf ("Determinant 10: (%20g,%20g)^%d\n", re, im, p) ;
#endif

    /* Convert base 10 numbers to base 2 numbers, for comparison */
    y = p * M_LN10 / M_LN2;
    x = (int) y;
    y -= x;

    /* ASSERT
     *	x = integral part of exponent, y = fraction part of exponent
     */

    /* Fold in the fractional part */
#ifdef debug_print
    printf (" ** base10 -> base2 int =  %g, frac = %20g\n", x, y) ;
#endif
    z = pow (2.0, y) ;
    re *= z ;
    im *= z ;
#ifdef debug_print
    printf (" ** multiplier = %20g\n", z) ;
#endif

    /* Re-normalize (re or im may be > 2.0 or both < 1.0 */
    if (re != 0.0)
    {
	y = logb (re) ;
	if (im != 0.0)
	    z = logb (im) ;
	else
	    z = 0 ;
    } else if (im != 0.0) {
	z = logb (im) ;
	y = 0 ;
    } else {
	/* Singular */
	/*printf("10 -> singular\n");*/
	y = 0 ;
	z = 0 ;
    }

#ifdef debug_print
    printf (" ** renormalize changes = %g,%g\n", y, z) ;
#endif
    if (y < z)
	y = z ;

    *pExponent = (int)(x + y) ;
    x = scalbn (re, (int) -y) ;
    z = scalbn (im, (int) -y) ;
#ifdef debug_print
    printf (" ** values are: re %g, im %g, y %g, re' %g, im' %g\n", re, im, y, x, z) ;
#endif
    pMantissa->real = scalbn (re, (int) -y) ;
    pMantissa->imag = scalbn (im, (int) -y) ;

#ifdef debug_print
    printf ("Determinant 10->2: (%20g,%20g)^%d\n", pMantissa->real, pMantissa->imag, *pExponent) ;
#endif

    if (Matrix->CKTkluMODE)
    {
        return 0 ;
    } else {
        return spError (Matrix->SPmatrix) ;
    }
}



/*
 *  The following routines need internal knowledge of the Sparse data
 *  structures.
 */

/*
 *  LOAD GMIN
 *
 *  This routine adds Gmin to each diagonal element.  Because Gmin is
 *  added to the current diagonal, which may bear little relation to
 *  what the outside world thinks is a diagonal, and because the
 *  elements that are diagonals may change after calling spOrderAndFactor,
 *  use of this routine is not recommended.  It is included here simply
 *  for compatibility with Spice3.
 */


static void
LoadGmin_CSC (double **diag, int n, double Gmin)
{
    int i ;

    if (Gmin != 0.0)
        for (i = 0 ; i < n ; i++)
            if (diag [i] != NULL)
                *(diag [i]) += Gmin ;
}

static void
LoadGmin (SMPmatrix *eMatrix, double Gmin)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    int I ;
    ArrayOfElementPtrs Diag ;
    ElementPtr diag ;

    /* Begin `LoadGmin'. */
    assert (IS_SPARSE (Matrix)) ;

    if (Gmin != 0.0) {
	Diag = Matrix->Diag ;
	for (I = Matrix->Size ; I > 0 ; I--)
        {
	    if ((diag = Diag [I]) != NULL)
		diag->Real += Gmin ;
	}
    }
    return ;
}




/*
 *  FIND ELEMENT
 *
 *  This routine finds an element in the matrix by row and column number.
 *  If the element exists, a pointer to it is returned.  If not, then NULL
 *  is returned unless the CreateIfMissing flag is TRUE, in which case a
 *  pointer to the new element is returned.
 */

SMPelement *
SMPfindElt (SMPmatrix *eMatrix, int Row, int Col, int CreateIfMissing)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr Element ;

    /* Begin `SMPfindElt'. */
    assert (IS_SPARSE (Matrix)) ;
    Row = Matrix->ExtToIntRowMap [Row] ;
    Col = Matrix->ExtToIntColMap [Col] ;
    Element = Matrix->FirstInCol [Col] ;
    Element = spcFindElementInCol (Matrix, &Element, Row, Col, CreateIfMissing) ;
    return (SMPelement *)Element ;
}

/* XXX The following should probably be implemented in spUtils */

/*
 * SMPcZeroCol()
 */
int
SMPcZeroCol (SMPmatrix *eMatrix, int Col)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Element ;

    Col = Matrix->ExtToIntColMap [Col] ;

    if (eMatrix->CKTkluMODE)
    {
        int i ;
        for (i = eMatrix->CKTkluAp [Col - 1] ; i < eMatrix->CKTkluAp [Col] ; i++)
        {
            eMatrix->CKTkluAx_Complex [2 * i] = 0 ;
            eMatrix->CKTkluAx_Complex [2 * i + 1] = 0 ;
        }
        return 0 ;
    } else {
        for (Element = Matrix->FirstInCol [Col] ; Element != NULL ; Element = Element->NextInCol)
        {
            Element->Real = 0.0 ;
            Element->Imag = 0.0 ;
        }
        return spError (Matrix) ;
    }
}

/*
 * SMPcAddCol()
 */
int
SMPcAddCol (SMPmatrix *eMatrix, int Accum_Col, int Addend_Col)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Accum, Addend, *Prev ;

    Accum_Col = Matrix->ExtToIntColMap [Accum_Col] ;
    Addend_Col = Matrix->ExtToIntColMap [Addend_Col] ;

    Addend = Matrix->FirstInCol [Addend_Col] ;
    Prev = &Matrix->FirstInCol [Accum_Col] ;
    Accum = *Prev;

    while (Addend != NULL)
    {
	while (Accum && Accum->Row < Addend->Row)
        {
	    Prev = &Accum->NextInCol ;
	    Accum = *Prev ;
	}
	if (!Accum || Accum->Row > Addend->Row)
        {
	    Accum = spcCreateElement (Matrix, Addend->Row, Accum_Col, Prev, 0) ;
	}
	Accum->Real += Addend->Real ;
	Accum->Imag += Addend->Imag ;
	Addend = Addend->NextInCol ;
    }

    return spError (Matrix) ;
}

/*
 * SMPzeroRow()
 */
int
SMPzeroRow (SMPmatrix *eMatrix, int Row)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Element ;

    Row = Matrix->ExtToIntColMap [Row] ;

    if (Matrix->RowsLinked == NO)
	spcLinkRows (Matrix) ;

    if (Matrix->PreviousMatrixWasComplex || Matrix->Complex)
    {
	for (Element = Matrix->FirstInRow[Row] ; Element != NULL; Element = Element->NextInRow)
	{
	    Element->Real = 0.0 ;
	    Element->Imag = 0.0 ;
	}
    } else {
	for (Element = Matrix->FirstInRow [Row] ; Element != NULL ; Element = Element->NextInRow)
	{
	    Element->Real = 0.0 ;
	}
    }

    return spError (Matrix) ;
}
