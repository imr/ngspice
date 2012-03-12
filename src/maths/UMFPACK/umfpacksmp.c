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
static void LoadGmin(SMPmatrix *eMatrix, double Gmin);

void
SMPmatrix_CSC (SMPmatrix *Matrix)
{
    spMatrix_CSC (Matrix->SPmatrix, Matrix->CKTumfpackAp, Matrix->CKTumfpackAi, Matrix->CKTumfpackAx,
                  Matrix->CKTumfpackN, Matrix->CKTbind_Sparse, Matrix->CKTbind_CSC, Matrix->CKTdiag_CSC) ;
    return ;
}

void
SMPnnz (SMPmatrix *Matrix)
{
    Matrix->CKTumfpackN = spGetSize (Matrix->SPmatrix, 1) ;
    Matrix->CKTumfpacknz = Matrix->SPmatrix->Elements ;
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
    spClear (Matrix->SPmatrix) ;
}

/*
 * SMPclear()
 */

void
SMPclear (SMPmatrix *Matrix)
{
    int i ;
    if (Matrix->CKTumfpackMODE)
    {
        spClear (Matrix->SPmatrix) ;
        if (Matrix->CKTumfpackAx != NULL)
        {
            for (i = 0 ; i < Matrix->CKTumfpacknz ; i++)
                Matrix->CKTumfpackAx [i] = 0 ;
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
    NG_IGNORE(PivTol);

    spSetComplex (Matrix->SPmatrix) ;
    return spFactor (Matrix->SPmatrix) ;
}

/*
 * SMPluFac()
 */
/*ARGSUSED*/

int
SMPluFac (SMPmatrix *Matrix, double PivTol, double Gmin)
{//printf("ReFactor\n");
    int status ;

    NG_IGNORE (PivTol) ;

    if (Matrix->CKTumfpackMODE)
    {
	spSetReal (Matrix->SPmatrix) ;
	LoadGmin_CSC (Matrix->CKTdiag_CSC, Matrix->CKTumfpackN, Gmin) ;

	status = umfpack_di_numeric (Matrix->CKTumfpackAp, Matrix->CKTumfpackAi, Matrix->CKTumfpackAx,
                                     Matrix->CKTumfpackSymbolic, &(Matrix->CKTumfpackNumeric), Matrix->CKTumfpackControl, Matrix->CKTumfpackInfo) ;

	if (status < 0) {
	    umfpack_di_report_info (Matrix->CKTumfpackControl, Matrix->CKTumfpackInfo) ;
	    umfpack_di_report_status (Matrix->CKTumfpackControl, status) ;
	    return 1 ;
	}

	return 0 ;

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
    *NumSwaps = 1 ;
    spSetComplex (Matrix->SPmatrix) ;
    return spOrderAndFactor (Matrix->SPmatrix, NULL,
                             (spREAL)PivRel, (spREAL)PivTol, YES) ;
}

/*
 * SMPreorder()
 */

int
SMPreorder (SMPmatrix *Matrix, double PivTol, double PivRel, double Gmin)
{//printf("Factor\n");
    int status ;
    if (Matrix->CKTumfpackMODE)
    {
	spSetReal (Matrix->SPmatrix) ;
	LoadGmin_CSC (Matrix->CKTdiag_CSC, Matrix->CKTumfpackN, Gmin) ;

	status = umfpack_di_numeric (Matrix->CKTumfpackAp, Matrix->CKTumfpackAi, Matrix->CKTumfpackAx,
                                     Matrix->CKTumfpackSymbolic, &(Matrix->CKTumfpackNumeric), Matrix->CKTumfpackControl, Matrix->CKTumfpackInfo) ;

	if (status < 0) {
	    umfpack_di_report_info (Matrix->CKTumfpackControl, Matrix->CKTumfpackInfo) ;
	    umfpack_di_report_status (Matrix->CKTumfpackControl, status) ;
	    return 1 ;
	}

	return 0 ;
    }
    else {
	spSetReal (Matrix->SPmatrix) ;
	LoadGmin (Matrix, Gmin) ;
	return spOrderAndFactor (Matrix->SPmatrix, NULL,
				 (spREAL)PivRel, (spREAL)PivTol, YES) ;
    }
}

/*
 * SMPcaSolve()
 */

void
SMPcaSolve (SMPmatrix *Matrix, double RHS[], double iRHS[], double Spare[], double iSpare[])
{
    printf ("SMPcaSolve\n") ;
    NG_IGNORE (iSpare) ;
    NG_IGNORE (Spare) ;

    spSolveTransposed (Matrix->SPmatrix, RHS, RHS, iRHS, iRHS) ;
}

/*
 * SMPcSolve()
 */

void
SMPcSolve (SMPmatrix *Matrix, double RHS[], double iRHS[], double Spare[], double iSpare[])
{
    NG_IGNORE (iSpare) ;
    NG_IGNORE (Spare) ;

    spSolve (Matrix->SPmatrix, RHS, RHS, iRHS, iRHS) ;
}

/*
 * SMPsolve()
 */

void
SMPsolve (SMPmatrix *Matrix, double RHS[], double Spare[])
{//printf("Solve\n");
    int i, *pExtOrder, status ;
    if (Matrix->CKTumfpackMODE)
    {
	NG_IGNORE (Spare) ;

	pExtOrder = &Matrix->SPmatrix->IntToExtRowMap[Matrix->CKTumfpackN] ;
	for (i = Matrix->CKTumfpackN - 1 ; i >= 0 ; i--)
            Matrix->CKTumfpackIntermediate [i] = RHS [*(pExtOrder--)] ;

	status = umfpack_di_solve (UMFPACK_A, Matrix->CKTumfpackAp, Matrix->CKTumfpackAi, Matrix->CKTumfpackAx,
                                   Matrix->CKTumfpackX, Matrix->CKTumfpackIntermediate, Matrix->CKTumfpackNumeric,
                                   Matrix->CKTumfpackControl, Matrix->CKTumfpackInfo) ;

	umfpack_di_report_info (Matrix->CKTumfpackControl, Matrix->CKTumfpackInfo) ;
	umfpack_di_report_status (Matrix->CKTumfpackControl, status) ;
	if (status < 0) printf ("SMPsolve using UMFPACK error\n") ;

	pExtOrder = &Matrix->SPmatrix->IntToExtColMap[Matrix->CKTumfpackN] ;
	for (i = Matrix->CKTumfpackN - 1 ; i >= 0 ; i--)
            RHS [*(pExtOrder--)] = Matrix->CKTumfpackX [i] ;
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
SMPnewMatrix (SMPmatrix *Matrix)
{
    int Error;
    Matrix->SPmatrix = spCreate (0, 1, &Error) ;
    return Error ;
}

/*
 * SMPdestroy()
 */

void
SMPdestroy (SMPmatrix *Matrix)
{
    spDestroy (Matrix->SPmatrix) ;
}

/*
 * SMPpreOrder()
 */

int
SMPpreOrder (SMPmatrix *Matrix)
{//printf("PreOrder\n");
    int status ;
    if (Matrix->CKTumfpackMODE)
    {
	umfpack_di_defaults (Matrix->CKTumfpackControl) ;
	Matrix->CKTumfpackControl [UMFPACK_SCALE] = UMFPACK_SCALE_MAX ;
	Matrix->CKTumfpackControl [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC ;
	Matrix->CKTumfpackControl [UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD ;
	Matrix->CKTumfpackControl [UMFPACK_SYM_PIVOT_TOLERANCE] = 0.001 ;

	status = umfpack_di_symbolic (Matrix->CKTumfpackN, Matrix->CKTumfpackN, Matrix->CKTumfpackAp, Matrix->CKTumfpackAi, Matrix->CKTumfpackAx,
                                      &(Matrix->CKTumfpackSymbolic), Matrix->CKTumfpackControl, Matrix->CKTumfpackInfo) ;

	if (status < 0) {
	    umfpack_di_report_info (Matrix->CKTumfpackControl, Matrix->CKTumfpackInfo) ;
	    umfpack_di_report_status (Matrix->CKTumfpackControl, status) ;
	    return 1 ;
	}

	return 0 ;

    } else {
	spMNA_Preorder (Matrix->SPmatrix) ;
	return spError (Matrix->SPmatrix) ;
    }
}

/*
 * SMPprint()
 */
/*ARGSUSED*/
void
SMPprint (SMPmatrix *Matrix, FILE *File)
{
    NG_IGNORE (File) ;

    spPrint (Matrix->SPmatrix, 0, 1, 1) ;
}

/*
 * SMPgetError()
 */
void
SMPgetError (SMPmatrix *Matrix, int *Col, int *Row)
{
    spWhereSingular (Matrix->SPmatrix, Row, Col) ;
}

/*
 * SMPcProdDiag()
 *    note: obsolete for Spice3d2 and later
 */
int
SMPcProdDiag (SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent)
{
    spDeterminant (Matrix->SPmatrix, pExponent, &(pMantissa->real), &(pMantissa->imag)) ;
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

    spDeterminant (Matrix->SPmatrix, &p, &re, &im) ;

#ifndef M_LN2
#define M_LN2   0.69314718055994530942
#endif
#ifndef M_LN10
#define M_LN10  2.30258509299404568402
#endif

#ifdef debug_print
    printf("Determinant 10: (%20g,%20g)^%d\n", re, im, p);
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
    printf(" ** base10 -> base2 int =  %g, frac = %20g\n", x, y);
#endif
    z = pow(2.0, y);
    re *= z;
    im *= z;
#ifdef debug_print
    printf(" ** multiplier = %20g\n", z);
#endif

    /* Re-normalize (re or im may be > 2.0 or both < 1.0 */
    if (re != 0.0) {
	y = logb(re);
	if (im != 0.0)
	    z = logb(im);
	else
	    z = 0;
    } else if (im != 0.0) {
	z = logb(im);
	y = 0;
    } else {
	/* Singular */
	/*printf("10 -> singular\n");*/
	y = 0;
	z = 0;
    }

#ifdef debug_print
    printf(" ** renormalize changes = %g,%g\n", y, z);
#endif
    if (y < z)
	y = z;

    *pExponent = (int)(x + y);
    x = scalbn(re, (int) -y);
    z = scalbn(im, (int) -y);
#ifdef debug_print
    printf(" ** values are: re %g, im %g, y %g, re' %g, im' %g\n",
	    re, im, y, x, z);
#endif
    pMantissa->real = scalbn(re, (int) -y);
    pMantissa->imag = scalbn(im, (int) -y);

#ifdef debug_print
    printf("Determinant 10->2: (%20g,%20g)^%d\n", pMantissa->real,
	pMantissa->imag, *pExponent);
#endif
    return spError (Matrix->SPmatrix) ;
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
	if (Gmin != 0.0) {
		for (i = 0 ; i < n ; i++) {
			if (diag [i] != NULL) *(diag [i]) += Gmin ;
		}
	}

	return ;
}

static void
LoadGmin(SMPmatrix *eMatrix, double Gmin)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    int I;
    ArrayOfElementPtrs Diag;
    ElementPtr diag;

    /* Begin `LoadGmin'. */
    assert (IS_SPARSE (Matrix)) ;

    if (Gmin != 0.0) {
	Diag = Matrix->Diag;
	for (I = Matrix->Size; I > 0; I--) {
	    if ((diag = Diag[I]) != NULL)
		diag->Real += Gmin;
	}
    }
    return;
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
SMPfindElt(SMPmatrix *eMatrix, int Row, int Col, int CreateIfMissing)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr Element;

    /* Begin `SMPfindElt'. */
    assert( IS_SPARSE( Matrix ) );
    Row = Matrix->ExtToIntRowMap[Row];
    Col = Matrix->ExtToIntColMap[Col];
    Element = Matrix->FirstInCol[Col];
    Element = spcFindElementInCol(Matrix, &Element, Row, Col, CreateIfMissing);
    return (SMPelement *)Element;
}

/* XXX The following should probably be implemented in spUtils */

/*
 * SMPcZeroCol()
 */
int
SMPcZeroCol(SMPmatrix *eMatrix, int Col)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Element;

    Col = Matrix->ExtToIntColMap[Col];

    for (Element = Matrix->FirstInCol[Col];
	Element != NULL;
	Element = Element->NextInCol)
    {
	Element->Real = 0.0;
	Element->Imag = 0.0;
    }

    return spError( Matrix );
}

/*
 * SMPcAddCol()
 */
int
SMPcAddCol(SMPmatrix *eMatrix, int Accum_Col, int Addend_Col)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Accum, Addend, *Prev;

    Accum_Col = Matrix->ExtToIntColMap[Accum_Col];
    Addend_Col = Matrix->ExtToIntColMap[Addend_Col];

    Addend = Matrix->FirstInCol[Addend_Col];
    Prev = &Matrix->FirstInCol[Accum_Col];
    Accum = *Prev;;

    while (Addend != NULL) {
	while (Accum && Accum->Row < Addend->Row) {
	    Prev = &Accum->NextInCol;
	    Accum = *Prev;
	}
	if (!Accum || Accum->Row > Addend->Row) {
	    Accum = spcCreateElement(Matrix, Addend->Row, Accum_Col, Prev, 0);
	}
	Accum->Real += Addend->Real;
	Accum->Imag += Addend->Imag;
	Addend = Addend->NextInCol;
    }

    return spError( Matrix );
}

/*
 * SMPzeroRow()
 */
int
SMPzeroRow(SMPmatrix *eMatrix, int Row)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Element;

    Row = Matrix->ExtToIntColMap[Row];

    if (Matrix->RowsLinked == NO)
	spcLinkRows(Matrix);

    if (Matrix->PreviousMatrixWasComplex || Matrix->Complex) {
	for (Element = Matrix->FirstInRow[Row];
	    Element != NULL;
	    Element = Element->NextInRow)
	{
	    Element->Real = 0.0;
	    Element->Imag = 0.0;
	}
    } else {
	for (Element = Matrix->FirstInRow[Row];
	    Element != NULL;
	    Element = Element->NextInRow)
	{
	    Element->Real = 0.0;
	}
    }

    return spError( Matrix );
}

#ifdef PARALLEL_ARCH
/*
 * SMPcombine()
 */
void
SMPcombine(SMPmatrix *Matrix, double RHS[], double Spare[])
{
    spSetReal (Matrix->SPmatrix) ;
    spCombine (Matrix->SPmatrix, RHS, Spare, NULL, NULL) ;
}

/*
 * SMPcCombine()
 */
void
SMPcCombine (SMPmatrix *Matrix, double RHS[], double Spare[], double iRHS[], double iSpare[])
{
    spSetComplex (Matrix->SPmatrix) ;
    spCombine (Matrix->SPmatrix, RHS, Spare, iRHS, iSpare) ;
}
#endif /* PARALLEL_ARCH */
