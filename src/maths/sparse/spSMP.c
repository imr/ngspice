/*
 *  Spice3 COMPATIBILITY MODULE
 *
 *  Author:                     Advising professor:
 *     Kenneth S. Kundert           Alberto Sangiovanni-Vincentelli
 *     UC Berkeley
 *
 *  This module contains routines that make Sparse1.4 a direct
 *  replacement for the SMP sparse matrix package in Spice3c1 and Spice3d1.
 *  Sparse1.4 is in general a faster and more robust package than SMP.
 *  These advantages become significant on large circuits.
 *
 *  This module is provided for convience only. It has not been tested
 *  with the recent version of Spice3 and is not supported.
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
 *      REAL                            YES
 *      EXPANDABLE                      YES
 *      TRANSLATE                       NO
 *      INITIALIZE                      NO or YES, YES for use with test prog.
 *      DIAGONAL_PIVOTING               YES
 *      ARRAY_OFFSET                    YES
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
 *      FORTRAN                         NO
 *      DEBUG                           YES
 *      spCOMPLEX                       1
 *      spSEPARATED_COMPLEX_VECTORS     1
 *
 *      spREAL  double
 */

/*
 *  Revision and copyright information.
 *
 *  Copyright (c) 1985-2003 by Kenneth S. Kundert
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

#include "ngspice/spmatrix.h"
#include "ngspice/smpdefs.h"

#define NO   0
#define YES  1

typedef  spREAL  RealNumber, *RealVector;

static void LoadGmin(char *Matrix, double Gmin);

/*
 * SMPaddElt()
 */
int
SMPaddElt( Matrix, Row, Col, Value )
SMPmatrix *Matrix;
int Row, Col;
double Value;
{
    *spGetElement( (char *)Matrix, Row, Col ) = Value;
    return spErrorState( (char *)Matrix );
}

/*
 * SMPmakeElt()
 */
double *
SMPmakeElt( Matrix, Row, Col )
SMPmatrix *Matrix;
int Row, Col;
{
    return spGetElement( (char *)Matrix, Row, Col );
}

/*
 * SMPcClear()
 */
void
SMPcClear( Matrix )
SMPmatrix *Matrix;
{
    spClear( (char *)Matrix );
}

/*
 * SMPclear()
 */
void
SMPclear( Matrix )
SMPmatrix *Matrix;
{
    spClear( (char *)Matrix );
}

/*
 * SMPcLUfac()
 */
/*ARGSUSED*/
int
SMPcLUfac( Matrix, PivTol )
SMPmatrix *Matrix;
double PivTol;
{
    spSetComplex( (char *)Matrix );
    return spFactor( (char *)Matrix );
}

/*
 * SMPluFac()
 */
/*ARGSUSED*/
int
SMPluFac( Matrix, PivTol, Gmin )
SMPmatrix *Matrix;
double PivTol, Gmin;
{
    spSetReal( (char *)Matrix );
    LoadGmin( (char *)Matrix, Gmin );
    return spFactor( (char *)Matrix );
}

/*
 * SMPcReorder()
 */
int
SMPcReorder( Matrix, PivTol, PivRel, NumSwaps )
SMPmatrix *Matrix;
double PivTol, PivRel;
int *NumSwaps;
{
    *NumSwaps = 0;
    spSetComplex( (char *)Matrix );
    return spOrderAndFactor( (char *)Matrix, (spREAL*)NULL,
                             (spREAL)PivRel, (spREAL)PivTol, YES );
}

/*
 * SMPreorder()
 */
int
SMPreorder( Matrix, PivTol, PivRel, Gmin )
SMPmatrix *Matrix;
double PivTol, PivRel, Gmin;
{
    spSetComplex( (char *)Matrix );
    LoadGmin( (char *)Matrix, Gmin );
    return spOrderAndFactor( (char *)Matrix, (spREAL*)NULL,
                             (spREAL)PivRel, (spREAL)PivTol, YES );
}

/*
 * SMPcaSolve()
 */
void
SMPcaSolve( Matrix, RHS, iRHS, Spare, iSpare)
SMPmatrix *Matrix;
double RHS[], iRHS[], Spare[], iSpare[];
{
    spSolveTransposed( (char *)Matrix, RHS, RHS, iRHS, iRHS );
}

/*
 * SMPcSolve()
 */
void
SMPcSolve( Matrix, RHS, iRHS, Spare, iSpare)
SMPmatrix *Matrix;
double RHS[], iRHS[], Spare[], iSpare[];
{
    spSolve( (char *)Matrix, RHS, RHS, iRHS, iRHS );
}

/*
 * SMPsolve()
 */
void
SMPsolve( Matrix, RHS, Spare )
SMPmatrix *Matrix;
double RHS[], Spare[];
{
    spSolve( (char *)Matrix, RHS, RHS, (spREAL*)NULL, (spREAL*)NULL );
}

/*
 * SMPmatSize()
 */
int
SMPmatSize( Matrix )
SMPmatrix *Matrix;
{
    return spGetSize( (char *)Matrix, 1 );
}

/*
 * SMPnewMatrix()
 */
int
SMPnewMatrix( pMatrix, dummy )
SMPmatrix **pMatrix;
int dummy;
{
int Error;
    *pMatrix = (SMPmatrix *)spCreate( 0, 1, &Error );
    return Error;
}

/*
 * SMPdestroy()
 */
void
SMPdestroy( Matrix )
SMPmatrix *Matrix;
{
    spDestroy( (char *)Matrix );
}

/*
 * SMPpreOrder()
 */
int
SMPpreOrder( Matrix )
SMPmatrix *Matrix;
{
    spMNA_Preorder( (char *)Matrix );
    return spErrorState( (char *)Matrix );
}

/*
 * SMPprintRHS()
 */
void
SMPprintRHS(SMPmatrix *Matrix, char *Filename, RealVector RHS, RealVector iRHS)
{
    spFileVector( Matrix, Filename, RHS, iRHS );
}

/*
 * SMPprint()
 */
/*ARGSUSED*/
void
SMPprint( Matrix, File )
SMPmatrix *Matrix;
char *File;
{
    spPrint( (char *)Matrix, 0, 1, 1 );
}

/*
 * SMPgetError()
 */
void
SMPgetError( Matrix, Col, Row)
SMPmatrix *Matrix;
int *Row, *Col;
{
    spWhereSingular( (char *)Matrix, Row, Col );
}

/*
 * SMPcProdDiag()
 */
int
SMPcProdDiag( Matrix, pMantissa, pExponent)
SMPmatrix *Matrix;
SPcomplex *pMantissa;
int *pExponent;
{
    spDeterminant( (char *)Matrix, pExponent, &(pMantissa->real),
                                              &(pMantissa->imag) );
    return spErrorState( (char *)Matrix );
}

/*
 * SMPcDProd()
 */
int
SMPcDProd(SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent)
{
    double  re, im, x, y, z;
    int     p;

    spDeterminant( Matrix, &p, &re, &im);

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
     *  x = integral part of exponent, y = fraction part of exponent
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
    return spErrorState( Matrix );
}


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
#include "spDefs.h"
void
LoadGmin( eMatrix, Gmin )
char *eMatrix;
register double Gmin;
{
MatrixPtr Matrix = (MatrixPtr)eMatrix;
register int I;
register ArrayOfElementPtrs Diag;

/* Begin `spLoadGmin'. */
    ASSERT_IS_SPARSE( Matrix );

    Diag = Matrix->Diag;
    for (I = Matrix->Size; I > 0; I--)
        Diag[I]->Real += Gmin;
    return;
}





/*
 *  FIND ELEMENT
 *
 *  This routine finds an element in the matrix by row and column number.
 *  If the element exists, a pointer to it is returned.  If not, then NULL
 *  is returned unless the CreateIfMissing flag is true, in which case a
 *  pointer to the new element is returned.
 */

//SMPelement *
//SMPfindElt( Matrix, Row, Col, CreateIfMissing )
//
//SMPmatrix *Matrix;
//int Row, Col;
//int CreateIfMissing;
//{
////MatrixPtr Matrix = (MatrixPtr)eMatrix;
//spREAL *Element = (spREAL *)Matrix->FirstInCol[Col];
//
///* Begin `SMPfindElt'. */
//    ASSERT_IS_SPARSE( Matrix );
//    if (CreateIfMissing)
//    {   Element = spcCreateElement( Matrix, Row, Col,
//                    &Matrix->FirstInRow[Row],
//                    &Matrix->FirstInCol[Col], NO );
//    }
//    else Element = spcFindElement( Matrix, Row, Col );
//    return (SMPelement *)Element;
//}

SMPelement *
SMPfindElt(SMPmatrix *Matrix, int Row, int Col, int CreateIfMissing)
{
    ElementPtr Element;

    /* Begin `SMPfindElt'. */
    ASSERT_IS_SPARSE( Matrix );
    Row = Matrix->ExtToIntRowMap[Row];

    Col = Matrix->ExtToIntColMap[Col];

    if (Col == -1)
    /* No element available */
        return NULL;

    Element = Matrix->FirstInCol[Col];
    Element = spcFindElementInCol(Matrix, &Element, Row, Col, CreateIfMissing);
    return Element;
}


/*
 * SMPcZeroCol()
 */
int
SMPcZeroCol(SMPmatrix *Matrix, int Col)
{
    ElementPtr  Element;

    Col = Matrix->ExtToIntColMap[Col];

    for (Element = Matrix->FirstInCol[Col];
    Element != NULL;
    Element = Element->NextInCol)
    {
    Element->Real = 0.0;
    Element->Imag = 0.0;
    }

    return spErrorState( Matrix );
}

/*
 * SMPcAddCol()
 */
int
SMPcAddCol(SMPmatrix *Matrix, int Accum_Col, int Addend_Col)
{
    ElementPtr  Accum, Addend, *Prev;

    Accum_Col = Matrix->ExtToIntColMap[Accum_Col];
    Addend_Col = Matrix->ExtToIntColMap[Addend_Col];

    Addend = Matrix->FirstInCol[Addend_Col];
    Prev = &Matrix->FirstInCol[Accum_Col];
    Accum = *Prev;

    while (Addend != NULL) {
    while (Accum && Accum->Row < Addend->Row) {
        Prev = &Accum->NextInCol;
        Accum = *Prev;
    }
    if (!Accum || Accum->Row > Addend->Row) {
        Accum = spcCreateElement(Matrix, Addend->Row, Accum_Col, Prev, 0, 0);
    }
    Accum->Real += Addend->Real;
    Accum->Imag += Addend->Imag;
    Addend = Addend->NextInCol;
    }

    return spErrorState( Matrix );
}


/*
 * SMPmultiply()
 */
void
SMPmultiply(SMPmatrix *Matrix, double *RHS, double *Solution, double *iRHS, double *iSolution)
{
    spMultiply(Matrix, RHS, Solution, iRHS, iSolution);
}

/*
 * SMPconstMult()
 */
void
SMPconstMult(SMPmatrix *Matrix, double constant)
{
    spConstMult(Matrix, constant);
}
