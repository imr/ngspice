/*
 *  MATRIX SOLVE MODULE
 *
 *  Author:                     Advising professor:
 *      Kenneth S. Kundert          Alberto Sangiovanni-Vincentelli
 *      UC Berkeley
 *
 *  This file contains the forward and backward substitution routines for
 *  the sparse matrix routines.
 *
 *  >>> User accessible functions contained in this file:
 *  spSolve
 *  spSolveTransposed
 *
 *  >>> Other functions contained in this file:
 *  SolveComplexMatrix
 *  SolveComplexTransposedMatrix
 */


/*
 *  Revision and copyright information.
 *
 *  Copyright (c) 1985,86,87,88,89,90
 *  by Kenneth S. Kundert and the University of California.
 *
 *  Permission to use, copy, modify, and distribute this software and
 *  its documentation for any purpose and without fee is hereby granted,
 *  provided that the copyright notices appear in all copies and
 *  supporting documentation and that the authors and the University of
 *  California are properly credited.  The authors and the University of
 *  California make no representations as to the suitability of this
 *  software for any purpose.  It is provided `as is', without express
 *  or implied warranty.
 */

/*
 *  IMPORTS
 *
 *  >>> Import descriptions:
 *  spConfig.h
 *     Macros that customize the sparse matrix routines.
 *  spMatrix.h
 *     Macros and declarations to be imported by the user.
 *  spDefs.h
 *     Matrix type and macro definitions for the sparse matrix routines.
 */
#include <assert.h>

#define spINSIDE_SPARSE
#include "spconfig.h"
#include "ngspice/spmatrix.h"
#include "spdefs.h"




/*
 * Function declarations
 */

static void SolveComplexMatrix( MatrixPtr,
                        RealVector, RealVector, RealVector, RealVector );
static void SolveComplexTransposedMatrix( MatrixPtr,
                        RealVector, RealVector, RealVector, RealVector );







/*
 *  SOLVE MATRIX EQUATION
 *
 *  Performs forward elimination and back substitution to find the
 *  unknown vector from the RHS vector and factored matrix.  This
 *  routine assumes that the pivots are associated with the lower
 *  triangular (L) matrix and that the diagonal of the upper triangular
 *  (U) matrix consists of ones.  This routine arranges the computation
 *  in different way than is traditionally used in order to exploit the
 *  sparsity of the right-hand side.  See the reference in spRevision.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (char *)
 *      Pointer to matrix.
 *  RHS  <input>  (RealVector)
 *      RHS is the input data array, the right hand side. This data is
 *      undisturbed and may be reused for other solves.
 *  Solution  <output>  (RealVector)
 *      Solution is the output data array. This routine is constructed such that
 *      RHS and Solution can be the same array.
 *  iRHS  <input>  (RealVector)
 *      iRHS is the imaginary portion of the input data array, the right
 *      hand side. This data is undisturbed and may be reused for other solves.
 *  iSolution  <output>  (RealVector)
 *      iSolution is the imaginary portion of the output data array. This
 *      routine is constructed such that iRHS and iSolution can be
 *      the same array.
 *
 *  >>> Local variables:
 *  Intermediate  (RealVector)
 *      Temporary storage for use in forward elimination and backward
 *      substitution.  Commonly referred to as c, when the LU factorization
 *      equations are given as  Ax = b, Lc = b, Ux = c Local version of
 *      Matrix->Intermediate, which was created during the initial
 *      factorization in function spcCreateInternalVectors() in the matrix
 *      factorization module.
 *  pElement  (ElementPtr)
 *      Pointer used to address elements in both the lower and upper triangle
 *      matrices.
 *  pExtOrder  (int *)
 *      Pointer used to sequentially access each entry in IntToExtRowMap
 *      and IntToExtColMap arrays.  Used to quickly scramble and unscramble
 *      RHS and Solution to account for row and column interchanges.
 *  pPivot  (ElementPtr)
 *      Pointer that points to current pivot or diagonal element.
 *  Size  (int)
 *      Size of matrix. Made local to reduce indirection.
 *  Temp  (RealNumber)
 *      Temporary storage for entries in arrays.
 */

/*VARARGS3*/

void
spSolve(MatrixPtr Matrix, RealVector RHS, RealVector Solution,
	RealVector iRHS, RealVector iSolution)
{
    ElementPtr  pElement;
    RealVector  Intermediate;
    RealNumber  Temp;
    int  I, *pExtOrder, Size;
    ElementPtr  pPivot;

    /* Begin `spSolve'. */
    assert( IS_VALID(Matrix) && IS_FACTORED(Matrix) );

    if (Matrix->Complex)
    {
	SolveComplexMatrix( Matrix, RHS, Solution, iRHS, iSolution );
        return;
    }

    Intermediate = Matrix->Intermediate;
    Size = Matrix->Size;

    /* Initialize Intermediate vector. */
    pExtOrder = &Matrix->IntToExtRowMap[Size];
    for (I = Size; I > 0; I--)
        Intermediate[I] = RHS[*(pExtOrder--)];

    /* Forward elimination. Solves Lc = b.*/
    for (I = 1; I <= Size; I++)
    {
   
	/* This step of the elimination is skipped if Temp equals zero. */
        if ((Temp = Intermediate[I]) != 0.0)
        {
	    pPivot = Matrix->Diag[I];
            Intermediate[I] = (Temp *= pPivot->Real);

            pElement = pPivot->NextInCol;
            while (pElement != NULL)
            {
		Intermediate[pElement->Row] -= Temp * pElement->Real;
                pElement = pElement->NextInCol;
            }
        }
    }

    /* Backward Substitution. Solves Ux = c.*/
    for (I = Size; I > 0; I--)
    {
	Temp = Intermediate[I];
        pElement = Matrix->Diag[I]->NextInRow;
        while (pElement != NULL)
        {
	    Temp -= pElement->Real * Intermediate[pElement->Col];
            pElement = pElement->NextInRow;
        }
        Intermediate[I] = Temp;
    }

    /* Unscramble Intermediate vector while placing data in to Solution vector. */
    pExtOrder = &Matrix->IntToExtColMap[Size];
    for (I = Size; I > 0; I--)
        Solution[*(pExtOrder--)] = Intermediate[I];

    return;
}











/*
 *  SOLVE COMPLEX MATRIX EQUATION
 *
 *  Performs forward elimination and back substitution to find the
 *  unknown vector from the RHS vector and factored matrix.  This
 *  routine assumes that the pivots are associated with the lower
 *  triangular (L) matrix and that the diagonal of the upper triangular
 *  (U) matrix consists of ones.  This routine arranges the computation
 *  in different way than is traditionally used in order to exploit the
 *  sparsity of the right-hand side.  See the reference in spRevision.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (char *)
 *      Pointer to matrix.
 *  RHS  <input>  (RealVector)
 *      RHS is the real portion of the input data array, the right hand
 *      side. This data is undisturbed and may be reused for other solves.
 *  Solution  <output>  (RealVector)
 *      Solution is the real portion of the output data array. This routine
 *      is constructed such that RHS and Solution can be the same
 *      array.
 *  iRHS  <input>  (RealVector)
 *      iRHS is the imaginary portion of the input data array, the right
 *      hand side. This data is undisturbed and may be reused for other solves.
 *      If spSEPARATED_COMPLEX_VECTOR is set FALSE, there is no need to
 *      supply this array.
 *  iSolution  <output>  (RealVector)
 *      iSolution is the imaginary portion of the output data array. This
 *      routine is constructed such that iRHS and iSolution can be
 *      the same array.  If spSEPARATED_COMPLEX_VECTOR is set FALSE, there is no
 *      need to supply this array.
 *
 *  >>> Local variables:
 *  Intermediate  (ComplexVector)
 *      Temporary storage for use in forward elimination and backward
 *      substitution.  Commonly referred to as c, when the LU factorization
 *      equations are given as  Ax = b, Lc = b, Ux = c.
 *      Local version of Matrix->Intermediate, which was created during
 *      the initial factorization in function spcCreateInternalVectors() in the
 *      matrix factorization module.
 *  pElement  (ElementPtr)
 *      Pointer used to address elements in both the lower and upper triangle
 *      matrices.
 *  pExtOrder  (int *)
 *      Pointer used to sequentially access each entry in IntToExtRowMap
 *      and IntToExtColMap arrays.  Used to quickly scramble and unscramble
 *      RHS and Solution to account for row and column interchanges.
 *  pPivot  (ElementPtr)
 *      Pointer that points to current pivot or diagonal element.
 *  Size  (int)
 *      Size of matrix. Made local to reduce indirection.
 *  Temp  (ComplexNumber)
 *      Temporary storage for entries in arrays.
 */

static void
SolveComplexMatrix( MatrixPtr Matrix, RealVector RHS, RealVector Solution , RealVector iRHS, RealVector iSolution )
{
    ElementPtr  pElement;
    ComplexVector  Intermediate;
    int  I, *pExtOrder, Size;
    ElementPtr  pPivot;
    ComplexNumber  Temp;

    /* Begin `SolveComplexMatrix'. */

    Size = Matrix->Size;
    Intermediate = (ComplexVector)Matrix->Intermediate;

    /* Initialize Intermediate vector. */
    pExtOrder = &Matrix->IntToExtRowMap[Size];

    for (I = Size; I > 0; I--)
    {
	Intermediate[I].Real = RHS[*(pExtOrder)];
        Intermediate[I].Imag = iRHS[*(pExtOrder--)];
    }

    /* Forward substitution. Solves Lc = b.*/
    for (I = 1; I <= Size; I++)
    {
	Temp = Intermediate[I];

	/* This step of the substitution is skipped if Temp equals zero. */
        if ((Temp.Real != 0.0) || (Temp.Imag != 0.0))
        {
	    pPivot = Matrix->Diag[I];
	    /* Cmplx expr: Temp *= (1.0 / Pivot). */
            CMPLX_MULT_ASSIGN(Temp, *pPivot);
            Intermediate[I] = Temp;
            pElement = pPivot->NextInCol;
            while (pElement != NULL)
            {
		/* Cmplx expr: Intermediate[Element->Row] -= Temp * *Element. */
                CMPLX_MULT_SUBT_ASSIGN(Intermediate[pElement->Row],
                                       Temp, *pElement);
                pElement = pElement->NextInCol;
            }
        }
    }

    /* Backward Substitution. Solves Ux = c.*/
    for (I = Size; I > 0; I--)
    {
	Temp = Intermediate[I];
        pElement = Matrix->Diag[I]->NextInRow;

        while (pElement != NULL)
        {
	    /* Cmplx expr: Temp -= *Element * Intermediate[Element->Col]. */
            CMPLX_MULT_SUBT_ASSIGN(Temp, *pElement,Intermediate[pElement->Col]);
            pElement = pElement->NextInRow;
        }
        Intermediate[I] = Temp;
    }

    /* Unscramble Intermediate vector while placing data in to Solution vector. */
    pExtOrder = &Matrix->IntToExtColMap[Size];

    for (I = Size; I > 0; I--)
    {
	Solution[*(pExtOrder)] = Intermediate[I].Real;
        iSolution[*(pExtOrder--)] = Intermediate[I].Imag;
    }

    return;
}














#if TRANSPOSE
/*
 *  SOLVE TRANSPOSED MATRIX EQUATION
 *
 *  Performs forward elimination and back substitution to find the
 *  unknown vector from the RHS vector and transposed factored
 *  matrix. This routine is useful when performing sensitivity analysis
 *  on a circuit using the adjoint method.  This routine assumes that
 *  the pivots are associated with the untransposed lower triangular
 *  (L) matrix and that the diagonal of the untransposed upper
 *  triangular (U) matrix consists of ones.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (char *)
 *      Pointer to matrix.
 *  RHS  <input>  (RealVector)
 *      RHS is the input data array, the right hand side. This data is
 *      undisturbed and may be reused for other solves.
 *  Solution  <output>  (RealVector)
 *      Solution is the output data array. This routine is constructed such that
 *      RHS and Solution can be the same array.
 *  iRHS  <input>  (RealVector)
 *      iRHS is the imaginary portion of the input data array, the right
 *      hand side. This data is undisturbed and may be reused for other solves.
 *      If spSEPARATED_COMPLEX_VECTOR is set FALSE, or if matrix is real, there
 *      is no need to supply this array.
 *  iSolution  <output>  (RealVector)
 *      iSolution is the imaginary portion of the output data array. This
 *      routine is constructed such that iRHS and iSolution can be
 *      the same array.  If spSEPARATED_COMPLEX_VECTOR is set FALSE, or if
 *      matrix is real, there is no need to supply this array.
 *
 *  >>> Local variables:
 *  Intermediate  (RealVector)
 *      Temporary storage for use in forward elimination and backward
 *      substitution.  Commonly referred to as c, when the LU factorization
 *      equations are given as  Ax = b, Lc = b, Ux = c.  Local version of
 *      Matrix->Intermediate, which was created during the initial
 *      factorization in function spcCreateInternalVectors() in the matrix
 *      factorization module.
 *  pElement  (ElementPtr)
 *      Pointer used to address elements in both the lower and upper triangle
 *      matrices.
 *  pExtOrder  (int *)
 *      Pointer used to sequentially access each entry in IntToExtRowMap
 *      and IntToExtRowMap arrays.  Used to quickly scramble and unscramble
 *      RHS and Solution to account for row and column interchanges.
 *  pPivot  (ElementPtr)
 *      Pointer that points to current pivot or diagonal element.
 *  Size  (int)
 *      Size of matrix. Made local to reduce indirection.
 *  Temp  (RealNumber)
 *      Temporary storage for entries in arrays.
 */

/*VARARGS3*/

void
spSolveTransposed(MatrixPtr Matrix, RealVector RHS, RealVector Solution,
		  RealVector iRHS, RealVector iSolution)
{
    ElementPtr  pElement;
    RealVector  Intermediate;
    int  I, *pExtOrder, Size;
    ElementPtr  pPivot;
    RealNumber  Temp;

    /* Begin `spSolveTransposed'. */
    assert( IS_VALID(Matrix) && IS_FACTORED(Matrix) );

    if (Matrix->Complex)
    {
	SolveComplexTransposedMatrix( Matrix, RHS, Solution , iRHS, iSolution );
        return;
    }

    Size = Matrix->Size;
    Intermediate = Matrix->Intermediate;

    /* Initialize Intermediate vector. */
    pExtOrder = &Matrix->IntToExtColMap[Size];
    for (I = Size; I > 0; I--)
        Intermediate[I] = RHS[*(pExtOrder--)];

    /* Forward elimination. */
    for (I = 1; I <= Size; I++)
    {
   
	/* This step of the elimination is skipped if Temp equals zero. */
        if ((Temp = Intermediate[I]) != 0.0)
        {
	    pElement = Matrix->Diag[I]->NextInRow;
            while (pElement != NULL)
            {
		Intermediate[pElement->Col] -= Temp * pElement->Real;
                pElement = pElement->NextInRow;
            }

        }
    }

    /* Backward Substitution. */
    for (I = Size; I > 0; I--)
    {
	pPivot = Matrix->Diag[I];
        Temp = Intermediate[I];
        pElement = pPivot->NextInCol;
        while (pElement != NULL)
        {
	    Temp -= pElement->Real * Intermediate[pElement->Row];
            pElement = pElement->NextInCol;
        }
        Intermediate[I] = Temp * pPivot->Real;
    }

    /* Unscramble Intermediate vector while placing data in to
       Solution vector. */
    pExtOrder = &Matrix->IntToExtRowMap[Size];
    for (I = Size; I > 0; I--)
        Solution[*(pExtOrder--)] = Intermediate[I];

    return;
}
#endif /* TRANSPOSE */










#if TRANSPOSE
/*
 *  SOLVE COMPLEX TRANSPOSED MATRIX EQUATION
 *
 *  Performs forward elimination and back substitution to find the
 *  unknown vector from the RHS vector and transposed factored
 *  matrix. This routine is useful when performing sensitivity analysis
 *  on a circuit using the adjoint method.  This routine assumes that
 *  the pivots are associated with the untransposed lower triangular
 *  (L) matrix and that the diagonal of the untransposed upper
 *  triangular (U) matrix consists of ones.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (char *)
 *      Pointer to matrix.
 *  RHS  <input>  (RealVector)
 *      RHS is the input data array, the right hand
 *      side. This data is undisturbed and may be reused for other solves.
 *      This vector is only the real portion if the matrix is complex.
 *  Solution  <output>  (RealVector)
 *      Solution is the real portion of the output data array. This routine
 *      is constructed such that RHS and Solution can be the same array.
 *      This vector is only the real portion if the matrix is complex.
 *  iRHS  <input>  (RealVector)
 *      iRHS is the imaginary portion of the input data array, the right
 *      hand side. This data is undisturbed and may be reused for other solves.
 *  iSolution  <output>  (RealVector)
 *      iSolution is the imaginary portion of the output data array. This
 *      routine is constructed such that iRHS and iSolution can be
 *      the same array.
 *
 *  >>> Local variables:
 *  Intermediate  (ComplexVector)
 *      Temporary storage for use in forward elimination and backward
 *      substitution.  Commonly referred to as c, when the LU factorization
 *      equations are given as  Ax = b, Lc = b, Ux = c.  Local version of
 *      Matrix->Intermediate, which was created during
 *      the initial factorization in function spcCreateInternalVectors() in the
 *      matrix factorization module.
 *  pElement  (ElementPtr)
 *      Pointer used to address elements in both the lower and upper triangle
 *      matrices.
 *  pExtOrder  (int *)
 *      Pointer used to sequentially access each entry in IntToExtRowMap
 *      and IntToExtColMap arrays.  Used to quickly scramble and unscramble
 *      RHS and Solution to account for row and column interchanges.
 *  pPivot  (ElementPtr)
 *      Pointer that points to current pivot or diagonal element.
 *  Size  (int)
 *      Size of matrix. Made local to reduce indirection.
 *  Temp  (ComplexNumber)
 *      Temporary storage for entries in arrays.
 */

static void
SolveComplexTransposedMatrix(MatrixPtr Matrix, RealVector RHS, RealVector Solution , RealVector iRHS, RealVector iSolution )
{
    ElementPtr  pElement;
    ComplexVector  Intermediate;
    int  I, *pExtOrder, Size;
    ElementPtr  pPivot;
    ComplexNumber  Temp;

    /* Begin `SolveComplexTransposedMatrix'. */

    Size = Matrix->Size;
    Intermediate = (ComplexVector)Matrix->Intermediate;

    /* Initialize Intermediate vector. */
    pExtOrder = &Matrix->IntToExtColMap[Size];

    for (I = Size; I > 0; I--)
    {
	Intermediate[I].Real = RHS[*(pExtOrder)];
        Intermediate[I].Imag = iRHS[*(pExtOrder--)];
    }

    /* Forward elimination. */
    for (I = 1; I <= Size; I++)
    {
	Temp = Intermediate[I];

	/* This step of the elimination is skipped if Temp equals zero. */
        if ((Temp.Real != 0.0) || (Temp.Imag != 0.0))
        {
	    pElement = Matrix->Diag[I]->NextInRow;
            while (pElement != NULL)
            {
		/* Cmplx expr: Intermediate[Element->Col] -= Temp * *Element. */
                CMPLX_MULT_SUBT_ASSIGN( Intermediate[pElement->Col],
                                        Temp, *pElement);
                pElement = pElement->NextInRow;
            }
        }
    }

    /* Backward Substitution. */
    for (I = Size; I > 0; I--)
    {
	pPivot = Matrix->Diag[I];
        Temp = Intermediate[I];
        pElement = pPivot->NextInCol;

        while (pElement != NULL)
        {
	    /* Cmplx expr: Temp -= Intermediate[Element->Row] * *Element. */
            CMPLX_MULT_SUBT_ASSIGN(Temp,Intermediate[pElement->Row],*pElement);

            pElement = pElement->NextInCol;
        }
	/* Cmplx expr: Intermediate = Temp * (1.0 / *pPivot). */
        CMPLX_MULT(Intermediate[I], Temp, *pPivot);
    }

    /* Unscramble Intermediate vector while placing data in to
       Solution vector. */
    pExtOrder = &Matrix->IntToExtRowMap[Size];

    for (I = Size; I > 0; I--)
    {
	Solution[*(pExtOrder)] = Intermediate[I].Real;
        iSolution[*(pExtOrder--)] = Intermediate[I].Imag;
    }

    return;
}
#endif /* TRANSPOSE */
