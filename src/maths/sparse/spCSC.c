/* Sparse Matrix to CSC Matrix Conversion Routines
 * Including Dump Routines
 *
 * Author: Francesco Lannutti 2011-2012
 *
 * Instructions:
 * spMatrix_CSC_dump and spRHS_CSC_dump are the dump routines;
 * insert them in a point in your code after that the Sparse Matrix
 * is filled in to dump the whole matrix in the CSC format.
 * To solve correctly the resulting CSC linear system, it's crucial
 * to perform another inversion of the Solution Vector following this code:
 *
 * pExtOrder = IntToExtColMap [n] ;
 * for (i = n - 1 ; i >= 0 ; i--)
 *     RHS [*(pExtOrder--)] = Intermediate [i] ;
 */

/* Includes */
#include "ngspice/spmatrix.h"
#include "spdefs.h"

/* Body */
int
WriteCol_original (MatrixPtr Matrix, int Col, spREAL *CSC_Element, spREAL *CSC_Element_Complex, int *CSC_Row, BindElement *BindSparseCSC, spREAL **diag)
{
    int i ;
    ElementPtr current ;

    i = 0 ;
    current = Matrix->FirstInCol [Col] ;

    while (current != NULL) {
        BindSparseCSC [i].Sparse = (double *)current ;
        BindSparseCSC [i].CSC = &(CSC_Element [i]) ;
        BindSparseCSC [i].CSC_Complex = &(CSC_Element_Complex [2 * i]) ;
        CSC_Row [i] = (current->Row) - 1 ;
        if (CSC_Row [i] == Col - 1)
            diag [0] = &(CSC_Element [i]) ;
        i++ ;
        current = current->NextInCol ;
    }

    return i ;
}

int
WriteCol_original_dump (MatrixPtr Matrix, int Col, spREAL *CSC_Element, int *CSC_Row, unsigned int complex)
{
    int i ;
    ElementPtr current ;
    i = 0 ;
    current = Matrix->FirstInCol [Col] ;

    while (current != NULL)
    {
        if (complex)
        {
            CSC_Element [2 * i] = current->Real ;
            CSC_Element [2 * i + 1] = current->Imag ;
        } else {
            CSC_Element [i] = current->Real ;
        }
        CSC_Row [i] = (current->Row) - 1 ;
        i++ ;
        current = current->NextInCol ;
    }

    return i ;
}

void
spMatrix_CSC (MatrixPtr Matrix, int *Ap, int *Ai, double *Ax, double *Ax_Complex, int n, BindElement *BindSparseCSC, double **diag)
{
    int offset, i ;

    offset = 0 ;
    Ap[0] = offset ;
    for (i = 1 ; i <= n ; i++) {
        offset += WriteCol_original (Matrix, i, (spREAL *)(Ax + offset), (spREAL *)(Ax_Complex + 2 * offset),
                                     (int *)(Ai + offset), BindSparseCSC + offset, (spREAL **)(diag + (i - 1))) ;

        Ap[i] = offset ;
    }
}

void
spMatrix_CSC_dump (MatrixPtr Matrix, unsigned int complex, char *CSC_output)
{
    FILE *output ;
    int offset, i, j, *Ap, *Ai, n, nz ;
    double *Ax ;

    n = spGetSize (Matrix, 1) ;
    nz = Matrix->Elements ;
    Ap = (int *) SP_MALLOC (int, n + 1) ;
    Ai = (int *) SP_MALLOC (int, nz) ;
    if (complex)
    {
        Ax = (double *) SP_MALLOC (double, 2 * nz) ;
    } else {
        Ax = (double *) SP_MALLOC (double, nz) ;
    }

    offset = 0 ;
    Ap [0] = offset ;
    for (i = 1 ; i <= n ; i++) {
        offset += WriteCol_original_dump (Matrix, i, (spREAL *)(Ax + offset), (int *)(Ai + offset), complex) ;
        Ap [i] = offset ;
    }

    if (!complex && CSC_output != NULL)
    {
        output = fopen (CSC_output, "w") ;
        fprintf (output, "%%%%MatrixMarket matrix coordinate real general\n") ;
        fprintf (output, "%%-------------------------------------------------------------------------------\n") ;
        fprintf (output, "%% Transient Matrix Dump\n%% Family: ISCAS Circuit\n") ;
        fprintf (output, "%%-------------------------------------------------------------------------------\n") ;
        fprintf (output, "%d %d %d\n", n, n, offset) ;
        for (i = 0 ; i < n ; i++)
            for (j = Ap [i] ; j < Ap [i + 1] ; j++)
                fprintf (output, "%d %d %-.9g\n", Ai [j] + 1, i + 1, Ax [j]) ;
        fclose (output) ;

        output = fopen ("IntToExtColMap.txt", "w") ;
        for (i = 1 ; i <= n ; i++)
            fprintf (output, "%d\n", Matrix->IntToExtColMap [i]) ;
        fclose (output) ;
    } else {
        fprintf (stderr, "CSC Matrix converted from SPARSE 1.3 matrix\n") ;
        for (i = 0 ; i < n ; i++)
        {
            for (j = Ap [i] ; j < Ap [i + 1] ; j++)
            {
                if (complex)
                {
                    fprintf (stderr, "Row: %d\tCol: %d\tValue: %-.9g j%-.9g\n", Ai [j] + 1, i + 1, Ax [2 * j], Ax [2 * j + 1]) ;
                } else {
                    fprintf (stderr, "Row: %d\tCol: %d\tValue: %-.9g\n", Ai [j] + 1, i + 1, Ax [j]) ;
                }
            }
        }
    }

    SP_FREE (Ap) ;
    SP_FREE (Ai) ;
    SP_FREE (Ax) ;
}

void
spRHS_CSC_dump (RealNumber *RHS, char *CSC_output_b, MatrixPtr Matrix)
{
    FILE *output ;
    int i, n, *pExtOrder ;
    double *Intermediate ;

    n = spGetSize (Matrix, 1) ;
    Intermediate = (double *) SP_MALLOC (double, n) ;

    pExtOrder = &Matrix->IntToExtRowMap [n] ;
    for (i = n - 1 ; i >= 0 ; i--)
        Intermediate [i] = RHS [*(pExtOrder--)] ;

    output = fopen (CSC_output_b, "w") ;
    fprintf (output, "%%%%MatrixMarket matrix array real general\n") ;
    fprintf (output, "%%-------------------------------------------------------------------------------\n") ;
    fprintf (output, "%% Transient RHS Vector Dump\n%% Family: ISCAS Circuit\n") ;
    fprintf (output, "%%-------------------------------------------------------------------------------\n") ;
    fprintf (output, "%d %d\n", n, 1) ;
    for (i = 1 ; i < n + 1 ; i++)
        fprintf (output, "%-.9g\n", Intermediate [i]) ;
    fclose (output) ;

    SP_FREE (Intermediate) ;
}
