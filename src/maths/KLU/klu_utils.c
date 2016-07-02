#include "klu_internal.h"

typedef struct sElement {
  Int row ;
  Int col ;
  Entry val ;
} Element ;

static int
CompareRow (const void *a, const void *b)
{
    Element *A = (Element *) a ;
    Element *B = (Element *) b ;

    return
        (A->row > B->row) ?  1 :
        (A->row < B->row) ? -1 :
        0 ;
}

static int
CompareColumn (const void *a, const void *b)
{
    Element *A = (Element *) a ;
    Element *B = (Element *) b ;

    return
        (A->col > B->col) ?  1 :
        (A->col < B->col) ? -1 :
        0 ;
}

static void
Compress (Int *Ai, Int *Bp, int num_rows, int n_COO)
{
    int i, j ;

    for (i = 0 ; i <= Ai [0] ; i++)
        Bp [i] = 0 ;

    j = Ai [0] + 1 ;
    for (i = 1 ; i < n_COO ; i++)
    {
        if (Ai [i] == Ai [i - 1] + 1)
        {
            Bp [j] = i ;
            j++ ;
        }
        else if (Ai [i] > Ai [i - 1] + 1)
        {
            for ( ; j <= Ai [i] ; j++)
                Bp [j] = i ;
        }
    }

    for ( ; j <= num_rows ; j++)
        Bp [j] = i ;
}

Int
KLU_convert_matrix_in_CSR         /* return TRUE if successful, FALSE otherwise */
(
    Int *Ap_CSC,    /* CSC */
    Int *Ai_CSC,    /* CSC */
    double *Ax_CSC, /* CSC */
    Int *Ap_CSR,    /* CSR */
    Int *Ai_CSR,    /* CSR */
    double *Ax_CSR, /* CSR */
    Int n,
    Int nz,
    KLU_common *Common
)
{
    Element *MatrixCOO ;
    Entry *Az_CSC, *Az_CSR ;
    Int *Ap_COO, i, j, k ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (Common == NULL)
    {
        return (FALSE) ;
    }

    if (Ap_CSC != NULL)
    {
        if (Ai_CSC == NULL || Ax_CSC == NULL)
        {
            Ap_CSR [0] = 0 ;
            Ap_CSR [1] = 0 ;
            Common->status = KLU_OK ;
            return (TRUE) ;
        }
    } else {
        Common->status = KLU_INVALID ;
        return (FALSE) ;
    }

    Common->status = KLU_OK ;


    MatrixCOO = (Element *) malloc ((size_t)nz * sizeof (Element)) ;
    Ap_COO = (Int *) malloc ((size_t)nz * sizeof (Int)) ;

    Az_CSC = (Entry *)Ax_CSC ;
    Az_CSR = (Entry *)Ax_CSR ;

    k = 0 ;
    for (i = 0 ; i < n ; i++)
    {
        for (j = Ap_CSC [i] ; j < Ap_CSC [i + 1] ; j++)
        {
            MatrixCOO [k].row = Ai_CSC [j] ;
            MatrixCOO [k].col = i ;

#ifdef COMPLEX
            MatrixCOO [k].val.Real = Az_CSC [j].Real ;
            MatrixCOO [k].val.Imag = Az_CSC [j].Imag ;
#else
            MatrixCOO [k].val = Az_CSC [j] ;
#endif

            k++ ;
        }
    }

    /* Order the MatrixCOO along the rows */
    qsort (MatrixCOO, (size_t)nz, sizeof(Element), CompareRow) ;

    /* Order the MatrixCOO along the columns */
    i = 0 ;
    while (i < nz)
    {
        for (j = i + 1 ; j < nz ; j++)
        {
            if (MatrixCOO [j].row != MatrixCOO [i].row)
            {
                break ;
            }
        }

        qsort (MatrixCOO + i, (size_t)(j - i), sizeof(Element), CompareColumn) ;

        i = j ;
    }

    for (i = 0 ; i < nz ; i++)
    {
        Ap_COO [i] = MatrixCOO [i].row ;
        Ai_CSR [i] = MatrixCOO [i].col ;

#ifdef COMPLEX
        Az_CSR [i].Real = MatrixCOO [i].val.Real ;
        Az_CSR [i].Imag = MatrixCOO [i].val.Imag ;
#else
        Az_CSR [i] = MatrixCOO [i].val ;
#endif

    }

    Compress (Ap_COO, Ap_CSR, n, nz) ;

    free (MatrixCOO) ;
    free (Ap_COO) ;

    return (TRUE) ;
}
