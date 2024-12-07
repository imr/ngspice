#include <stdio.h>
#include "ngspice/memory.h"
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
            /* CSC Matrix is empty, so the CSR one */
            for (i = 0 ; i <= n ; i++)
            {
                Ap_CSR [i] = 0 ;
            }
            Common->status = KLU_OK ;
            return (TRUE) ;
        }
    } else {
        Common->status = KLU_INVALID ;
        return (FALSE) ;
    }

    Common->status = KLU_OK ;


    MatrixCOO = TMALLOC(Element, nz);
    Ap_COO = TMALLOC(Int, nz);

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

/* Francesco - Extract only Udiag */
Int KLU_extract_Udiag     /* returns TRUE if successful, FALSE otherwise */
(
    /* inputs: */
    KLU_numeric *Numeric,
    KLU_symbolic *Symbolic,

    /* outputs, all of which must be allocated on input */

    /* U */
    double *Ux,     /* size nnz(U) */
#ifdef COMPLEX
    double *Uz,     /* size nnz(U) for the complex case, ignored if real */
#endif

    Int *P,
    Int *Q,
    double *Rs,

    KLU_common *Common
)
{
    Entry *Ukk ;
    Int block, k1, k2, kk, i, n, nk, nblocks, nz ;

    if (Common == NULL)
    {
        return (FALSE) ;
    }

    if (Common->status == KLU_EMPTY_MATRIX)
    {
        return (FALSE) ;
    }

    if (Symbolic == NULL || Numeric == NULL)
    {
        Common->status = KLU_INVALID ;
        return (FALSE) ;
    }

    Common->status = KLU_OK ;
    n = Symbolic->n ;
    nblocks = Symbolic->nblocks ;

    /* ---------------------------------------------------------------------- */
    /* extract scale factors */
    /* ---------------------------------------------------------------------- */

    if (Rs != NULL)
    {
        if (Numeric->Rs != NULL)
        {
            for (i = 0 ; i < n ; i++)
            {
                Rs [i] = Numeric->Rs [i] ;
            }
        }
        else
        {
            /* no scaling */
            for (i = 0 ; i < n ; i++)
            {
                Rs [i] = 1 ;
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* extract final row permutation */
    /* ---------------------------------------------------------------------- */

    if (P != NULL)
    {
        for (i = 0 ; i < n ; i++)
        {
            P [i] = Numeric->Pnum [i] ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* extract column permutation */
    /* ---------------------------------------------------------------------- */

    if (Q != NULL)
    {
        for (i = 0 ; i < n ; i++)
        {
            Q [i] = Symbolic->Q [i] ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* extract each block of U */
    /* ---------------------------------------------------------------------- */

    if (Ux != NULL
#ifdef COMPLEX
        && Uz != NULL
#endif
    )
    {
        nz = 0 ;
        for (block = 0 ; block < nblocks ; block++)
        {
            k1 = Symbolic->R [block] ;
            k2 = Symbolic->R [block+1] ;
            nk = k2 - k1 ;
            Ukk = ((Entry *) Numeric->Udiag) + k1 ;
            if (nk == 1)
            {
                /* singleton block */
                Ux [nz] = REAL (Ukk [0]) ;
#ifdef COMPLEX
                Uz [nz] = IMAG (Ukk [0]) ;
#endif
                nz++ ;
            }
            else
            {
                /* non-singleton block */
                for (kk = 0 ; kk < nk ; kk++)
                {
                    /* add the diagonal entry */
                    Ux [nz] = REAL (Ukk [kk]) ;
#ifdef COMPLEX
                    Uz [nz] = IMAG (Ukk [kk]) ;
#endif
                    nz++ ;
                }
            }
        }
        ASSERT (nz == Numeric->unz) ;

    }

    return (TRUE) ;
}

Int KLU_print
(
    Int *Ap,
    Int *Ai,
    double *Ax,
    int n,
    int *IntToExtRowMap,
    int *IntToExtColMap
)
{
    Entry *Az ;
    int i, j ;

    Az = (Entry *)Ax ;
    for (i = 0 ; i < n ; i++)
    {
        for (j = Ap [i] ; j < Ap [i + 1] ; j++)
        {

#ifdef COMPLEX
            if (IntToExtRowMap && IntToExtColMap) {
                fprintf (stderr, "Row: %d\tCol: %d\tValue: %-.9g j%-.9g\n", IntToExtRowMap [Ai [j] + 1], IntToExtColMap [i + 1], Az [j].Real, Az [j].Imag) ;
            } else {
                fprintf (stderr, "Row: %d\tCol: %d\tValue: %-.9g j%-.9g\n", Ai [j] + 1, i + 1, Az [j].Real, Az [j].Imag) ;
            }
#else
            if (IntToExtRowMap && IntToExtColMap) {
                fprintf (stderr, "Row: %d\tCol: %d\tValue: %-.9g\n", IntToExtRowMap [Ai [j] + 1], IntToExtColMap [i + 1], Az [j]) ;
            } else {
                fprintf (stderr, "Row: %d\tCol: %d\tValue: %-.9g\n", Ai [j] + 1, i + 1, Az [j]) ;
            }
#endif

        }
    }

    return 0 ;
}
