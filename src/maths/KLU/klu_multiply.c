#include "klu_internal.h"

Int KLU_constant_multiply         /* return TRUE if successful, FALSE otherwise */
(
    Int *Ap,
    double *Ax,
    Int n,
    KLU_common *Common,
    double constant
)
{
    Entry *Az ;
    int i, j ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (Common == NULL)
    {
        return (FALSE) ;
    }

    if (Ap == NULL || Ax == NULL)
    {
        Common->status = KLU_INVALID ;
        return (FALSE) ;
    }

    Common->status = KLU_OK ;

    Az = (Entry *)Ax ;

    for (i = 0 ; i < n ; i++)
    {
        for (j = Ap [i] ; j < Ap [i + 1] ; j++)
        {

#ifdef COMPLEX
            Az [j].Real *= constant ;
            Az [j].Imag *= constant ;
#else
            Az [j] *= constant ;
#endif

        }
    }

    return (TRUE) ;
}

/* Macro function that multiplies two complex numbers and then adds them
 * to another. to += mult_a * mult_b */
#define  CMPLX_MULT_ADD_ASSIGN(to,from_a,from_b)        \
{   (to).Real += (from_a).Real * (from_b).Real -        \
                 (from_a).Imag * (from_b).Imag;         \
    (to).Imag += (from_a).Real * (from_b).Imag +        \
                 (from_a).Imag * (from_b).Real;         \
}

Int KLU_matrix_vector_multiply         /* return TRUE if successful, FALSE otherwise */
(
    Int *Ap,    /* CSR */
    Int *Ai,    /* CSR */
    double *Ax, /* CSR */
    double *RHS,
    double *Solution,
#ifdef COMPLEX
    double *iRHS,
    double *iSolution,
#endif
    Int *IntToExtRowMap,
    Int *IntToExtColMap,
    Int n,
    KLU_common *Common
)
{
    Entry *Az, *Intermediate, Sum ;
    Int i, j, *pExtOrder ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (Common == NULL)
    {
        return (FALSE) ;
    }

    if (Ap == NULL || Ai == NULL || Ax == NULL || RHS == NULL || Solution == NULL

#ifdef COMPLEX
        || iRHS == NULL || iSolution == NULL
#endif

    )
    {
        Common->status = KLU_INVALID ;
        return (FALSE) ;
    }

    Common->status = KLU_OK ;


    Intermediate = (Entry *) malloc ((size_t)n * sizeof (Entry)) ;

    Az = (Entry *)Ax ;

    pExtOrder = &IntToExtColMap [n] ;
    for (i = n - 1 ; i >= 0 ; i--)
    {

#ifdef COMPLEX
        Intermediate [i].Real = Solution [*(pExtOrder)] ;
        Intermediate [i].Imag = iSolution [*(pExtOrder--)] ;
#else
        Intermediate [i] = Solution [*(pExtOrder--)] ;
#endif
    }

    pExtOrder = &IntToExtRowMap [n] ;
    for (i = n - 1 ; i >= 0 ; i--)
    {

#ifdef COMPLEX
        Sum.Real = 0.0 ;
        Sum.Imag = 0.0 ;
#else
        Sum = 0.0 ;
#endif

        for (j = Ap [i] ; j < Ap [i + 1] ; j++)
        {

#ifdef COMPLEX
            CMPLX_MULT_ADD_ASSIGN (Sum, Az [j], Intermediate [Ai [j]]) ;
#else
	    Sum += Az [j] * Intermediate [Ai [j]] ;
#endif

        }

#ifdef COMPLEX
        RHS [*(pExtOrder)] = Sum.Real ;
        iRHS [*(pExtOrder--)] = Sum.Imag ;
#else
        RHS [*(pExtOrder--)] = Sum ;
#endif

    }

    free (Intermediate) ;

    return (TRUE) ;
}
