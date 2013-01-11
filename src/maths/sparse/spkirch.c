/**
 * Routine to Verify the KCL
 */

#include <math.h>
#define spINSIDE_SPARSE
#include "spconfig.h"
#include "ngspice/spmatrix.h"
#include "spdefs.h"

int KCL_verification (MatrixPtr Matrix, spREAL *rhsOld, spREAL *rhs, double gmin, int i, double RelTol, double AbsTol, double maximum)
{
    ElementPtr element ;
    spREAL current ;

    current = 0 ;

#ifdef TRANSLATE
    element = Matrix->FirstInRow [Matrix->ExtToIntRowMap [i]] ;
#else
    element = Matrix->FirstInRow [i] ;
#endif

    /* A*x */
    while (element != NULL)
    {
        current += element->Real * rhsOld [Matrix->IntToExtColMap [element->Col]] ;
        element = element->NextInRow ;
    }

#ifdef TRANSLATE
    current += gmin * rhsOld [Matrix->IntToExtColMap [Matrix->Diag [Matrix->ExtToIntRowMap [i]]->Col]] ;
#else
    current += gmin * rhsOld [Matrix->IntToExtColMap [Matrix->Diag [i]->Col]] ;
#endif

    if (fabs (current - rhs [i]) > RelTol * maximum + AbsTol)
        return 1 ;

    return 0 ;
}
