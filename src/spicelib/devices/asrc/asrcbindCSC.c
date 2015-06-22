/**********
Author: 2015 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"

#include <stdlib.h>

static
int
BindCompare (const void *a, const void *b)
{
    BindElement *A, *B ;
    A = (BindElement *)a ;
    B = (BindElement *)b ;

    return ((int)(A->Sparse - B->Sparse)) ;
}

int
ASRCbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    ASRCmodel *model = (ASRCmodel *)inModel ;
    ASRCinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;
    int j, k ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the ASRC models */
    for ( ; model != NULL ; model = model->ASRCnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->ASRCinstances ; here != NULL ; here = here->ASRCnextInstance)
        {
            j = 0 ;
            if (here->ASRCtype == ASRC_VOLTAGE)
            {
                if ((here->ASRCposNode != 0) && (here->ASRCbranch != 0))
                {
                    i = here->ASRCposptr [j] ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->ASRCposptrStructPtr [j] = matched ;
                    here->ASRCposptr [j] = matched->CSC ;
                }
                j++ ;

                if ((here->ASRCnegNode != 0) && (here->ASRCbranch != 0))
                {
                    i = here->ASRCposptr [j] ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->ASRCposptrStructPtr [j] = matched ;
                    here->ASRCposptr [j] = matched->CSC ;
                }
                j++ ;

                if ((here->ASRCbranch != 0) && (here->ASRCnegNode != 0))
                {
                    i = here->ASRCposptr [j] ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->ASRCposptrStructPtr [j] = matched ;
                    here->ASRCposptr [j] = matched->CSC ;
                }
                j++ ;

                if ((here->ASRCbranch != 0) && (here->ASRCposNode != 0))
                {
                    i = here->ASRCposptr [j] ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->ASRCposptrStructPtr [j] = matched ;
                    here->ASRCposptr [j] = matched->CSC ;
                }
                j++ ;
            }

            for (k = 0 ; k < here->ASRCtree->numVars ; k++)
            {
                if (here->ASRCtype == ASRC_VOLTAGE)
                {
                    if ((here->ASRCbranch != 0) && (here->ASRCvars [k] != 0))
                    {
                        i = here->ASRCposptr [j] ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->ASRCposptrStructPtr [j] = matched ;
                        here->ASRCposptr [j] = matched->CSC ;
                    }
                    j++ ;
                } else {
                    if ((here->ASRCposNode != 0) && (here->ASRCvars [k] != 0))
                    {
                        i = here->ASRCposptr [j] ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->ASRCposptrStructPtr [j] = matched ;
                        here->ASRCposptr [j] = matched->CSC ;
                    }
                    j++ ;

                    if ((here->ASRCnegNode != 0) && (here->ASRCvars [k] != 0))
                    {
                        i = here->ASRCposptr [j] ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->ASRCposptrStructPtr [j] = matched ;
                        here->ASRCposptr [j] = matched->CSC ;
                    }
                    j++ ;
                }
            }
        }
    }

    return (OK) ;
}

int
ASRCbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    ASRCmodel *model = (ASRCmodel *)inModel ;
    ASRCinstance *here ;
    int j, k ;

    NG_IGNORE (ckt) ;

    /* loop through all the ASRC models */
    for ( ; model != NULL ; model = model->ASRCnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->ASRCinstances ; here != NULL ; here = here->ASRCnextInstance)
        {
            j = 0 ;
            if (here->ASRCtype == ASRC_VOLTAGE)
            {
                if ((here->ASRCposNode != 0) && (here->ASRCbranch != 0))
                {
                    here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC_Complex ;
                }
                j++ ;

                if ((here->ASRCnegNode != 0) && (here->ASRCbranch != 0))
                {
                    here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC_Complex ;
                }
                j++ ;

                if ((here->ASRCbranch != 0) && (here->ASRCnegNode != 0))
                {
                    here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC_Complex ;
                }
                j++ ;

                if ((here->ASRCbranch != 0) && (here->ASRCposNode != 0))
                {
                    here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC_Complex ;
                }
                j++ ;
            }

            for (k = 0 ; k < here->ASRCtree->numVars ; k++)
            {
                if (here->ASRCtype == ASRC_VOLTAGE)
                {
                    if ((here->ASRCbranch != 0) && (here->ASRCvars [k] != 0))
                    {
                        here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC_Complex ;
                    }
                    j++ ;
                } else {
                    if ((here->ASRCposNode != 0) && (here->ASRCvars [k] != 0))
                    {
                        here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC_Complex ;
                    }
                    j++ ;

                    if ((here->ASRCnegNode != 0) && (here->ASRCvars [k] != 0))
                    {
                        here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC_Complex ;
                    }
                    j++ ;
                }
            }
        }
    }

    return (OK) ;
}

int
ASRCbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    ASRCmodel *model = (ASRCmodel *)inModel ;
    ASRCinstance *here ;
    int j, k ;

    NG_IGNORE (ckt) ;

    /* loop through all the ASRC models */
    for ( ; model != NULL ; model = model->ASRCnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->ASRCinstances ; here != NULL ; here = here->ASRCnextInstance)
        {
            j = 0 ;
            if (here->ASRCtype == ASRC_VOLTAGE)
            {
                if ((here->ASRCposNode != 0) && (here->ASRCbranch != 0))
                {
                    here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC ;
                }
                j++ ;

                if ((here->ASRCnegNode != 0) && (here->ASRCbranch != 0))
                {
                    here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC ;
                }
                j++ ;

                if ((here->ASRCbranch != 0) && (here->ASRCnegNode != 0))
                {
                    here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC ;
                }
                j++ ;

                if ((here->ASRCbranch != 0) && (here->ASRCposNode != 0))
                {
                    here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC ;
                }
                j++ ;
            }

            for (k = 0 ; k < here->ASRCtree->numVars ; k++)
            {
                if (here->ASRCtype == ASRC_VOLTAGE)
                {
                    if ((here->ASRCbranch != 0) && (here->ASRCvars [k] != 0))
                    {
                        here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC ;
                    }
                    j++ ;
                } else {
                    if ((here->ASRCposNode != 0) && (here->ASRCvars [k] != 0))
                    {
                        here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC ;
                    }
                    j++ ;

                    if ((here->ASRCnegNode != 0) && (here->ASRCvars [k] != 0))
                    {
                        here->ASRCposptr [j] = here->ASRCposptrStructPtr [j]->CSC ;
                    }
                    j++ ;
                }
            }
        }
    }

    return (OK) ;
}
