/**********
Author: 2015 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

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
                CREATE_KLU_BINDING_TABLE(ASRCposPtr [j], ASRCposBinding [j], ASRCposNode, ASRCbranch);
                j++ ;

                CREATE_KLU_BINDING_TABLE(ASRCposPtr [j], ASRCposBinding [j], ASRCnegNode, ASRCbranch);
                j++ ;

                CREATE_KLU_BINDING_TABLE(ASRCposPtr [j], ASRCposBinding [j], ASRCbranch, ASRCnegNode);
                j++ ;

                CREATE_KLU_BINDING_TABLE(ASRCposPtr [j], ASRCposBinding [j], ASRCbranch, ASRCposNode);
                j++ ;
            }

            for (k = 0 ; k < here->ASRCtree->numVars ; k++)
            {
                if (here->ASRCtype == ASRC_VOLTAGE)
                {
                    CREATE_KLU_BINDING_TABLE(ASRCposPtr [j], ASRCposBinding [j], ASRCbranch, ASRCvars [k]);
                    j++ ;
                } else {
                    CREATE_KLU_BINDING_TABLE(ASRCposPtr [j], ASRCposBinding [j], ASRCposNode, ASRCvars [k]);
                    j++ ;

                    CREATE_KLU_BINDING_TABLE(ASRCposPtr [j], ASRCposBinding [j], ASRCnegNode, ASRCvars [k]);
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
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ASRCposPtr [j], ASRCposBinding [j], ASRCposNode, ASRCbranch);
                j++ ;

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ASRCposPtr [j], ASRCposBinding [j], ASRCnegNode, ASRCbranch);
                j++ ;

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ASRCposPtr [j], ASRCposBinding [j], ASRCbranch, ASRCnegNode);
                j++ ;

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ASRCposPtr [j], ASRCposBinding [j], ASRCbranch, ASRCposNode);
                j++ ;
            }

            for (k = 0 ; k < here->ASRCtree->numVars ; k++)
            {
                if (here->ASRCtype == ASRC_VOLTAGE)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ASRCposPtr [j], ASRCposBinding [j], ASRCbranch, ASRCvars [k]);
                    j++ ;
                } else {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ASRCposPtr [j], ASRCposBinding [j], ASRCposNode, ASRCvars [k]);
                    j++ ;

                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ASRCposPtr [j], ASRCposBinding [j], ASRCnegNode, ASRCvars [k]);
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
                CONVERT_KLU_BINDING_TABLE_TO_REAL(ASRCposPtr [j], ASRCposBinding [j], ASRCposNode, ASRCbranch);
                j++ ;

                CONVERT_KLU_BINDING_TABLE_TO_REAL(ASRCposPtr [j], ASRCposBinding [j], ASRCnegNode, ASRCbranch);
                j++ ;

                CONVERT_KLU_BINDING_TABLE_TO_REAL(ASRCposPtr [j], ASRCposBinding [j], ASRCbranch, ASRCnegNode);
                j++ ;

                CONVERT_KLU_BINDING_TABLE_TO_REAL(ASRCposPtr [j], ASRCposBinding [j], ASRCbranch, ASRCposNode);
                j++ ;
            }

            for (k = 0 ; k < here->ASRCtree->numVars ; k++)
            {
                if (here->ASRCtype == ASRC_VOLTAGE)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(ASRCposPtr [j], ASRCposBinding [j], ASRCbranch, ASRCvars [k]);
                    j++ ;
                } else {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(ASRCposPtr [j], ASRCposBinding [j], ASRCposNode, ASRCvars [k]);
                    j++ ;

                    CONVERT_KLU_BINDING_TABLE_TO_REAL(ASRCposPtr [j], ASRCposBinding [j], ASRCnegNode, ASRCvars [k]);
                    j++ ;
                }
            }
        }
    }

    return (OK) ;
}
