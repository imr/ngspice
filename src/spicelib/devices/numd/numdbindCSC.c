/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "numddefs.h"
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
NUMDbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMDmodel *model = (NUMDmodel *)inModel ;
    NUMDinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the NUMD models */
    for ( ; model != NULL ; model = NUMDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = NUMDinstances(model); here != NULL ; here = NUMDnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(NUMDposPosPtr, NUMDposPosBinding, NUMDposNode, NUMDposNode);
            CREATE_KLU_BINDING_TABLE(NUMDnegNegPtr, NUMDnegNegBinding, NUMDnegNode, NUMDnegNode);
            CREATE_KLU_BINDING_TABLE(NUMDnegPosPtr, NUMDnegPosBinding, NUMDnegNode, NUMDposNode);
            CREATE_KLU_BINDING_TABLE(NUMDposNegPtr, NUMDposNegBinding, NUMDposNode, NUMDnegNode);
        }
    }

    return (OK) ;
}

int
NUMDbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMDmodel *model = (NUMDmodel *)inModel ;
    NUMDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NUMD models */
    for ( ; model != NULL ; model = NUMDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = NUMDinstances(model); here != NULL ; here = NUMDnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMDposPosPtr, NUMDposPosBinding, NUMDposNode, NUMDposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMDnegNegPtr, NUMDnegNegBinding, NUMDnegNode, NUMDnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMDnegPosPtr, NUMDnegPosBinding, NUMDnegNode, NUMDposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMDposNegPtr, NUMDposNegBinding, NUMDposNode, NUMDnegNode);
        }
    }

    return (OK) ;
}

int
NUMDbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMDmodel *model = (NUMDmodel *)inModel ;
    NUMDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NUMD models */
    for ( ; model != NULL ; model = NUMDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = NUMDinstances(model); here != NULL ; here = NUMDnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMDposPosPtr, NUMDposPosBinding, NUMDposNode, NUMDposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMDnegNegPtr, NUMDnegNegBinding, NUMDnegNode, NUMDnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMDnegPosPtr, NUMDnegPosBinding, NUMDnegNode, NUMDposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMDposNegPtr, NUMDposNegBinding, NUMDposNode, NUMDnegNode);
        }
    }

    return (OK) ;
}
