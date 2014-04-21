/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cswdefs.h"
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
CSWbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *)inModel ;
    CSWinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the CSW models */
    for ( ; model != NULL ; model = CSWnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = CSWinstances(model); here != NULL ; here = CSWnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(CSWposPosPtr, CSWposPosBinding, CSWposNode, CSWposNode);
            CREATE_KLU_BINDING_TABLE(CSWposNegPtr, CSWposNegBinding, CSWposNode, CSWnegNode);
            CREATE_KLU_BINDING_TABLE(CSWnegPosPtr, CSWnegPosBinding, CSWnegNode, CSWposNode);
            CREATE_KLU_BINDING_TABLE(CSWnegNegPtr, CSWnegNegBinding, CSWnegNode, CSWnegNode);
        }
    }

    return (OK) ;
}

int
CSWbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *)inModel ;
    CSWinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the CSW models */
    for ( ; model != NULL ; model = CSWnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = CSWinstances(model); here != NULL ; here = CSWnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CSWposPosPtr, CSWposPosBinding, CSWposNode, CSWposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CSWposNegPtr, CSWposNegBinding, CSWposNode, CSWnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CSWnegPosPtr, CSWnegPosBinding, CSWnegNode, CSWposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CSWnegNegPtr, CSWnegNegBinding, CSWnegNode, CSWnegNode);
        }
    }

    return (OK) ;
}

int
CSWbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *)inModel ;
    CSWinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the CSW models */
    for ( ; model != NULL ; model = CSWnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = CSWinstances(model); here != NULL ; here = CSWnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CSWposPosPtr, CSWposPosBinding, CSWposNode, CSWposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CSWposNegPtr, CSWposNegBinding, CSWposNode, CSWnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CSWnegPosPtr, CSWnegPosBinding, CSWnegNode, CSWposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CSWnegNegPtr, CSWnegNegBinding, CSWnegNode, CSWnegNode);
        }
    }

    return (OK) ;
}
