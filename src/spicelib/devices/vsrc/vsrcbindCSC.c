/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
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
VSRCbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model = (VSRCmodel *)inModel ;
    VSRCinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the VSRC models */
    for ( ; model != NULL ; model = VSRCnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ; here = VSRCnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE_DYNAMIC(VSRCposIbrPtr, VSRCposIbrBinding, VSRCposNode, VSRCbranch);
            CREATE_KLU_BINDING_TABLE_DYNAMIC(VSRCnegIbrPtr, VSRCnegIbrBinding, VSRCnegNode, VSRCbranch);
            CREATE_KLU_BINDING_TABLE_DYNAMIC(VSRCibrNegPtr, VSRCibrNegBinding, VSRCbranch, VSRCnegNode);
            CREATE_KLU_BINDING_TABLE_DYNAMIC(VSRCibrPosPtr, VSRCibrPosBinding, VSRCbranch, VSRCposNode);

            /* Pole-Zero Analysis */
            if (here->VSRCibrIbrPtr)
            {
                i = here->VSRCibrIbrPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VSRCibrIbrBinding = matched ;
                if (matched != NULL)
                {
                    here->VSRCibrIbrPtr = matched->CSC_LinearDynamic ;
                }
            }
        }
    }

    return (OK) ;
}

int
VSRCbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model = (VSRCmodel *)inModel ;
    VSRCinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VSRC models */
    for ( ; model != NULL ; model = VSRCnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ; here = VSRCnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_DYNAMIC(VSRCposIbrPtr, VSRCposIbrBinding, VSRCposNode, VSRCbranch);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_DYNAMIC(VSRCnegIbrPtr, VSRCnegIbrBinding, VSRCnegNode, VSRCbranch);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_DYNAMIC(VSRCibrNegPtr, VSRCibrNegBinding, VSRCbranch, VSRCnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_DYNAMIC(VSRCibrPosPtr, VSRCibrPosBinding, VSRCbranch, VSRCposNode);
            /* Pole-Zero Analysis */
            if ((here-> VSRCbranch != 0) && (here-> VSRCbranch != 0))
            {
                if (here->VSRCibrIbrBinding)
                {
                    here->VSRCibrIbrPtr = here->VSRCibrIbrBinding->CSC_Complex_LinearDynamic ;
                }
            }
        }
    }

    return (OK) ;
}

int
VSRCbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model = (VSRCmodel *)inModel ;
    VSRCinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VSRC models */
    for ( ; model != NULL ; model = VSRCnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ; here = VSRCnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL_DYNAMIC(VSRCposIbrPtr, VSRCposIbrBinding, VSRCposNode, VSRCbranch);
            CONVERT_KLU_BINDING_TABLE_TO_REAL_DYNAMIC(VSRCnegIbrPtr, VSRCnegIbrBinding, VSRCnegNode, VSRCbranch);
            CONVERT_KLU_BINDING_TABLE_TO_REAL_DYNAMIC(VSRCibrNegPtr, VSRCibrNegBinding, VSRCbranch, VSRCnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL_DYNAMIC(VSRCibrPosPtr, VSRCibrPosBinding, VSRCbranch, VSRCposNode);
            /* Pole-Zero Analysis */
            if ((here-> VSRCbranch != 0) && (here-> VSRCbranch != 0))
            {
                if (here->VSRCibrIbrBinding)
                {
                    here->VSRCibrIbrPtr = here->VSRCibrIbrBinding->CSC_LinearDynamic ;
                }
            }
        }
    }

    return (OK) ;
}
