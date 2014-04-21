/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
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
VBICbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    VBICmodel *model = (VBICmodel *)inModel ;
    VBICinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the VBIC models */
    for ( ; model != NULL ; model = model->VBICnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VBICinstances ; here != NULL ; here = here->VBICnextInstance)
        {
            if ((here->VBICcollNode != 0) && (here->VBICcollNode != 0))
            {
                i = here->VBICcollCollPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCollStructPtr = matched ;
                here->VBICcollCollPtr = matched->CSC ;
            }

            if ((here->VBICbaseNode != 0) && (here->VBICbaseNode != 0))
            {
                i = here->VBICbaseBasePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBaseStructPtr = matched ;
                here->VBICbaseBasePtr = matched->CSC ;
            }

            if ((here->VBICemitNode != 0) && (here->VBICemitNode != 0))
            {
                i = here->VBICemitEmitPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICemitEmitStructPtr = matched ;
                here->VBICemitEmitPtr = matched->CSC ;
            }

            if ((here->VBICsubsNode != 0) && (here->VBICsubsNode != 0))
            {
                i = here->VBICsubsSubsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICsubsSubsStructPtr = matched ;
                here->VBICsubsSubsPtr = matched->CSC ;
            }

            if ((here->VBICcollCXNode != 0) && (here->VBICcollCXNode != 0))
            {
                i = here->VBICcollCXCollCXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCXCollCXStructPtr = matched ;
                here->VBICcollCXCollCXPtr = matched->CSC ;
            }

            if ((here->VBICcollCINode != 0) && (here->VBICcollCINode != 0))
            {
                i = here->VBICcollCICollCIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCICollCIStructPtr = matched ;
                here->VBICcollCICollCIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBXNode != 0))
            {
                i = here->VBICbaseBXBaseBXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBXBaseBXStructPtr = matched ;
                here->VBICbaseBXBaseBXPtr = matched->CSC ;
            }

            if ((here->VBICbaseBINode != 0) && (here->VBICbaseBINode != 0))
            {
                i = here->VBICbaseBIBaseBIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBIBaseBIStructPtr = matched ;
                here->VBICbaseBIBaseBIPtr = matched->CSC ;
            }

            if ((here->VBICemitEINode != 0) && (here->VBICemitEINode != 0))
            {
                i = here->VBICemitEIEmitEIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICemitEIEmitEIStructPtr = matched ;
                here->VBICemitEIEmitEIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBPNode != 0))
            {
                i = here->VBICbaseBPBaseBPPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBPBaseBPStructPtr = matched ;
                here->VBICbaseBPBaseBPPtr = matched->CSC ;
            }

            if ((here->VBICsubsSINode != 0) && (here->VBICsubsSINode != 0))
            {
                i = here->VBICsubsSISubsSIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICsubsSISubsSIStructPtr = matched ;
                here->VBICsubsSISubsSIPtr = matched->CSC ;
            }

            if ((here->VBICbaseNode != 0) && (here->VBICemitNode != 0))
            {
                i = here->VBICbaseEmitPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseEmitStructPtr = matched ;
                here->VBICbaseEmitPtr = matched->CSC ;
            }

            if ((here->VBICemitNode != 0) && (here->VBICbaseNode != 0))
            {
                i = here->VBICemitBasePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICemitBaseStructPtr = matched ;
                here->VBICemitBasePtr = matched->CSC ;
            }

            if ((here->VBICbaseNode != 0) && (here->VBICcollNode != 0))
            {
                i = here->VBICbaseCollPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseCollStructPtr = matched ;
                here->VBICbaseCollPtr = matched->CSC ;
            }

            if ((here->VBICcollNode != 0) && (here->VBICbaseNode != 0))
            {
                i = here->VBICcollBasePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollBaseStructPtr = matched ;
                here->VBICcollBasePtr = matched->CSC ;
            }

            if ((here->VBICcollNode != 0) && (here->VBICcollCXNode != 0))
            {
                i = here->VBICcollCollCXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCollCXStructPtr = matched ;
                here->VBICcollCollCXPtr = matched->CSC ;
            }

            if ((here->VBICbaseNode != 0) && (here->VBICbaseBXNode != 0))
            {
                i = here->VBICbaseBaseBXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBaseBXStructPtr = matched ;
                here->VBICbaseBaseBXPtr = matched->CSC ;
            }

            if ((here->VBICemitNode != 0) && (here->VBICemitEINode != 0))
            {
                i = here->VBICemitEmitEIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICemitEmitEIStructPtr = matched ;
                here->VBICemitEmitEIPtr = matched->CSC ;
            }

            if ((here->VBICsubsNode != 0) && (here->VBICsubsSINode != 0))
            {
                i = here->VBICsubsSubsSIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICsubsSubsSIStructPtr = matched ;
                here->VBICsubsSubsSIPtr = matched->CSC ;
            }

            if ((here->VBICcollCXNode != 0) && (here->VBICcollCINode != 0))
            {
                i = here->VBICcollCXCollCIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCXCollCIStructPtr = matched ;
                here->VBICcollCXCollCIPtr = matched->CSC ;
            }

            if ((here->VBICcollCXNode != 0) && (here->VBICbaseBXNode != 0))
            {
                i = here->VBICcollCXBaseBXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCXBaseBXStructPtr = matched ;
                here->VBICcollCXBaseBXPtr = matched->CSC ;
            }

            if ((here->VBICcollCXNode != 0) && (here->VBICbaseBINode != 0))
            {
                i = here->VBICcollCXBaseBIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCXBaseBIStructPtr = matched ;
                here->VBICcollCXBaseBIPtr = matched->CSC ;
            }

            if ((here->VBICcollCXNode != 0) && (here->VBICbaseBPNode != 0))
            {
                i = here->VBICcollCXBaseBPPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCXBaseBPStructPtr = matched ;
                here->VBICcollCXBaseBPPtr = matched->CSC ;
            }

            if ((here->VBICcollCINode != 0) && (here->VBICbaseBINode != 0))
            {
                i = here->VBICcollCIBaseBIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCIBaseBIStructPtr = matched ;
                here->VBICcollCIBaseBIPtr = matched->CSC ;
            }

            if ((here->VBICcollCINode != 0) && (here->VBICemitEINode != 0))
            {
                i = here->VBICcollCIEmitEIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCIEmitEIStructPtr = matched ;
                here->VBICcollCIEmitEIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBINode != 0))
            {
                i = here->VBICbaseBXBaseBIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBXBaseBIStructPtr = matched ;
                here->VBICbaseBXBaseBIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBXNode != 0) && (here->VBICemitEINode != 0))
            {
                i = here->VBICbaseBXEmitEIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBXEmitEIStructPtr = matched ;
                here->VBICbaseBXEmitEIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBPNode != 0))
            {
                i = here->VBICbaseBXBaseBPPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBXBaseBPStructPtr = matched ;
                here->VBICbaseBXBaseBPPtr = matched->CSC ;
            }

            if ((here->VBICbaseBXNode != 0) && (here->VBICsubsSINode != 0))
            {
                i = here->VBICbaseBXSubsSIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBXSubsSIStructPtr = matched ;
                here->VBICbaseBXSubsSIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBINode != 0) && (here->VBICemitEINode != 0))
            {
                i = here->VBICbaseBIEmitEIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBIEmitEIStructPtr = matched ;
                here->VBICbaseBIEmitEIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBPNode != 0) && (here->VBICsubsSINode != 0))
            {
                i = here->VBICbaseBPSubsSIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBPSubsSIStructPtr = matched ;
                here->VBICbaseBPSubsSIPtr = matched->CSC ;
            }

            if ((here->VBICcollCXNode != 0) && (here->VBICcollNode != 0))
            {
                i = here->VBICcollCXCollPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCXCollStructPtr = matched ;
                here->VBICcollCXCollPtr = matched->CSC ;
            }

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseNode != 0))
            {
                i = here->VBICbaseBXBasePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBXBaseStructPtr = matched ;
                here->VBICbaseBXBasePtr = matched->CSC ;
            }

            if ((here->VBICemitEINode != 0) && (here->VBICemitNode != 0))
            {
                i = here->VBICemitEIEmitPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICemitEIEmitStructPtr = matched ;
                here->VBICemitEIEmitPtr = matched->CSC ;
            }

            if ((here->VBICsubsSINode != 0) && (here->VBICsubsNode != 0))
            {
                i = here->VBICsubsSISubsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICsubsSISubsStructPtr = matched ;
                here->VBICsubsSISubsPtr = matched->CSC ;
            }

            if ((here->VBICcollCINode != 0) && (here->VBICcollCXNode != 0))
            {
                i = here->VBICcollCICollCXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICcollCICollCXStructPtr = matched ;
                here->VBICcollCICollCXPtr = matched->CSC ;
            }

            if ((here->VBICbaseBINode != 0) && (here->VBICcollCXNode != 0))
            {
                i = here->VBICbaseBICollCXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBICollCXStructPtr = matched ;
                here->VBICbaseBICollCXPtr = matched->CSC ;
            }

            if ((here->VBICbaseBPNode != 0) && (here->VBICcollCXNode != 0))
            {
                i = here->VBICbaseBPCollCXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBPCollCXStructPtr = matched ;
                here->VBICbaseBPCollCXPtr = matched->CSC ;
            }

            if ((here->VBICbaseBXNode != 0) && (here->VBICcollCINode != 0))
            {
                i = here->VBICbaseBXCollCIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBXCollCIStructPtr = matched ;
                here->VBICbaseBXCollCIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBINode != 0) && (here->VBICcollCINode != 0))
            {
                i = here->VBICbaseBICollCIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBICollCIStructPtr = matched ;
                here->VBICbaseBICollCIPtr = matched->CSC ;
            }

            if ((here->VBICemitEINode != 0) && (here->VBICcollCINode != 0))
            {
                i = here->VBICemitEICollCIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICemitEICollCIStructPtr = matched ;
                here->VBICemitEICollCIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBPNode != 0) && (here->VBICcollCINode != 0))
            {
                i = here->VBICbaseBPCollCIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBPCollCIStructPtr = matched ;
                here->VBICbaseBPCollCIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBINode != 0) && (here->VBICbaseBXNode != 0))
            {
                i = here->VBICbaseBIBaseBXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBIBaseBXStructPtr = matched ;
                here->VBICbaseBIBaseBXPtr = matched->CSC ;
            }

            if ((here->VBICemitEINode != 0) && (here->VBICbaseBXNode != 0))
            {
                i = here->VBICemitEIBaseBXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICemitEIBaseBXStructPtr = matched ;
                here->VBICemitEIBaseBXPtr = matched->CSC ;
            }

            if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBXNode != 0))
            {
                i = here->VBICbaseBPBaseBXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBPBaseBXStructPtr = matched ;
                here->VBICbaseBPBaseBXPtr = matched->CSC ;
            }

            if ((here->VBICsubsSINode != 0) && (here->VBICbaseBXNode != 0))
            {
                i = here->VBICsubsSIBaseBXPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICsubsSIBaseBXStructPtr = matched ;
                here->VBICsubsSIBaseBXPtr = matched->CSC ;
            }

            if ((here->VBICemitEINode != 0) && (here->VBICbaseBINode != 0))
            {
                i = here->VBICemitEIBaseBIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICemitEIBaseBIStructPtr = matched ;
                here->VBICemitEIBaseBIPtr = matched->CSC ;
            }

            if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBINode != 0))
            {
                i = here->VBICbaseBPBaseBIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICbaseBPBaseBIStructPtr = matched ;
                here->VBICbaseBPBaseBIPtr = matched->CSC ;
            }

            if ((here->VBICsubsSINode != 0) && (here->VBICcollCINode != 0))
            {
                i = here->VBICsubsSICollCIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICsubsSICollCIStructPtr = matched ;
                here->VBICsubsSICollCIPtr = matched->CSC ;
            }

            if ((here->VBICsubsSINode != 0) && (here->VBICbaseBINode != 0))
            {
                i = here->VBICsubsSIBaseBIPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICsubsSIBaseBIStructPtr = matched ;
                here->VBICsubsSIBaseBIPtr = matched->CSC ;
            }

            if ((here->VBICsubsSINode != 0) && (here->VBICbaseBPNode != 0))
            {
                i = here->VBICsubsSIBaseBPPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VBICsubsSIBaseBPStructPtr = matched ;
                here->VBICsubsSIBaseBPPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
VBICbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    VBICmodel *model = (VBICmodel *)inModel ;
    VBICinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VBIC models */
    for ( ; model != NULL ; model = model->VBICnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VBICinstances ; here != NULL ; here = here->VBICnextInstance)
        {
            if ((here->VBICcollNode != 0) && (here->VBICcollNode != 0))
                here->VBICcollCollPtr = here->VBICcollCollStructPtr->CSC_Complex ;

            if ((here->VBICbaseNode != 0) && (here->VBICbaseNode != 0))
                here->VBICbaseBasePtr = here->VBICbaseBaseStructPtr->CSC_Complex ;

            if ((here->VBICemitNode != 0) && (here->VBICemitNode != 0))
                here->VBICemitEmitPtr = here->VBICemitEmitStructPtr->CSC_Complex ;

            if ((here->VBICsubsNode != 0) && (here->VBICsubsNode != 0))
                here->VBICsubsSubsPtr = here->VBICsubsSubsStructPtr->CSC_Complex ;

            if ((here->VBICcollCXNode != 0) && (here->VBICcollCXNode != 0))
                here->VBICcollCXCollCXPtr = here->VBICcollCXCollCXStructPtr->CSC_Complex ;

            if ((here->VBICcollCINode != 0) && (here->VBICcollCINode != 0))
                here->VBICcollCICollCIPtr = here->VBICcollCICollCIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICbaseBXBaseBXPtr = here->VBICbaseBXBaseBXStructPtr->CSC_Complex ;

            if ((here->VBICbaseBINode != 0) && (here->VBICbaseBINode != 0))
                here->VBICbaseBIBaseBIPtr = here->VBICbaseBIBaseBIStructPtr->CSC_Complex ;

            if ((here->VBICemitEINode != 0) && (here->VBICemitEINode != 0))
                here->VBICemitEIEmitEIPtr = here->VBICemitEIEmitEIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBPNode != 0))
                here->VBICbaseBPBaseBPPtr = here->VBICbaseBPBaseBPStructPtr->CSC_Complex ;

            if ((here->VBICsubsSINode != 0) && (here->VBICsubsSINode != 0))
                here->VBICsubsSISubsSIPtr = here->VBICsubsSISubsSIStructPtr->CSC_Complex ;

            if ((here->VBICbaseNode != 0) && (here->VBICemitNode != 0))
                here->VBICbaseEmitPtr = here->VBICbaseEmitStructPtr->CSC_Complex ;

            if ((here->VBICemitNode != 0) && (here->VBICbaseNode != 0))
                here->VBICemitBasePtr = here->VBICemitBaseStructPtr->CSC_Complex ;

            if ((here->VBICbaseNode != 0) && (here->VBICcollNode != 0))
                here->VBICbaseCollPtr = here->VBICbaseCollStructPtr->CSC_Complex ;

            if ((here->VBICcollNode != 0) && (here->VBICbaseNode != 0))
                here->VBICcollBasePtr = here->VBICcollBaseStructPtr->CSC_Complex ;

            if ((here->VBICcollNode != 0) && (here->VBICcollCXNode != 0))
                here->VBICcollCollCXPtr = here->VBICcollCollCXStructPtr->CSC_Complex ;

            if ((here->VBICbaseNode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICbaseBaseBXPtr = here->VBICbaseBaseBXStructPtr->CSC_Complex ;

            if ((here->VBICemitNode != 0) && (here->VBICemitEINode != 0))
                here->VBICemitEmitEIPtr = here->VBICemitEmitEIStructPtr->CSC_Complex ;

            if ((here->VBICsubsNode != 0) && (here->VBICsubsSINode != 0))
                here->VBICsubsSubsSIPtr = here->VBICsubsSubsSIStructPtr->CSC_Complex ;

            if ((here->VBICcollCXNode != 0) && (here->VBICcollCINode != 0))
                here->VBICcollCXCollCIPtr = here->VBICcollCXCollCIStructPtr->CSC_Complex ;

            if ((here->VBICcollCXNode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICcollCXBaseBXPtr = here->VBICcollCXBaseBXStructPtr->CSC_Complex ;

            if ((here->VBICcollCXNode != 0) && (here->VBICbaseBINode != 0))
                here->VBICcollCXBaseBIPtr = here->VBICcollCXBaseBIStructPtr->CSC_Complex ;

            if ((here->VBICcollCXNode != 0) && (here->VBICbaseBPNode != 0))
                here->VBICcollCXBaseBPPtr = here->VBICcollCXBaseBPStructPtr->CSC_Complex ;

            if ((here->VBICcollCINode != 0) && (here->VBICbaseBINode != 0))
                here->VBICcollCIBaseBIPtr = here->VBICcollCIBaseBIStructPtr->CSC_Complex ;

            if ((here->VBICcollCINode != 0) && (here->VBICemitEINode != 0))
                here->VBICcollCIEmitEIPtr = here->VBICcollCIEmitEIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBINode != 0))
                here->VBICbaseBXBaseBIPtr = here->VBICbaseBXBaseBIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICemitEINode != 0))
                here->VBICbaseBXEmitEIPtr = here->VBICbaseBXEmitEIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBPNode != 0))
                here->VBICbaseBXBaseBPPtr = here->VBICbaseBXBaseBPStructPtr->CSC_Complex ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICsubsSINode != 0))
                here->VBICbaseBXSubsSIPtr = here->VBICbaseBXSubsSIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBINode != 0) && (here->VBICemitEINode != 0))
                here->VBICbaseBIEmitEIPtr = here->VBICbaseBIEmitEIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICsubsSINode != 0))
                here->VBICbaseBPSubsSIPtr = here->VBICbaseBPSubsSIStructPtr->CSC_Complex ;

            if ((here->VBICcollCXNode != 0) && (here->VBICcollNode != 0))
                here->VBICcollCXCollPtr = here->VBICcollCXCollStructPtr->CSC_Complex ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseNode != 0))
                here->VBICbaseBXBasePtr = here->VBICbaseBXBaseStructPtr->CSC_Complex ;

            if ((here->VBICemitEINode != 0) && (here->VBICemitNode != 0))
                here->VBICemitEIEmitPtr = here->VBICemitEIEmitStructPtr->CSC_Complex ;

            if ((here->VBICsubsSINode != 0) && (here->VBICsubsNode != 0))
                here->VBICsubsSISubsPtr = here->VBICsubsSISubsStructPtr->CSC_Complex ;

            if ((here->VBICcollCINode != 0) && (here->VBICcollCXNode != 0))
                here->VBICcollCICollCXPtr = here->VBICcollCICollCXStructPtr->CSC_Complex ;

            if ((here->VBICbaseBINode != 0) && (here->VBICcollCXNode != 0))
                here->VBICbaseBICollCXPtr = here->VBICbaseBICollCXStructPtr->CSC_Complex ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICcollCXNode != 0))
                here->VBICbaseBPCollCXPtr = here->VBICbaseBPCollCXStructPtr->CSC_Complex ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICcollCINode != 0))
                here->VBICbaseBXCollCIPtr = here->VBICbaseBXCollCIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBINode != 0) && (here->VBICcollCINode != 0))
                here->VBICbaseBICollCIPtr = here->VBICbaseBICollCIStructPtr->CSC_Complex ;

            if ((here->VBICemitEINode != 0) && (here->VBICcollCINode != 0))
                here->VBICemitEICollCIPtr = here->VBICemitEICollCIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICcollCINode != 0))
                here->VBICbaseBPCollCIPtr = here->VBICbaseBPCollCIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBINode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICbaseBIBaseBXPtr = here->VBICbaseBIBaseBXStructPtr->CSC_Complex ;

            if ((here->VBICemitEINode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICemitEIBaseBXPtr = here->VBICemitEIBaseBXStructPtr->CSC_Complex ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICbaseBPBaseBXPtr = here->VBICbaseBPBaseBXStructPtr->CSC_Complex ;

            if ((here->VBICsubsSINode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICsubsSIBaseBXPtr = here->VBICsubsSIBaseBXStructPtr->CSC_Complex ;

            if ((here->VBICemitEINode != 0) && (here->VBICbaseBINode != 0))
                here->VBICemitEIBaseBIPtr = here->VBICemitEIBaseBIStructPtr->CSC_Complex ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBINode != 0))
                here->VBICbaseBPBaseBIPtr = here->VBICbaseBPBaseBIStructPtr->CSC_Complex ;

            if ((here->VBICsubsSINode != 0) && (here->VBICcollCINode != 0))
                here->VBICsubsSICollCIPtr = here->VBICsubsSICollCIStructPtr->CSC_Complex ;

            if ((here->VBICsubsSINode != 0) && (here->VBICbaseBINode != 0))
                here->VBICsubsSIBaseBIPtr = here->VBICsubsSIBaseBIStructPtr->CSC_Complex ;

            if ((here->VBICsubsSINode != 0) && (here->VBICbaseBPNode != 0))
                here->VBICsubsSIBaseBPPtr = here->VBICsubsSIBaseBPStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
VBICbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    VBICmodel *model = (VBICmodel *)inModel ;
    VBICinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VBIC models */
    for ( ; model != NULL ; model = model->VBICnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VBICinstances ; here != NULL ; here = here->VBICnextInstance)
        {
            if ((here->VBICcollNode != 0) && (here->VBICcollNode != 0))
                here->VBICcollCollPtr = here->VBICcollCollStructPtr->CSC ;

            if ((here->VBICbaseNode != 0) && (here->VBICbaseNode != 0))
                here->VBICbaseBasePtr = here->VBICbaseBaseStructPtr->CSC ;

            if ((here->VBICemitNode != 0) && (here->VBICemitNode != 0))
                here->VBICemitEmitPtr = here->VBICemitEmitStructPtr->CSC ;

            if ((here->VBICsubsNode != 0) && (here->VBICsubsNode != 0))
                here->VBICsubsSubsPtr = here->VBICsubsSubsStructPtr->CSC ;

            if ((here->VBICcollCXNode != 0) && (here->VBICcollCXNode != 0))
                here->VBICcollCXCollCXPtr = here->VBICcollCXCollCXStructPtr->CSC ;

            if ((here->VBICcollCINode != 0) && (here->VBICcollCINode != 0))
                here->VBICcollCICollCIPtr = here->VBICcollCICollCIStructPtr->CSC ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICbaseBXBaseBXPtr = here->VBICbaseBXBaseBXStructPtr->CSC ;

            if ((here->VBICbaseBINode != 0) && (here->VBICbaseBINode != 0))
                here->VBICbaseBIBaseBIPtr = here->VBICbaseBIBaseBIStructPtr->CSC ;

            if ((here->VBICemitEINode != 0) && (here->VBICemitEINode != 0))
                here->VBICemitEIEmitEIPtr = here->VBICemitEIEmitEIStructPtr->CSC ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBPNode != 0))
                here->VBICbaseBPBaseBPPtr = here->VBICbaseBPBaseBPStructPtr->CSC ;

            if ((here->VBICsubsSINode != 0) && (here->VBICsubsSINode != 0))
                here->VBICsubsSISubsSIPtr = here->VBICsubsSISubsSIStructPtr->CSC ;

            if ((here->VBICbaseNode != 0) && (here->VBICemitNode != 0))
                here->VBICbaseEmitPtr = here->VBICbaseEmitStructPtr->CSC ;

            if ((here->VBICemitNode != 0) && (here->VBICbaseNode != 0))
                here->VBICemitBasePtr = here->VBICemitBaseStructPtr->CSC ;

            if ((here->VBICbaseNode != 0) && (here->VBICcollNode != 0))
                here->VBICbaseCollPtr = here->VBICbaseCollStructPtr->CSC ;

            if ((here->VBICcollNode != 0) && (here->VBICbaseNode != 0))
                here->VBICcollBasePtr = here->VBICcollBaseStructPtr->CSC ;

            if ((here->VBICcollNode != 0) && (here->VBICcollCXNode != 0))
                here->VBICcollCollCXPtr = here->VBICcollCollCXStructPtr->CSC ;

            if ((here->VBICbaseNode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICbaseBaseBXPtr = here->VBICbaseBaseBXStructPtr->CSC ;

            if ((here->VBICemitNode != 0) && (here->VBICemitEINode != 0))
                here->VBICemitEmitEIPtr = here->VBICemitEmitEIStructPtr->CSC ;

            if ((here->VBICsubsNode != 0) && (here->VBICsubsSINode != 0))
                here->VBICsubsSubsSIPtr = here->VBICsubsSubsSIStructPtr->CSC ;

            if ((here->VBICcollCXNode != 0) && (here->VBICcollCINode != 0))
                here->VBICcollCXCollCIPtr = here->VBICcollCXCollCIStructPtr->CSC ;

            if ((here->VBICcollCXNode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICcollCXBaseBXPtr = here->VBICcollCXBaseBXStructPtr->CSC ;

            if ((here->VBICcollCXNode != 0) && (here->VBICbaseBINode != 0))
                here->VBICcollCXBaseBIPtr = here->VBICcollCXBaseBIStructPtr->CSC ;

            if ((here->VBICcollCXNode != 0) && (here->VBICbaseBPNode != 0))
                here->VBICcollCXBaseBPPtr = here->VBICcollCXBaseBPStructPtr->CSC ;

            if ((here->VBICcollCINode != 0) && (here->VBICbaseBINode != 0))
                here->VBICcollCIBaseBIPtr = here->VBICcollCIBaseBIStructPtr->CSC ;

            if ((here->VBICcollCINode != 0) && (here->VBICemitEINode != 0))
                here->VBICcollCIEmitEIPtr = here->VBICcollCIEmitEIStructPtr->CSC ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBINode != 0))
                here->VBICbaseBXBaseBIPtr = here->VBICbaseBXBaseBIStructPtr->CSC ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICemitEINode != 0))
                here->VBICbaseBXEmitEIPtr = here->VBICbaseBXEmitEIStructPtr->CSC ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBPNode != 0))
                here->VBICbaseBXBaseBPPtr = here->VBICbaseBXBaseBPStructPtr->CSC ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICsubsSINode != 0))
                here->VBICbaseBXSubsSIPtr = here->VBICbaseBXSubsSIStructPtr->CSC ;

            if ((here->VBICbaseBINode != 0) && (here->VBICemitEINode != 0))
                here->VBICbaseBIEmitEIPtr = here->VBICbaseBIEmitEIStructPtr->CSC ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICsubsSINode != 0))
                here->VBICbaseBPSubsSIPtr = here->VBICbaseBPSubsSIStructPtr->CSC ;

            if ((here->VBICcollCXNode != 0) && (here->VBICcollNode != 0))
                here->VBICcollCXCollPtr = here->VBICcollCXCollStructPtr->CSC ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICbaseNode != 0))
                here->VBICbaseBXBasePtr = here->VBICbaseBXBaseStructPtr->CSC ;

            if ((here->VBICemitEINode != 0) && (here->VBICemitNode != 0))
                here->VBICemitEIEmitPtr = here->VBICemitEIEmitStructPtr->CSC ;

            if ((here->VBICsubsSINode != 0) && (here->VBICsubsNode != 0))
                here->VBICsubsSISubsPtr = here->VBICsubsSISubsStructPtr->CSC ;

            if ((here->VBICcollCINode != 0) && (here->VBICcollCXNode != 0))
                here->VBICcollCICollCXPtr = here->VBICcollCICollCXStructPtr->CSC ;

            if ((here->VBICbaseBINode != 0) && (here->VBICcollCXNode != 0))
                here->VBICbaseBICollCXPtr = here->VBICbaseBICollCXStructPtr->CSC ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICcollCXNode != 0))
                here->VBICbaseBPCollCXPtr = here->VBICbaseBPCollCXStructPtr->CSC ;

            if ((here->VBICbaseBXNode != 0) && (here->VBICcollCINode != 0))
                here->VBICbaseBXCollCIPtr = here->VBICbaseBXCollCIStructPtr->CSC ;

            if ((here->VBICbaseBINode != 0) && (here->VBICcollCINode != 0))
                here->VBICbaseBICollCIPtr = here->VBICbaseBICollCIStructPtr->CSC ;

            if ((here->VBICemitEINode != 0) && (here->VBICcollCINode != 0))
                here->VBICemitEICollCIPtr = here->VBICemitEICollCIStructPtr->CSC ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICcollCINode != 0))
                here->VBICbaseBPCollCIPtr = here->VBICbaseBPCollCIStructPtr->CSC ;

            if ((here->VBICbaseBINode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICbaseBIBaseBXPtr = here->VBICbaseBIBaseBXStructPtr->CSC ;

            if ((here->VBICemitEINode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICemitEIBaseBXPtr = here->VBICemitEIBaseBXStructPtr->CSC ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICbaseBPBaseBXPtr = here->VBICbaseBPBaseBXStructPtr->CSC ;

            if ((here->VBICsubsSINode != 0) && (here->VBICbaseBXNode != 0))
                here->VBICsubsSIBaseBXPtr = here->VBICsubsSIBaseBXStructPtr->CSC ;

            if ((here->VBICemitEINode != 0) && (here->VBICbaseBINode != 0))
                here->VBICemitEIBaseBIPtr = here->VBICemitEIBaseBIStructPtr->CSC ;

            if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBINode != 0))
                here->VBICbaseBPBaseBIPtr = here->VBICbaseBPBaseBIStructPtr->CSC ;

            if ((here->VBICsubsSINode != 0) && (here->VBICcollCINode != 0))
                here->VBICsubsSICollCIPtr = here->VBICsubsSICollCIStructPtr->CSC ;

            if ((here->VBICsubsSINode != 0) && (here->VBICbaseBINode != 0))
                here->VBICsubsSIBaseBIPtr = here->VBICsubsSIBaseBIStructPtr->CSC ;

            if ((here->VBICsubsSINode != 0) && (here->VBICbaseBPNode != 0))
                here->VBICsubsSIBaseBPPtr = here->VBICsubsSIBaseBPStructPtr->CSC ;

        }
    }

    return (OK) ;
}
