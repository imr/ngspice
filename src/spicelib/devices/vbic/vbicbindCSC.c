/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
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
            CREATE_KLU_BINDING_TABLE(VBICcollCollPtr, VBICcollCollBinding, VBICcollNode, VBICcollNode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBasePtr, VBICbaseBaseBinding, VBICbaseNode, VBICbaseNode);
            CREATE_KLU_BINDING_TABLE(VBICemitEmitPtr, VBICemitEmitBinding, VBICemitNode, VBICemitNode);
            CREATE_KLU_BINDING_TABLE(VBICsubsSubsPtr, VBICsubsSubsBinding, VBICsubsNode, VBICsubsNode);
            CREATE_KLU_BINDING_TABLE(VBICcollCXCollCXPtr, VBICcollCXCollCXBinding, VBICcollCXNode, VBICcollCXNode);
            CREATE_KLU_BINDING_TABLE(VBICcollCICollCIPtr, VBICcollCICollCIBinding, VBICcollCINode, VBICcollCINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBXBaseBXPtr, VBICbaseBXBaseBXBinding, VBICbaseBXNode, VBICbaseBXNode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBIBaseBIPtr, VBICbaseBIBaseBIBinding, VBICbaseBINode, VBICbaseBINode);
            CREATE_KLU_BINDING_TABLE(VBICemitEIEmitEIPtr, VBICemitEIEmitEIBinding, VBICemitEINode, VBICemitEINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBPBaseBPPtr, VBICbaseBPBaseBPBinding, VBICbaseBPNode, VBICbaseBPNode);
            CREATE_KLU_BINDING_TABLE(VBICsubsSISubsSIPtr, VBICsubsSISubsSIBinding, VBICsubsSINode, VBICsubsSINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseEmitPtr, VBICbaseEmitBinding, VBICbaseNode, VBICemitNode);
            CREATE_KLU_BINDING_TABLE(VBICemitBasePtr, VBICemitBaseBinding, VBICemitNode, VBICbaseNode);
            CREATE_KLU_BINDING_TABLE(VBICbaseCollPtr, VBICbaseCollBinding, VBICbaseNode, VBICcollNode);
            CREATE_KLU_BINDING_TABLE(VBICcollBasePtr, VBICcollBaseBinding, VBICcollNode, VBICbaseNode);
            CREATE_KLU_BINDING_TABLE(VBICcollCollCXPtr, VBICcollCollCXBinding, VBICcollNode, VBICcollCXNode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBaseBXPtr, VBICbaseBaseBXBinding, VBICbaseNode, VBICbaseBXNode);
            CREATE_KLU_BINDING_TABLE(VBICemitEmitEIPtr, VBICemitEmitEIBinding, VBICemitNode, VBICemitEINode);
            CREATE_KLU_BINDING_TABLE(VBICsubsSubsSIPtr, VBICsubsSubsSIBinding, VBICsubsNode, VBICsubsSINode);
            CREATE_KLU_BINDING_TABLE(VBICcollCXCollCIPtr, VBICcollCXCollCIBinding, VBICcollCXNode, VBICcollCINode);
            CREATE_KLU_BINDING_TABLE(VBICcollCXBaseBXPtr, VBICcollCXBaseBXBinding, VBICcollCXNode, VBICbaseBXNode);
            CREATE_KLU_BINDING_TABLE(VBICcollCXBaseBIPtr, VBICcollCXBaseBIBinding, VBICcollCXNode, VBICbaseBINode);
            CREATE_KLU_BINDING_TABLE(VBICcollCXBaseBPPtr, VBICcollCXBaseBPBinding, VBICcollCXNode, VBICbaseBPNode);
            CREATE_KLU_BINDING_TABLE(VBICcollCIBaseBIPtr, VBICcollCIBaseBIBinding, VBICcollCINode, VBICbaseBINode);
            CREATE_KLU_BINDING_TABLE(VBICcollCIEmitEIPtr, VBICcollCIEmitEIBinding, VBICcollCINode, VBICemitEINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBXBaseBIPtr, VBICbaseBXBaseBIBinding, VBICbaseBXNode, VBICbaseBINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBXEmitEIPtr, VBICbaseBXEmitEIBinding, VBICbaseBXNode, VBICemitEINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBXBaseBPPtr, VBICbaseBXBaseBPBinding, VBICbaseBXNode, VBICbaseBPNode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBXSubsSIPtr, VBICbaseBXSubsSIBinding, VBICbaseBXNode, VBICsubsSINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBIEmitEIPtr, VBICbaseBIEmitEIBinding, VBICbaseBINode, VBICemitEINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBPSubsSIPtr, VBICbaseBPSubsSIBinding, VBICbaseBPNode, VBICsubsSINode);
            CREATE_KLU_BINDING_TABLE(VBICcollCXCollPtr, VBICcollCXCollBinding, VBICcollCXNode, VBICcollNode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBXBasePtr, VBICbaseBXBaseBinding, VBICbaseBXNode, VBICbaseNode);
            CREATE_KLU_BINDING_TABLE(VBICemitEIEmitPtr, VBICemitEIEmitBinding, VBICemitEINode, VBICemitNode);
            CREATE_KLU_BINDING_TABLE(VBICsubsSISubsPtr, VBICsubsSISubsBinding, VBICsubsSINode, VBICsubsNode);
            CREATE_KLU_BINDING_TABLE(VBICcollCICollCXPtr, VBICcollCICollCXBinding, VBICcollCINode, VBICcollCXNode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBICollCXPtr, VBICbaseBICollCXBinding, VBICbaseBINode, VBICcollCXNode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBPCollCXPtr, VBICbaseBPCollCXBinding, VBICbaseBPNode, VBICcollCXNode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBXCollCIPtr, VBICbaseBXCollCIBinding, VBICbaseBXNode, VBICcollCINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBICollCIPtr, VBICbaseBICollCIBinding, VBICbaseBINode, VBICcollCINode);
            CREATE_KLU_BINDING_TABLE(VBICemitEICollCIPtr, VBICemitEICollCIBinding, VBICemitEINode, VBICcollCINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBPCollCIPtr, VBICbaseBPCollCIBinding, VBICbaseBPNode, VBICcollCINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBIBaseBXPtr, VBICbaseBIBaseBXBinding, VBICbaseBINode, VBICbaseBXNode);
            CREATE_KLU_BINDING_TABLE(VBICemitEIBaseBXPtr, VBICemitEIBaseBXBinding, VBICemitEINode, VBICbaseBXNode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBPBaseBXPtr, VBICbaseBPBaseBXBinding, VBICbaseBPNode, VBICbaseBXNode);
            CREATE_KLU_BINDING_TABLE(VBICsubsSIBaseBXPtr, VBICsubsSIBaseBXBinding, VBICsubsSINode, VBICbaseBXNode);
            CREATE_KLU_BINDING_TABLE(VBICemitEIBaseBIPtr, VBICemitEIBaseBIBinding, VBICemitEINode, VBICbaseBINode);
            CREATE_KLU_BINDING_TABLE(VBICbaseBPBaseBIPtr, VBICbaseBPBaseBIBinding, VBICbaseBPNode, VBICbaseBINode);
            CREATE_KLU_BINDING_TABLE(VBICsubsSICollCIPtr, VBICsubsSICollCIBinding, VBICsubsSINode, VBICcollCINode);
            CREATE_KLU_BINDING_TABLE(VBICsubsSIBaseBIPtr, VBICsubsSIBaseBIBinding, VBICsubsSINode, VBICbaseBINode);
            CREATE_KLU_BINDING_TABLE(VBICsubsSIBaseBPPtr, VBICsubsSIBaseBPBinding, VBICsubsSINode, VBICbaseBPNode);
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
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCollPtr, VBICcollCollBinding, VBICcollNode, VBICcollNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBasePtr, VBICbaseBaseBinding, VBICbaseNode, VBICbaseNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICemitEmitPtr, VBICemitEmitBinding, VBICemitNode, VBICemitNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICsubsSubsPtr, VBICsubsSubsBinding, VBICsubsNode, VBICsubsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCXCollCXPtr, VBICcollCXCollCXBinding, VBICcollCXNode, VBICcollCXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCICollCIPtr, VBICcollCICollCIBinding, VBICcollCINode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBXBaseBXPtr, VBICbaseBXBaseBXBinding, VBICbaseBXNode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBIBaseBIPtr, VBICbaseBIBaseBIBinding, VBICbaseBINode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICemitEIEmitEIPtr, VBICemitEIEmitEIBinding, VBICemitEINode, VBICemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBPBaseBPPtr, VBICbaseBPBaseBPBinding, VBICbaseBPNode, VBICbaseBPNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICsubsSISubsSIPtr, VBICsubsSISubsSIBinding, VBICsubsSINode, VBICsubsSINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseEmitPtr, VBICbaseEmitBinding, VBICbaseNode, VBICemitNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICemitBasePtr, VBICemitBaseBinding, VBICemitNode, VBICbaseNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseCollPtr, VBICbaseCollBinding, VBICbaseNode, VBICcollNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollBasePtr, VBICcollBaseBinding, VBICcollNode, VBICbaseNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCollCXPtr, VBICcollCollCXBinding, VBICcollNode, VBICcollCXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBaseBXPtr, VBICbaseBaseBXBinding, VBICbaseNode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICemitEmitEIPtr, VBICemitEmitEIBinding, VBICemitNode, VBICemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICsubsSubsSIPtr, VBICsubsSubsSIBinding, VBICsubsNode, VBICsubsSINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCXCollCIPtr, VBICcollCXCollCIBinding, VBICcollCXNode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCXBaseBXPtr, VBICcollCXBaseBXBinding, VBICcollCXNode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCXBaseBIPtr, VBICcollCXBaseBIBinding, VBICcollCXNode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCXBaseBPPtr, VBICcollCXBaseBPBinding, VBICcollCXNode, VBICbaseBPNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCIBaseBIPtr, VBICcollCIBaseBIBinding, VBICcollCINode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCIEmitEIPtr, VBICcollCIEmitEIBinding, VBICcollCINode, VBICemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBXBaseBIPtr, VBICbaseBXBaseBIBinding, VBICbaseBXNode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBXEmitEIPtr, VBICbaseBXEmitEIBinding, VBICbaseBXNode, VBICemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBXBaseBPPtr, VBICbaseBXBaseBPBinding, VBICbaseBXNode, VBICbaseBPNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBXSubsSIPtr, VBICbaseBXSubsSIBinding, VBICbaseBXNode, VBICsubsSINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBIEmitEIPtr, VBICbaseBIEmitEIBinding, VBICbaseBINode, VBICemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBPSubsSIPtr, VBICbaseBPSubsSIBinding, VBICbaseBPNode, VBICsubsSINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCXCollPtr, VBICcollCXCollBinding, VBICcollCXNode, VBICcollNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBXBasePtr, VBICbaseBXBaseBinding, VBICbaseBXNode, VBICbaseNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICemitEIEmitPtr, VBICemitEIEmitBinding, VBICemitEINode, VBICemitNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICsubsSISubsPtr, VBICsubsSISubsBinding, VBICsubsSINode, VBICsubsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCICollCXPtr, VBICcollCICollCXBinding, VBICcollCINode, VBICcollCXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBICollCXPtr, VBICbaseBICollCXBinding, VBICbaseBINode, VBICcollCXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBPCollCXPtr, VBICbaseBPCollCXBinding, VBICbaseBPNode, VBICcollCXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBXCollCIPtr, VBICbaseBXCollCIBinding, VBICbaseBXNode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBICollCIPtr, VBICbaseBICollCIBinding, VBICbaseBINode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICemitEICollCIPtr, VBICemitEICollCIBinding, VBICemitEINode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBPCollCIPtr, VBICbaseBPCollCIBinding, VBICbaseBPNode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBIBaseBXPtr, VBICbaseBIBaseBXBinding, VBICbaseBINode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICemitEIBaseBXPtr, VBICemitEIBaseBXBinding, VBICemitEINode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBPBaseBXPtr, VBICbaseBPBaseBXBinding, VBICbaseBPNode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICsubsSIBaseBXPtr, VBICsubsSIBaseBXBinding, VBICsubsSINode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICemitEIBaseBIPtr, VBICemitEIBaseBIBinding, VBICemitEINode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBPBaseBIPtr, VBICbaseBPBaseBIBinding, VBICbaseBPNode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICsubsSICollCIPtr, VBICsubsSICollCIBinding, VBICsubsSINode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICsubsSIBaseBIPtr, VBICsubsSIBaseBIBinding, VBICsubsSINode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICsubsSIBaseBPPtr, VBICsubsSIBaseBPBinding, VBICsubsSINode, VBICbaseBPNode);
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
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCollPtr, VBICcollCollBinding, VBICcollNode, VBICcollNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBasePtr, VBICbaseBaseBinding, VBICbaseNode, VBICbaseNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICemitEmitPtr, VBICemitEmitBinding, VBICemitNode, VBICemitNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICsubsSubsPtr, VBICsubsSubsBinding, VBICsubsNode, VBICsubsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCXCollCXPtr, VBICcollCXCollCXBinding, VBICcollCXNode, VBICcollCXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCICollCIPtr, VBICcollCICollCIBinding, VBICcollCINode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBXBaseBXPtr, VBICbaseBXBaseBXBinding, VBICbaseBXNode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBIBaseBIPtr, VBICbaseBIBaseBIBinding, VBICbaseBINode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICemitEIEmitEIPtr, VBICemitEIEmitEIBinding, VBICemitEINode, VBICemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBPBaseBPPtr, VBICbaseBPBaseBPBinding, VBICbaseBPNode, VBICbaseBPNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICsubsSISubsSIPtr, VBICsubsSISubsSIBinding, VBICsubsSINode, VBICsubsSINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseEmitPtr, VBICbaseEmitBinding, VBICbaseNode, VBICemitNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICemitBasePtr, VBICemitBaseBinding, VBICemitNode, VBICbaseNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseCollPtr, VBICbaseCollBinding, VBICbaseNode, VBICcollNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollBasePtr, VBICcollBaseBinding, VBICcollNode, VBICbaseNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCollCXPtr, VBICcollCollCXBinding, VBICcollNode, VBICcollCXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBaseBXPtr, VBICbaseBaseBXBinding, VBICbaseNode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICemitEmitEIPtr, VBICemitEmitEIBinding, VBICemitNode, VBICemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICsubsSubsSIPtr, VBICsubsSubsSIBinding, VBICsubsNode, VBICsubsSINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCXCollCIPtr, VBICcollCXCollCIBinding, VBICcollCXNode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCXBaseBXPtr, VBICcollCXBaseBXBinding, VBICcollCXNode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCXBaseBIPtr, VBICcollCXBaseBIBinding, VBICcollCXNode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCXBaseBPPtr, VBICcollCXBaseBPBinding, VBICcollCXNode, VBICbaseBPNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCIBaseBIPtr, VBICcollCIBaseBIBinding, VBICcollCINode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCIEmitEIPtr, VBICcollCIEmitEIBinding, VBICcollCINode, VBICemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBXBaseBIPtr, VBICbaseBXBaseBIBinding, VBICbaseBXNode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBXEmitEIPtr, VBICbaseBXEmitEIBinding, VBICbaseBXNode, VBICemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBXBaseBPPtr, VBICbaseBXBaseBPBinding, VBICbaseBXNode, VBICbaseBPNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBXSubsSIPtr, VBICbaseBXSubsSIBinding, VBICbaseBXNode, VBICsubsSINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBIEmitEIPtr, VBICbaseBIEmitEIBinding, VBICbaseBINode, VBICemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBPSubsSIPtr, VBICbaseBPSubsSIBinding, VBICbaseBPNode, VBICsubsSINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCXCollPtr, VBICcollCXCollBinding, VBICcollCXNode, VBICcollNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBXBasePtr, VBICbaseBXBaseBinding, VBICbaseBXNode, VBICbaseNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICemitEIEmitPtr, VBICemitEIEmitBinding, VBICemitEINode, VBICemitNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICsubsSISubsPtr, VBICsubsSISubsBinding, VBICsubsSINode, VBICsubsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCICollCXPtr, VBICcollCICollCXBinding, VBICcollCINode, VBICcollCXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBICollCXPtr, VBICbaseBICollCXBinding, VBICbaseBINode, VBICcollCXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBPCollCXPtr, VBICbaseBPCollCXBinding, VBICbaseBPNode, VBICcollCXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBXCollCIPtr, VBICbaseBXCollCIBinding, VBICbaseBXNode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBICollCIPtr, VBICbaseBICollCIBinding, VBICbaseBINode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICemitEICollCIPtr, VBICemitEICollCIBinding, VBICemitEINode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBPCollCIPtr, VBICbaseBPCollCIBinding, VBICbaseBPNode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBIBaseBXPtr, VBICbaseBIBaseBXBinding, VBICbaseBINode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICemitEIBaseBXPtr, VBICemitEIBaseBXBinding, VBICemitEINode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBPBaseBXPtr, VBICbaseBPBaseBXBinding, VBICbaseBPNode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICsubsSIBaseBXPtr, VBICsubsSIBaseBXBinding, VBICsubsSINode, VBICbaseBXNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICemitEIBaseBIPtr, VBICemitEIBaseBIBinding, VBICemitEINode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBPBaseBIPtr, VBICbaseBPBaseBIBinding, VBICbaseBPNode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICsubsSICollCIPtr, VBICsubsSICollCIBinding, VBICsubsSINode, VBICcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICsubsSIBaseBIPtr, VBICsubsSIBaseBIBinding, VBICsubsSINode, VBICbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICsubsSIBaseBPPtr, VBICsubsSIBaseBPBinding, VBICsubsSINode, VBICbaseBPNode);
        }
    }

    return (OK) ;
}
