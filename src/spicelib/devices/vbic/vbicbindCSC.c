/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

int
VBICbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    VBICmodel *model = (VBICmodel *)inModel ;
    VBICinstance *here ;
    BindElement i, *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
    nz = (size_t)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixLinkedListNZ ;

    /* loop through all the VBIC models */
    for ( ; model != NULL ; model = VBICnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VBICinstances(model); here != NULL ; here = VBICnextInstance(here))
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

            if (here->VBIC_selfheat) {
                CREATE_KLU_BINDING_TABLE(VBICcollTempPtr, VBICcollTempBinding, VBICcollNode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICbaseTempPtr, VBICbaseTempBinding, VBICbaseNode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICemitTempPtr, VBICemitTempBinding, VBICemitNode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICsubsTempPtr, VBICsubsTempBinding, VBICsubsNode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICcollCItempPtr, VBICcollCItempBinding, VBICcollCINode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICcollCXtempPtr, VBICcollCXtempBinding, VBICcollCXNode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICbaseBItempPtr, VBICbaseBItempBinding, VBICbaseBINode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICbaseBXtempPtr, VBICbaseBXtempBinding, VBICbaseBXNode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICbaseBPtempPtr, VBICbaseBPtempBinding, VBICbaseBPNode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICemitEItempPtr, VBICemitEItempBinding, VBICemitEINode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICsubsSItempPtr, VBICsubsSItempBinding, VBICsubsSINode, VBICtempNode);
                CREATE_KLU_BINDING_TABLE(VBICtempCollPtr, VBICtempCollBinding, VBICtempNode, VBICcollNode);
                CREATE_KLU_BINDING_TABLE(VBICtempCollCIPtr, VBICtempCollCIBinding, VBICtempNode, VBICcollCINode);
                CREATE_KLU_BINDING_TABLE(VBICtempCollCXPtr, VBICtempCollCXBinding, VBICtempNode, VBICcollCXNode);
                CREATE_KLU_BINDING_TABLE(VBICtempBaseBIPtr, VBICtempBaseBIBinding, VBICtempNode, VBICbaseBINode);
                CREATE_KLU_BINDING_TABLE(VBICtempBasePtr, VBICtempBaseBinding, VBICtempNode, VBICbaseNode);
                CREATE_KLU_BINDING_TABLE(VBICtempBaseBXPtr, VBICtempBaseBXBinding, VBICtempNode, VBICbaseBXNode);
                CREATE_KLU_BINDING_TABLE(VBICtempBaseBPPtr, VBICtempBaseBPBinding, VBICtempNode, VBICbaseBPNode);
                CREATE_KLU_BINDING_TABLE(VBICtempEmitPtr, VBICtempEmitBinding, VBICtempNode, VBICemitNode);
                CREATE_KLU_BINDING_TABLE(VBICtempEmitEIPtr, VBICtempEmitEIBinding, VBICtempNode, VBICemitEINode);
                CREATE_KLU_BINDING_TABLE(VBICtempSubsPtr, VBICtempSubsBinding, VBICtempNode, VBICsubsNode);
                CREATE_KLU_BINDING_TABLE(VBICtempSubsSIPtr, VBICtempSubsSIBinding, VBICtempNode, VBICsubsSINode);
                CREATE_KLU_BINDING_TABLE(VBICtempTempPtr, VBICtempTempBinding, VBICtempNode, VBICtempNode);
                if (here->VBIC_excessPhase) {
                    CREATE_KLU_BINDING_TABLE(VBICtempXf2Ptr, VBICtempXf2Binding, VBICtempNode, VBICxf2Node);
                    CREATE_KLU_BINDING_TABLE(VBICxf1TempPtr, VBICxf1TempBinding, VBICxf1Node ,VBICtempNode);
                }
            }

            if (here->VBIC_excessPhase) {
                CREATE_KLU_BINDING_TABLE(VBICxf1Xf1Ptr   , VBICxf1Xf1Binding   , VBICxf1Node   , VBICxf1Node);
                CREATE_KLU_BINDING_TABLE(VBICxf1Xf2Ptr   , VBICxf1Xf2Binding   , VBICxf1Node   , VBICxf2Node);
                CREATE_KLU_BINDING_TABLE(VBICxf1CollCIPtr, VBICxf1CollCIBinding, VBICxf1Node   , VBICcollCINode);
                CREATE_KLU_BINDING_TABLE(VBICxf1BaseBIPtr, VBICxf1BaseBIBinding, VBICxf1Node   , VBICbaseBINode);
                CREATE_KLU_BINDING_TABLE(VBICxf1EmitEIPtr, VBICxf1EmitEIBinding, VBICxf1Node   , VBICemitEINode);
                CREATE_KLU_BINDING_TABLE(VBICxf2Xf2Ptr   , VBICxf2Xf2Binding   , VBICxf2Node   , VBICxf2Node);
                CREATE_KLU_BINDING_TABLE(VBICxf2Xf1Ptr   , VBICxf2Xf1Binding   , VBICxf2Node   , VBICxf1Node);
                CREATE_KLU_BINDING_TABLE(VBICcollCIXf2Ptr, VBICcollCIXf2Binding, VBICcollCINode, VBICxf2Node);
                CREATE_KLU_BINDING_TABLE(VBICbaseBIXf2Ptr, VBICbaseBIXf2Binding, VBICbaseBINode, VBICxf2Node);
                CREATE_KLU_BINDING_TABLE(VBICemitEIXf2Ptr, VBICemitEIXf2Binding, VBICemitEINode, VBICxf2Node);
                CREATE_KLU_BINDING_TABLE(VBICxf1IbrPtr   , VBICxf1IbrBinding   , VBICxf1Node   , VBICbrEq);
                CREATE_KLU_BINDING_TABLE(VBICxf2IbrPtr   , VBICxf2IbrBinding   , VBICxf2Node   , VBICbrEq);
                CREATE_KLU_BINDING_TABLE(VBICibrXf2Ptr   , VBICibrXf2Binding   , VBICbrEq      , VBICxf2Node);
                CREATE_KLU_BINDING_TABLE(VBICibrXf1Ptr   , VBICibrXf1Binding   , VBICbrEq      , VBICxf1Node);
                CREATE_KLU_BINDING_TABLE(VBICibrIbrPtr   , VBICibrIbrBinding   , VBICbrEq      , VBICbrEq);
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
    for ( ; model != NULL ; model = VBICnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VBICinstances(model); here != NULL ; here = VBICnextInstance(here))
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

            if (here->VBIC_selfheat) {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollTempPtr, VBICcollTempBinding, VBICcollNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseTempPtr, VBICbaseTempBinding, VBICbaseNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICemitTempPtr, VBICemitTempBinding, VBICemitNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICsubsTempPtr, VBICsubsTempBinding, VBICsubsNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCItempPtr, VBICcollCItempBinding, VBICcollCINode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICcollCXtempPtr, VBICcollCXtempBinding, VBICcollCXNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBItempPtr, VBICbaseBItempBinding, VBICbaseBINode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBXtempPtr, VBICbaseBXtempBinding, VBICbaseBXNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICbaseBPtempPtr, VBICbaseBPtempBinding, VBICbaseBPNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICemitEItempPtr, VBICemitEItempBinding, VBICemitEINode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICsubsSItempPtr, VBICsubsSItempBinding, VBICsubsSINode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempCollPtr, VBICtempCollBinding, VBICtempNode, VBICcollNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempCollCIPtr, VBICtempCollCIBinding, VBICtempNode, VBICcollCINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempCollCXPtr, VBICtempCollCXBinding, VBICtempNode, VBICcollCXNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempBaseBIPtr, VBICtempBaseBIBinding, VBICtempNode, VBICbaseBINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempBasePtr, VBICtempBaseBinding, VBICtempNode, VBICbaseNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempBaseBXPtr, VBICtempBaseBXBinding, VBICtempNode, VBICbaseBXNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempBaseBPPtr, VBICtempBaseBPBinding, VBICtempNode, VBICbaseBPNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempEmitPtr, VBICtempEmitBinding, VBICtempNode, VBICemitNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempEmitEIPtr, VBICtempEmitEIBinding, VBICtempNode, VBICemitEINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempSubsPtr, VBICtempSubsBinding, VBICtempNode, VBICsubsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempSubsSIPtr, VBICtempSubsSIBinding, VBICtempNode, VBICsubsSINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VBICtempTempPtr, VBICtempTempBinding, VBICtempNode, VBICtempNode);
            }
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
    for ( ; model != NULL ; model = VBICnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VBICinstances(model); here != NULL ; here = VBICnextInstance(here))
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

            if (here->VBIC_selfheat) {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollTempPtr, VBICcollTempBinding, VBICcollNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseTempPtr, VBICbaseTempBinding, VBICbaseNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICemitTempPtr, VBICemitTempBinding, VBICemitNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICsubsTempPtr, VBICsubsTempBinding, VBICsubsNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCItempPtr, VBICcollCItempBinding, VBICcollCINode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICcollCXtempPtr, VBICcollCXtempBinding, VBICcollCXNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBItempPtr, VBICbaseBItempBinding, VBICbaseBINode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBXtempPtr, VBICbaseBXtempBinding, VBICbaseBXNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICbaseBPtempPtr, VBICbaseBPtempBinding, VBICbaseBPNode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICemitEItempPtr, VBICemitEItempBinding, VBICemitEINode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICsubsSItempPtr, VBICsubsSItempBinding, VBICsubsSINode, VBICtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempCollPtr, VBICtempCollBinding, VBICtempNode, VBICcollNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempCollCIPtr, VBICtempCollCIBinding, VBICtempNode, VBICcollCINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempCollCXPtr, VBICtempCollCXBinding, VBICtempNode, VBICcollCXNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempBaseBIPtr, VBICtempBaseBIBinding, VBICtempNode, VBICbaseBINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempBasePtr, VBICtempBaseBinding, VBICtempNode, VBICbaseNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempBaseBXPtr, VBICtempBaseBXBinding, VBICtempNode, VBICbaseBXNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempBaseBPPtr, VBICtempBaseBPBinding, VBICtempNode, VBICbaseBPNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempEmitPtr, VBICtempEmitBinding, VBICtempNode, VBICemitNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempEmitEIPtr, VBICtempEmitEIBinding, VBICtempNode, VBICemitEINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempSubsPtr, VBICtempSubsBinding, VBICtempNode, VBICsubsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempSubsSIPtr, VBICtempSubsSIBinding, VBICtempNode, VBICsubsSINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VBICtempTempPtr, VBICtempTempBinding, VBICtempNode, VBICtempNode);
            }
        }
    }

    return (OK) ;
}
