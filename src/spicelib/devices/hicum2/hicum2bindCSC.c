/**********
Author: 2022 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hicum2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

int
HICUMbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    HICUMmodel *model = (HICUMmodel *)inModel ;
    HICUMinstance *here ;
    BindElement i, *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
    nz = (size_t)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixLinkedListNZ ;

    /* loop through all the HICUM models */
    for ( ; model != NULL ; model = HICUMnextModel(model))
    {
        int selfheat = (((model->HICUMflsh == 1) || (model->HICUMflsh == 2)) && (model->HICUMrthGiven) && (model->HICUMrth > 0.0));
        int nqs      = ( (model->HICUMflnqs != 0 || model->HICUMflcomp < 2.3) && (model->HICUMalit > 0 || model->HICUMalqf > 0));

        /* loop through all the instances of the model */
        for (here = HICUMinstances(model); here != NULL ; here = HICUMnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(HICUMcollCollPtr, HICUMcollCollBinding, HICUMcollNode, HICUMcollNode);
            CREATE_KLU_BINDING_TABLE(HICUMbaseBasePtr, HICUMbaseBaseBinding, HICUMbaseNode, HICUMbaseNode);
            CREATE_KLU_BINDING_TABLE(HICUMemitEmitPtr, HICUMemitEmitBinding, HICUMemitNode, HICUMemitNode);
            CREATE_KLU_BINDING_TABLE(HICUMsubsSubsPtr, HICUMsubsSubsBinding, HICUMsubsNode, HICUMsubsNode);

            CREATE_KLU_BINDING_TABLE(HICUMcollCICollCIPtr, HICUMcollCICollCIBinding, HICUMcollCINode, HICUMcollCINode);
            CREATE_KLU_BINDING_TABLE(HICUMbaseBIBaseBIPtr, HICUMbaseBIBaseBIBinding, HICUMbaseBINode, HICUMbaseBINode);
            CREATE_KLU_BINDING_TABLE(HICUMemitEIEmitEIPtr, HICUMemitEIEmitEIBinding, HICUMemitEINode, HICUMemitEINode);
            CREATE_KLU_BINDING_TABLE(HICUMbaseBPBaseBPPtr, HICUMbaseBPBaseBPBinding, HICUMbaseBPNode, HICUMbaseBPNode);
            CREATE_KLU_BINDING_TABLE(HICUMsubsSISubsSIPtr, HICUMsubsSISubsSIBinding, HICUMsubsSINode, HICUMsubsSINode);

            CREATE_KLU_BINDING_TABLE(HICUMbaseEmitPtr, HICUMbaseEmitBinding, HICUMbaseNode, HICUMemitNode); //b-e
            CREATE_KLU_BINDING_TABLE(HICUMemitBasePtr, HICUMemitBaseBinding, HICUMemitNode, HICUMbaseNode); //e-b

            CREATE_KLU_BINDING_TABLE(HICUMbaseBaseBPPtr, HICUMbaseBaseBPBinding, HICUMbaseNode, HICUMbaseBPNode); //b-bp
            CREATE_KLU_BINDING_TABLE(HICUMbaseBPBasePtr, HICUMbaseBPBaseBinding, HICUMbaseBPNode, HICUMbaseNode); //bp-b
            CREATE_KLU_BINDING_TABLE(HICUMemitEmitEIPtr, HICUMemitEmitEIBinding, HICUMemitNode, HICUMemitEINode); //e-ei
            CREATE_KLU_BINDING_TABLE(HICUMemitEIEmitPtr, HICUMemitEIEmitBinding, HICUMemitEINode, HICUMemitNode); //ei-e

            CREATE_KLU_BINDING_TABLE(HICUMsubsSubsSIPtr, HICUMsubsSubsSIBinding, HICUMsubsNode, HICUMsubsSINode); //s-si
            CREATE_KLU_BINDING_TABLE(HICUMsubsSISubsPtr, HICUMsubsSISubsBinding, HICUMsubsSINode, HICUMsubsNode); //si-s

            CREATE_KLU_BINDING_TABLE(HICUMcollCIBasePtr, HICUMcollCIBaseBinding, HICUMcollCINode, HICUMbaseNode); //b-ci
            CREATE_KLU_BINDING_TABLE(HICUMbaseCollCIPtr, HICUMbaseCollCIBinding, HICUMbaseNode, HICUMcollCINode); //ci-b

            CREATE_KLU_BINDING_TABLE(HICUMcollCIEmitEIPtr, HICUMcollCIEmitEIBinding, HICUMcollCINode, HICUMemitEINode); //ci-ei
            CREATE_KLU_BINDING_TABLE(HICUMemitEICollCIPtr, HICUMemitEICollCIBinding, HICUMemitEINode, HICUMcollCINode); //ei-ci

            CREATE_KLU_BINDING_TABLE(HICUMbaseBPBaseBIPtr, HICUMbaseBPBaseBIBinding, HICUMbaseBPNode, HICUMbaseBINode); //bp-bi
            CREATE_KLU_BINDING_TABLE(HICUMbaseBIBaseBPPtr, HICUMbaseBIBaseBPBinding, HICUMbaseBINode, HICUMbaseBPNode); //bi-bp

            CREATE_KLU_BINDING_TABLE(HICUMbaseBPEmitEIPtr, HICUMbaseBPEmitEIBinding, HICUMbaseBPNode, HICUMemitEINode); //bp-ei
            CREATE_KLU_BINDING_TABLE(HICUMemitEIBaseBPPtr, HICUMemitEIBaseBPBinding, HICUMemitEINode, HICUMbaseBPNode); //ei-bp

            CREATE_KLU_BINDING_TABLE(HICUMbaseBPEmitPtr, HICUMbaseBPEmitBinding, HICUMbaseBPNode, HICUMemitNode); //bp-e
            CREATE_KLU_BINDING_TABLE(HICUMemitBaseBPPtr, HICUMemitBaseBPBinding, HICUMemitNode, HICUMbaseBPNode); //e-bp

            CREATE_KLU_BINDING_TABLE(HICUMbaseBPSubsSIPtr, HICUMbaseBPSubsSIBinding, HICUMbaseBPNode, HICUMsubsSINode); //bp-si
            CREATE_KLU_BINDING_TABLE(HICUMsubsSIBaseBPPtr, HICUMsubsSIBaseBPBinding, HICUMsubsSINode, HICUMbaseBPNode); //si-bp

            CREATE_KLU_BINDING_TABLE(HICUMbaseBIEmitEIPtr, HICUMbaseBIEmitEIBinding, HICUMbaseBINode, HICUMemitEINode); //ei-bi
            CREATE_KLU_BINDING_TABLE(HICUMemitEIBaseBIPtr, HICUMemitEIBaseBIBinding, HICUMemitEINode, HICUMbaseBINode); //bi-ei
            if (nqs) {
                CREATE_KLU_BINDING_TABLE(HICUMbaseBIXfPtr, HICUMbaseBIXfBinding, HICUMbaseBINode, HICUMxfNode); //bi - xf
                CREATE_KLU_BINDING_TABLE(HICUMemitEIXfPtr, HICUMemitEIXfBinding, HICUMemitEINode, HICUMxfNode); //ei - xf
            }

            CREATE_KLU_BINDING_TABLE(HICUMbaseBICollCIPtr, HICUMbaseBICollCIBinding, HICUMbaseBINode, HICUMcollCINode);
            CREATE_KLU_BINDING_TABLE(HICUMcollCIBaseBIPtr, HICUMcollCIBaseBIBinding, HICUMcollCINode, HICUMbaseBINode);

            CREATE_KLU_BINDING_TABLE(HICUMbaseBPCollCIPtr, HICUMbaseBPCollCIBinding, HICUMbaseBPNode, HICUMcollCINode); //bp-ci
            CREATE_KLU_BINDING_TABLE(HICUMcollCIBaseBPPtr, HICUMcollCIBaseBPBinding, HICUMcollCINode, HICUMbaseBPNode); //ci-bp

            CREATE_KLU_BINDING_TABLE(HICUMsubsSICollCIPtr, HICUMsubsSICollCIBinding, HICUMsubsSINode, HICUMcollCINode); //si-ci
            CREATE_KLU_BINDING_TABLE(HICUMcollCISubsSIPtr, HICUMcollCISubsSIBinding, HICUMcollCINode, HICUMsubsSINode); //ci-si

            CREATE_KLU_BINDING_TABLE(HICUMcollCICollPtr, HICUMcollCICollBinding, HICUMcollCINode, HICUMcollNode); //ci-c
            CREATE_KLU_BINDING_TABLE(HICUMcollCollCIPtr, HICUMcollCollCIBinding, HICUMcollNode, HICUMcollCINode); //c-ci

            CREATE_KLU_BINDING_TABLE(HICUMsubsCollPtr, HICUMsubsCollBinding, HICUMsubsNode, HICUMcollNode); //s-c
            CREATE_KLU_BINDING_TABLE(HICUMcollSubsPtr, HICUMcollSubsBinding, HICUMcollNode, HICUMsubsNode); //c-s

            if (nqs) {
                CREATE_KLU_BINDING_TABLE(HICUMxf1Xf1Ptr, HICUMxf1Xf1Binding, HICUMxf1Node, HICUMxf1Node);
                CREATE_KLU_BINDING_TABLE(HICUMxf1BaseBIPtr, HICUMxf1BaseBIBinding, HICUMxf1Node, HICUMbaseBINode);
                CREATE_KLU_BINDING_TABLE(HICUMxf1EmitEIPtr, HICUMxf1EmitEIBinding, HICUMxf1Node, HICUMemitEINode);
                CREATE_KLU_BINDING_TABLE(HICUMxf1CollCIPtr, HICUMxf1CollCIBinding, HICUMxf1Node, HICUMcollCINode);
                CREATE_KLU_BINDING_TABLE(HICUMxf1Xf2Ptr, HICUMxf1Xf2Binding, HICUMxf1Node, HICUMxf2Node);

                CREATE_KLU_BINDING_TABLE(HICUMxf2Xf1Ptr, HICUMxf2Xf1Binding, HICUMxf2Node, HICUMxf1Node);
                CREATE_KLU_BINDING_TABLE(HICUMxf2BaseBIPtr, HICUMxf2BaseBIBinding, HICUMxf2Node, HICUMbaseBINode);
                CREATE_KLU_BINDING_TABLE(HICUMxf2EmitEIPtr, HICUMxf2EmitEIBinding, HICUMxf2Node, HICUMemitEINode);
                CREATE_KLU_BINDING_TABLE(HICUMxf2CollCIPtr, HICUMxf2CollCIBinding, HICUMxf2Node, HICUMcollCINode);
                CREATE_KLU_BINDING_TABLE(HICUMxf2Xf2Ptr, HICUMxf2Xf2Binding, HICUMxf2Node, HICUMxf2Node);
                CREATE_KLU_BINDING_TABLE(HICUMemitEIXf2Ptr, HICUMemitEIXf2Binding, HICUMemitEINode, HICUMxf2Node);
                CREATE_KLU_BINDING_TABLE(HICUMcollCIXf2Ptr, HICUMcollCIXf2Binding, HICUMcollCINode, HICUMxf2Node);

                CREATE_KLU_BINDING_TABLE(HICUMxfXfPtr, HICUMxfXfBinding, HICUMxfNode, HICUMxfNode);
                CREATE_KLU_BINDING_TABLE(HICUMxfEmitEIPtr, HICUMxfEmitEIBinding, HICUMxfNode, HICUMemitEINode);
                CREATE_KLU_BINDING_TABLE(HICUMxfCollCIPtr, HICUMxfCollCIBinding, HICUMxfNode, HICUMcollCINode);
                CREATE_KLU_BINDING_TABLE(HICUMxfBaseBIPtr, HICUMxfBaseBIBinding, HICUMxfNode, HICUMbaseBINode);
            }

            if (selfheat) {
                CREATE_KLU_BINDING_TABLE(HICUMcollTempPtr, HICUMcollTempBinding, HICUMcollNode, HICUMtempNode);
                CREATE_KLU_BINDING_TABLE(HICUMbaseTempPtr, HICUMbaseTempBinding, HICUMbaseNode, HICUMtempNode);
                CREATE_KLU_BINDING_TABLE(HICUMemitTempPtr, HICUMemitTempBinding, HICUMemitNode, HICUMtempNode);

                CREATE_KLU_BINDING_TABLE(HICUMcollCItempPtr, HICUMcollCItempBinding, HICUMcollCINode, HICUMtempNode);
                CREATE_KLU_BINDING_TABLE(HICUMbaseBItempPtr, HICUMbaseBItempBinding, HICUMbaseBINode, HICUMtempNode);
                CREATE_KLU_BINDING_TABLE(HICUMbaseBPtempPtr, HICUMbaseBPtempBinding, HICUMbaseBPNode, HICUMtempNode);
                CREATE_KLU_BINDING_TABLE(HICUMemitEItempPtr, HICUMemitEItempBinding, HICUMemitEINode, HICUMtempNode);
                CREATE_KLU_BINDING_TABLE(HICUMsubsSItempPtr, HICUMsubsSItempBinding, HICUMsubsSINode, HICUMtempNode);
                CREATE_KLU_BINDING_TABLE(HICUMsubsTempPtr, HICUMsubsTempBinding, HICUMsubsNode, HICUMtempNode);
                CREATE_KLU_BINDING_TABLE(HICUMcollTempPtr, HICUMcollTempBinding, HICUMcollNode, HICUMtempNode);

                CREATE_KLU_BINDING_TABLE(HICUMtempCollPtr, HICUMtempCollBinding, HICUMtempNode, HICUMcollNode);
                CREATE_KLU_BINDING_TABLE(HICUMtempBasePtr, HICUMtempBaseBinding, HICUMtempNode, HICUMbaseNode);
                CREATE_KLU_BINDING_TABLE(HICUMtempEmitPtr, HICUMtempEmitBinding, HICUMtempNode, HICUMemitNode);

                CREATE_KLU_BINDING_TABLE(HICUMtempCollCIPtr, HICUMtempCollCIBinding, HICUMtempNode, HICUMcollCINode);
                CREATE_KLU_BINDING_TABLE(HICUMtempBaseBIPtr, HICUMtempBaseBIBinding, HICUMtempNode, HICUMbaseBINode);
                CREATE_KLU_BINDING_TABLE(HICUMtempBaseBPPtr, HICUMtempBaseBPBinding, HICUMtempNode, HICUMbaseBPNode);
                CREATE_KLU_BINDING_TABLE(HICUMtempEmitEIPtr, HICUMtempEmitEIBinding, HICUMtempNode, HICUMemitEINode);
                CREATE_KLU_BINDING_TABLE(HICUMtempSubsSIPtr, HICUMtempSubsSIBinding, HICUMtempNode, HICUMsubsSINode);

                CREATE_KLU_BINDING_TABLE(HICUMtempTempPtr, HICUMtempTempBinding, HICUMtempNode, HICUMtempNode);

                if (nqs) {
                    CREATE_KLU_BINDING_TABLE(HICUMxfTempPtr, HICUMxfTempBinding, HICUMxfNode, HICUMtempNode);
                    CREATE_KLU_BINDING_TABLE(HICUMxf2TempPtr, HICUMxf2TempBinding, HICUMxf2Node, HICUMtempNode);
                    CREATE_KLU_BINDING_TABLE(HICUMxf1TempPtr, HICUMxf1TempBinding, HICUMxf1Node, HICUMtempNode);
                }
            }
        }
    }

    return (OK) ;
}

int
HICUMbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    HICUMmodel *model = (HICUMmodel *)inModel ;
    HICUMinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HICUM models */
    for ( ; model != NULL ; model = HICUMnextModel(model))
    {
        int selfheat = (((model->HICUMflsh == 1) || (model->HICUMflsh == 2)) && (model->HICUMrthGiven) && (model->HICUMrth > 0.0));
        int nqs      = ( (model->HICUMflnqs != 0 || model->HICUMflcomp < 2.3) && (model->HICUMalit > 0 || model->HICUMalqf > 0));

	/* loop through all the instances of the model */
	for (here = HICUMinstances(model); here != NULL ; here = HICUMnextInstance(here))
	{
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCollPtr, HICUMcollCollBinding, HICUMcollNode, HICUMcollNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBasePtr, HICUMbaseBaseBinding, HICUMbaseNode, HICUMbaseNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitEmitPtr, HICUMemitEmitBinding, HICUMemitNode, HICUMemitNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMsubsSubsPtr, HICUMsubsSubsBinding, HICUMsubsNode, HICUMsubsNode);

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCICollCIPtr, HICUMcollCICollCIBinding, HICUMcollCINode, HICUMcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBIBaseBIPtr, HICUMbaseBIBaseBIBinding, HICUMbaseBINode, HICUMbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitEIEmitEIPtr, HICUMemitEIEmitEIBinding, HICUMemitEINode, HICUMemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBPBaseBPPtr, HICUMbaseBPBaseBPBinding, HICUMbaseBPNode, HICUMbaseBPNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMsubsSISubsSIPtr, HICUMsubsSISubsSIBinding, HICUMsubsSINode, HICUMsubsSINode);

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseEmitPtr, HICUMbaseEmitBinding, HICUMbaseNode, HICUMemitNode); //b-e
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitBasePtr, HICUMemitBaseBinding, HICUMemitNode, HICUMbaseNode); //e-b

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBaseBPPtr, HICUMbaseBaseBPBinding, HICUMbaseNode, HICUMbaseBPNode); //b-bp
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBPBasePtr, HICUMbaseBPBaseBinding, HICUMbaseBPNode, HICUMbaseNode); //bp-b
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitEmitEIPtr, HICUMemitEmitEIBinding, HICUMemitNode, HICUMemitEINode); //e-ei
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitEIEmitPtr, HICUMemitEIEmitBinding, HICUMemitEINode, HICUMemitNode); //ei-e

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMsubsSubsSIPtr, HICUMsubsSubsSIBinding, HICUMsubsNode, HICUMsubsSINode); //s-si
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMsubsSISubsPtr, HICUMsubsSISubsBinding, HICUMsubsSINode, HICUMsubsNode); //si-s

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCIBasePtr, HICUMcollCIBaseBinding, HICUMcollCINode, HICUMbaseNode); //b-ci
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseCollCIPtr, HICUMbaseCollCIBinding, HICUMbaseNode, HICUMcollCINode); //ci-b

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCIEmitEIPtr, HICUMcollCIEmitEIBinding, HICUMcollCINode, HICUMemitEINode); //ci-ei
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitEICollCIPtr, HICUMemitEICollCIBinding, HICUMemitEINode, HICUMcollCINode); //ei-ci

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBPBaseBIPtr, HICUMbaseBPBaseBIBinding, HICUMbaseBPNode, HICUMbaseBINode); //bp-bi
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBIBaseBPPtr, HICUMbaseBIBaseBPBinding, HICUMbaseBINode, HICUMbaseBPNode); //bi-bp

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBPEmitEIPtr, HICUMbaseBPEmitEIBinding, HICUMbaseBPNode, HICUMemitEINode); //bp-ei
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitEIBaseBPPtr, HICUMemitEIBaseBPBinding, HICUMemitEINode, HICUMbaseBPNode); //ei-bp

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBPEmitPtr, HICUMbaseBPEmitBinding, HICUMbaseBPNode, HICUMemitNode); //bp-e
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitBaseBPPtr, HICUMemitBaseBPBinding, HICUMemitNode, HICUMbaseBPNode); //e-bp

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBPSubsSIPtr, HICUMbaseBPSubsSIBinding, HICUMbaseBPNode, HICUMsubsSINode); //bp-si
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMsubsSIBaseBPPtr, HICUMsubsSIBaseBPBinding, HICUMsubsSINode, HICUMbaseBPNode); //si-bp

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBIEmitEIPtr, HICUMbaseBIEmitEIBinding, HICUMbaseBINode, HICUMemitEINode); //ei-bi
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitEIBaseBIPtr, HICUMemitEIBaseBIBinding, HICUMemitEINode, HICUMbaseBINode); //bi-ei
            if (nqs) {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBIXfPtr, HICUMbaseBIXfBinding, HICUMbaseBINode, HICUMxfNode); //bi - xf
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitEIXfPtr, HICUMemitEIXfBinding, HICUMemitEINode, HICUMxfNode); //ei - xf
            }

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBICollCIPtr, HICUMbaseBICollCIBinding, HICUMbaseBINode, HICUMcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCIBaseBIPtr, HICUMcollCIBaseBIBinding, HICUMcollCINode, HICUMbaseBINode);

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBPCollCIPtr, HICUMbaseBPCollCIBinding, HICUMbaseBPNode, HICUMcollCINode); //bp-ci
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCIBaseBPPtr, HICUMcollCIBaseBPBinding, HICUMcollCINode, HICUMbaseBPNode); //ci-bp

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMsubsSICollCIPtr, HICUMsubsSICollCIBinding, HICUMsubsSINode, HICUMcollCINode); //si-ci
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCISubsSIPtr, HICUMcollCISubsSIBinding, HICUMcollCINode, HICUMsubsSINode); //ci-si

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCICollPtr, HICUMcollCICollBinding, HICUMcollCINode, HICUMcollNode); //ci-c
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCollCIPtr, HICUMcollCollCIBinding, HICUMcollNode, HICUMcollCINode); //c-ci

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMsubsCollPtr, HICUMsubsCollBinding, HICUMsubsNode, HICUMcollNode); //s-c
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollSubsPtr, HICUMcollSubsBinding, HICUMcollNode, HICUMsubsNode); //c-s

            if (nqs) {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf1Xf1Ptr, HICUMxf1Xf1Binding, HICUMxf1Node, HICUMxf1Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf1BaseBIPtr, HICUMxf1BaseBIBinding, HICUMxf1Node, HICUMbaseBINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf1EmitEIPtr, HICUMxf1EmitEIBinding, HICUMxf1Node, HICUMemitEINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf1CollCIPtr, HICUMxf1CollCIBinding, HICUMxf1Node, HICUMcollCINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf1Xf2Ptr, HICUMxf1Xf2Binding, HICUMxf1Node, HICUMxf2Node);

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf2Xf1Ptr, HICUMxf2Xf1Binding, HICUMxf2Node, HICUMxf1Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf2BaseBIPtr, HICUMxf2BaseBIBinding, HICUMxf2Node, HICUMbaseBINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf2EmitEIPtr, HICUMxf2EmitEIBinding, HICUMxf2Node, HICUMemitEINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf2CollCIPtr, HICUMxf2CollCIBinding, HICUMxf2Node, HICUMcollCINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf2Xf2Ptr, HICUMxf2Xf2Binding, HICUMxf2Node, HICUMxf2Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitEIXf2Ptr, HICUMemitEIXf2Binding, HICUMemitEINode, HICUMxf2Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCIXf2Ptr, HICUMcollCIXf2Binding, HICUMcollCINode, HICUMxf2Node);

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxfXfPtr, HICUMxfXfBinding, HICUMxfNode, HICUMxfNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxfEmitEIPtr, HICUMxfEmitEIBinding, HICUMxfNode, HICUMemitEINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxfCollCIPtr, HICUMxfCollCIBinding, HICUMxfNode, HICUMcollCINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxfBaseBIPtr, HICUMxfBaseBIBinding, HICUMxfNode, HICUMbaseBINode);
            }

            if (selfheat) {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollTempPtr, HICUMcollTempBinding, HICUMcollNode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseTempPtr, HICUMbaseTempBinding, HICUMbaseNode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitTempPtr, HICUMemitTempBinding, HICUMemitNode, HICUMtempNode);

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollCItempPtr, HICUMcollCItempBinding, HICUMcollCINode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBItempPtr, HICUMbaseBItempBinding, HICUMbaseBINode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMbaseBPtempPtr, HICUMbaseBPtempBinding, HICUMbaseBPNode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMemitEItempPtr, HICUMemitEItempBinding, HICUMemitEINode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMsubsSItempPtr, HICUMsubsSItempBinding, HICUMsubsSINode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMsubsTempPtr, HICUMsubsTempBinding, HICUMsubsNode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMcollTempPtr, HICUMcollTempBinding, HICUMcollNode, HICUMtempNode);

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMtempCollPtr, HICUMtempCollBinding, HICUMtempNode, HICUMcollNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMtempBasePtr, HICUMtempBaseBinding, HICUMtempNode, HICUMbaseNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMtempEmitPtr, HICUMtempEmitBinding, HICUMtempNode, HICUMemitNode);

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMtempCollCIPtr, HICUMtempCollCIBinding, HICUMtempNode, HICUMcollCINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMtempBaseBIPtr, HICUMtempBaseBIBinding, HICUMtempNode, HICUMbaseBINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMtempBaseBPPtr, HICUMtempBaseBPBinding, HICUMtempNode, HICUMbaseBPNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMtempEmitEIPtr, HICUMtempEmitEIBinding, HICUMtempNode, HICUMemitEINode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMtempSubsSIPtr, HICUMtempSubsSIBinding, HICUMtempNode, HICUMsubsSINode);

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMtempTempPtr, HICUMtempTempBinding, HICUMtempNode, HICUMtempNode);

                if (nqs) {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxfTempPtr, HICUMxfTempBinding, HICUMxfNode, HICUMtempNode);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf2TempPtr, HICUMxf2TempBinding, HICUMxf2Node, HICUMtempNode);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HICUMxf1TempPtr, HICUMxf1TempBinding, HICUMxf1Node, HICUMtempNode);
                }
            }
        }
    }

    return (OK) ;
}

int
HICUMbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HICUMmodel *model = (HICUMmodel *)inModel ;
    HICUMinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HICUM models */
    for ( ; model != NULL ; model = HICUMnextModel(model))
    {
        int selfheat = (((model->HICUMflsh == 1) || (model->HICUMflsh == 2)) && (model->HICUMrthGiven) && (model->HICUMrth > 0.0));
        int nqs      = ( (model->HICUMflnqs != 0 || model->HICUMflcomp < 2.3) && (model->HICUMalit > 0 || model->HICUMalqf > 0));

        /* loop through all the instances of the model */
        for (here = HICUMinstances(model); here != NULL ; here = HICUMnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCollPtr, HICUMcollCollBinding, HICUMcollNode, HICUMcollNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBasePtr, HICUMbaseBaseBinding, HICUMbaseNode, HICUMbaseNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitEmitPtr, HICUMemitEmitBinding, HICUMemitNode, HICUMemitNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMsubsSubsPtr, HICUMsubsSubsBinding, HICUMsubsNode, HICUMsubsNode);

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCICollCIPtr, HICUMcollCICollCIBinding, HICUMcollCINode, HICUMcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBIBaseBIPtr, HICUMbaseBIBaseBIBinding, HICUMbaseBINode, HICUMbaseBINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitEIEmitEIPtr, HICUMemitEIEmitEIBinding, HICUMemitEINode, HICUMemitEINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBPBaseBPPtr, HICUMbaseBPBaseBPBinding, HICUMbaseBPNode, HICUMbaseBPNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMsubsSISubsSIPtr, HICUMsubsSISubsSIBinding, HICUMsubsSINode, HICUMsubsSINode);

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseEmitPtr, HICUMbaseEmitBinding, HICUMbaseNode, HICUMemitNode); //b-e
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitBasePtr, HICUMemitBaseBinding, HICUMemitNode, HICUMbaseNode); //e-b

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBaseBPPtr, HICUMbaseBaseBPBinding, HICUMbaseNode, HICUMbaseBPNode); //b-bp
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBPBasePtr, HICUMbaseBPBaseBinding, HICUMbaseBPNode, HICUMbaseNode); //bp-b
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitEmitEIPtr, HICUMemitEmitEIBinding, HICUMemitNode, HICUMemitEINode); //e-ei
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitEIEmitPtr, HICUMemitEIEmitBinding, HICUMemitEINode, HICUMemitNode); //ei-e

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMsubsSubsSIPtr, HICUMsubsSubsSIBinding, HICUMsubsNode, HICUMsubsSINode); //s-si
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMsubsSISubsPtr, HICUMsubsSISubsBinding, HICUMsubsSINode, HICUMsubsNode); //si-s

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCIBasePtr, HICUMcollCIBaseBinding, HICUMcollCINode, HICUMbaseNode); //b-ci
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseCollCIPtr, HICUMbaseCollCIBinding, HICUMbaseNode, HICUMcollCINode); //ci-b

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCIEmitEIPtr, HICUMcollCIEmitEIBinding, HICUMcollCINode, HICUMemitEINode); //ci-ei
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitEICollCIPtr, HICUMemitEICollCIBinding, HICUMemitEINode, HICUMcollCINode); //ei-ci

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBPBaseBIPtr, HICUMbaseBPBaseBIBinding, HICUMbaseBPNode, HICUMbaseBINode); //bp-bi
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBIBaseBPPtr, HICUMbaseBIBaseBPBinding, HICUMbaseBINode, HICUMbaseBPNode); //bi-bp

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBPEmitEIPtr, HICUMbaseBPEmitEIBinding, HICUMbaseBPNode, HICUMemitEINode); //bp-ei
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitEIBaseBPPtr, HICUMemitEIBaseBPBinding, HICUMemitEINode, HICUMbaseBPNode); //ei-bp

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBPEmitPtr, HICUMbaseBPEmitBinding, HICUMbaseBPNode, HICUMemitNode); //bp-e
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitBaseBPPtr, HICUMemitBaseBPBinding, HICUMemitNode, HICUMbaseBPNode); //e-bp

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBPSubsSIPtr, HICUMbaseBPSubsSIBinding, HICUMbaseBPNode, HICUMsubsSINode); //bp-si
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMsubsSIBaseBPPtr, HICUMsubsSIBaseBPBinding, HICUMsubsSINode, HICUMbaseBPNode); //si-bp

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBIEmitEIPtr, HICUMbaseBIEmitEIBinding, HICUMbaseBINode, HICUMemitEINode); //ei-bi
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitEIBaseBIPtr, HICUMemitEIBaseBIBinding, HICUMemitEINode, HICUMbaseBINode); //bi-ei
            if (nqs) {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBIXfPtr, HICUMbaseBIXfBinding, HICUMbaseBINode, HICUMxfNode); //bi - xf
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitEIXfPtr, HICUMemitEIXfBinding, HICUMemitEINode, HICUMxfNode); //ei - xf
            }

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBICollCIPtr, HICUMbaseBICollCIBinding, HICUMbaseBINode, HICUMcollCINode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCIBaseBIPtr, HICUMcollCIBaseBIBinding, HICUMcollCINode, HICUMbaseBINode);

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBPCollCIPtr, HICUMbaseBPCollCIBinding, HICUMbaseBPNode, HICUMcollCINode); //bp-ci
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCIBaseBPPtr, HICUMcollCIBaseBPBinding, HICUMcollCINode, HICUMbaseBPNode); //ci-bp

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMsubsSICollCIPtr, HICUMsubsSICollCIBinding, HICUMsubsSINode, HICUMcollCINode); //si-ci
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCISubsSIPtr, HICUMcollCISubsSIBinding, HICUMcollCINode, HICUMsubsSINode); //ci-si

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCICollPtr, HICUMcollCICollBinding, HICUMcollCINode, HICUMcollNode); //ci-c
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCollCIPtr, HICUMcollCollCIBinding, HICUMcollNode, HICUMcollCINode); //c-ci

            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMsubsCollPtr, HICUMsubsCollBinding, HICUMsubsNode, HICUMcollNode); //s-c
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollSubsPtr, HICUMcollSubsBinding, HICUMcollNode, HICUMsubsNode); //c-s

            if (nqs) {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf1Xf1Ptr, HICUMxf1Xf1Binding, HICUMxf1Node, HICUMxf1Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf1BaseBIPtr, HICUMxf1BaseBIBinding, HICUMxf1Node, HICUMbaseBINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf1EmitEIPtr, HICUMxf1EmitEIBinding, HICUMxf1Node, HICUMemitEINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf1CollCIPtr, HICUMxf1CollCIBinding, HICUMxf1Node, HICUMcollCINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf1Xf2Ptr, HICUMxf1Xf2Binding, HICUMxf1Node, HICUMxf2Node);

                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf2Xf1Ptr, HICUMxf2Xf1Binding, HICUMxf2Node, HICUMxf1Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf2BaseBIPtr, HICUMxf2BaseBIBinding, HICUMxf2Node, HICUMbaseBINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf2EmitEIPtr, HICUMxf2EmitEIBinding, HICUMxf2Node, HICUMemitEINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf2CollCIPtr, HICUMxf2CollCIBinding, HICUMxf2Node, HICUMcollCINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf2Xf2Ptr, HICUMxf2Xf2Binding, HICUMxf2Node, HICUMxf2Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitEIXf2Ptr, HICUMemitEIXf2Binding, HICUMemitEINode, HICUMxf2Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCIXf2Ptr, HICUMcollCIXf2Binding, HICUMcollCINode, HICUMxf2Node);

                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxfXfPtr, HICUMxfXfBinding, HICUMxfNode, HICUMxfNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxfEmitEIPtr, HICUMxfEmitEIBinding, HICUMxfNode, HICUMemitEINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxfCollCIPtr, HICUMxfCollCIBinding, HICUMxfNode, HICUMcollCINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxfBaseBIPtr, HICUMxfBaseBIBinding, HICUMxfNode, HICUMbaseBINode);
            }

            if (selfheat) {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollTempPtr, HICUMcollTempBinding, HICUMcollNode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseTempPtr, HICUMbaseTempBinding, HICUMbaseNode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitTempPtr, HICUMemitTempBinding, HICUMemitNode, HICUMtempNode);

                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollCItempPtr, HICUMcollCItempBinding, HICUMcollCINode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBItempPtr, HICUMbaseBItempBinding, HICUMbaseBINode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMbaseBPtempPtr, HICUMbaseBPtempBinding, HICUMbaseBPNode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMemitEItempPtr, HICUMemitEItempBinding, HICUMemitEINode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMsubsSItempPtr, HICUMsubsSItempBinding, HICUMsubsSINode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMsubsTempPtr, HICUMsubsTempBinding, HICUMsubsNode, HICUMtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMcollTempPtr, HICUMcollTempBinding, HICUMcollNode, HICUMtempNode);

                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMtempCollPtr, HICUMtempCollBinding, HICUMtempNode, HICUMcollNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMtempBasePtr, HICUMtempBaseBinding, HICUMtempNode, HICUMbaseNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMtempEmitPtr, HICUMtempEmitBinding, HICUMtempNode, HICUMemitNode);

                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMtempCollCIPtr, HICUMtempCollCIBinding, HICUMtempNode, HICUMcollCINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMtempBaseBIPtr, HICUMtempBaseBIBinding, HICUMtempNode, HICUMbaseBINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMtempBaseBPPtr, HICUMtempBaseBPBinding, HICUMtempNode, HICUMbaseBPNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMtempEmitEIPtr, HICUMtempEmitEIBinding, HICUMtempNode, HICUMemitEINode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMtempSubsSIPtr, HICUMtempSubsSIBinding, HICUMtempNode, HICUMsubsSINode);

                CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMtempTempPtr, HICUMtempTempBinding, HICUMtempNode, HICUMtempNode);

                if (nqs) {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxfTempPtr, HICUMxfTempBinding, HICUMxfNode, HICUMtempNode);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf2TempPtr, HICUMxf2TempBinding, HICUMxf2Node, HICUMtempNode);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(HICUMxf1TempPtr, HICUMxf1TempBinding, HICUMxf1Node, HICUMtempNode);
                }
            }
        }
    }

    return (OK) ;
}
