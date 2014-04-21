/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soifddef.h"
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
B3SOIFDbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIFDmodel *model = (B3SOIFDmodel *)inModel ;
    B3SOIFDinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the B3SOIFD models */
    for ( ; model != NULL ; model = B3SOIFDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B3SOIFDinstances(model); here != NULL ; here = B3SOIFDnextInstance(here))
        {
            if ((model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0 != 0.0))
            {
                CREATE_KLU_BINDING_TABLE(B3SOIFDTemptempPtr, B3SOIFDTemptempBinding, B3SOIFDtempNode, B3SOIFDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDTempdpPtr, B3SOIFDTempdpBinding, B3SOIFDtempNode, B3SOIFDdNodePrime);
                CREATE_KLU_BINDING_TABLE(B3SOIFDTempspPtr, B3SOIFDTempspBinding, B3SOIFDtempNode, B3SOIFDsNodePrime);
                CREATE_KLU_BINDING_TABLE(B3SOIFDTempgPtr, B3SOIFDTempgBinding, B3SOIFDtempNode, B3SOIFDgNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDTempbPtr, B3SOIFDTempbBinding, B3SOIFDtempNode, B3SOIFDbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDTempePtr, B3SOIFDTempeBinding, B3SOIFDtempNode, B3SOIFDeNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDGtempPtr, B3SOIFDGtempBinding, B3SOIFDgNode, B3SOIFDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDDPtempPtr, B3SOIFDDPtempBinding, B3SOIFDdNodePrime, B3SOIFDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDSPtempPtr, B3SOIFDSPtempBinding, B3SOIFDsNodePrime, B3SOIFDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDEtempPtr, B3SOIFDEtempBinding, B3SOIFDeNode, B3SOIFDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDBtempPtr, B3SOIFDBtempBinding, B3SOIFDbNode, B3SOIFDtempNode);
                if (here->B3SOIFDbodyMod == 1)
                {
                    CREATE_KLU_BINDING_TABLE(B3SOIFDPtempPtr, B3SOIFDPtempBinding, B3SOIFDpNode, B3SOIFDtempNode);
                }
            }
            if (here->B3SOIFDbodyMod == 2)
            {
            }
            else if (here->B3SOIFDbodyMod == 1)
            {
                CREATE_KLU_BINDING_TABLE(B3SOIFDBpPtr, B3SOIFDBpBinding, B3SOIFDbNode, B3SOIFDpNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDPbPtr, B3SOIFDPbBinding, B3SOIFDpNode, B3SOIFDbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDPpPtr, B3SOIFDPpBinding, B3SOIFDpNode, B3SOIFDpNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDPgPtr, B3SOIFDPgBinding, B3SOIFDpNode, B3SOIFDgNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDPdpPtr, B3SOIFDPdpBinding, B3SOIFDpNode, B3SOIFDdNodePrime);
                CREATE_KLU_BINDING_TABLE(B3SOIFDPspPtr, B3SOIFDPspBinding, B3SOIFDpNode, B3SOIFDsNodePrime);
                CREATE_KLU_BINDING_TABLE(B3SOIFDPePtr, B3SOIFDPeBinding, B3SOIFDpNode, B3SOIFDeNode);
            }
            CREATE_KLU_BINDING_TABLE(B3SOIFDEgPtr, B3SOIFDEgBinding, B3SOIFDeNode, B3SOIFDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDEdpPtr, B3SOIFDEdpBinding, B3SOIFDeNode, B3SOIFDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIFDEspPtr, B3SOIFDEspBinding, B3SOIFDeNode, B3SOIFDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIFDGePtr, B3SOIFDGeBinding, B3SOIFDgNode, B3SOIFDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDDPePtr, B3SOIFDDPeBinding, B3SOIFDdNodePrime, B3SOIFDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDSPePtr, B3SOIFDSPeBinding, B3SOIFDsNodePrime, B3SOIFDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDEbPtr, B3SOIFDEbBinding, B3SOIFDeNode, B3SOIFDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDEePtr, B3SOIFDEeBinding, B3SOIFDeNode, B3SOIFDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDGgPtr, B3SOIFDGgBinding, B3SOIFDgNode, B3SOIFDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDGdpPtr, B3SOIFDGdpBinding, B3SOIFDgNode, B3SOIFDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIFDGspPtr, B3SOIFDGspBinding, B3SOIFDgNode, B3SOIFDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIFDDPgPtr, B3SOIFDDPgBinding, B3SOIFDdNodePrime, B3SOIFDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDDPdpPtr, B3SOIFDDPdpBinding, B3SOIFDdNodePrime, B3SOIFDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIFDDPspPtr, B3SOIFDDPspBinding, B3SOIFDdNodePrime, B3SOIFDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIFDDPdPtr, B3SOIFDDPdBinding, B3SOIFDdNodePrime, B3SOIFDdNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDSPgPtr, B3SOIFDSPgBinding, B3SOIFDsNodePrime, B3SOIFDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDSPdpPtr, B3SOIFDSPdpBinding, B3SOIFDsNodePrime, B3SOIFDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIFDSPspPtr, B3SOIFDSPspBinding, B3SOIFDsNodePrime, B3SOIFDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIFDSPsPtr, B3SOIFDSPsBinding, B3SOIFDsNodePrime, B3SOIFDsNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDDdPtr, B3SOIFDDdBinding, B3SOIFDdNode, B3SOIFDdNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDDdpPtr, B3SOIFDDdpBinding, B3SOIFDdNode, B3SOIFDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIFDSsPtr, B3SOIFDSsBinding, B3SOIFDsNode, B3SOIFDsNode);
            CREATE_KLU_BINDING_TABLE(B3SOIFDSspPtr, B3SOIFDSspBinding, B3SOIFDsNode, B3SOIFDsNodePrime);
            if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1))
            {
                CREATE_KLU_BINDING_TABLE(B3SOIFDVbsPtr, B3SOIFDVbsBinding, B3SOIFDvbsNode, B3SOIFDvbsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDIdsPtr, B3SOIFDIdsBinding, B3SOIFDidsNode, B3SOIFDidsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDIcPtr, B3SOIFDIcBinding, B3SOIFDicNode, B3SOIFDicNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDIbsPtr, B3SOIFDIbsBinding, B3SOIFDibsNode, B3SOIFDibsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDIbdPtr, B3SOIFDIbdBinding, B3SOIFDibdNode, B3SOIFDibdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDIiiPtr, B3SOIFDIiiBinding, B3SOIFDiiiNode, B3SOIFDiiiNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDIgidlPtr, B3SOIFDIgidlBinding, B3SOIFDigidlNode, B3SOIFDigidlNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDItunPtr, B3SOIFDItunBinding, B3SOIFDitunNode, B3SOIFDitunNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDIbpPtr, B3SOIFDIbpBinding, B3SOIFDibpNode, B3SOIFDibpNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDAbeffPtr, B3SOIFDAbeffBinding, B3SOIFDabeffNode, B3SOIFDabeffNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDVbs0effPtr, B3SOIFDVbs0effBinding, B3SOIFDvbs0effNode, B3SOIFDvbs0effNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDVbseffPtr, B3SOIFDVbseffBinding, B3SOIFDvbseffNode, B3SOIFDvbseffNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDXcPtr, B3SOIFDXcBinding, B3SOIFDxcNode, B3SOIFDxcNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDCbbPtr, B3SOIFDCbbBinding, B3SOIFDcbbNode, B3SOIFDcbbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDCbdPtr, B3SOIFDCbdBinding, B3SOIFDcbdNode, B3SOIFDcbdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDCbgPtr, B3SOIFDCbgBinding, B3SOIFDcbgNode, B3SOIFDcbgNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDqbPtr, B3SOIFDqbBinding, B3SOIFDqbNode, B3SOIFDqbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDQbfPtr, B3SOIFDQbfBinding, B3SOIFDqbfNode, B3SOIFDqbfNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDQjsPtr, B3SOIFDQjsBinding, B3SOIFDqjsNode, B3SOIFDqjsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDQjdPtr, B3SOIFDQjdBinding, B3SOIFDqjdNode, B3SOIFDqjdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDGmPtr, B3SOIFDGmBinding, B3SOIFDgmNode, B3SOIFDgmNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDGmbsPtr, B3SOIFDGmbsBinding, B3SOIFDgmbsNode, B3SOIFDgmbsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDGdsPtr, B3SOIFDGdsBinding, B3SOIFDgdsNode, B3SOIFDgdsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDGmePtr, B3SOIFDGmeBinding, B3SOIFDgmeNode, B3SOIFDgmeNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDVbs0teffPtr, B3SOIFDVbs0teffBinding, B3SOIFDvbs0teffNode, B3SOIFDvbs0teffNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDVthPtr, B3SOIFDVthBinding, B3SOIFDvthNode, B3SOIFDvthNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDVgsteffPtr, B3SOIFDVgsteffBinding, B3SOIFDvgsteffNode, B3SOIFDvgsteffNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDXcsatPtr, B3SOIFDXcsatBinding, B3SOIFDxcsatNode, B3SOIFDxcsatNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDVcscvPtr, B3SOIFDVcscvBinding, B3SOIFDvcscvNode, B3SOIFDvcscvNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDVdscvPtr, B3SOIFDVdscvBinding, B3SOIFDvdscvNode, B3SOIFDvdscvNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDCbePtr, B3SOIFDCbeBinding, B3SOIFDcbeNode, B3SOIFDcbeNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDDum1Ptr, B3SOIFDDum1Binding, B3SOIFDdum1Node, B3SOIFDdum1Node);
                CREATE_KLU_BINDING_TABLE(B3SOIFDDum2Ptr, B3SOIFDDum2Binding, B3SOIFDdum2Node, B3SOIFDdum2Node);
                CREATE_KLU_BINDING_TABLE(B3SOIFDDum3Ptr, B3SOIFDDum3Binding, B3SOIFDdum3Node, B3SOIFDdum3Node);
                CREATE_KLU_BINDING_TABLE(B3SOIFDDum4Ptr, B3SOIFDDum4Binding, B3SOIFDdum4Node, B3SOIFDdum4Node);
                CREATE_KLU_BINDING_TABLE(B3SOIFDDum5Ptr, B3SOIFDDum5Binding, B3SOIFDdum5Node, B3SOIFDdum5Node);
                CREATE_KLU_BINDING_TABLE(B3SOIFDQaccPtr, B3SOIFDQaccBinding, B3SOIFDqaccNode, B3SOIFDqaccNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDQsub0Ptr, B3SOIFDQsub0Binding, B3SOIFDqsub0Node, B3SOIFDqsub0Node);
                CREATE_KLU_BINDING_TABLE(B3SOIFDQsubs1Ptr, B3SOIFDQsubs1Binding, B3SOIFDqsubs1Node, B3SOIFDqsubs1Node);
                CREATE_KLU_BINDING_TABLE(B3SOIFDQsubs2Ptr, B3SOIFDQsubs2Binding, B3SOIFDqsubs2Node, B3SOIFDqsubs2Node);
                CREATE_KLU_BINDING_TABLE(B3SOIFDqePtr, B3SOIFDqeBinding, B3SOIFDqeNode, B3SOIFDqeNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDqdPtr, B3SOIFDqdBinding, B3SOIFDqdNode, B3SOIFDqdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIFDqgPtr, B3SOIFDqgBinding, B3SOIFDqgNode, B3SOIFDqgNode);
            }
        }
    }

    return (OK) ;
}

int
B3SOIFDbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIFDmodel *model = (B3SOIFDmodel *)inModel ;
    B3SOIFDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B3SOIFD models */
    for ( ; model != NULL ; model = B3SOIFDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B3SOIFDinstances(model); here != NULL ; here = B3SOIFDnextInstance(here))
        {
            if ((model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0 != 0.0))
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDTemptempPtr, B3SOIFDTemptempBinding, B3SOIFDtempNode, B3SOIFDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDTempdpPtr, B3SOIFDTempdpBinding, B3SOIFDtempNode, B3SOIFDdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDTempspPtr, B3SOIFDTempspBinding, B3SOIFDtempNode, B3SOIFDsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDTempgPtr, B3SOIFDTempgBinding, B3SOIFDtempNode, B3SOIFDgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDTempbPtr, B3SOIFDTempbBinding, B3SOIFDtempNode, B3SOIFDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDTempePtr, B3SOIFDTempeBinding, B3SOIFDtempNode, B3SOIFDeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDGtempPtr, B3SOIFDGtempBinding, B3SOIFDgNode, B3SOIFDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDPtempPtr, B3SOIFDDPtempBinding, B3SOIFDdNodePrime, B3SOIFDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDSPtempPtr, B3SOIFDSPtempBinding, B3SOIFDsNodePrime, B3SOIFDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDEtempPtr, B3SOIFDEtempBinding, B3SOIFDeNode, B3SOIFDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDBtempPtr, B3SOIFDBtempBinding, B3SOIFDbNode, B3SOIFDtempNode);
                if (here->B3SOIFDbodyMod == 1)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDPtempPtr, B3SOIFDPtempBinding, B3SOIFDpNode, B3SOIFDtempNode);
                }
            }
            if (here->B3SOIFDbodyMod == 2)
            {
            }
            else if (here->B3SOIFDbodyMod == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDBpPtr, B3SOIFDBpBinding, B3SOIFDbNode, B3SOIFDpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDPbPtr, B3SOIFDPbBinding, B3SOIFDpNode, B3SOIFDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDPpPtr, B3SOIFDPpBinding, B3SOIFDpNode, B3SOIFDpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDPgPtr, B3SOIFDPgBinding, B3SOIFDpNode, B3SOIFDgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDPdpPtr, B3SOIFDPdpBinding, B3SOIFDpNode, B3SOIFDdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDPspPtr, B3SOIFDPspBinding, B3SOIFDpNode, B3SOIFDsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDPePtr, B3SOIFDPeBinding, B3SOIFDpNode, B3SOIFDeNode);
            }
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDEgPtr, B3SOIFDEgBinding, B3SOIFDeNode, B3SOIFDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDEdpPtr, B3SOIFDEdpBinding, B3SOIFDeNode, B3SOIFDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDEspPtr, B3SOIFDEspBinding, B3SOIFDeNode, B3SOIFDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDGePtr, B3SOIFDGeBinding, B3SOIFDgNode, B3SOIFDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDPePtr, B3SOIFDDPeBinding, B3SOIFDdNodePrime, B3SOIFDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDSPePtr, B3SOIFDSPeBinding, B3SOIFDsNodePrime, B3SOIFDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDEbPtr, B3SOIFDEbBinding, B3SOIFDeNode, B3SOIFDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDEePtr, B3SOIFDEeBinding, B3SOIFDeNode, B3SOIFDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDGgPtr, B3SOIFDGgBinding, B3SOIFDgNode, B3SOIFDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDGdpPtr, B3SOIFDGdpBinding, B3SOIFDgNode, B3SOIFDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDGspPtr, B3SOIFDGspBinding, B3SOIFDgNode, B3SOIFDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDPgPtr, B3SOIFDDPgBinding, B3SOIFDdNodePrime, B3SOIFDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDPdpPtr, B3SOIFDDPdpBinding, B3SOIFDdNodePrime, B3SOIFDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDPspPtr, B3SOIFDDPspBinding, B3SOIFDdNodePrime, B3SOIFDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDPdPtr, B3SOIFDDPdBinding, B3SOIFDdNodePrime, B3SOIFDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDSPgPtr, B3SOIFDSPgBinding, B3SOIFDsNodePrime, B3SOIFDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDSPdpPtr, B3SOIFDSPdpBinding, B3SOIFDsNodePrime, B3SOIFDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDSPspPtr, B3SOIFDSPspBinding, B3SOIFDsNodePrime, B3SOIFDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDSPsPtr, B3SOIFDSPsBinding, B3SOIFDsNodePrime, B3SOIFDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDdPtr, B3SOIFDDdBinding, B3SOIFDdNode, B3SOIFDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDdpPtr, B3SOIFDDdpBinding, B3SOIFDdNode, B3SOIFDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDSsPtr, B3SOIFDSsBinding, B3SOIFDsNode, B3SOIFDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDSspPtr, B3SOIFDSspBinding, B3SOIFDsNode, B3SOIFDsNodePrime);
            if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1))
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDVbsPtr, B3SOIFDVbsBinding, B3SOIFDvbsNode, B3SOIFDvbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDIdsPtr, B3SOIFDIdsBinding, B3SOIFDidsNode, B3SOIFDidsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDIcPtr, B3SOIFDIcBinding, B3SOIFDicNode, B3SOIFDicNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDIbsPtr, B3SOIFDIbsBinding, B3SOIFDibsNode, B3SOIFDibsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDIbdPtr, B3SOIFDIbdBinding, B3SOIFDibdNode, B3SOIFDibdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDIiiPtr, B3SOIFDIiiBinding, B3SOIFDiiiNode, B3SOIFDiiiNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDIgidlPtr, B3SOIFDIgidlBinding, B3SOIFDigidlNode, B3SOIFDigidlNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDItunPtr, B3SOIFDItunBinding, B3SOIFDitunNode, B3SOIFDitunNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDIbpPtr, B3SOIFDIbpBinding, B3SOIFDibpNode, B3SOIFDibpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDAbeffPtr, B3SOIFDAbeffBinding, B3SOIFDabeffNode, B3SOIFDabeffNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDVbs0effPtr, B3SOIFDVbs0effBinding, B3SOIFDvbs0effNode, B3SOIFDvbs0effNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDVbseffPtr, B3SOIFDVbseffBinding, B3SOIFDvbseffNode, B3SOIFDvbseffNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDXcPtr, B3SOIFDXcBinding, B3SOIFDxcNode, B3SOIFDxcNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDCbbPtr, B3SOIFDCbbBinding, B3SOIFDcbbNode, B3SOIFDcbbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDCbdPtr, B3SOIFDCbdBinding, B3SOIFDcbdNode, B3SOIFDcbdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDCbgPtr, B3SOIFDCbgBinding, B3SOIFDcbgNode, B3SOIFDcbgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDqbPtr, B3SOIFDqbBinding, B3SOIFDqbNode, B3SOIFDqbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDQbfPtr, B3SOIFDQbfBinding, B3SOIFDqbfNode, B3SOIFDqbfNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDQjsPtr, B3SOIFDQjsBinding, B3SOIFDqjsNode, B3SOIFDqjsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDQjdPtr, B3SOIFDQjdBinding, B3SOIFDqjdNode, B3SOIFDqjdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDGmPtr, B3SOIFDGmBinding, B3SOIFDgmNode, B3SOIFDgmNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDGmbsPtr, B3SOIFDGmbsBinding, B3SOIFDgmbsNode, B3SOIFDgmbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDGdsPtr, B3SOIFDGdsBinding, B3SOIFDgdsNode, B3SOIFDgdsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDGmePtr, B3SOIFDGmeBinding, B3SOIFDgmeNode, B3SOIFDgmeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDVbs0teffPtr, B3SOIFDVbs0teffBinding, B3SOIFDvbs0teffNode, B3SOIFDvbs0teffNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDVthPtr, B3SOIFDVthBinding, B3SOIFDvthNode, B3SOIFDvthNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDVgsteffPtr, B3SOIFDVgsteffBinding, B3SOIFDvgsteffNode, B3SOIFDvgsteffNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDXcsatPtr, B3SOIFDXcsatBinding, B3SOIFDxcsatNode, B3SOIFDxcsatNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDVcscvPtr, B3SOIFDVcscvBinding, B3SOIFDvcscvNode, B3SOIFDvcscvNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDVdscvPtr, B3SOIFDVdscvBinding, B3SOIFDvdscvNode, B3SOIFDvdscvNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDCbePtr, B3SOIFDCbeBinding, B3SOIFDcbeNode, B3SOIFDcbeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDum1Ptr, B3SOIFDDum1Binding, B3SOIFDdum1Node, B3SOIFDdum1Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDum2Ptr, B3SOIFDDum2Binding, B3SOIFDdum2Node, B3SOIFDdum2Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDum3Ptr, B3SOIFDDum3Binding, B3SOIFDdum3Node, B3SOIFDdum3Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDum4Ptr, B3SOIFDDum4Binding, B3SOIFDdum4Node, B3SOIFDdum4Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDDum5Ptr, B3SOIFDDum5Binding, B3SOIFDdum5Node, B3SOIFDdum5Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDQaccPtr, B3SOIFDQaccBinding, B3SOIFDqaccNode, B3SOIFDqaccNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDQsub0Ptr, B3SOIFDQsub0Binding, B3SOIFDqsub0Node, B3SOIFDqsub0Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDQsubs1Ptr, B3SOIFDQsubs1Binding, B3SOIFDqsubs1Node, B3SOIFDqsubs1Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDQsubs2Ptr, B3SOIFDQsubs2Binding, B3SOIFDqsubs2Node, B3SOIFDqsubs2Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDqePtr, B3SOIFDqeBinding, B3SOIFDqeNode, B3SOIFDqeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDqdPtr, B3SOIFDqdBinding, B3SOIFDqdNode, B3SOIFDqdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIFDqgPtr, B3SOIFDqgBinding, B3SOIFDqgNode, B3SOIFDqgNode);
            }
        }
    }

    return (OK) ;
}

int
B3SOIFDbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIFDmodel *model = (B3SOIFDmodel *)inModel ;
    B3SOIFDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B3SOIFD models */
    for ( ; model != NULL ; model = B3SOIFDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B3SOIFDinstances(model); here != NULL ; here = B3SOIFDnextInstance(here))
        {
            if ((model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0 != 0.0))
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDTemptempPtr, B3SOIFDTemptempBinding, B3SOIFDtempNode, B3SOIFDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDTempdpPtr, B3SOIFDTempdpBinding, B3SOIFDtempNode, B3SOIFDdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDTempspPtr, B3SOIFDTempspBinding, B3SOIFDtempNode, B3SOIFDsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDTempgPtr, B3SOIFDTempgBinding, B3SOIFDtempNode, B3SOIFDgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDTempbPtr, B3SOIFDTempbBinding, B3SOIFDtempNode, B3SOIFDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDTempePtr, B3SOIFDTempeBinding, B3SOIFDtempNode, B3SOIFDeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDGtempPtr, B3SOIFDGtempBinding, B3SOIFDgNode, B3SOIFDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDPtempPtr, B3SOIFDDPtempBinding, B3SOIFDdNodePrime, B3SOIFDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDSPtempPtr, B3SOIFDSPtempBinding, B3SOIFDsNodePrime, B3SOIFDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDEtempPtr, B3SOIFDEtempBinding, B3SOIFDeNode, B3SOIFDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDBtempPtr, B3SOIFDBtempBinding, B3SOIFDbNode, B3SOIFDtempNode);
                if (here->B3SOIFDbodyMod == 1)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDPtempPtr, B3SOIFDPtempBinding, B3SOIFDpNode, B3SOIFDtempNode);
                }
            }
            if (here->B3SOIFDbodyMod == 2)
            {
            }
            else if (here->B3SOIFDbodyMod == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDBpPtr, B3SOIFDBpBinding, B3SOIFDbNode, B3SOIFDpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDPbPtr, B3SOIFDPbBinding, B3SOIFDpNode, B3SOIFDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDPpPtr, B3SOIFDPpBinding, B3SOIFDpNode, B3SOIFDpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDPgPtr, B3SOIFDPgBinding, B3SOIFDpNode, B3SOIFDgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDPdpPtr, B3SOIFDPdpBinding, B3SOIFDpNode, B3SOIFDdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDPspPtr, B3SOIFDPspBinding, B3SOIFDpNode, B3SOIFDsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDPePtr, B3SOIFDPeBinding, B3SOIFDpNode, B3SOIFDeNode);
            }
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDEgPtr, B3SOIFDEgBinding, B3SOIFDeNode, B3SOIFDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDEdpPtr, B3SOIFDEdpBinding, B3SOIFDeNode, B3SOIFDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDEspPtr, B3SOIFDEspBinding, B3SOIFDeNode, B3SOIFDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDGePtr, B3SOIFDGeBinding, B3SOIFDgNode, B3SOIFDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDPePtr, B3SOIFDDPeBinding, B3SOIFDdNodePrime, B3SOIFDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDSPePtr, B3SOIFDSPeBinding, B3SOIFDsNodePrime, B3SOIFDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDEbPtr, B3SOIFDEbBinding, B3SOIFDeNode, B3SOIFDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDEePtr, B3SOIFDEeBinding, B3SOIFDeNode, B3SOIFDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDGgPtr, B3SOIFDGgBinding, B3SOIFDgNode, B3SOIFDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDGdpPtr, B3SOIFDGdpBinding, B3SOIFDgNode, B3SOIFDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDGspPtr, B3SOIFDGspBinding, B3SOIFDgNode, B3SOIFDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDPgPtr, B3SOIFDDPgBinding, B3SOIFDdNodePrime, B3SOIFDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDPdpPtr, B3SOIFDDPdpBinding, B3SOIFDdNodePrime, B3SOIFDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDPspPtr, B3SOIFDDPspBinding, B3SOIFDdNodePrime, B3SOIFDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDPdPtr, B3SOIFDDPdBinding, B3SOIFDdNodePrime, B3SOIFDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDSPgPtr, B3SOIFDSPgBinding, B3SOIFDsNodePrime, B3SOIFDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDSPdpPtr, B3SOIFDSPdpBinding, B3SOIFDsNodePrime, B3SOIFDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDSPspPtr, B3SOIFDSPspBinding, B3SOIFDsNodePrime, B3SOIFDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDSPsPtr, B3SOIFDSPsBinding, B3SOIFDsNodePrime, B3SOIFDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDdPtr, B3SOIFDDdBinding, B3SOIFDdNode, B3SOIFDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDdpPtr, B3SOIFDDdpBinding, B3SOIFDdNode, B3SOIFDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDSsPtr, B3SOIFDSsBinding, B3SOIFDsNode, B3SOIFDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDSspPtr, B3SOIFDSspBinding, B3SOIFDsNode, B3SOIFDsNodePrime);
            if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1))
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDVbsPtr, B3SOIFDVbsBinding, B3SOIFDvbsNode, B3SOIFDvbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDIdsPtr, B3SOIFDIdsBinding, B3SOIFDidsNode, B3SOIFDidsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDIcPtr, B3SOIFDIcBinding, B3SOIFDicNode, B3SOIFDicNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDIbsPtr, B3SOIFDIbsBinding, B3SOIFDibsNode, B3SOIFDibsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDIbdPtr, B3SOIFDIbdBinding, B3SOIFDibdNode, B3SOIFDibdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDIiiPtr, B3SOIFDIiiBinding, B3SOIFDiiiNode, B3SOIFDiiiNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDIgidlPtr, B3SOIFDIgidlBinding, B3SOIFDigidlNode, B3SOIFDigidlNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDItunPtr, B3SOIFDItunBinding, B3SOIFDitunNode, B3SOIFDitunNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDIbpPtr, B3SOIFDIbpBinding, B3SOIFDibpNode, B3SOIFDibpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDAbeffPtr, B3SOIFDAbeffBinding, B3SOIFDabeffNode, B3SOIFDabeffNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDVbs0effPtr, B3SOIFDVbs0effBinding, B3SOIFDvbs0effNode, B3SOIFDvbs0effNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDVbseffPtr, B3SOIFDVbseffBinding, B3SOIFDvbseffNode, B3SOIFDvbseffNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDXcPtr, B3SOIFDXcBinding, B3SOIFDxcNode, B3SOIFDxcNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDCbbPtr, B3SOIFDCbbBinding, B3SOIFDcbbNode, B3SOIFDcbbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDCbdPtr, B3SOIFDCbdBinding, B3SOIFDcbdNode, B3SOIFDcbdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDCbgPtr, B3SOIFDCbgBinding, B3SOIFDcbgNode, B3SOIFDcbgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDqbPtr, B3SOIFDqbBinding, B3SOIFDqbNode, B3SOIFDqbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDQbfPtr, B3SOIFDQbfBinding, B3SOIFDqbfNode, B3SOIFDqbfNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDQjsPtr, B3SOIFDQjsBinding, B3SOIFDqjsNode, B3SOIFDqjsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDQjdPtr, B3SOIFDQjdBinding, B3SOIFDqjdNode, B3SOIFDqjdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDGmPtr, B3SOIFDGmBinding, B3SOIFDgmNode, B3SOIFDgmNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDGmbsPtr, B3SOIFDGmbsBinding, B3SOIFDgmbsNode, B3SOIFDgmbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDGdsPtr, B3SOIFDGdsBinding, B3SOIFDgdsNode, B3SOIFDgdsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDGmePtr, B3SOIFDGmeBinding, B3SOIFDgmeNode, B3SOIFDgmeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDVbs0teffPtr, B3SOIFDVbs0teffBinding, B3SOIFDvbs0teffNode, B3SOIFDvbs0teffNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDVthPtr, B3SOIFDVthBinding, B3SOIFDvthNode, B3SOIFDvthNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDVgsteffPtr, B3SOIFDVgsteffBinding, B3SOIFDvgsteffNode, B3SOIFDvgsteffNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDXcsatPtr, B3SOIFDXcsatBinding, B3SOIFDxcsatNode, B3SOIFDxcsatNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDVcscvPtr, B3SOIFDVcscvBinding, B3SOIFDvcscvNode, B3SOIFDvcscvNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDVdscvPtr, B3SOIFDVdscvBinding, B3SOIFDvdscvNode, B3SOIFDvdscvNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDCbePtr, B3SOIFDCbeBinding, B3SOIFDcbeNode, B3SOIFDcbeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDum1Ptr, B3SOIFDDum1Binding, B3SOIFDdum1Node, B3SOIFDdum1Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDum2Ptr, B3SOIFDDum2Binding, B3SOIFDdum2Node, B3SOIFDdum2Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDum3Ptr, B3SOIFDDum3Binding, B3SOIFDdum3Node, B3SOIFDdum3Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDum4Ptr, B3SOIFDDum4Binding, B3SOIFDdum4Node, B3SOIFDdum4Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDDum5Ptr, B3SOIFDDum5Binding, B3SOIFDdum5Node, B3SOIFDdum5Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDQaccPtr, B3SOIFDQaccBinding, B3SOIFDqaccNode, B3SOIFDqaccNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDQsub0Ptr, B3SOIFDQsub0Binding, B3SOIFDqsub0Node, B3SOIFDqsub0Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDQsubs1Ptr, B3SOIFDQsubs1Binding, B3SOIFDqsubs1Node, B3SOIFDqsubs1Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDQsubs2Ptr, B3SOIFDQsubs2Binding, B3SOIFDqsubs2Node, B3SOIFDqsubs2Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDqePtr, B3SOIFDqeBinding, B3SOIFDqeNode, B3SOIFDqeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDqdPtr, B3SOIFDqdBinding, B3SOIFDqdNode, B3SOIFDqdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIFDqgPtr, B3SOIFDqgBinding, B3SOIFDqgNode, B3SOIFDqgNode);
            }
        }
    }

    return (OK) ;
}
