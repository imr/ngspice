/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
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
B3SOIDDbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIDDmodel *model = (B3SOIDDmodel *)inModel ;
    B3SOIDDinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the B3SOIDD models */
    for ( ; model != NULL ; model = B3SOIDDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B3SOIDDinstances(model); here != NULL ; here = B3SOIDDnextInstance(here))
        {
            if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0 != 0.0))
            {
                CREATE_KLU_BINDING_TABLE(B3SOIDDTemptempPtr, B3SOIDDTemptempBinding, B3SOIDDtempNode, B3SOIDDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDTempdpPtr, B3SOIDDTempdpBinding, B3SOIDDtempNode, B3SOIDDdNodePrime);
                CREATE_KLU_BINDING_TABLE(B3SOIDDTempspPtr, B3SOIDDTempspBinding, B3SOIDDtempNode, B3SOIDDsNodePrime);
                CREATE_KLU_BINDING_TABLE(B3SOIDDTempgPtr, B3SOIDDTempgBinding, B3SOIDDtempNode, B3SOIDDgNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDTempbPtr, B3SOIDDTempbBinding, B3SOIDDtempNode, B3SOIDDbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDTempePtr, B3SOIDDTempeBinding, B3SOIDDtempNode, B3SOIDDeNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDGtempPtr, B3SOIDDGtempBinding, B3SOIDDgNode, B3SOIDDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDDPtempPtr, B3SOIDDDPtempBinding, B3SOIDDdNodePrime, B3SOIDDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDSPtempPtr, B3SOIDDSPtempBinding, B3SOIDDsNodePrime, B3SOIDDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDEtempPtr, B3SOIDDEtempBinding, B3SOIDDeNode, B3SOIDDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDBtempPtr, B3SOIDDBtempBinding, B3SOIDDbNode, B3SOIDDtempNode);
                if (here->B3SOIDDbodyMod == 1)
                {
                    CREATE_KLU_BINDING_TABLE(B3SOIDDPtempPtr, B3SOIDDPtempBinding, B3SOIDDpNode, B3SOIDDtempNode);
                }
            }
            if (here->B3SOIDDbodyMod == 2)
            {
            }
            else if (here->B3SOIDDbodyMod == 1)
            {
                CREATE_KLU_BINDING_TABLE(B3SOIDDBpPtr, B3SOIDDBpBinding, B3SOIDDbNode, B3SOIDDpNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDPbPtr, B3SOIDDPbBinding, B3SOIDDpNode, B3SOIDDbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDPpPtr, B3SOIDDPpBinding, B3SOIDDpNode, B3SOIDDpNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDPgPtr, B3SOIDDPgBinding, B3SOIDDpNode, B3SOIDDgNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDPdpPtr, B3SOIDDPdpBinding, B3SOIDDpNode, B3SOIDDdNodePrime);
                CREATE_KLU_BINDING_TABLE(B3SOIDDPspPtr, B3SOIDDPspBinding, B3SOIDDpNode, B3SOIDDsNodePrime);
                CREATE_KLU_BINDING_TABLE(B3SOIDDPePtr, B3SOIDDPeBinding, B3SOIDDpNode, B3SOIDDeNode);
            }
            CREATE_KLU_BINDING_TABLE(B3SOIDDEgPtr, B3SOIDDEgBinding, B3SOIDDeNode, B3SOIDDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDEdpPtr, B3SOIDDEdpBinding, B3SOIDDeNode, B3SOIDDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDEspPtr, B3SOIDDEspBinding, B3SOIDDeNode, B3SOIDDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDGePtr, B3SOIDDGeBinding, B3SOIDDgNode, B3SOIDDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDDPePtr, B3SOIDDDPeBinding, B3SOIDDdNodePrime, B3SOIDDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDSPePtr, B3SOIDDSPeBinding, B3SOIDDsNodePrime, B3SOIDDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDEbPtr, B3SOIDDEbBinding, B3SOIDDeNode, B3SOIDDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDGbPtr, B3SOIDDGbBinding, B3SOIDDgNode, B3SOIDDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDDPbPtr, B3SOIDDDPbBinding, B3SOIDDdNodePrime, B3SOIDDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDSPbPtr, B3SOIDDSPbBinding, B3SOIDDsNodePrime, B3SOIDDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDBePtr, B3SOIDDBeBinding, B3SOIDDbNode, B3SOIDDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDBgPtr, B3SOIDDBgBinding, B3SOIDDbNode, B3SOIDDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDBdpPtr, B3SOIDDBdpBinding, B3SOIDDbNode, B3SOIDDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDBspPtr, B3SOIDDBspBinding, B3SOIDDbNode, B3SOIDDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDBbPtr, B3SOIDDBbBinding, B3SOIDDbNode, B3SOIDDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDEePtr, B3SOIDDEeBinding, B3SOIDDeNode, B3SOIDDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDGgPtr, B3SOIDDGgBinding, B3SOIDDgNode, B3SOIDDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDGdpPtr, B3SOIDDGdpBinding, B3SOIDDgNode, B3SOIDDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDGspPtr, B3SOIDDGspBinding, B3SOIDDgNode, B3SOIDDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDDPgPtr, B3SOIDDDPgBinding, B3SOIDDdNodePrime, B3SOIDDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDDPdpPtr, B3SOIDDDPdpBinding, B3SOIDDdNodePrime, B3SOIDDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDDPspPtr, B3SOIDDDPspBinding, B3SOIDDdNodePrime, B3SOIDDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDDPdPtr, B3SOIDDDPdBinding, B3SOIDDdNodePrime, B3SOIDDdNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDSPgPtr, B3SOIDDSPgBinding, B3SOIDDsNodePrime, B3SOIDDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDSPdpPtr, B3SOIDDSPdpBinding, B3SOIDDsNodePrime, B3SOIDDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDSPspPtr, B3SOIDDSPspBinding, B3SOIDDsNodePrime, B3SOIDDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDSPsPtr, B3SOIDDSPsBinding, B3SOIDDsNodePrime, B3SOIDDsNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDDdPtr, B3SOIDDDdBinding, B3SOIDDdNode, B3SOIDDdNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDDdpPtr, B3SOIDDDdpBinding, B3SOIDDdNode, B3SOIDDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIDDSsPtr, B3SOIDDSsBinding, B3SOIDDsNode, B3SOIDDsNode);
            CREATE_KLU_BINDING_TABLE(B3SOIDDSspPtr, B3SOIDDSspBinding, B3SOIDDsNode, B3SOIDDsNodePrime);
            if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
            {
                CREATE_KLU_BINDING_TABLE(B3SOIDDVbsPtr, B3SOIDDVbsBinding, B3SOIDDvbsNode, B3SOIDDvbsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDIdsPtr, B3SOIDDIdsBinding, B3SOIDDidsNode, B3SOIDDidsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDIcPtr, B3SOIDDIcBinding, B3SOIDDicNode, B3SOIDDicNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDIbsPtr, B3SOIDDIbsBinding, B3SOIDDibsNode, B3SOIDDibsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDIbdPtr, B3SOIDDIbdBinding, B3SOIDDibdNode, B3SOIDDibdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDIiiPtr, B3SOIDDIiiBinding, B3SOIDDiiiNode, B3SOIDDiiiNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDIgidlPtr, B3SOIDDIgidlBinding, B3SOIDDigidlNode, B3SOIDDigidlNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDItunPtr, B3SOIDDItunBinding, B3SOIDDitunNode, B3SOIDDitunNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDIbpPtr, B3SOIDDIbpBinding, B3SOIDDibpNode, B3SOIDDibpNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDAbeffPtr, B3SOIDDAbeffBinding, B3SOIDDabeffNode, B3SOIDDabeffNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDVbs0effPtr, B3SOIDDVbs0effBinding, B3SOIDDvbs0effNode, B3SOIDDvbs0effNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDVbseffPtr, B3SOIDDVbseffBinding, B3SOIDDvbseffNode, B3SOIDDvbseffNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDXcPtr, B3SOIDDXcBinding, B3SOIDDxcNode, B3SOIDDxcNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDCbbPtr, B3SOIDDCbbBinding, B3SOIDDcbbNode, B3SOIDDcbbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDCbdPtr, B3SOIDDCbdBinding, B3SOIDDcbdNode, B3SOIDDcbdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDCbgPtr, B3SOIDDCbgBinding, B3SOIDDcbgNode, B3SOIDDcbgNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDqbPtr, B3SOIDDqbBinding, B3SOIDDqbNode, B3SOIDDqbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDQbfPtr, B3SOIDDQbfBinding, B3SOIDDqbfNode, B3SOIDDqbfNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDQjsPtr, B3SOIDDQjsBinding, B3SOIDDqjsNode, B3SOIDDqjsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDQjdPtr, B3SOIDDQjdBinding, B3SOIDDqjdNode, B3SOIDDqjdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDGmPtr, B3SOIDDGmBinding, B3SOIDDgmNode, B3SOIDDgmNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDGmbsPtr, B3SOIDDGmbsBinding, B3SOIDDgmbsNode, B3SOIDDgmbsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDGdsPtr, B3SOIDDGdsBinding, B3SOIDDgdsNode, B3SOIDDgdsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDGmePtr, B3SOIDDGmeBinding, B3SOIDDgmeNode, B3SOIDDgmeNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDVbs0teffPtr, B3SOIDDVbs0teffBinding, B3SOIDDvbs0teffNode, B3SOIDDvbs0teffNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDVthPtr, B3SOIDDVthBinding, B3SOIDDvthNode, B3SOIDDvthNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDVgsteffPtr, B3SOIDDVgsteffBinding, B3SOIDDvgsteffNode, B3SOIDDvgsteffNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDXcsatPtr, B3SOIDDXcsatBinding, B3SOIDDxcsatNode, B3SOIDDxcsatNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDVcscvPtr, B3SOIDDVcscvBinding, B3SOIDDvcscvNode, B3SOIDDvcscvNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDVdscvPtr, B3SOIDDVdscvBinding, B3SOIDDvdscvNode, B3SOIDDvdscvNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDCbePtr, B3SOIDDCbeBinding, B3SOIDDcbeNode, B3SOIDDcbeNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDDum1Ptr, B3SOIDDDum1Binding, B3SOIDDdum1Node, B3SOIDDdum1Node);
                CREATE_KLU_BINDING_TABLE(B3SOIDDDum2Ptr, B3SOIDDDum2Binding, B3SOIDDdum2Node, B3SOIDDdum2Node);
                CREATE_KLU_BINDING_TABLE(B3SOIDDDum3Ptr, B3SOIDDDum3Binding, B3SOIDDdum3Node, B3SOIDDdum3Node);
                CREATE_KLU_BINDING_TABLE(B3SOIDDDum4Ptr, B3SOIDDDum4Binding, B3SOIDDdum4Node, B3SOIDDdum4Node);
                CREATE_KLU_BINDING_TABLE(B3SOIDDDum5Ptr, B3SOIDDDum5Binding, B3SOIDDdum5Node, B3SOIDDdum5Node);
                CREATE_KLU_BINDING_TABLE(B3SOIDDQaccPtr, B3SOIDDQaccBinding, B3SOIDDqaccNode, B3SOIDDqaccNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDQsub0Ptr, B3SOIDDQsub0Binding, B3SOIDDqsub0Node, B3SOIDDqsub0Node);
                CREATE_KLU_BINDING_TABLE(B3SOIDDQsubs1Ptr, B3SOIDDQsubs1Binding, B3SOIDDqsubs1Node, B3SOIDDqsubs1Node);
                CREATE_KLU_BINDING_TABLE(B3SOIDDQsubs2Ptr, B3SOIDDQsubs2Binding, B3SOIDDqsubs2Node, B3SOIDDqsubs2Node);
                CREATE_KLU_BINDING_TABLE(B3SOIDDqePtr, B3SOIDDqeBinding, B3SOIDDqeNode, B3SOIDDqeNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDqdPtr, B3SOIDDqdBinding, B3SOIDDqdNode, B3SOIDDqdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIDDqgPtr, B3SOIDDqgBinding, B3SOIDDqgNode, B3SOIDDqgNode);
            }
        }
    }

    return (OK) ;
}

int
B3SOIDDbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIDDmodel *model = (B3SOIDDmodel *)inModel ;
    B3SOIDDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B3SOIDD models */
    for ( ; model != NULL ; model = B3SOIDDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B3SOIDDinstances(model); here != NULL ; here = B3SOIDDnextInstance(here))
        {
            if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0 != 0.0))
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDTemptempPtr, B3SOIDDTemptempBinding, B3SOIDDtempNode, B3SOIDDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDTempdpPtr, B3SOIDDTempdpBinding, B3SOIDDtempNode, B3SOIDDdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDTempspPtr, B3SOIDDTempspBinding, B3SOIDDtempNode, B3SOIDDsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDTempgPtr, B3SOIDDTempgBinding, B3SOIDDtempNode, B3SOIDDgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDTempbPtr, B3SOIDDTempbBinding, B3SOIDDtempNode, B3SOIDDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDTempePtr, B3SOIDDTempeBinding, B3SOIDDtempNode, B3SOIDDeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDGtempPtr, B3SOIDDGtempBinding, B3SOIDDgNode, B3SOIDDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDPtempPtr, B3SOIDDDPtempBinding, B3SOIDDdNodePrime, B3SOIDDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDSPtempPtr, B3SOIDDSPtempBinding, B3SOIDDsNodePrime, B3SOIDDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDEtempPtr, B3SOIDDEtempBinding, B3SOIDDeNode, B3SOIDDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDBtempPtr, B3SOIDDBtempBinding, B3SOIDDbNode, B3SOIDDtempNode);
                if (here->B3SOIDDbodyMod == 1)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDPtempPtr, B3SOIDDPtempBinding, B3SOIDDpNode, B3SOIDDtempNode);
                }
            }
            if (here->B3SOIDDbodyMod == 2)
            {
            }
            else if (here->B3SOIDDbodyMod == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDBpPtr, B3SOIDDBpBinding, B3SOIDDbNode, B3SOIDDpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDPbPtr, B3SOIDDPbBinding, B3SOIDDpNode, B3SOIDDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDPpPtr, B3SOIDDPpBinding, B3SOIDDpNode, B3SOIDDpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDPgPtr, B3SOIDDPgBinding, B3SOIDDpNode, B3SOIDDgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDPdpPtr, B3SOIDDPdpBinding, B3SOIDDpNode, B3SOIDDdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDPspPtr, B3SOIDDPspBinding, B3SOIDDpNode, B3SOIDDsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDPePtr, B3SOIDDPeBinding, B3SOIDDpNode, B3SOIDDeNode);
            }
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDEgPtr, B3SOIDDEgBinding, B3SOIDDeNode, B3SOIDDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDEdpPtr, B3SOIDDEdpBinding, B3SOIDDeNode, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDEspPtr, B3SOIDDEspBinding, B3SOIDDeNode, B3SOIDDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDGePtr, B3SOIDDGeBinding, B3SOIDDgNode, B3SOIDDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDPePtr, B3SOIDDDPeBinding, B3SOIDDdNodePrime, B3SOIDDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDSPePtr, B3SOIDDSPeBinding, B3SOIDDsNodePrime, B3SOIDDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDEbPtr, B3SOIDDEbBinding, B3SOIDDeNode, B3SOIDDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDGbPtr, B3SOIDDGbBinding, B3SOIDDgNode, B3SOIDDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDPbPtr, B3SOIDDDPbBinding, B3SOIDDdNodePrime, B3SOIDDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDSPbPtr, B3SOIDDSPbBinding, B3SOIDDsNodePrime, B3SOIDDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDBePtr, B3SOIDDBeBinding, B3SOIDDbNode, B3SOIDDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDBgPtr, B3SOIDDBgBinding, B3SOIDDbNode, B3SOIDDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDBdpPtr, B3SOIDDBdpBinding, B3SOIDDbNode, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDBspPtr, B3SOIDDBspBinding, B3SOIDDbNode, B3SOIDDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDBbPtr, B3SOIDDBbBinding, B3SOIDDbNode, B3SOIDDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDEePtr, B3SOIDDEeBinding, B3SOIDDeNode, B3SOIDDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDGgPtr, B3SOIDDGgBinding, B3SOIDDgNode, B3SOIDDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDGdpPtr, B3SOIDDGdpBinding, B3SOIDDgNode, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDGspPtr, B3SOIDDGspBinding, B3SOIDDgNode, B3SOIDDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDPgPtr, B3SOIDDDPgBinding, B3SOIDDdNodePrime, B3SOIDDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDPdpPtr, B3SOIDDDPdpBinding, B3SOIDDdNodePrime, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDPspPtr, B3SOIDDDPspBinding, B3SOIDDdNodePrime, B3SOIDDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDPdPtr, B3SOIDDDPdBinding, B3SOIDDdNodePrime, B3SOIDDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDSPgPtr, B3SOIDDSPgBinding, B3SOIDDsNodePrime, B3SOIDDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDSPdpPtr, B3SOIDDSPdpBinding, B3SOIDDsNodePrime, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDSPspPtr, B3SOIDDSPspBinding, B3SOIDDsNodePrime, B3SOIDDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDSPsPtr, B3SOIDDSPsBinding, B3SOIDDsNodePrime, B3SOIDDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDdPtr, B3SOIDDDdBinding, B3SOIDDdNode, B3SOIDDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDdpPtr, B3SOIDDDdpBinding, B3SOIDDdNode, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDSsPtr, B3SOIDDSsBinding, B3SOIDDsNode, B3SOIDDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDSspPtr, B3SOIDDSspBinding, B3SOIDDsNode, B3SOIDDsNodePrime);
            if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDVbsPtr, B3SOIDDVbsBinding, B3SOIDDvbsNode, B3SOIDDvbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDIdsPtr, B3SOIDDIdsBinding, B3SOIDDidsNode, B3SOIDDidsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDIcPtr, B3SOIDDIcBinding, B3SOIDDicNode, B3SOIDDicNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDIbsPtr, B3SOIDDIbsBinding, B3SOIDDibsNode, B3SOIDDibsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDIbdPtr, B3SOIDDIbdBinding, B3SOIDDibdNode, B3SOIDDibdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDIiiPtr, B3SOIDDIiiBinding, B3SOIDDiiiNode, B3SOIDDiiiNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDIgidlPtr, B3SOIDDIgidlBinding, B3SOIDDigidlNode, B3SOIDDigidlNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDItunPtr, B3SOIDDItunBinding, B3SOIDDitunNode, B3SOIDDitunNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDIbpPtr, B3SOIDDIbpBinding, B3SOIDDibpNode, B3SOIDDibpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDAbeffPtr, B3SOIDDAbeffBinding, B3SOIDDabeffNode, B3SOIDDabeffNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDVbs0effPtr, B3SOIDDVbs0effBinding, B3SOIDDvbs0effNode, B3SOIDDvbs0effNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDVbseffPtr, B3SOIDDVbseffBinding, B3SOIDDvbseffNode, B3SOIDDvbseffNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDXcPtr, B3SOIDDXcBinding, B3SOIDDxcNode, B3SOIDDxcNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDCbbPtr, B3SOIDDCbbBinding, B3SOIDDcbbNode, B3SOIDDcbbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDCbdPtr, B3SOIDDCbdBinding, B3SOIDDcbdNode, B3SOIDDcbdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDCbgPtr, B3SOIDDCbgBinding, B3SOIDDcbgNode, B3SOIDDcbgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDqbPtr, B3SOIDDqbBinding, B3SOIDDqbNode, B3SOIDDqbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDQbfPtr, B3SOIDDQbfBinding, B3SOIDDqbfNode, B3SOIDDqbfNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDQjsPtr, B3SOIDDQjsBinding, B3SOIDDqjsNode, B3SOIDDqjsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDQjdPtr, B3SOIDDQjdBinding, B3SOIDDqjdNode, B3SOIDDqjdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDGmPtr, B3SOIDDGmBinding, B3SOIDDgmNode, B3SOIDDgmNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDGmbsPtr, B3SOIDDGmbsBinding, B3SOIDDgmbsNode, B3SOIDDgmbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDGdsPtr, B3SOIDDGdsBinding, B3SOIDDgdsNode, B3SOIDDgdsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDGmePtr, B3SOIDDGmeBinding, B3SOIDDgmeNode, B3SOIDDgmeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDVbs0teffPtr, B3SOIDDVbs0teffBinding, B3SOIDDvbs0teffNode, B3SOIDDvbs0teffNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDVthPtr, B3SOIDDVthBinding, B3SOIDDvthNode, B3SOIDDvthNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDVgsteffPtr, B3SOIDDVgsteffBinding, B3SOIDDvgsteffNode, B3SOIDDvgsteffNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDXcsatPtr, B3SOIDDXcsatBinding, B3SOIDDxcsatNode, B3SOIDDxcsatNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDVcscvPtr, B3SOIDDVcscvBinding, B3SOIDDvcscvNode, B3SOIDDvcscvNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDVdscvPtr, B3SOIDDVdscvBinding, B3SOIDDvdscvNode, B3SOIDDvdscvNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDCbePtr, B3SOIDDCbeBinding, B3SOIDDcbeNode, B3SOIDDcbeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDum1Ptr, B3SOIDDDum1Binding, B3SOIDDdum1Node, B3SOIDDdum1Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDum2Ptr, B3SOIDDDum2Binding, B3SOIDDdum2Node, B3SOIDDdum2Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDum3Ptr, B3SOIDDDum3Binding, B3SOIDDdum3Node, B3SOIDDdum3Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDum4Ptr, B3SOIDDDum4Binding, B3SOIDDdum4Node, B3SOIDDdum4Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDDum5Ptr, B3SOIDDDum5Binding, B3SOIDDdum5Node, B3SOIDDdum5Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDQaccPtr, B3SOIDDQaccBinding, B3SOIDDqaccNode, B3SOIDDqaccNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDQsub0Ptr, B3SOIDDQsub0Binding, B3SOIDDqsub0Node, B3SOIDDqsub0Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDQsubs1Ptr, B3SOIDDQsubs1Binding, B3SOIDDqsubs1Node, B3SOIDDqsubs1Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDQsubs2Ptr, B3SOIDDQsubs2Binding, B3SOIDDqsubs2Node, B3SOIDDqsubs2Node);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDqePtr, B3SOIDDqeBinding, B3SOIDDqeNode, B3SOIDDqeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDqdPtr, B3SOIDDqdBinding, B3SOIDDqdNode, B3SOIDDqdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIDDqgPtr, B3SOIDDqgBinding, B3SOIDDqgNode, B3SOIDDqgNode);
            }
        }
    }

    return (OK) ;
}

int
B3SOIDDbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIDDmodel *model = (B3SOIDDmodel *)inModel ;
    B3SOIDDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B3SOIDD models */
    for ( ; model != NULL ; model = B3SOIDDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B3SOIDDinstances(model); here != NULL ; here = B3SOIDDnextInstance(here))
        {
            if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0 != 0.0))
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDTemptempPtr, B3SOIDDTemptempBinding, B3SOIDDtempNode, B3SOIDDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDTempdpPtr, B3SOIDDTempdpBinding, B3SOIDDtempNode, B3SOIDDdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDTempspPtr, B3SOIDDTempspBinding, B3SOIDDtempNode, B3SOIDDsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDTempgPtr, B3SOIDDTempgBinding, B3SOIDDtempNode, B3SOIDDgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDTempbPtr, B3SOIDDTempbBinding, B3SOIDDtempNode, B3SOIDDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDTempePtr, B3SOIDDTempeBinding, B3SOIDDtempNode, B3SOIDDeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDGtempPtr, B3SOIDDGtempBinding, B3SOIDDgNode, B3SOIDDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDPtempPtr, B3SOIDDDPtempBinding, B3SOIDDdNodePrime, B3SOIDDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDSPtempPtr, B3SOIDDSPtempBinding, B3SOIDDsNodePrime, B3SOIDDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDEtempPtr, B3SOIDDEtempBinding, B3SOIDDeNode, B3SOIDDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDBtempPtr, B3SOIDDBtempBinding, B3SOIDDbNode, B3SOIDDtempNode);
                if (here->B3SOIDDbodyMod == 1)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDPtempPtr, B3SOIDDPtempBinding, B3SOIDDpNode, B3SOIDDtempNode);
                }
            }
            if (here->B3SOIDDbodyMod == 2)
            {
            }
            else if (here->B3SOIDDbodyMod == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDBpPtr, B3SOIDDBpBinding, B3SOIDDbNode, B3SOIDDpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDPbPtr, B3SOIDDPbBinding, B3SOIDDpNode, B3SOIDDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDPpPtr, B3SOIDDPpBinding, B3SOIDDpNode, B3SOIDDpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDPgPtr, B3SOIDDPgBinding, B3SOIDDpNode, B3SOIDDgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDPdpPtr, B3SOIDDPdpBinding, B3SOIDDpNode, B3SOIDDdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDPspPtr, B3SOIDDPspBinding, B3SOIDDpNode, B3SOIDDsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDPePtr, B3SOIDDPeBinding, B3SOIDDpNode, B3SOIDDeNode);
            }
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDEgPtr, B3SOIDDEgBinding, B3SOIDDeNode, B3SOIDDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDEdpPtr, B3SOIDDEdpBinding, B3SOIDDeNode, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDEspPtr, B3SOIDDEspBinding, B3SOIDDeNode, B3SOIDDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDGePtr, B3SOIDDGeBinding, B3SOIDDgNode, B3SOIDDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDPePtr, B3SOIDDDPeBinding, B3SOIDDdNodePrime, B3SOIDDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDSPePtr, B3SOIDDSPeBinding, B3SOIDDsNodePrime, B3SOIDDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDEbPtr, B3SOIDDEbBinding, B3SOIDDeNode, B3SOIDDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDGbPtr, B3SOIDDGbBinding, B3SOIDDgNode, B3SOIDDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDPbPtr, B3SOIDDDPbBinding, B3SOIDDdNodePrime, B3SOIDDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDSPbPtr, B3SOIDDSPbBinding, B3SOIDDsNodePrime, B3SOIDDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDBePtr, B3SOIDDBeBinding, B3SOIDDbNode, B3SOIDDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDBgPtr, B3SOIDDBgBinding, B3SOIDDbNode, B3SOIDDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDBdpPtr, B3SOIDDBdpBinding, B3SOIDDbNode, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDBspPtr, B3SOIDDBspBinding, B3SOIDDbNode, B3SOIDDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDBbPtr, B3SOIDDBbBinding, B3SOIDDbNode, B3SOIDDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDEePtr, B3SOIDDEeBinding, B3SOIDDeNode, B3SOIDDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDGgPtr, B3SOIDDGgBinding, B3SOIDDgNode, B3SOIDDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDGdpPtr, B3SOIDDGdpBinding, B3SOIDDgNode, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDGspPtr, B3SOIDDGspBinding, B3SOIDDgNode, B3SOIDDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDPgPtr, B3SOIDDDPgBinding, B3SOIDDdNodePrime, B3SOIDDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDPdpPtr, B3SOIDDDPdpBinding, B3SOIDDdNodePrime, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDPspPtr, B3SOIDDDPspBinding, B3SOIDDdNodePrime, B3SOIDDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDPdPtr, B3SOIDDDPdBinding, B3SOIDDdNodePrime, B3SOIDDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDSPgPtr, B3SOIDDSPgBinding, B3SOIDDsNodePrime, B3SOIDDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDSPdpPtr, B3SOIDDSPdpBinding, B3SOIDDsNodePrime, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDSPspPtr, B3SOIDDSPspBinding, B3SOIDDsNodePrime, B3SOIDDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDSPsPtr, B3SOIDDSPsBinding, B3SOIDDsNodePrime, B3SOIDDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDdPtr, B3SOIDDDdBinding, B3SOIDDdNode, B3SOIDDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDdpPtr, B3SOIDDDdpBinding, B3SOIDDdNode, B3SOIDDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDSsPtr, B3SOIDDSsBinding, B3SOIDDsNode, B3SOIDDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDSspPtr, B3SOIDDSspBinding, B3SOIDDsNode, B3SOIDDsNodePrime);
            if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDVbsPtr, B3SOIDDVbsBinding, B3SOIDDvbsNode, B3SOIDDvbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDIdsPtr, B3SOIDDIdsBinding, B3SOIDDidsNode, B3SOIDDidsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDIcPtr, B3SOIDDIcBinding, B3SOIDDicNode, B3SOIDDicNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDIbsPtr, B3SOIDDIbsBinding, B3SOIDDibsNode, B3SOIDDibsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDIbdPtr, B3SOIDDIbdBinding, B3SOIDDibdNode, B3SOIDDibdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDIiiPtr, B3SOIDDIiiBinding, B3SOIDDiiiNode, B3SOIDDiiiNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDIgidlPtr, B3SOIDDIgidlBinding, B3SOIDDigidlNode, B3SOIDDigidlNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDItunPtr, B3SOIDDItunBinding, B3SOIDDitunNode, B3SOIDDitunNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDIbpPtr, B3SOIDDIbpBinding, B3SOIDDibpNode, B3SOIDDibpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDAbeffPtr, B3SOIDDAbeffBinding, B3SOIDDabeffNode, B3SOIDDabeffNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDVbs0effPtr, B3SOIDDVbs0effBinding, B3SOIDDvbs0effNode, B3SOIDDvbs0effNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDVbseffPtr, B3SOIDDVbseffBinding, B3SOIDDvbseffNode, B3SOIDDvbseffNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDXcPtr, B3SOIDDXcBinding, B3SOIDDxcNode, B3SOIDDxcNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDCbbPtr, B3SOIDDCbbBinding, B3SOIDDcbbNode, B3SOIDDcbbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDCbdPtr, B3SOIDDCbdBinding, B3SOIDDcbdNode, B3SOIDDcbdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDCbgPtr, B3SOIDDCbgBinding, B3SOIDDcbgNode, B3SOIDDcbgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDqbPtr, B3SOIDDqbBinding, B3SOIDDqbNode, B3SOIDDqbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDQbfPtr, B3SOIDDQbfBinding, B3SOIDDqbfNode, B3SOIDDqbfNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDQjsPtr, B3SOIDDQjsBinding, B3SOIDDqjsNode, B3SOIDDqjsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDQjdPtr, B3SOIDDQjdBinding, B3SOIDDqjdNode, B3SOIDDqjdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDGmPtr, B3SOIDDGmBinding, B3SOIDDgmNode, B3SOIDDgmNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDGmbsPtr, B3SOIDDGmbsBinding, B3SOIDDgmbsNode, B3SOIDDgmbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDGdsPtr, B3SOIDDGdsBinding, B3SOIDDgdsNode, B3SOIDDgdsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDGmePtr, B3SOIDDGmeBinding, B3SOIDDgmeNode, B3SOIDDgmeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDVbs0teffPtr, B3SOIDDVbs0teffBinding, B3SOIDDvbs0teffNode, B3SOIDDvbs0teffNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDVthPtr, B3SOIDDVthBinding, B3SOIDDvthNode, B3SOIDDvthNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDVgsteffPtr, B3SOIDDVgsteffBinding, B3SOIDDvgsteffNode, B3SOIDDvgsteffNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDXcsatPtr, B3SOIDDXcsatBinding, B3SOIDDxcsatNode, B3SOIDDxcsatNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDVcscvPtr, B3SOIDDVcscvBinding, B3SOIDDvcscvNode, B3SOIDDvcscvNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDVdscvPtr, B3SOIDDVdscvBinding, B3SOIDDvdscvNode, B3SOIDDvdscvNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDCbePtr, B3SOIDDCbeBinding, B3SOIDDcbeNode, B3SOIDDcbeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDum1Ptr, B3SOIDDDum1Binding, B3SOIDDdum1Node, B3SOIDDdum1Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDum2Ptr, B3SOIDDDum2Binding, B3SOIDDdum2Node, B3SOIDDdum2Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDum3Ptr, B3SOIDDDum3Binding, B3SOIDDdum3Node, B3SOIDDdum3Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDum4Ptr, B3SOIDDDum4Binding, B3SOIDDdum4Node, B3SOIDDdum4Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDDum5Ptr, B3SOIDDDum5Binding, B3SOIDDdum5Node, B3SOIDDdum5Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDQaccPtr, B3SOIDDQaccBinding, B3SOIDDqaccNode, B3SOIDDqaccNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDQsub0Ptr, B3SOIDDQsub0Binding, B3SOIDDqsub0Node, B3SOIDDqsub0Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDQsubs1Ptr, B3SOIDDQsubs1Binding, B3SOIDDqsubs1Node, B3SOIDDqsubs1Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDQsubs2Ptr, B3SOIDDQsubs2Binding, B3SOIDDqsubs2Node, B3SOIDDqsubs2Node);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDqePtr, B3SOIDDqeBinding, B3SOIDDqeNode, B3SOIDDqeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDqdPtr, B3SOIDDqdBinding, B3SOIDDqdNode, B3SOIDDqdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIDDqgPtr, B3SOIDDqgBinding, B3SOIDDqgNode, B3SOIDDqgNode);
            }
        }
    }

    return (OK) ;
}
