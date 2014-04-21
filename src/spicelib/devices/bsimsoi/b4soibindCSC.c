/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b4soidef.h"
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
B4SOIbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    B4SOImodel *model = (B4SOImodel *)inModel ;
    B4SOIinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the B4SOI models */
    for ( ; model != NULL ; model = B4SOInextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B4SOIinstances(model); here != NULL ; here = B4SOInextInstance(here))
        {
            if ((model->B4SOIshMod == 1) && (here->B4SOIrth0 != 0.0))
            {
                CREATE_KLU_BINDING_TABLE(B4SOITemptempPtr, B4SOITemptempBinding, B4SOItempNode, B4SOItempNode);
                CREATE_KLU_BINDING_TABLE(B4SOITempdpPtr, B4SOITempdpBinding, B4SOItempNode, B4SOIdNodePrime);
                CREATE_KLU_BINDING_TABLE(B4SOITempspPtr, B4SOITempspBinding, B4SOItempNode, B4SOIsNodePrime);
                CREATE_KLU_BINDING_TABLE(B4SOITempgPtr, B4SOITempgBinding, B4SOItempNode, B4SOIgNode);
                CREATE_KLU_BINDING_TABLE(B4SOITempbPtr, B4SOITempbBinding, B4SOItempNode, B4SOIbNode);
                CREATE_KLU_BINDING_TABLE(B4SOIGtempPtr, B4SOIGtempBinding, B4SOIgNode, B4SOItempNode);
                CREATE_KLU_BINDING_TABLE(B4SOIDPtempPtr, B4SOIDPtempBinding, B4SOIdNodePrime, B4SOItempNode);
                CREATE_KLU_BINDING_TABLE(B4SOISPtempPtr, B4SOISPtempBinding, B4SOIsNodePrime, B4SOItempNode);
                CREATE_KLU_BINDING_TABLE(B4SOIEtempPtr, B4SOIEtempBinding, B4SOIeNode, B4SOItempNode);
                CREATE_KLU_BINDING_TABLE(B4SOIBtempPtr, B4SOIBtempBinding, B4SOIbNode, B4SOItempNode);
                if (here->B4SOIbodyMod == 1)
                {
                    CREATE_KLU_BINDING_TABLE(B4SOIPtempPtr, B4SOIPtempBinding, B4SOIpNode, B4SOItempNode);
                }
                if (here->B4SOIsoiMod != 0)
                {
                    CREATE_KLU_BINDING_TABLE(B4SOITempePtr, B4SOITempeBinding, B4SOItempNode, B4SOIeNode);
                }
            }
            if (here->B4SOIbodyMod == 2)
            {
            }
            else if (here->B4SOIbodyMod == 1)
            {
                CREATE_KLU_BINDING_TABLE(B4SOIBpPtr, B4SOIBpBinding, B4SOIbNode, B4SOIpNode);
                CREATE_KLU_BINDING_TABLE(B4SOIPbPtr, B4SOIPbBinding, B4SOIpNode, B4SOIbNode);
                CREATE_KLU_BINDING_TABLE(B4SOIPpPtr, B4SOIPpBinding, B4SOIpNode, B4SOIpNode);
                CREATE_KLU_BINDING_TABLE(B4SOIPgPtr, B4SOIPgBinding, B4SOIpNode, B4SOIgNode);
                CREATE_KLU_BINDING_TABLE(B4SOIGpPtr, B4SOIGpBinding, B4SOIgNode, B4SOIpNode);
            }
            if (here->B4SOIrgateMod != 0)
            {
                CREATE_KLU_BINDING_TABLE(B4SOIGEgePtr, B4SOIGEgeBinding, B4SOIgNodeExt, B4SOIgNodeExt);
                CREATE_KLU_BINDING_TABLE(B4SOIGEgPtr, B4SOIGEgBinding, B4SOIgNodeExt, B4SOIgNode);
                CREATE_KLU_BINDING_TABLE(B4SOIGgePtr, B4SOIGgeBinding, B4SOIgNode, B4SOIgNodeExt);
                CREATE_KLU_BINDING_TABLE(B4SOIGEdpPtr, B4SOIGEdpBinding, B4SOIgNodeExt, B4SOIdNodePrime);
                CREATE_KLU_BINDING_TABLE(B4SOIGEspPtr, B4SOIGEspBinding, B4SOIgNodeExt, B4SOIsNodePrime);
                if (here->B4SOIsoiMod != 2)
                {
                    CREATE_KLU_BINDING_TABLE(B4SOIGEbPtr, B4SOIGEbBinding, B4SOIgNodeExt, B4SOIbNode);
                }
                CREATE_KLU_BINDING_TABLE(B4SOIGMdpPtr, B4SOIGMdpBinding, B4SOIgNodeMid, B4SOIdNodePrime);
                CREATE_KLU_BINDING_TABLE(B4SOIGMgPtr, B4SOIGMgBinding, B4SOIgNodeMid, B4SOIgNode);
                CREATE_KLU_BINDING_TABLE(B4SOIGMgmPtr, B4SOIGMgmBinding, B4SOIgNodeMid, B4SOIgNodeMid);
                CREATE_KLU_BINDING_TABLE(B4SOIGMgePtr, B4SOIGMgeBinding, B4SOIgNodeMid, B4SOIgNodeExt);
                CREATE_KLU_BINDING_TABLE(B4SOIGMspPtr, B4SOIGMspBinding, B4SOIgNodeMid, B4SOIsNodePrime);
                if (here->B4SOIsoiMod != 2)
                {
                    CREATE_KLU_BINDING_TABLE(B4SOIGMbPtr, B4SOIGMbBinding, B4SOIgNodeMid, B4SOIbNode);
                }
                CREATE_KLU_BINDING_TABLE(B4SOIGMePtr, B4SOIGMeBinding, B4SOIgNodeMid, B4SOIeNode);
                CREATE_KLU_BINDING_TABLE(B4SOIDPgmPtr, B4SOIDPgmBinding, B4SOIdNodePrime, B4SOIgNodeMid);
                CREATE_KLU_BINDING_TABLE(B4SOIGgmPtr, B4SOIGgmBinding, B4SOIgNode, B4SOIgNodeMid);
                CREATE_KLU_BINDING_TABLE(B4SOIGEgmPtr, B4SOIGEgmBinding, B4SOIgNodeExt, B4SOIgNodeMid);
                CREATE_KLU_BINDING_TABLE(B4SOISPgmPtr, B4SOISPgmBinding, B4SOIsNodePrime, B4SOIgNodeMid);
                CREATE_KLU_BINDING_TABLE(B4SOIEgmPtr, B4SOIEgmBinding, B4SOIeNode, B4SOIgNodeMid);
            }
            if (here->B4SOIsoiMod != 2) /* v3.2 */
            {
                CREATE_KLU_BINDING_TABLE(B4SOIEbPtr, B4SOIEbBinding, B4SOIeNode, B4SOIbNode);
                CREATE_KLU_BINDING_TABLE(B4SOIGbPtr, B4SOIGbBinding, B4SOIgNode, B4SOIbNode);
                CREATE_KLU_BINDING_TABLE(B4SOIDPbPtr, B4SOIDPbBinding, B4SOIdNodePrime, B4SOIbNode);
                CREATE_KLU_BINDING_TABLE(B4SOISPbPtr, B4SOISPbBinding, B4SOIsNodePrime, B4SOIbNode);
                CREATE_KLU_BINDING_TABLE(B4SOIBePtr, B4SOIBeBinding, B4SOIbNode, B4SOIeNode);
                CREATE_KLU_BINDING_TABLE(B4SOIBgPtr, B4SOIBgBinding, B4SOIbNode, B4SOIgNode);
                CREATE_KLU_BINDING_TABLE(B4SOIBdpPtr, B4SOIBdpBinding, B4SOIbNode, B4SOIdNodePrime);
                CREATE_KLU_BINDING_TABLE(B4SOIBspPtr, B4SOIBspBinding, B4SOIbNode, B4SOIsNodePrime);
                CREATE_KLU_BINDING_TABLE(B4SOIBbPtr, B4SOIBbBinding, B4SOIbNode, B4SOIbNode);
            }
            CREATE_KLU_BINDING_TABLE(B4SOIEgPtr, B4SOIEgBinding, B4SOIeNode, B4SOIgNode);
            CREATE_KLU_BINDING_TABLE(B4SOIEdpPtr, B4SOIEdpBinding, B4SOIeNode, B4SOIdNodePrime);
            CREATE_KLU_BINDING_TABLE(B4SOIEspPtr, B4SOIEspBinding, B4SOIeNode, B4SOIsNodePrime);
            CREATE_KLU_BINDING_TABLE(B4SOIGePtr, B4SOIGeBinding, B4SOIgNode, B4SOIeNode);
            CREATE_KLU_BINDING_TABLE(B4SOIDPePtr, B4SOIDPeBinding, B4SOIdNodePrime, B4SOIeNode);
            CREATE_KLU_BINDING_TABLE(B4SOISPePtr, B4SOISPeBinding, B4SOIsNodePrime, B4SOIeNode);
            CREATE_KLU_BINDING_TABLE(B4SOIEePtr, B4SOIEeBinding, B4SOIeNode, B4SOIeNode);
            CREATE_KLU_BINDING_TABLE(B4SOIGgPtr, B4SOIGgBinding, B4SOIgNode, B4SOIgNode);
            CREATE_KLU_BINDING_TABLE(B4SOIGdpPtr, B4SOIGdpBinding, B4SOIgNode, B4SOIdNodePrime);
            CREATE_KLU_BINDING_TABLE(B4SOIGspPtr, B4SOIGspBinding, B4SOIgNode, B4SOIsNodePrime);
            CREATE_KLU_BINDING_TABLE(B4SOIDPgPtr, B4SOIDPgBinding, B4SOIdNodePrime, B4SOIgNode);
            CREATE_KLU_BINDING_TABLE(B4SOIDPdpPtr, B4SOIDPdpBinding, B4SOIdNodePrime, B4SOIdNodePrime);
            CREATE_KLU_BINDING_TABLE(B4SOIDPspPtr, B4SOIDPspBinding, B4SOIdNodePrime, B4SOIsNodePrime);
            CREATE_KLU_BINDING_TABLE(B4SOIDPdPtr, B4SOIDPdBinding, B4SOIdNodePrime, B4SOIdNode);
            CREATE_KLU_BINDING_TABLE(B4SOISPgPtr, B4SOISPgBinding, B4SOIsNodePrime, B4SOIgNode);
            CREATE_KLU_BINDING_TABLE(B4SOISPdpPtr, B4SOISPdpBinding, B4SOIsNodePrime, B4SOIdNodePrime);
            CREATE_KLU_BINDING_TABLE(B4SOISPspPtr, B4SOISPspBinding, B4SOIsNodePrime, B4SOIsNodePrime);
            CREATE_KLU_BINDING_TABLE(B4SOISPsPtr, B4SOISPsBinding, B4SOIsNodePrime, B4SOIsNode);
            CREATE_KLU_BINDING_TABLE(B4SOIDdPtr, B4SOIDdBinding, B4SOIdNode, B4SOIdNode);
            CREATE_KLU_BINDING_TABLE(B4SOIDdpPtr, B4SOIDdpBinding, B4SOIdNode, B4SOIdNodePrime);
            CREATE_KLU_BINDING_TABLE(B4SOISsPtr, B4SOISsBinding, B4SOIsNode, B4SOIsNode);
            CREATE_KLU_BINDING_TABLE(B4SOISspPtr, B4SOISspBinding, B4SOIsNode, B4SOIsNodePrime);
            if (here->B4SOIrbodyMod == 1)
            {
                CREATE_KLU_BINDING_TABLE(B4SOIDPdbPtr, B4SOIDPdbBinding, B4SOIdNodePrime, B4SOIdbNode);
                CREATE_KLU_BINDING_TABLE(B4SOISPsbPtr, B4SOISPsbBinding, B4SOIsNodePrime, B4SOIsbNode);
                CREATE_KLU_BINDING_TABLE(B4SOIDBdpPtr, B4SOIDBdpBinding, B4SOIdbNode, B4SOIdNodePrime);
                CREATE_KLU_BINDING_TABLE(B4SOIDBdbPtr, B4SOIDBdbBinding, B4SOIdbNode, B4SOIdbNode);
                CREATE_KLU_BINDING_TABLE(B4SOIDBbPtr, B4SOIDBbBinding, B4SOIdbNode, B4SOIbNode);
                CREATE_KLU_BINDING_TABLE(B4SOISBspPtr, B4SOISBspBinding, B4SOIsbNode, B4SOIsNodePrime);
                CREATE_KLU_BINDING_TABLE(B4SOISBsbPtr, B4SOISBsbBinding, B4SOIsbNode, B4SOIsbNode);
                CREATE_KLU_BINDING_TABLE(B4SOISBbPtr, B4SOISBbBinding, B4SOIsbNode, B4SOIbNode);
                CREATE_KLU_BINDING_TABLE(B4SOIBdbPtr, B4SOIBdbBinding, B4SOIbNode, B4SOIdbNode);
                CREATE_KLU_BINDING_TABLE(B4SOIBsbPtr, B4SOIBsbBinding, B4SOIbNode, B4SOIsbNode);
            }
            if (model->B4SOIrdsMod)
            {
                CREATE_KLU_BINDING_TABLE(B4SOIDgPtr, B4SOIDgBinding, B4SOIdNode, B4SOIgNode);
                CREATE_KLU_BINDING_TABLE(B4SOIDspPtr, B4SOIDspBinding, B4SOIdNode, B4SOIsNodePrime);
                CREATE_KLU_BINDING_TABLE(B4SOISdpPtr, B4SOISdpBinding, B4SOIsNode, B4SOIdNodePrime);
                CREATE_KLU_BINDING_TABLE(B4SOISgPtr, B4SOISgBinding, B4SOIsNode, B4SOIgNode);
                if (model->B4SOIsoiMod != 2)
                {
                    CREATE_KLU_BINDING_TABLE(B4SOIDbPtr, B4SOIDbBinding, B4SOIdNode, B4SOIbNode);
                    CREATE_KLU_BINDING_TABLE(B4SOISbPtr, B4SOISbBinding, B4SOIsNode, B4SOIbNode);
                }
            }
            if (here->B4SOIdebugMod != 0)
            {
                CREATE_KLU_BINDING_TABLE(B4SOIVbsPtr, B4SOIVbsBinding, B4SOIvbsNode, B4SOIvbsNode);
                CREATE_KLU_BINDING_TABLE(B4SOIIdsPtr, B4SOIIdsBinding, B4SOIidsNode, B4SOIidsNode);
                CREATE_KLU_BINDING_TABLE(B4SOIIcPtr, B4SOIIcBinding, B4SOIicNode, B4SOIicNode);
                CREATE_KLU_BINDING_TABLE(B4SOIIbsPtr, B4SOIIbsBinding, B4SOIibsNode, B4SOIibsNode);
                CREATE_KLU_BINDING_TABLE(B4SOIIbdPtr, B4SOIIbdBinding, B4SOIibdNode, B4SOIibdNode);
                CREATE_KLU_BINDING_TABLE(B4SOIIiiPtr, B4SOIIiiBinding, B4SOIiiiNode, B4SOIiiiNode);
                CREATE_KLU_BINDING_TABLE(B4SOIIgPtr, B4SOIIgBinding, B4SOIigNode, B4SOIigNode);
                CREATE_KLU_BINDING_TABLE(B4SOIGiggPtr, B4SOIGiggBinding, B4SOIgiggNode, B4SOIgiggNode);
                CREATE_KLU_BINDING_TABLE(B4SOIGigdPtr, B4SOIGigdBinding, B4SOIgigdNode, B4SOIgigdNode);
                CREATE_KLU_BINDING_TABLE(B4SOIGigbPtr, B4SOIGigbBinding, B4SOIgigbNode, B4SOIgigbNode);
                CREATE_KLU_BINDING_TABLE(B4SOIIgidlPtr, B4SOIIgidlBinding, B4SOIigidlNode, B4SOIigidlNode);
                CREATE_KLU_BINDING_TABLE(B4SOIItunPtr, B4SOIItunBinding, B4SOIitunNode, B4SOIitunNode);
                CREATE_KLU_BINDING_TABLE(B4SOIIbpPtr, B4SOIIbpBinding, B4SOIibpNode, B4SOIibpNode);
                CREATE_KLU_BINDING_TABLE(B4SOICbbPtr, B4SOICbbBinding, B4SOIcbbNode, B4SOIcbbNode);
                CREATE_KLU_BINDING_TABLE(B4SOICbdPtr, B4SOICbdBinding, B4SOIcbdNode, B4SOIcbdNode);
                CREATE_KLU_BINDING_TABLE(B4SOICbgPtr, B4SOICbgBinding, B4SOIcbgNode, B4SOIcbgNode);
                CREATE_KLU_BINDING_TABLE(B4SOIQbfPtr, B4SOIQbfBinding, B4SOIqbfNode, B4SOIqbfNode);
                CREATE_KLU_BINDING_TABLE(B4SOIQjsPtr, B4SOIQjsBinding, B4SOIqjsNode, B4SOIqjsNode);
                CREATE_KLU_BINDING_TABLE(B4SOIQjdPtr, B4SOIQjdBinding, B4SOIqjdNode, B4SOIqjdNode);
            }
        }
    }

    return (OK) ;
}

int
B4SOIbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    B4SOImodel *model = (B4SOImodel *)inModel ;
    B4SOIinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B4SOI models */
    for ( ; model != NULL ; model = B4SOInextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B4SOIinstances(model); here != NULL ; here = B4SOInextInstance(here))
        {
            if ((model->B4SOIshMod == 1) && (here->B4SOIrth0 != 0.0))
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOITemptempPtr, B4SOITemptempBinding, B4SOItempNode, B4SOItempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOITempdpPtr, B4SOITempdpBinding, B4SOItempNode, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOITempspPtr, B4SOITempspBinding, B4SOItempNode, B4SOIsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOITempgPtr, B4SOITempgBinding, B4SOItempNode, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOITempbPtr, B4SOITempbBinding, B4SOItempNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGtempPtr, B4SOIGtempBinding, B4SOIgNode, B4SOItempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDPtempPtr, B4SOIDPtempBinding, B4SOIdNodePrime, B4SOItempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISPtempPtr, B4SOISPtempBinding, B4SOIsNodePrime, B4SOItempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIEtempPtr, B4SOIEtempBinding, B4SOIeNode, B4SOItempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIBtempPtr, B4SOIBtempBinding, B4SOIbNode, B4SOItempNode);
                if (here->B4SOIbodyMod == 1)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIPtempPtr, B4SOIPtempBinding, B4SOIpNode, B4SOItempNode);
                }
                if (here->B4SOIsoiMod != 0)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOITempePtr, B4SOITempeBinding, B4SOItempNode, B4SOIeNode);
                }
            }
            if (here->B4SOIbodyMod == 2)
            {
            }
            else if (here->B4SOIbodyMod == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIBpPtr, B4SOIBpBinding, B4SOIbNode, B4SOIpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIPbPtr, B4SOIPbBinding, B4SOIpNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIPpPtr, B4SOIPpBinding, B4SOIpNode, B4SOIpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIPgPtr, B4SOIPgBinding, B4SOIpNode, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGpPtr, B4SOIGpBinding, B4SOIgNode, B4SOIpNode);
            }
            if (here->B4SOIrgateMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGEgePtr, B4SOIGEgeBinding, B4SOIgNodeExt, B4SOIgNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGEgPtr, B4SOIGEgBinding, B4SOIgNodeExt, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGgePtr, B4SOIGgeBinding, B4SOIgNode, B4SOIgNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGEdpPtr, B4SOIGEdpBinding, B4SOIgNodeExt, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGEspPtr, B4SOIGEspBinding, B4SOIgNodeExt, B4SOIsNodePrime);
                if (here->B4SOIsoiMod != 2)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGEbPtr, B4SOIGEbBinding, B4SOIgNodeExt, B4SOIbNode);
                }
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGMdpPtr, B4SOIGMdpBinding, B4SOIgNodeMid, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGMgPtr, B4SOIGMgBinding, B4SOIgNodeMid, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGMgmPtr, B4SOIGMgmBinding, B4SOIgNodeMid, B4SOIgNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGMgePtr, B4SOIGMgeBinding, B4SOIgNodeMid, B4SOIgNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGMspPtr, B4SOIGMspBinding, B4SOIgNodeMid, B4SOIsNodePrime);
                if (here->B4SOIsoiMod != 2)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGMbPtr, B4SOIGMbBinding, B4SOIgNodeMid, B4SOIbNode);
                }
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGMePtr, B4SOIGMeBinding, B4SOIgNodeMid, B4SOIeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDPgmPtr, B4SOIDPgmBinding, B4SOIdNodePrime, B4SOIgNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGgmPtr, B4SOIGgmBinding, B4SOIgNode, B4SOIgNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGEgmPtr, B4SOIGEgmBinding, B4SOIgNodeExt, B4SOIgNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISPgmPtr, B4SOISPgmBinding, B4SOIsNodePrime, B4SOIgNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIEgmPtr, B4SOIEgmBinding, B4SOIeNode, B4SOIgNodeMid);
            }
            if (here->B4SOIsoiMod != 2) /* v3.2 */
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIEbPtr, B4SOIEbBinding, B4SOIeNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGbPtr, B4SOIGbBinding, B4SOIgNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDPbPtr, B4SOIDPbBinding, B4SOIdNodePrime, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISPbPtr, B4SOISPbBinding, B4SOIsNodePrime, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIBePtr, B4SOIBeBinding, B4SOIbNode, B4SOIeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIBgPtr, B4SOIBgBinding, B4SOIbNode, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIBdpPtr, B4SOIBdpBinding, B4SOIbNode, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIBspPtr, B4SOIBspBinding, B4SOIbNode, B4SOIsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIBbPtr, B4SOIBbBinding, B4SOIbNode, B4SOIbNode);
            }
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIEgPtr, B4SOIEgBinding, B4SOIeNode, B4SOIgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIEdpPtr, B4SOIEdpBinding, B4SOIeNode, B4SOIdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIEspPtr, B4SOIEspBinding, B4SOIeNode, B4SOIsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGePtr, B4SOIGeBinding, B4SOIgNode, B4SOIeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDPePtr, B4SOIDPeBinding, B4SOIdNodePrime, B4SOIeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISPePtr, B4SOISPeBinding, B4SOIsNodePrime, B4SOIeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIEePtr, B4SOIEeBinding, B4SOIeNode, B4SOIeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGgPtr, B4SOIGgBinding, B4SOIgNode, B4SOIgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGdpPtr, B4SOIGdpBinding, B4SOIgNode, B4SOIdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGspPtr, B4SOIGspBinding, B4SOIgNode, B4SOIsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDPgPtr, B4SOIDPgBinding, B4SOIdNodePrime, B4SOIgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDPdpPtr, B4SOIDPdpBinding, B4SOIdNodePrime, B4SOIdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDPspPtr, B4SOIDPspBinding, B4SOIdNodePrime, B4SOIsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDPdPtr, B4SOIDPdBinding, B4SOIdNodePrime, B4SOIdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISPgPtr, B4SOISPgBinding, B4SOIsNodePrime, B4SOIgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISPdpPtr, B4SOISPdpBinding, B4SOIsNodePrime, B4SOIdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISPspPtr, B4SOISPspBinding, B4SOIsNodePrime, B4SOIsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISPsPtr, B4SOISPsBinding, B4SOIsNodePrime, B4SOIsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDdPtr, B4SOIDdBinding, B4SOIdNode, B4SOIdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDdpPtr, B4SOIDdpBinding, B4SOIdNode, B4SOIdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISsPtr, B4SOISsBinding, B4SOIsNode, B4SOIsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISspPtr, B4SOISspBinding, B4SOIsNode, B4SOIsNodePrime);
            if (here->B4SOIrbodyMod == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDPdbPtr, B4SOIDPdbBinding, B4SOIdNodePrime, B4SOIdbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISPsbPtr, B4SOISPsbBinding, B4SOIsNodePrime, B4SOIsbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDBdpPtr, B4SOIDBdpBinding, B4SOIdbNode, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDBdbPtr, B4SOIDBdbBinding, B4SOIdbNode, B4SOIdbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDBbPtr, B4SOIDBbBinding, B4SOIdbNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISBspPtr, B4SOISBspBinding, B4SOIsbNode, B4SOIsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISBsbPtr, B4SOISBsbBinding, B4SOIsbNode, B4SOIsbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISBbPtr, B4SOISBbBinding, B4SOIsbNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIBdbPtr, B4SOIBdbBinding, B4SOIbNode, B4SOIdbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIBsbPtr, B4SOIBsbBinding, B4SOIbNode, B4SOIsbNode);
            }
            if (model->B4SOIrdsMod)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDgPtr, B4SOIDgBinding, B4SOIdNode, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDspPtr, B4SOIDspBinding, B4SOIdNode, B4SOIsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISdpPtr, B4SOISdpBinding, B4SOIsNode, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISgPtr, B4SOISgBinding, B4SOIsNode, B4SOIgNode);
                if (model->B4SOIsoiMod != 2)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIDbPtr, B4SOIDbBinding, B4SOIdNode, B4SOIbNode);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOISbPtr, B4SOISbBinding, B4SOIsNode, B4SOIbNode);
                }
            }
            if (here->B4SOIdebugMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIVbsPtr, B4SOIVbsBinding, B4SOIvbsNode, B4SOIvbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIIdsPtr, B4SOIIdsBinding, B4SOIidsNode, B4SOIidsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIIcPtr, B4SOIIcBinding, B4SOIicNode, B4SOIicNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIIbsPtr, B4SOIIbsBinding, B4SOIibsNode, B4SOIibsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIIbdPtr, B4SOIIbdBinding, B4SOIibdNode, B4SOIibdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIIiiPtr, B4SOIIiiBinding, B4SOIiiiNode, B4SOIiiiNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIIgPtr, B4SOIIgBinding, B4SOIigNode, B4SOIigNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGiggPtr, B4SOIGiggBinding, B4SOIgiggNode, B4SOIgiggNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGigdPtr, B4SOIGigdBinding, B4SOIgigdNode, B4SOIgigdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIGigbPtr, B4SOIGigbBinding, B4SOIgigbNode, B4SOIgigbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIIgidlPtr, B4SOIIgidlBinding, B4SOIigidlNode, B4SOIigidlNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIItunPtr, B4SOIItunBinding, B4SOIitunNode, B4SOIitunNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIIbpPtr, B4SOIIbpBinding, B4SOIibpNode, B4SOIibpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOICbbPtr, B4SOICbbBinding, B4SOIcbbNode, B4SOIcbbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOICbdPtr, B4SOICbdBinding, B4SOIcbdNode, B4SOIcbdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOICbgPtr, B4SOICbgBinding, B4SOIcbgNode, B4SOIcbgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIQbfPtr, B4SOIQbfBinding, B4SOIqbfNode, B4SOIqbfNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIQjsPtr, B4SOIQjsBinding, B4SOIqjsNode, B4SOIqjsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B4SOIQjdPtr, B4SOIQjdBinding, B4SOIqjdNode, B4SOIqjdNode);
            }
        }
    }

    return (OK) ;
}

int
B4SOIbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B4SOImodel *model = (B4SOImodel *)inModel ;
    B4SOIinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B4SOI models */
    for ( ; model != NULL ; model = B4SOInextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B4SOIinstances(model); here != NULL ; here = B4SOInextInstance(here))
        {
            if ((model->B4SOIshMod == 1) && (here->B4SOIrth0 != 0.0))
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOITemptempPtr, B4SOITemptempBinding, B4SOItempNode, B4SOItempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOITempdpPtr, B4SOITempdpBinding, B4SOItempNode, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOITempspPtr, B4SOITempspBinding, B4SOItempNode, B4SOIsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOITempgPtr, B4SOITempgBinding, B4SOItempNode, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOITempbPtr, B4SOITempbBinding, B4SOItempNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGtempPtr, B4SOIGtempBinding, B4SOIgNode, B4SOItempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDPtempPtr, B4SOIDPtempBinding, B4SOIdNodePrime, B4SOItempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISPtempPtr, B4SOISPtempBinding, B4SOIsNodePrime, B4SOItempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIEtempPtr, B4SOIEtempBinding, B4SOIeNode, B4SOItempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIBtempPtr, B4SOIBtempBinding, B4SOIbNode, B4SOItempNode);
                if (here->B4SOIbodyMod == 1)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIPtempPtr, B4SOIPtempBinding, B4SOIpNode, B4SOItempNode);
                }
                if (here->B4SOIsoiMod != 0)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOITempePtr, B4SOITempeBinding, B4SOItempNode, B4SOIeNode);
                }
            }
            if (here->B4SOIbodyMod == 2)
            {
            }
            else if (here->B4SOIbodyMod == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIBpPtr, B4SOIBpBinding, B4SOIbNode, B4SOIpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIPbPtr, B4SOIPbBinding, B4SOIpNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIPpPtr, B4SOIPpBinding, B4SOIpNode, B4SOIpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIPgPtr, B4SOIPgBinding, B4SOIpNode, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGpPtr, B4SOIGpBinding, B4SOIgNode, B4SOIpNode);
            }
            if (here->B4SOIrgateMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGEgePtr, B4SOIGEgeBinding, B4SOIgNodeExt, B4SOIgNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGEgPtr, B4SOIGEgBinding, B4SOIgNodeExt, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGgePtr, B4SOIGgeBinding, B4SOIgNode, B4SOIgNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGEdpPtr, B4SOIGEdpBinding, B4SOIgNodeExt, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGEspPtr, B4SOIGEspBinding, B4SOIgNodeExt, B4SOIsNodePrime);
                if (here->B4SOIsoiMod != 2)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGEbPtr, B4SOIGEbBinding, B4SOIgNodeExt, B4SOIbNode);
                }
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGMdpPtr, B4SOIGMdpBinding, B4SOIgNodeMid, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGMgPtr, B4SOIGMgBinding, B4SOIgNodeMid, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGMgmPtr, B4SOIGMgmBinding, B4SOIgNodeMid, B4SOIgNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGMgePtr, B4SOIGMgeBinding, B4SOIgNodeMid, B4SOIgNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGMspPtr, B4SOIGMspBinding, B4SOIgNodeMid, B4SOIsNodePrime);
                if (here->B4SOIsoiMod != 2)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGMbPtr, B4SOIGMbBinding, B4SOIgNodeMid, B4SOIbNode);
                }
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGMePtr, B4SOIGMeBinding, B4SOIgNodeMid, B4SOIeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDPgmPtr, B4SOIDPgmBinding, B4SOIdNodePrime, B4SOIgNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGgmPtr, B4SOIGgmBinding, B4SOIgNode, B4SOIgNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGEgmPtr, B4SOIGEgmBinding, B4SOIgNodeExt, B4SOIgNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISPgmPtr, B4SOISPgmBinding, B4SOIsNodePrime, B4SOIgNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIEgmPtr, B4SOIEgmBinding, B4SOIeNode, B4SOIgNodeMid);
            }
            if (here->B4SOIsoiMod != 2) /* v3.2 */
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIEbPtr, B4SOIEbBinding, B4SOIeNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGbPtr, B4SOIGbBinding, B4SOIgNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDPbPtr, B4SOIDPbBinding, B4SOIdNodePrime, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISPbPtr, B4SOISPbBinding, B4SOIsNodePrime, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIBePtr, B4SOIBeBinding, B4SOIbNode, B4SOIeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIBgPtr, B4SOIBgBinding, B4SOIbNode, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIBdpPtr, B4SOIBdpBinding, B4SOIbNode, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIBspPtr, B4SOIBspBinding, B4SOIbNode, B4SOIsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIBbPtr, B4SOIBbBinding, B4SOIbNode, B4SOIbNode);
            }
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIEgPtr, B4SOIEgBinding, B4SOIeNode, B4SOIgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIEdpPtr, B4SOIEdpBinding, B4SOIeNode, B4SOIdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIEspPtr, B4SOIEspBinding, B4SOIeNode, B4SOIsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGePtr, B4SOIGeBinding, B4SOIgNode, B4SOIeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDPePtr, B4SOIDPeBinding, B4SOIdNodePrime, B4SOIeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISPePtr, B4SOISPeBinding, B4SOIsNodePrime, B4SOIeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIEePtr, B4SOIEeBinding, B4SOIeNode, B4SOIeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGgPtr, B4SOIGgBinding, B4SOIgNode, B4SOIgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGdpPtr, B4SOIGdpBinding, B4SOIgNode, B4SOIdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGspPtr, B4SOIGspBinding, B4SOIgNode, B4SOIsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDPgPtr, B4SOIDPgBinding, B4SOIdNodePrime, B4SOIgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDPdpPtr, B4SOIDPdpBinding, B4SOIdNodePrime, B4SOIdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDPspPtr, B4SOIDPspBinding, B4SOIdNodePrime, B4SOIsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDPdPtr, B4SOIDPdBinding, B4SOIdNodePrime, B4SOIdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISPgPtr, B4SOISPgBinding, B4SOIsNodePrime, B4SOIgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISPdpPtr, B4SOISPdpBinding, B4SOIsNodePrime, B4SOIdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISPspPtr, B4SOISPspBinding, B4SOIsNodePrime, B4SOIsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISPsPtr, B4SOISPsBinding, B4SOIsNodePrime, B4SOIsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDdPtr, B4SOIDdBinding, B4SOIdNode, B4SOIdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDdpPtr, B4SOIDdpBinding, B4SOIdNode, B4SOIdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISsPtr, B4SOISsBinding, B4SOIsNode, B4SOIsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISspPtr, B4SOISspBinding, B4SOIsNode, B4SOIsNodePrime);
            if (here->B4SOIrbodyMod == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDPdbPtr, B4SOIDPdbBinding, B4SOIdNodePrime, B4SOIdbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISPsbPtr, B4SOISPsbBinding, B4SOIsNodePrime, B4SOIsbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDBdpPtr, B4SOIDBdpBinding, B4SOIdbNode, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDBdbPtr, B4SOIDBdbBinding, B4SOIdbNode, B4SOIdbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDBbPtr, B4SOIDBbBinding, B4SOIdbNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISBspPtr, B4SOISBspBinding, B4SOIsbNode, B4SOIsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISBsbPtr, B4SOISBsbBinding, B4SOIsbNode, B4SOIsbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISBbPtr, B4SOISBbBinding, B4SOIsbNode, B4SOIbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIBdbPtr, B4SOIBdbBinding, B4SOIbNode, B4SOIdbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIBsbPtr, B4SOIBsbBinding, B4SOIbNode, B4SOIsbNode);
            }
            if (model->B4SOIrdsMod)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDgPtr, B4SOIDgBinding, B4SOIdNode, B4SOIgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDspPtr, B4SOIDspBinding, B4SOIdNode, B4SOIsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISdpPtr, B4SOISdpBinding, B4SOIsNode, B4SOIdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISgPtr, B4SOISgBinding, B4SOIsNode, B4SOIgNode);
                if (model->B4SOIsoiMod != 2)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIDbPtr, B4SOIDbBinding, B4SOIdNode, B4SOIbNode);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOISbPtr, B4SOISbBinding, B4SOIsNode, B4SOIbNode);
                }
            }
            if (here->B4SOIdebugMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIVbsPtr, B4SOIVbsBinding, B4SOIvbsNode, B4SOIvbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIIdsPtr, B4SOIIdsBinding, B4SOIidsNode, B4SOIidsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIIcPtr, B4SOIIcBinding, B4SOIicNode, B4SOIicNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIIbsPtr, B4SOIIbsBinding, B4SOIibsNode, B4SOIibsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIIbdPtr, B4SOIIbdBinding, B4SOIibdNode, B4SOIibdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIIiiPtr, B4SOIIiiBinding, B4SOIiiiNode, B4SOIiiiNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIIgPtr, B4SOIIgBinding, B4SOIigNode, B4SOIigNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGiggPtr, B4SOIGiggBinding, B4SOIgiggNode, B4SOIgiggNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGigdPtr, B4SOIGigdBinding, B4SOIgigdNode, B4SOIgigdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIGigbPtr, B4SOIGigbBinding, B4SOIgigbNode, B4SOIgigbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIIgidlPtr, B4SOIIgidlBinding, B4SOIigidlNode, B4SOIigidlNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIItunPtr, B4SOIItunBinding, B4SOIitunNode, B4SOIitunNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIIbpPtr, B4SOIIbpBinding, B4SOIibpNode, B4SOIibpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOICbbPtr, B4SOICbbBinding, B4SOIcbbNode, B4SOIcbbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOICbdPtr, B4SOICbdBinding, B4SOIcbdNode, B4SOIcbdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOICbgPtr, B4SOICbgBinding, B4SOIcbgNode, B4SOIcbgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIQbfPtr, B4SOIQbfBinding, B4SOIqbfNode, B4SOIqbfNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIQjsPtr, B4SOIQjsBinding, B4SOIqjsNode, B4SOIqjsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B4SOIQjdPtr, B4SOIQjdBinding, B4SOIqjdNode, B4SOIqjdNode);
            }
        }
    }

    return (OK) ;
}
