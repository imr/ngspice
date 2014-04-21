/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
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
B3SOIPDbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIPDmodel *model = (B3SOIPDmodel *)inModel ;
    B3SOIPDinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the B3SOIPD models */
    for ( ; model != NULL ; model = B3SOIPDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B3SOIPDinstances(model); here != NULL ; here = B3SOIPDnextInstance(here))
        {
            if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0 != 0.0))
            {
                CREATE_KLU_BINDING_TABLE(B3SOIPDTemptempPtr, B3SOIPDTemptempBinding, B3SOIPDtempNode, B3SOIPDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDTempdpPtr, B3SOIPDTempdpBinding, B3SOIPDtempNode, B3SOIPDdNodePrime);
                CREATE_KLU_BINDING_TABLE(B3SOIPDTempspPtr, B3SOIPDTempspBinding, B3SOIPDtempNode, B3SOIPDsNodePrime);
                CREATE_KLU_BINDING_TABLE(B3SOIPDTempgPtr, B3SOIPDTempgBinding, B3SOIPDtempNode, B3SOIPDgNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDTempbPtr, B3SOIPDTempbBinding, B3SOIPDtempNode, B3SOIPDbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDGtempPtr, B3SOIPDGtempBinding, B3SOIPDgNode, B3SOIPDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDDPtempPtr, B3SOIPDDPtempBinding, B3SOIPDdNodePrime, B3SOIPDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDSPtempPtr, B3SOIPDSPtempBinding, B3SOIPDsNodePrime, B3SOIPDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDEtempPtr, B3SOIPDEtempBinding, B3SOIPDeNode, B3SOIPDtempNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDBtempPtr, B3SOIPDBtempBinding, B3SOIPDbNode, B3SOIPDtempNode);
                if (here->B3SOIPDbodyMod == 1)
                {
                    CREATE_KLU_BINDING_TABLE(B3SOIPDPtempPtr, B3SOIPDPtempBinding, B3SOIPDpNode, B3SOIPDtempNode);
                }
            }
            if (here->B3SOIPDbodyMod == 2)
            {
            }
            else if (here->B3SOIPDbodyMod == 1)
            {
                CREATE_KLU_BINDING_TABLE(B3SOIPDBpPtr, B3SOIPDBpBinding, B3SOIPDbNode, B3SOIPDpNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDPbPtr, B3SOIPDPbBinding, B3SOIPDpNode, B3SOIPDbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDPpPtr, B3SOIPDPpBinding, B3SOIPDpNode, B3SOIPDpNode);
            }
            CREATE_KLU_BINDING_TABLE(B3SOIPDEbPtr, B3SOIPDEbBinding, B3SOIPDeNode, B3SOIPDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDGbPtr, B3SOIPDGbBinding, B3SOIPDgNode, B3SOIPDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDDPbPtr, B3SOIPDDPbBinding, B3SOIPDdNodePrime, B3SOIPDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDSPbPtr, B3SOIPDSPbBinding, B3SOIPDsNodePrime, B3SOIPDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDBePtr, B3SOIPDBeBinding, B3SOIPDbNode, B3SOIPDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDBgPtr, B3SOIPDBgBinding, B3SOIPDbNode, B3SOIPDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDBdpPtr, B3SOIPDBdpBinding, B3SOIPDbNode, B3SOIPDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDBspPtr, B3SOIPDBspBinding, B3SOIPDbNode, B3SOIPDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDBbPtr, B3SOIPDBbBinding, B3SOIPDbNode, B3SOIPDbNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDEgPtr, B3SOIPDEgBinding, B3SOIPDeNode, B3SOIPDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDEdpPtr, B3SOIPDEdpBinding, B3SOIPDeNode, B3SOIPDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDEspPtr, B3SOIPDEspBinding, B3SOIPDeNode, B3SOIPDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDGePtr, B3SOIPDGeBinding, B3SOIPDgNode, B3SOIPDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDDPePtr, B3SOIPDDPeBinding, B3SOIPDdNodePrime, B3SOIPDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDSPePtr, B3SOIPDSPeBinding, B3SOIPDsNodePrime, B3SOIPDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDEePtr, B3SOIPDEeBinding, B3SOIPDeNode, B3SOIPDeNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDGgPtr, B3SOIPDGgBinding, B3SOIPDgNode, B3SOIPDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDGdpPtr, B3SOIPDGdpBinding, B3SOIPDgNode, B3SOIPDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDGspPtr, B3SOIPDGspBinding, B3SOIPDgNode, B3SOIPDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDDPgPtr, B3SOIPDDPgBinding, B3SOIPDdNodePrime, B3SOIPDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDDPdpPtr, B3SOIPDDPdpBinding, B3SOIPDdNodePrime, B3SOIPDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDDPspPtr, B3SOIPDDPspBinding, B3SOIPDdNodePrime, B3SOIPDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDDPdPtr, B3SOIPDDPdBinding, B3SOIPDdNodePrime, B3SOIPDdNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDSPgPtr, B3SOIPDSPgBinding, B3SOIPDsNodePrime, B3SOIPDgNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDSPdpPtr, B3SOIPDSPdpBinding, B3SOIPDsNodePrime, B3SOIPDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDSPspPtr, B3SOIPDSPspBinding, B3SOIPDsNodePrime, B3SOIPDsNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDSPsPtr, B3SOIPDSPsBinding, B3SOIPDsNodePrime, B3SOIPDsNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDDdPtr, B3SOIPDDdBinding, B3SOIPDdNode, B3SOIPDdNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDDdpPtr, B3SOIPDDdpBinding, B3SOIPDdNode, B3SOIPDdNodePrime);
            CREATE_KLU_BINDING_TABLE(B3SOIPDSsPtr, B3SOIPDSsBinding, B3SOIPDsNode, B3SOIPDsNode);
            CREATE_KLU_BINDING_TABLE(B3SOIPDSspPtr, B3SOIPDSspBinding, B3SOIPDsNode, B3SOIPDsNodePrime);
            if (here->B3SOIPDdebugMod != 0)
            {
                CREATE_KLU_BINDING_TABLE(B3SOIPDVbsPtr, B3SOIPDVbsBinding, B3SOIPDvbsNode, B3SOIPDvbsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDIdsPtr, B3SOIPDIdsBinding, B3SOIPDidsNode, B3SOIPDidsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDIcPtr, B3SOIPDIcBinding, B3SOIPDicNode, B3SOIPDicNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDIbsPtr, B3SOIPDIbsBinding, B3SOIPDibsNode, B3SOIPDibsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDIbdPtr, B3SOIPDIbdBinding, B3SOIPDibdNode, B3SOIPDibdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDIiiPtr, B3SOIPDIiiBinding, B3SOIPDiiiNode, B3SOIPDiiiNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDIgPtr, B3SOIPDIgBinding, B3SOIPDigNode, B3SOIPDigNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDGiggPtr, B3SOIPDGiggBinding, B3SOIPDgiggNode, B3SOIPDgiggNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDGigdPtr, B3SOIPDGigdBinding, B3SOIPDgigdNode, B3SOIPDgigdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDGigbPtr, B3SOIPDGigbBinding, B3SOIPDgigbNode, B3SOIPDgigbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDIgidlPtr, B3SOIPDIgidlBinding, B3SOIPDigidlNode, B3SOIPDigidlNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDItunPtr, B3SOIPDItunBinding, B3SOIPDitunNode, B3SOIPDitunNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDIbpPtr, B3SOIPDIbpBinding, B3SOIPDibpNode, B3SOIPDibpNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDCbbPtr, B3SOIPDCbbBinding, B3SOIPDcbbNode, B3SOIPDcbbNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDCbdPtr, B3SOIPDCbdBinding, B3SOIPDcbdNode, B3SOIPDcbdNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDCbgPtr, B3SOIPDCbgBinding, B3SOIPDcbgNode, B3SOIPDcbgNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDQbfPtr, B3SOIPDQbfBinding, B3SOIPDqbfNode, B3SOIPDqbfNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDQjsPtr, B3SOIPDQjsBinding, B3SOIPDqjsNode, B3SOIPDqjsNode);
                CREATE_KLU_BINDING_TABLE(B3SOIPDQjdPtr, B3SOIPDQjdBinding, B3SOIPDqjdNode, B3SOIPDqjdNode);
            }
        }
    }

    return (OK) ;
}

int
B3SOIPDbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIPDmodel *model = (B3SOIPDmodel *)inModel ;
    B3SOIPDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B3SOIPD models */
    for ( ; model != NULL ; model = B3SOIPDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B3SOIPDinstances(model); here != NULL ; here = B3SOIPDnextInstance(here))
        {
            if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0 != 0.0))
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDTemptempPtr, B3SOIPDTemptempBinding, B3SOIPDtempNode, B3SOIPDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDTempdpPtr, B3SOIPDTempdpBinding, B3SOIPDtempNode, B3SOIPDdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDTempspPtr, B3SOIPDTempspBinding, B3SOIPDtempNode, B3SOIPDsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDTempgPtr, B3SOIPDTempgBinding, B3SOIPDtempNode, B3SOIPDgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDTempbPtr, B3SOIPDTempbBinding, B3SOIPDtempNode, B3SOIPDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDGtempPtr, B3SOIPDGtempBinding, B3SOIPDgNode, B3SOIPDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDDPtempPtr, B3SOIPDDPtempBinding, B3SOIPDdNodePrime, B3SOIPDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDSPtempPtr, B3SOIPDSPtempBinding, B3SOIPDsNodePrime, B3SOIPDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDEtempPtr, B3SOIPDEtempBinding, B3SOIPDeNode, B3SOIPDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDBtempPtr, B3SOIPDBtempBinding, B3SOIPDbNode, B3SOIPDtempNode);
                if (here->B3SOIPDbodyMod == 1)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDPtempPtr, B3SOIPDPtempBinding, B3SOIPDpNode, B3SOIPDtempNode);
                }
            }
            if (here->B3SOIPDbodyMod == 2)
            {
            }
            else if (here->B3SOIPDbodyMod == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDBpPtr, B3SOIPDBpBinding, B3SOIPDbNode, B3SOIPDpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDPbPtr, B3SOIPDPbBinding, B3SOIPDpNode, B3SOIPDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDPpPtr, B3SOIPDPpBinding, B3SOIPDpNode, B3SOIPDpNode);
            }
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDEbPtr, B3SOIPDEbBinding, B3SOIPDeNode, B3SOIPDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDGbPtr, B3SOIPDGbBinding, B3SOIPDgNode, B3SOIPDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDDPbPtr, B3SOIPDDPbBinding, B3SOIPDdNodePrime, B3SOIPDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDSPbPtr, B3SOIPDSPbBinding, B3SOIPDsNodePrime, B3SOIPDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDBePtr, B3SOIPDBeBinding, B3SOIPDbNode, B3SOIPDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDBgPtr, B3SOIPDBgBinding, B3SOIPDbNode, B3SOIPDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDBdpPtr, B3SOIPDBdpBinding, B3SOIPDbNode, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDBspPtr, B3SOIPDBspBinding, B3SOIPDbNode, B3SOIPDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDBbPtr, B3SOIPDBbBinding, B3SOIPDbNode, B3SOIPDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDEgPtr, B3SOIPDEgBinding, B3SOIPDeNode, B3SOIPDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDEdpPtr, B3SOIPDEdpBinding, B3SOIPDeNode, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDEspPtr, B3SOIPDEspBinding, B3SOIPDeNode, B3SOIPDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDGePtr, B3SOIPDGeBinding, B3SOIPDgNode, B3SOIPDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDDPePtr, B3SOIPDDPeBinding, B3SOIPDdNodePrime, B3SOIPDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDSPePtr, B3SOIPDSPeBinding, B3SOIPDsNodePrime, B3SOIPDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDEePtr, B3SOIPDEeBinding, B3SOIPDeNode, B3SOIPDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDGgPtr, B3SOIPDGgBinding, B3SOIPDgNode, B3SOIPDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDGdpPtr, B3SOIPDGdpBinding, B3SOIPDgNode, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDGspPtr, B3SOIPDGspBinding, B3SOIPDgNode, B3SOIPDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDDPgPtr, B3SOIPDDPgBinding, B3SOIPDdNodePrime, B3SOIPDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDDPdpPtr, B3SOIPDDPdpBinding, B3SOIPDdNodePrime, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDDPspPtr, B3SOIPDDPspBinding, B3SOIPDdNodePrime, B3SOIPDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDDPdPtr, B3SOIPDDPdBinding, B3SOIPDdNodePrime, B3SOIPDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDSPgPtr, B3SOIPDSPgBinding, B3SOIPDsNodePrime, B3SOIPDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDSPdpPtr, B3SOIPDSPdpBinding, B3SOIPDsNodePrime, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDSPspPtr, B3SOIPDSPspBinding, B3SOIPDsNodePrime, B3SOIPDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDSPsPtr, B3SOIPDSPsBinding, B3SOIPDsNodePrime, B3SOIPDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDDdPtr, B3SOIPDDdBinding, B3SOIPDdNode, B3SOIPDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDDdpPtr, B3SOIPDDdpBinding, B3SOIPDdNode, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDSsPtr, B3SOIPDSsBinding, B3SOIPDsNode, B3SOIPDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDSspPtr, B3SOIPDSspBinding, B3SOIPDsNode, B3SOIPDsNodePrime);
            if (here->B3SOIPDdebugMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDVbsPtr, B3SOIPDVbsBinding, B3SOIPDvbsNode, B3SOIPDvbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDIdsPtr, B3SOIPDIdsBinding, B3SOIPDidsNode, B3SOIPDidsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDIcPtr, B3SOIPDIcBinding, B3SOIPDicNode, B3SOIPDicNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDIbsPtr, B3SOIPDIbsBinding, B3SOIPDibsNode, B3SOIPDibsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDIbdPtr, B3SOIPDIbdBinding, B3SOIPDibdNode, B3SOIPDibdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDIiiPtr, B3SOIPDIiiBinding, B3SOIPDiiiNode, B3SOIPDiiiNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDIgPtr, B3SOIPDIgBinding, B3SOIPDigNode, B3SOIPDigNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDGiggPtr, B3SOIPDGiggBinding, B3SOIPDgiggNode, B3SOIPDgiggNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDGigdPtr, B3SOIPDGigdBinding, B3SOIPDgigdNode, B3SOIPDgigdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDGigbPtr, B3SOIPDGigbBinding, B3SOIPDgigbNode, B3SOIPDgigbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDIgidlPtr, B3SOIPDIgidlBinding, B3SOIPDigidlNode, B3SOIPDigidlNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDItunPtr, B3SOIPDItunBinding, B3SOIPDitunNode, B3SOIPDitunNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDIbpPtr, B3SOIPDIbpBinding, B3SOIPDibpNode, B3SOIPDibpNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDCbbPtr, B3SOIPDCbbBinding, B3SOIPDcbbNode, B3SOIPDcbbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDCbdPtr, B3SOIPDCbdBinding, B3SOIPDcbdNode, B3SOIPDcbdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDCbgPtr, B3SOIPDCbgBinding, B3SOIPDcbgNode, B3SOIPDcbgNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDQbfPtr, B3SOIPDQbfBinding, B3SOIPDqbfNode, B3SOIPDqbfNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDQjsPtr, B3SOIPDQjsBinding, B3SOIPDqjsNode, B3SOIPDqjsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B3SOIPDQjdPtr, B3SOIPDQjdBinding, B3SOIPDqjdNode, B3SOIPDqjdNode);
            }
        }
    }

    return (OK) ;
}

int
B3SOIPDbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIPDmodel *model = (B3SOIPDmodel *)inModel ;
    B3SOIPDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B3SOIPD models */
    for ( ; model != NULL ; model = B3SOIPDnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B3SOIPDinstances(model); here != NULL ; here = B3SOIPDnextInstance(here))
        {
            if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0 != 0.0))
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDTemptempPtr, B3SOIPDTemptempBinding, B3SOIPDtempNode, B3SOIPDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDTempdpPtr, B3SOIPDTempdpBinding, B3SOIPDtempNode, B3SOIPDdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDTempspPtr, B3SOIPDTempspBinding, B3SOIPDtempNode, B3SOIPDsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDTempgPtr, B3SOIPDTempgBinding, B3SOIPDtempNode, B3SOIPDgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDTempbPtr, B3SOIPDTempbBinding, B3SOIPDtempNode, B3SOIPDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDGtempPtr, B3SOIPDGtempBinding, B3SOIPDgNode, B3SOIPDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDDPtempPtr, B3SOIPDDPtempBinding, B3SOIPDdNodePrime, B3SOIPDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDSPtempPtr, B3SOIPDSPtempBinding, B3SOIPDsNodePrime, B3SOIPDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDEtempPtr, B3SOIPDEtempBinding, B3SOIPDeNode, B3SOIPDtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDBtempPtr, B3SOIPDBtempBinding, B3SOIPDbNode, B3SOIPDtempNode);
                if (here->B3SOIPDbodyMod == 1)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDPtempPtr, B3SOIPDPtempBinding, B3SOIPDpNode, B3SOIPDtempNode);
                }
            }
            if (here->B3SOIPDbodyMod == 2)
            {
            }
            else if (here->B3SOIPDbodyMod == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDBpPtr, B3SOIPDBpBinding, B3SOIPDbNode, B3SOIPDpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDPbPtr, B3SOIPDPbBinding, B3SOIPDpNode, B3SOIPDbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDPpPtr, B3SOIPDPpBinding, B3SOIPDpNode, B3SOIPDpNode);
            }
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDEbPtr, B3SOIPDEbBinding, B3SOIPDeNode, B3SOIPDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDGbPtr, B3SOIPDGbBinding, B3SOIPDgNode, B3SOIPDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDDPbPtr, B3SOIPDDPbBinding, B3SOIPDdNodePrime, B3SOIPDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDSPbPtr, B3SOIPDSPbBinding, B3SOIPDsNodePrime, B3SOIPDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDBePtr, B3SOIPDBeBinding, B3SOIPDbNode, B3SOIPDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDBgPtr, B3SOIPDBgBinding, B3SOIPDbNode, B3SOIPDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDBdpPtr, B3SOIPDBdpBinding, B3SOIPDbNode, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDBspPtr, B3SOIPDBspBinding, B3SOIPDbNode, B3SOIPDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDBbPtr, B3SOIPDBbBinding, B3SOIPDbNode, B3SOIPDbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDEgPtr, B3SOIPDEgBinding, B3SOIPDeNode, B3SOIPDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDEdpPtr, B3SOIPDEdpBinding, B3SOIPDeNode, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDEspPtr, B3SOIPDEspBinding, B3SOIPDeNode, B3SOIPDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDGePtr, B3SOIPDGeBinding, B3SOIPDgNode, B3SOIPDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDDPePtr, B3SOIPDDPeBinding, B3SOIPDdNodePrime, B3SOIPDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDSPePtr, B3SOIPDSPeBinding, B3SOIPDsNodePrime, B3SOIPDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDEePtr, B3SOIPDEeBinding, B3SOIPDeNode, B3SOIPDeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDGgPtr, B3SOIPDGgBinding, B3SOIPDgNode, B3SOIPDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDGdpPtr, B3SOIPDGdpBinding, B3SOIPDgNode, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDGspPtr, B3SOIPDGspBinding, B3SOIPDgNode, B3SOIPDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDDPgPtr, B3SOIPDDPgBinding, B3SOIPDdNodePrime, B3SOIPDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDDPdpPtr, B3SOIPDDPdpBinding, B3SOIPDdNodePrime, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDDPspPtr, B3SOIPDDPspBinding, B3SOIPDdNodePrime, B3SOIPDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDDPdPtr, B3SOIPDDPdBinding, B3SOIPDdNodePrime, B3SOIPDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDSPgPtr, B3SOIPDSPgBinding, B3SOIPDsNodePrime, B3SOIPDgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDSPdpPtr, B3SOIPDSPdpBinding, B3SOIPDsNodePrime, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDSPspPtr, B3SOIPDSPspBinding, B3SOIPDsNodePrime, B3SOIPDsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDSPsPtr, B3SOIPDSPsBinding, B3SOIPDsNodePrime, B3SOIPDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDDdPtr, B3SOIPDDdBinding, B3SOIPDdNode, B3SOIPDdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDDdpPtr, B3SOIPDDdpBinding, B3SOIPDdNode, B3SOIPDdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDSsPtr, B3SOIPDSsBinding, B3SOIPDsNode, B3SOIPDsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDSspPtr, B3SOIPDSspBinding, B3SOIPDsNode, B3SOIPDsNodePrime);
            if (here->B3SOIPDdebugMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDVbsPtr, B3SOIPDVbsBinding, B3SOIPDvbsNode, B3SOIPDvbsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDIdsPtr, B3SOIPDIdsBinding, B3SOIPDidsNode, B3SOIPDidsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDIcPtr, B3SOIPDIcBinding, B3SOIPDicNode, B3SOIPDicNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDIbsPtr, B3SOIPDIbsBinding, B3SOIPDibsNode, B3SOIPDibsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDIbdPtr, B3SOIPDIbdBinding, B3SOIPDibdNode, B3SOIPDibdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDIiiPtr, B3SOIPDIiiBinding, B3SOIPDiiiNode, B3SOIPDiiiNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDIgPtr, B3SOIPDIgBinding, B3SOIPDigNode, B3SOIPDigNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDGiggPtr, B3SOIPDGiggBinding, B3SOIPDgiggNode, B3SOIPDgiggNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDGigdPtr, B3SOIPDGigdBinding, B3SOIPDgigdNode, B3SOIPDgigdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDGigbPtr, B3SOIPDGigbBinding, B3SOIPDgigbNode, B3SOIPDgigbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDIgidlPtr, B3SOIPDIgidlBinding, B3SOIPDigidlNode, B3SOIPDigidlNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDItunPtr, B3SOIPDItunBinding, B3SOIPDitunNode, B3SOIPDitunNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDIbpPtr, B3SOIPDIbpBinding, B3SOIPDibpNode, B3SOIPDibpNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDCbbPtr, B3SOIPDCbbBinding, B3SOIPDcbbNode, B3SOIPDcbbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDCbdPtr, B3SOIPDCbdBinding, B3SOIPDcbdNode, B3SOIPDcbdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDCbgPtr, B3SOIPDCbgBinding, B3SOIPDcbgNode, B3SOIPDcbgNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDQbfPtr, B3SOIPDQbfBinding, B3SOIPDqbfNode, B3SOIPDqbfNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDQjsPtr, B3SOIPDQjsBinding, B3SOIPDqjsNode, B3SOIPDqjsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(B3SOIPDQjdPtr, B3SOIPDQjdBinding, B3SOIPDqjdNode, B3SOIPDqjdNode);
            }
        }
    }

    return (OK) ;
}
