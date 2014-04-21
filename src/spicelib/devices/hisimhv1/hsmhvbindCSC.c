/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhvdef.h"
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
HSMHVbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    HSMHVmodel *model = (HSMHVmodel *)inModel ;
    HSMHVinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the HSMHV models */
    for ( ; model != NULL ; model = HSMHVnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = HSMHVinstances(model); here != NULL ; here = HSMHVnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(HSMHVDPbpPtr, HSMHVDPbpBinding, HSMHVdNodePrime, HSMHVbNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVSPbpPtr, HSMHVSPbpBinding, HSMHVsNodePrime, HSMHVbNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVGPbpPtr, HSMHVGPbpBinding, HSMHVgNodePrime, HSMHVbNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVBPdPtr, HSMHVBPdBinding, HSMHVbNodePrime, HSMHVdNode);
            CREATE_KLU_BINDING_TABLE(HSMHVBPsPtr, HSMHVBPsBinding, HSMHVbNodePrime, HSMHVsNode);
            CREATE_KLU_BINDING_TABLE(HSMHVBPdpPtr, HSMHVBPdpBinding, HSMHVbNodePrime, HSMHVdNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVBPspPtr, HSMHVBPspBinding, HSMHVbNodePrime, HSMHVsNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVBPgpPtr, HSMHVBPgpBinding, HSMHVbNodePrime, HSMHVgNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVBPbpPtr, HSMHVBPbpBinding, HSMHVbNodePrime, HSMHVbNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVDdPtr, HSMHVDdBinding, HSMHVdNode, HSMHVdNode);
            CREATE_KLU_BINDING_TABLE(HSMHVGPgpPtr, HSMHVGPgpBinding, HSMHVgNodePrime, HSMHVgNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVSsPtr, HSMHVSsBinding, HSMHVsNode, HSMHVsNode);
            CREATE_KLU_BINDING_TABLE(HSMHVDPdpPtr, HSMHVDPdpBinding, HSMHVdNodePrime, HSMHVdNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVSPspPtr, HSMHVSPspBinding, HSMHVsNodePrime, HSMHVsNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVDdpPtr, HSMHVDdpBinding, HSMHVdNode, HSMHVdNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVGPdpPtr, HSMHVGPdpBinding, HSMHVgNodePrime, HSMHVdNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVGPspPtr, HSMHVGPspBinding, HSMHVgNodePrime, HSMHVsNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVSspPtr, HSMHVSspBinding, HSMHVsNode, HSMHVsNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVDPspPtr, HSMHVDPspBinding, HSMHVdNodePrime, HSMHVsNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVDPdPtr, HSMHVDPdBinding, HSMHVdNodePrime, HSMHVdNode);
            CREATE_KLU_BINDING_TABLE(HSMHVDPgpPtr, HSMHVDPgpBinding, HSMHVdNodePrime, HSMHVgNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVSPgpPtr, HSMHVSPgpBinding, HSMHVsNodePrime, HSMHVgNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVSPsPtr, HSMHVSPsBinding, HSMHVsNodePrime, HSMHVsNode);
            CREATE_KLU_BINDING_TABLE(HSMHVSPdpPtr, HSMHVSPdpBinding, HSMHVsNodePrime, HSMHVdNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVGgPtr, HSMHVGgBinding, HSMHVgNode, HSMHVgNode);
            CREATE_KLU_BINDING_TABLE(HSMHVGgpPtr, HSMHVGgpBinding, HSMHVgNode, HSMHVgNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVGPgPtr, HSMHVGPgBinding, HSMHVgNodePrime, HSMHVgNode);
            CREATE_KLU_BINDING_TABLE(HSMHVDdbPtr, HSMHVDdbBinding, HSMHVdNode, HSMHVdbNode);
            CREATE_KLU_BINDING_TABLE(HSMHVSsbPtr, HSMHVSsbBinding, HSMHVsNode, HSMHVsbNode);
            CREATE_KLU_BINDING_TABLE(HSMHVDBdPtr, HSMHVDBdBinding, HSMHVdbNode, HSMHVdNode);
            CREATE_KLU_BINDING_TABLE(HSMHVDBdbPtr, HSMHVDBdbBinding, HSMHVdbNode, HSMHVdbNode);
            CREATE_KLU_BINDING_TABLE(HSMHVDBbpPtr, HSMHVDBbpBinding, HSMHVdbNode, HSMHVbNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVBPdbPtr, HSMHVBPdbBinding, HSMHVbNodePrime, HSMHVdbNode);
            CREATE_KLU_BINDING_TABLE(HSMHVBPbPtr, HSMHVBPbBinding, HSMHVbNodePrime, HSMHVbNode);
            CREATE_KLU_BINDING_TABLE(HSMHVBPsbPtr, HSMHVBPsbBinding, HSMHVbNodePrime, HSMHVsbNode);
            CREATE_KLU_BINDING_TABLE(HSMHVSBsPtr, HSMHVSBsBinding, HSMHVsbNode, HSMHVsNode);
            CREATE_KLU_BINDING_TABLE(HSMHVSBbpPtr, HSMHVSBbpBinding, HSMHVsbNode, HSMHVbNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVSBsbPtr, HSMHVSBsbBinding, HSMHVsbNode, HSMHVsbNode);
            CREATE_KLU_BINDING_TABLE(HSMHVBbpPtr, HSMHVBbpBinding, HSMHVbNode, HSMHVbNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVBbPtr, HSMHVBbBinding, HSMHVbNode, HSMHVbNode);
            CREATE_KLU_BINDING_TABLE(HSMHVDgpPtr, HSMHVDgpBinding, HSMHVdNode, HSMHVgNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVDsPtr, HSMHVDsBinding, HSMHVdNode, HSMHVsNode);
            CREATE_KLU_BINDING_TABLE(HSMHVDbpPtr, HSMHVDbpBinding, HSMHVdNode, HSMHVbNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVDspPtr, HSMHVDspBinding, HSMHVdNode, HSMHVsNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVDPsPtr, HSMHVDPsBinding, HSMHVdNodePrime, HSMHVsNode);
            CREATE_KLU_BINDING_TABLE(HSMHVSgpPtr, HSMHVSgpBinding, HSMHVsNode, HSMHVgNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVSdPtr, HSMHVSdBinding, HSMHVsNode, HSMHVdNode);
            CREATE_KLU_BINDING_TABLE(HSMHVSbpPtr, HSMHVSbpBinding, HSMHVsNode, HSMHVbNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVSdpPtr, HSMHVSdpBinding, HSMHVsNode, HSMHVdNodePrime);
            CREATE_KLU_BINDING_TABLE(HSMHVSPdPtr, HSMHVSPdBinding, HSMHVsNodePrime, HSMHVdNode);
            CREATE_KLU_BINDING_TABLE(HSMHVGPdPtr, HSMHVGPdBinding, HSMHVgNodePrime, HSMHVdNode);
            CREATE_KLU_BINDING_TABLE(HSMHVGPsPtr, HSMHVGPsBinding, HSMHVgNodePrime, HSMHVsNode);
            if (here->HSMHVsubNode > 0)
            {
                CREATE_KLU_BINDING_TABLE(HSMHVDsubPtr, HSMHVDsubBinding, HSMHVdNode, HSMHVsubNode);
                CREATE_KLU_BINDING_TABLE(HSMHVDPsubPtr, HSMHVDPsubBinding, HSMHVdNodePrime, HSMHVsubNode);
                CREATE_KLU_BINDING_TABLE(HSMHVSsubPtr, HSMHVSsubBinding, HSMHVsNode, HSMHVsubNode);
                CREATE_KLU_BINDING_TABLE(HSMHVSPsubPtr, HSMHVSPsubBinding, HSMHVsNodePrime, HSMHVsubNode);
            }
            if (here->HSMHV_coselfheat > 0)
            {
                CREATE_KLU_BINDING_TABLE(HSMHVTemptempPtr, HSMHVTemptempBinding, HSMHVtempNode, HSMHVtempNode);
                CREATE_KLU_BINDING_TABLE(HSMHVTempdPtr, HSMHVTempdBinding, HSMHVtempNode, HSMHVdNode);
                CREATE_KLU_BINDING_TABLE(HSMHVTempdpPtr, HSMHVTempdpBinding, HSMHVtempNode, HSMHVdNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVTempsPtr, HSMHVTempsBinding, HSMHVtempNode, HSMHVsNode);
                CREATE_KLU_BINDING_TABLE(HSMHVTempspPtr, HSMHVTempspBinding, HSMHVtempNode, HSMHVsNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVDPtempPtr, HSMHVDPtempBinding, HSMHVdNodePrime, HSMHVtempNode);
                CREATE_KLU_BINDING_TABLE(HSMHVSPtempPtr, HSMHVSPtempBinding, HSMHVsNodePrime, HSMHVtempNode);
                CREATE_KLU_BINDING_TABLE(HSMHVTempgpPtr, HSMHVTempgpBinding, HSMHVtempNode, HSMHVgNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVTempbpPtr, HSMHVTempbpBinding, HSMHVtempNode, HSMHVbNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVGPtempPtr, HSMHVGPtempBinding, HSMHVgNodePrime, HSMHVtempNode);
                CREATE_KLU_BINDING_TABLE(HSMHVBPtempPtr, HSMHVBPtempBinding, HSMHVbNodePrime, HSMHVtempNode);
                CREATE_KLU_BINDING_TABLE(HSMHVDBtempPtr, HSMHVDBtempBinding, HSMHVdbNode, HSMHVtempNode);
                CREATE_KLU_BINDING_TABLE(HSMHVSBtempPtr, HSMHVSBtempBinding, HSMHVsbNode, HSMHVtempNode);
                CREATE_KLU_BINDING_TABLE(HSMHVDtempPtr, HSMHVDtempBinding, HSMHVdNode, HSMHVtempNode);
                CREATE_KLU_BINDING_TABLE(HSMHVStempPtr, HSMHVStempBinding, HSMHVsNode, HSMHVtempNode);
            }
            if (model->HSMHV_conqs)
            {
                CREATE_KLU_BINDING_TABLE(HSMHVDPqiPtr, HSMHVDPqiBinding, HSMHVdNodePrime, HSMHVqiNode);
                CREATE_KLU_BINDING_TABLE(HSMHVGPqiPtr, HSMHVGPqiBinding, HSMHVgNodePrime, HSMHVqiNode);
                CREATE_KLU_BINDING_TABLE(HSMHVGPqbPtr, HSMHVGPqbBinding, HSMHVgNodePrime, HSMHVqbNode);
                CREATE_KLU_BINDING_TABLE(HSMHVSPqiPtr, HSMHVSPqiBinding, HSMHVsNodePrime, HSMHVqiNode);
                CREATE_KLU_BINDING_TABLE(HSMHVBPqbPtr, HSMHVBPqbBinding, HSMHVbNodePrime, HSMHVqbNode);
                CREATE_KLU_BINDING_TABLE(HSMHVQIdpPtr, HSMHVQIdpBinding, HSMHVqiNode, HSMHVdNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVQIgpPtr, HSMHVQIgpBinding, HSMHVqiNode, HSMHVgNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVQIspPtr, HSMHVQIspBinding, HSMHVqiNode, HSMHVsNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVQIbpPtr, HSMHVQIbpBinding, HSMHVqiNode, HSMHVbNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVQIqiPtr, HSMHVQIqiBinding, HSMHVqiNode, HSMHVqiNode);
                CREATE_KLU_BINDING_TABLE(HSMHVQBdpPtr, HSMHVQBdpBinding, HSMHVqbNode, HSMHVdNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVQBgpPtr, HSMHVQBgpBinding, HSMHVqbNode, HSMHVgNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVQBspPtr, HSMHVQBspBinding, HSMHVqbNode, HSMHVsNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVQBbpPtr, HSMHVQBbpBinding, HSMHVqbNode, HSMHVbNodePrime);
                CREATE_KLU_BINDING_TABLE(HSMHVQBqbPtr, HSMHVQBqbBinding, HSMHVqbNode, HSMHVqbNode);
                if (here->HSMHV_coselfheat > 0)
                {
                    CREATE_KLU_BINDING_TABLE(HSMHVQItempPtr, HSMHVQItempBinding, HSMHVqiNode, HSMHVtempNode);
                    CREATE_KLU_BINDING_TABLE(HSMHVQBtempPtr, HSMHVQBtempBinding, HSMHVqbNode, HSMHVtempNode);
                }
            }
        }
    }

    return (OK) ;
}

int
HSMHVbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    HSMHVmodel *model = (HSMHVmodel *)inModel ;
    HSMHVinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HSMHV models */
    for ( ; model != NULL ; model = HSMHVnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = HSMHVinstances(model); here != NULL ; here = HSMHVnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDPbpPtr, HSMHVDPbpBinding, HSMHVdNodePrime, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSPbpPtr, HSMHVSPbpBinding, HSMHVsNodePrime, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGPbpPtr, HSMHVGPbpBinding, HSMHVgNodePrime, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPdPtr, HSMHVBPdBinding, HSMHVbNodePrime, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPsPtr, HSMHVBPsBinding, HSMHVbNodePrime, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPdpPtr, HSMHVBPdpBinding, HSMHVbNodePrime, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPspPtr, HSMHVBPspBinding, HSMHVbNodePrime, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPgpPtr, HSMHVBPgpBinding, HSMHVbNodePrime, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPbpPtr, HSMHVBPbpBinding, HSMHVbNodePrime, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDdPtr, HSMHVDdBinding, HSMHVdNode, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGPgpPtr, HSMHVGPgpBinding, HSMHVgNodePrime, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSsPtr, HSMHVSsBinding, HSMHVsNode, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDPdpPtr, HSMHVDPdpBinding, HSMHVdNodePrime, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSPspPtr, HSMHVSPspBinding, HSMHVsNodePrime, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDdpPtr, HSMHVDdpBinding, HSMHVdNode, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGPdpPtr, HSMHVGPdpBinding, HSMHVgNodePrime, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGPspPtr, HSMHVGPspBinding, HSMHVgNodePrime, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSspPtr, HSMHVSspBinding, HSMHVsNode, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDPspPtr, HSMHVDPspBinding, HSMHVdNodePrime, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDPdPtr, HSMHVDPdBinding, HSMHVdNodePrime, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDPgpPtr, HSMHVDPgpBinding, HSMHVdNodePrime, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSPgpPtr, HSMHVSPgpBinding, HSMHVsNodePrime, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSPsPtr, HSMHVSPsBinding, HSMHVsNodePrime, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSPdpPtr, HSMHVSPdpBinding, HSMHVsNodePrime, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGgPtr, HSMHVGgBinding, HSMHVgNode, HSMHVgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGgpPtr, HSMHVGgpBinding, HSMHVgNode, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGPgPtr, HSMHVGPgBinding, HSMHVgNodePrime, HSMHVgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDdbPtr, HSMHVDdbBinding, HSMHVdNode, HSMHVdbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSsbPtr, HSMHVSsbBinding, HSMHVsNode, HSMHVsbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDBdPtr, HSMHVDBdBinding, HSMHVdbNode, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDBdbPtr, HSMHVDBdbBinding, HSMHVdbNode, HSMHVdbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDBbpPtr, HSMHVDBbpBinding, HSMHVdbNode, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPdbPtr, HSMHVBPdbBinding, HSMHVbNodePrime, HSMHVdbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPbPtr, HSMHVBPbBinding, HSMHVbNodePrime, HSMHVbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPsbPtr, HSMHVBPsbBinding, HSMHVbNodePrime, HSMHVsbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSBsPtr, HSMHVSBsBinding, HSMHVsbNode, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSBbpPtr, HSMHVSBbpBinding, HSMHVsbNode, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSBsbPtr, HSMHVSBsbBinding, HSMHVsbNode, HSMHVsbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBbpPtr, HSMHVBbpBinding, HSMHVbNode, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBbPtr, HSMHVBbBinding, HSMHVbNode, HSMHVbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDgpPtr, HSMHVDgpBinding, HSMHVdNode, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDsPtr, HSMHVDsBinding, HSMHVdNode, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDbpPtr, HSMHVDbpBinding, HSMHVdNode, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDspPtr, HSMHVDspBinding, HSMHVdNode, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDPsPtr, HSMHVDPsBinding, HSMHVdNodePrime, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSgpPtr, HSMHVSgpBinding, HSMHVsNode, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSdPtr, HSMHVSdBinding, HSMHVsNode, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSbpPtr, HSMHVSbpBinding, HSMHVsNode, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSdpPtr, HSMHVSdpBinding, HSMHVsNode, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSPdPtr, HSMHVSPdBinding, HSMHVsNodePrime, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGPdPtr, HSMHVGPdBinding, HSMHVgNodePrime, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGPsPtr, HSMHVGPsBinding, HSMHVgNodePrime, HSMHVsNode);
            if (here->HSMHVsubNode > 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDsubPtr, HSMHVDsubBinding, HSMHVdNode, HSMHVsubNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDPsubPtr, HSMHVDPsubBinding, HSMHVdNodePrime, HSMHVsubNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSsubPtr, HSMHVSsubBinding, HSMHVsNode, HSMHVsubNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSPsubPtr, HSMHVSPsubBinding, HSMHVsNodePrime, HSMHVsubNode);
            }
            if (here->HSMHV_coselfheat > 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVTemptempPtr, HSMHVTemptempBinding, HSMHVtempNode, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVTempdPtr, HSMHVTempdBinding, HSMHVtempNode, HSMHVdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVTempdpPtr, HSMHVTempdpBinding, HSMHVtempNode, HSMHVdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVTempsPtr, HSMHVTempsBinding, HSMHVtempNode, HSMHVsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVTempspPtr, HSMHVTempspBinding, HSMHVtempNode, HSMHVsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDPtempPtr, HSMHVDPtempBinding, HSMHVdNodePrime, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSPtempPtr, HSMHVSPtempBinding, HSMHVsNodePrime, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVTempgpPtr, HSMHVTempgpBinding, HSMHVtempNode, HSMHVgNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVTempbpPtr, HSMHVTempbpBinding, HSMHVtempNode, HSMHVbNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGPtempPtr, HSMHVGPtempBinding, HSMHVgNodePrime, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPtempPtr, HSMHVBPtempBinding, HSMHVbNodePrime, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDBtempPtr, HSMHVDBtempBinding, HSMHVdbNode, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSBtempPtr, HSMHVSBtempBinding, HSMHVsbNode, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDtempPtr, HSMHVDtempBinding, HSMHVdNode, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVStempPtr, HSMHVStempBinding, HSMHVsNode, HSMHVtempNode);
            }
            if (model->HSMHV_conqs)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVDPqiPtr, HSMHVDPqiBinding, HSMHVdNodePrime, HSMHVqiNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGPqiPtr, HSMHVGPqiBinding, HSMHVgNodePrime, HSMHVqiNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVGPqbPtr, HSMHVGPqbBinding, HSMHVgNodePrime, HSMHVqbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVSPqiPtr, HSMHVSPqiBinding, HSMHVsNodePrime, HSMHVqiNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVBPqbPtr, HSMHVBPqbBinding, HSMHVbNodePrime, HSMHVqbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQIdpPtr, HSMHVQIdpBinding, HSMHVqiNode, HSMHVdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQIgpPtr, HSMHVQIgpBinding, HSMHVqiNode, HSMHVgNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQIspPtr, HSMHVQIspBinding, HSMHVqiNode, HSMHVsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQIbpPtr, HSMHVQIbpBinding, HSMHVqiNode, HSMHVbNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQIqiPtr, HSMHVQIqiBinding, HSMHVqiNode, HSMHVqiNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQBdpPtr, HSMHVQBdpBinding, HSMHVqbNode, HSMHVdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQBgpPtr, HSMHVQBgpBinding, HSMHVqbNode, HSMHVgNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQBspPtr, HSMHVQBspBinding, HSMHVqbNode, HSMHVsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQBbpPtr, HSMHVQBbpBinding, HSMHVqbNode, HSMHVbNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQBqbPtr, HSMHVQBqbBinding, HSMHVqbNode, HSMHVqbNode);
                if (here->HSMHV_coselfheat > 0)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQItempPtr, HSMHVQItempBinding, HSMHVqiNode, HSMHVtempNode);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSMHVQBtempPtr, HSMHVQBtempBinding, HSMHVqbNode, HSMHVtempNode);
                }
            }
        }
    }

    return (OK) ;
}

int
HSMHVbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HSMHVmodel *model = (HSMHVmodel *)inModel ;
    HSMHVinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HSMHV models */
    for ( ; model != NULL ; model = HSMHVnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = HSMHVinstances(model); here != NULL ; here = HSMHVnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDPbpPtr, HSMHVDPbpBinding, HSMHVdNodePrime, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSPbpPtr, HSMHVSPbpBinding, HSMHVsNodePrime, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGPbpPtr, HSMHVGPbpBinding, HSMHVgNodePrime, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPdPtr, HSMHVBPdBinding, HSMHVbNodePrime, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPsPtr, HSMHVBPsBinding, HSMHVbNodePrime, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPdpPtr, HSMHVBPdpBinding, HSMHVbNodePrime, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPspPtr, HSMHVBPspBinding, HSMHVbNodePrime, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPgpPtr, HSMHVBPgpBinding, HSMHVbNodePrime, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPbpPtr, HSMHVBPbpBinding, HSMHVbNodePrime, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDdPtr, HSMHVDdBinding, HSMHVdNode, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGPgpPtr, HSMHVGPgpBinding, HSMHVgNodePrime, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSsPtr, HSMHVSsBinding, HSMHVsNode, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDPdpPtr, HSMHVDPdpBinding, HSMHVdNodePrime, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSPspPtr, HSMHVSPspBinding, HSMHVsNodePrime, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDdpPtr, HSMHVDdpBinding, HSMHVdNode, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGPdpPtr, HSMHVGPdpBinding, HSMHVgNodePrime, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGPspPtr, HSMHVGPspBinding, HSMHVgNodePrime, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSspPtr, HSMHVSspBinding, HSMHVsNode, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDPspPtr, HSMHVDPspBinding, HSMHVdNodePrime, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDPdPtr, HSMHVDPdBinding, HSMHVdNodePrime, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDPgpPtr, HSMHVDPgpBinding, HSMHVdNodePrime, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSPgpPtr, HSMHVSPgpBinding, HSMHVsNodePrime, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSPsPtr, HSMHVSPsBinding, HSMHVsNodePrime, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSPdpPtr, HSMHVSPdpBinding, HSMHVsNodePrime, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGgPtr, HSMHVGgBinding, HSMHVgNode, HSMHVgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGgpPtr, HSMHVGgpBinding, HSMHVgNode, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGPgPtr, HSMHVGPgBinding, HSMHVgNodePrime, HSMHVgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDdbPtr, HSMHVDdbBinding, HSMHVdNode, HSMHVdbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSsbPtr, HSMHVSsbBinding, HSMHVsNode, HSMHVsbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDBdPtr, HSMHVDBdBinding, HSMHVdbNode, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDBdbPtr, HSMHVDBdbBinding, HSMHVdbNode, HSMHVdbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDBbpPtr, HSMHVDBbpBinding, HSMHVdbNode, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPdbPtr, HSMHVBPdbBinding, HSMHVbNodePrime, HSMHVdbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPbPtr, HSMHVBPbBinding, HSMHVbNodePrime, HSMHVbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPsbPtr, HSMHVBPsbBinding, HSMHVbNodePrime, HSMHVsbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSBsPtr, HSMHVSBsBinding, HSMHVsbNode, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSBbpPtr, HSMHVSBbpBinding, HSMHVsbNode, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSBsbPtr, HSMHVSBsbBinding, HSMHVsbNode, HSMHVsbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBbpPtr, HSMHVBbpBinding, HSMHVbNode, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBbPtr, HSMHVBbBinding, HSMHVbNode, HSMHVbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDgpPtr, HSMHVDgpBinding, HSMHVdNode, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDsPtr, HSMHVDsBinding, HSMHVdNode, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDbpPtr, HSMHVDbpBinding, HSMHVdNode, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDspPtr, HSMHVDspBinding, HSMHVdNode, HSMHVsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDPsPtr, HSMHVDPsBinding, HSMHVdNodePrime, HSMHVsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSgpPtr, HSMHVSgpBinding, HSMHVsNode, HSMHVgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSdPtr, HSMHVSdBinding, HSMHVsNode, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSbpPtr, HSMHVSbpBinding, HSMHVsNode, HSMHVbNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSdpPtr, HSMHVSdpBinding, HSMHVsNode, HSMHVdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSPdPtr, HSMHVSPdBinding, HSMHVsNodePrime, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGPdPtr, HSMHVGPdBinding, HSMHVgNodePrime, HSMHVdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGPsPtr, HSMHVGPsBinding, HSMHVgNodePrime, HSMHVsNode);
            if (here->HSMHVsubNode > 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDsubPtr, HSMHVDsubBinding, HSMHVdNode, HSMHVsubNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDPsubPtr, HSMHVDPsubBinding, HSMHVdNodePrime, HSMHVsubNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSsubPtr, HSMHVSsubBinding, HSMHVsNode, HSMHVsubNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSPsubPtr, HSMHVSPsubBinding, HSMHVsNodePrime, HSMHVsubNode);
            }
            if (here->HSMHV_coselfheat > 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVTemptempPtr, HSMHVTemptempBinding, HSMHVtempNode, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVTempdPtr, HSMHVTempdBinding, HSMHVtempNode, HSMHVdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVTempdpPtr, HSMHVTempdpBinding, HSMHVtempNode, HSMHVdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVTempsPtr, HSMHVTempsBinding, HSMHVtempNode, HSMHVsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVTempspPtr, HSMHVTempspBinding, HSMHVtempNode, HSMHVsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDPtempPtr, HSMHVDPtempBinding, HSMHVdNodePrime, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSPtempPtr, HSMHVSPtempBinding, HSMHVsNodePrime, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVTempgpPtr, HSMHVTempgpBinding, HSMHVtempNode, HSMHVgNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVTempbpPtr, HSMHVTempbpBinding, HSMHVtempNode, HSMHVbNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGPtempPtr, HSMHVGPtempBinding, HSMHVgNodePrime, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPtempPtr, HSMHVBPtempBinding, HSMHVbNodePrime, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDBtempPtr, HSMHVDBtempBinding, HSMHVdbNode, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSBtempPtr, HSMHVSBtempBinding, HSMHVsbNode, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDtempPtr, HSMHVDtempBinding, HSMHVdNode, HSMHVtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVStempPtr, HSMHVStempBinding, HSMHVsNode, HSMHVtempNode);
            }
            if (model->HSMHV_conqs)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVDPqiPtr, HSMHVDPqiBinding, HSMHVdNodePrime, HSMHVqiNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGPqiPtr, HSMHVGPqiBinding, HSMHVgNodePrime, HSMHVqiNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVGPqbPtr, HSMHVGPqbBinding, HSMHVgNodePrime, HSMHVqbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVSPqiPtr, HSMHVSPqiBinding, HSMHVsNodePrime, HSMHVqiNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVBPqbPtr, HSMHVBPqbBinding, HSMHVbNodePrime, HSMHVqbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQIdpPtr, HSMHVQIdpBinding, HSMHVqiNode, HSMHVdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQIgpPtr, HSMHVQIgpBinding, HSMHVqiNode, HSMHVgNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQIspPtr, HSMHVQIspBinding, HSMHVqiNode, HSMHVsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQIbpPtr, HSMHVQIbpBinding, HSMHVqiNode, HSMHVbNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQIqiPtr, HSMHVQIqiBinding, HSMHVqiNode, HSMHVqiNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQBdpPtr, HSMHVQBdpBinding, HSMHVqbNode, HSMHVdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQBgpPtr, HSMHVQBgpBinding, HSMHVqbNode, HSMHVgNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQBspPtr, HSMHVQBspBinding, HSMHVqbNode, HSMHVsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQBbpPtr, HSMHVQBbpBinding, HSMHVqbNode, HSMHVbNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQBqbPtr, HSMHVQBqbBinding, HSMHVqbNode, HSMHVqbNode);
                if (here->HSMHV_coselfheat > 0)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQItempPtr, HSMHVQItempBinding, HSMHVqiNode, HSMHVtempNode);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(HSMHVQBtempPtr, HSMHVQBtempBinding, HSMHVqbNode, HSMHVtempNode);
                }
            }
        }
    }

    return (OK) ;
}
