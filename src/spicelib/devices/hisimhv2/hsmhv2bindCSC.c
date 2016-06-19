/**********
Author: 2016 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhv2def.h"
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
HSMHV2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    HSMHV2model *model = (HSMHV2model *)inModel ;
    HSMHV2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the HSMHV2 models */
    for ( ; model != NULL ; model = HSMHV2nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = HSMHV2instances(model); here != NULL ; here = HSMHV2nextInstance(here))
        {
                CREATE_KLU_BINDING_TABLE (HSMHV2GPsPtr, HSMHV2GPsBinding, HSMHV2gNodePrime, HSMHV2sNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2GPspPtr, HSMHV2GPspBinding, HSMHV2gNodePrime, HSMHV2sNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SPgpPtr, HSMHV2SPgpBinding, HSMHV2sNodePrime, HSMHV2gNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DBdPtr, HSMHV2DBdBinding, HSMHV2dbNode, HSMHV2dNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SsPtr, HSMHV2SsBinding, HSMHV2sNode, HSMHV2sNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DPsPtr, HSMHV2DPsBinding, HSMHV2dNodePrime, HSMHV2sNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SsbPtr, HSMHV2SsbBinding, HSMHV2sNode, HSMHV2sbNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2GPdPtr, HSMHV2GPdBinding, HSMHV2gNodePrime, HSMHV2dNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPbPtr, HSMHV2BPbBinding, HSMHV2bNodePrime, HSMHV2bNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPspPtr, HSMHV2BPspBinding, HSMHV2bNodePrime, HSMHV2sNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SPspPtr, HSMHV2SPspBinding, HSMHV2sNodePrime, HSMHV2sNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2GgPtr, HSMHV2GgBinding, HSMHV2gNode, HSMHV2gNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DbpPtr, HSMHV2DbpBinding, HSMHV2dNode, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SdPtr, HSMHV2SdBinding, HSMHV2sNode, HSMHV2dNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DdpPtr, HSMHV2DdpBinding, HSMHV2dNode, HSMHV2dNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DPdPtr, HSMHV2DPdBinding, HSMHV2dNodePrime, HSMHV2dNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2GPbpPtr, HSMHV2GPbpBinding, HSMHV2gNodePrime, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2GPdpPtr, HSMHV2GPdpBinding, HSMHV2gNodePrime, HSMHV2dNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2GPgPtr, HSMHV2GPgBinding, HSMHV2gNodePrime, HSMHV2gNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DsPtr, HSMHV2DsBinding, HSMHV2dNode, HSMHV2sNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SgpPtr, HSMHV2SgpBinding, HSMHV2sNode, HSMHV2gNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPbpPtr, HSMHV2BPbpBinding, HSMHV2bNodePrime, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SPbpPtr, HSMHV2SPbpBinding, HSMHV2sNodePrime, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPdpPtr, HSMHV2BPdpBinding, HSMHV2bNodePrime, HSMHV2dNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SPdpPtr, HSMHV2SPdpBinding, HSMHV2sNodePrime, HSMHV2dNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SBbpPtr, HSMHV2SBbpBinding, HSMHV2sbNode, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DdPtr, HSMHV2DdBinding, HSMHV2dNode, HSMHV2dNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SBsPtr, HSMHV2SBsBinding, HSMHV2sbNode, HSMHV2sNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DPgpPtr, HSMHV2DPgpBinding, HSMHV2dNodePrime, HSMHV2gNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DBdbPtr, HSMHV2DBdbBinding, HSMHV2dbNode, HSMHV2dbNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SspPtr, HSMHV2SspBinding, HSMHV2sNode, HSMHV2sNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BbPtr, HSMHV2BbBinding, HSMHV2bNode, HSMHV2bNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SPsPtr, HSMHV2SPsBinding, HSMHV2sNodePrime, HSMHV2sNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2GgpPtr, HSMHV2GgpBinding, HSMHV2gNode, HSMHV2gNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DPspPtr, HSMHV2DPspBinding, HSMHV2dNodePrime, HSMHV2sNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPsbPtr, HSMHV2BPsbBinding, HSMHV2bNodePrime, HSMHV2sbNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DdbPtr, HSMHV2DdbBinding, HSMHV2dNode, HSMHV2dbNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SBsbPtr, HSMHV2SBsbBinding, HSMHV2sbNode, HSMHV2sbNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DgpPtr, HSMHV2DgpBinding, HSMHV2dNode, HSMHV2gNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPsPtr, HSMHV2BPsBinding, HSMHV2bNodePrime, HSMHV2sNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SbpPtr, HSMHV2SbpBinding, HSMHV2sNode, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2GPgpPtr, HSMHV2GPgpBinding, HSMHV2gNodePrime, HSMHV2gNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SPdPtr, HSMHV2SPdBinding, HSMHV2sNodePrime, HSMHV2dNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SdpPtr, HSMHV2SdpBinding, HSMHV2sNode, HSMHV2dNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BbpPtr, HSMHV2BbpBinding, HSMHV2bNode, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPdPtr, HSMHV2BPdBinding, HSMHV2bNodePrime, HSMHV2dNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DPbpPtr, HSMHV2DPbpBinding, HSMHV2dNodePrime, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DspPtr, HSMHV2DspBinding, HSMHV2dNode, HSMHV2sNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DPdpPtr, HSMHV2DPdpBinding, HSMHV2dNodePrime, HSMHV2dNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DBbpPtr, HSMHV2DBbpBinding, HSMHV2dbNode, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPdbPtr, HSMHV2BPdbBinding, HSMHV2bNodePrime, HSMHV2dbNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPgpPtr, HSMHV2BPgpBinding, HSMHV2bNodePrime, HSMHV2gNodePrime) ;

            if (here->HSMHV2subNode > 0)
            { /* 5th substrate node */
                CREATE_KLU_BINDING_TABLE (HSMHV2DPsubPtr, HSMHV2DPsubBinding, HSMHV2dNodePrime, HSMHV2subNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DsubPtr, HSMHV2DsubBinding, HSMHV2dNode, HSMHV2subNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SPsubPtr, HSMHV2SPsubBinding, HSMHV2sNodePrime, HSMHV2subNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SsubPtr, HSMHV2SsubBinding, HSMHV2sNode, HSMHV2subNode) ;
            }

            if (here->HSMHV2tempNode > 0)
            { /* self heating */
                CREATE_KLU_BINDING_TABLE (HSMHV2TemptempPtr, HSMHV2TemptempBinding, HSMHV2tempNode, HSMHV2tempNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2TempbpPtr, HSMHV2TempbpBinding, HSMHV2tempNode, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPtempPtr, HSMHV2BPtempBinding, HSMHV2bNodePrime, HSMHV2tempNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2TempspPtr, HSMHV2TempspBinding, HSMHV2tempNode, HSMHV2sNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SPtempPtr, HSMHV2SPtempBinding, HSMHV2sNodePrime, HSMHV2tempNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DtempPtr, HSMHV2DtempBinding, HSMHV2dNode, HSMHV2tempNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2TempdPtr, HSMHV2TempdBinding, HSMHV2tempNode, HSMHV2dNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2TempdpPtr, HSMHV2TempdpBinding, HSMHV2tempNode, HSMHV2dNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DPtempPtr, HSMHV2DPtempBinding, HSMHV2dNodePrime, HSMHV2tempNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SBtempPtr, HSMHV2SBtempBinding, HSMHV2sbNode, HSMHV2tempNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DBtempPtr, HSMHV2DBtempBinding, HSMHV2dbNode, HSMHV2tempNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2TempgpPtr, HSMHV2TempgpBinding, HSMHV2tempNode, HSMHV2gNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2GPtempPtr, HSMHV2GPtempBinding, HSMHV2gNodePrime, HSMHV2tempNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2StempPtr, HSMHV2StempBinding, HSMHV2sNode, HSMHV2tempNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2TempsPtr, HSMHV2TempsBinding, HSMHV2tempNode, HSMHV2sNode) ;
            }

            if (model->HSMHV2_conqs)
            { /* flat handling of NQS */
                CREATE_KLU_BINDING_TABLE (HSMHV2GPqbPtr, HSMHV2GPqbBinding, HSMHV2gNodePrime, HSMHV2qbNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2QBgpPtr, HSMHV2QBgpBinding, HSMHV2qbNode, HSMHV2gNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2QIbpPtr, HSMHV2QIbpBinding, HSMHV2qiNode, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2QIqiPtr, HSMHV2QIqiBinding, HSMHV2qiNode, HSMHV2qiNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2QBbpPtr, HSMHV2QBbpBinding, HSMHV2qbNode, HSMHV2bNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2SPqiPtr, HSMHV2SPqiBinding, HSMHV2sNodePrime, HSMHV2qiNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2BPqbPtr, HSMHV2BPqbBinding, HSMHV2bNodePrime, HSMHV2qbNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2QIspPtr, HSMHV2QIspBinding, HSMHV2qiNode, HSMHV2sNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2QBqbPtr, HSMHV2QBqbBinding, HSMHV2qbNode, HSMHV2qbNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2QBspPtr, HSMHV2QBspBinding, HSMHV2qbNode, HSMHV2sNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2DPqiPtr, HSMHV2DPqiBinding, HSMHV2dNodePrime, HSMHV2qiNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2QIdpPtr, HSMHV2QIdpBinding, HSMHV2qiNode, HSMHV2dNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2QBdpPtr, HSMHV2QBdpBinding, HSMHV2qbNode, HSMHV2dNodePrime) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2GPqiPtr, HSMHV2GPqiBinding, HSMHV2gNodePrime, HSMHV2qiNode) ;
                CREATE_KLU_BINDING_TABLE (HSMHV2QIgpPtr, HSMHV2QIgpBinding, HSMHV2qiNode, HSMHV2gNodePrime) ;

                if (here->HSMHV2tempNode > 0)
                { /* self heating */
                    CREATE_KLU_BINDING_TABLE (HSMHV2QItempPtr, HSMHV2QItempBinding, HSMHV2qiNode, HSMHV2tempNode) ;
                    CREATE_KLU_BINDING_TABLE (HSMHV2QBtempPtr, HSMHV2QBtempBinding, HSMHV2qbNode, HSMHV2tempNode) ;
                }
            }
        }
    }

    return (OK) ;
}

int
HSMHV2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    HSMHV2model *model = (HSMHV2model *)inModel ;
    HSMHV2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HSMHV2 models */
    for ( ; model != NULL ; model = HSMHV2nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = HSMHV2instances(model); here != NULL ; here = HSMHV2nextInstance(here))
        {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GPsPtr, HSMHV2GPsBinding, HSMHV2gNodePrime, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GPspPtr, HSMHV2GPspBinding, HSMHV2gNodePrime, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SPgpPtr, HSMHV2SPgpBinding, HSMHV2sNodePrime, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DBdPtr, HSMHV2DBdBinding, HSMHV2dbNode, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SsPtr, HSMHV2SsBinding, HSMHV2sNode, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DPsPtr, HSMHV2DPsBinding, HSMHV2dNodePrime, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SsbPtr, HSMHV2SsbBinding, HSMHV2sNode, HSMHV2sbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GPdPtr, HSMHV2GPdBinding, HSMHV2gNodePrime, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPbPtr, HSMHV2BPbBinding, HSMHV2bNodePrime, HSMHV2bNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPspPtr, HSMHV2BPspBinding, HSMHV2bNodePrime, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SPspPtr, HSMHV2SPspBinding, HSMHV2sNodePrime, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GgPtr, HSMHV2GgBinding, HSMHV2gNode, HSMHV2gNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DbpPtr, HSMHV2DbpBinding, HSMHV2dNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SdPtr, HSMHV2SdBinding, HSMHV2sNode, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DdpPtr, HSMHV2DdpBinding, HSMHV2dNode, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DPdPtr, HSMHV2DPdBinding, HSMHV2dNodePrime, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GPbpPtr, HSMHV2GPbpBinding, HSMHV2gNodePrime, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GPdpPtr, HSMHV2GPdpBinding, HSMHV2gNodePrime, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GPgPtr, HSMHV2GPgBinding, HSMHV2gNodePrime, HSMHV2gNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DsPtr, HSMHV2DsBinding, HSMHV2dNode, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SgpPtr, HSMHV2SgpBinding, HSMHV2sNode, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPbpPtr, HSMHV2BPbpBinding, HSMHV2bNodePrime, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SPbpPtr, HSMHV2SPbpBinding, HSMHV2sNodePrime, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPdpPtr, HSMHV2BPdpBinding, HSMHV2bNodePrime, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SPdpPtr, HSMHV2SPdpBinding, HSMHV2sNodePrime, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SBbpPtr, HSMHV2SBbpBinding, HSMHV2sbNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DdPtr, HSMHV2DdBinding, HSMHV2dNode, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SBsPtr, HSMHV2SBsBinding, HSMHV2sbNode, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DPgpPtr, HSMHV2DPgpBinding, HSMHV2dNodePrime, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DBdbPtr, HSMHV2DBdbBinding, HSMHV2dbNode, HSMHV2dbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SspPtr, HSMHV2SspBinding, HSMHV2sNode, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BbPtr, HSMHV2BbBinding, HSMHV2bNode, HSMHV2bNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SPsPtr, HSMHV2SPsBinding, HSMHV2sNodePrime, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GgpPtr, HSMHV2GgpBinding, HSMHV2gNode, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DPspPtr, HSMHV2DPspBinding, HSMHV2dNodePrime, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPsbPtr, HSMHV2BPsbBinding, HSMHV2bNodePrime, HSMHV2sbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DdbPtr, HSMHV2DdbBinding, HSMHV2dNode, HSMHV2dbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SBsbPtr, HSMHV2SBsbBinding, HSMHV2sbNode, HSMHV2sbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DgpPtr, HSMHV2DgpBinding, HSMHV2dNode, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPsPtr, HSMHV2BPsBinding, HSMHV2bNodePrime, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SbpPtr, HSMHV2SbpBinding, HSMHV2sNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GPgpPtr, HSMHV2GPgpBinding, HSMHV2gNodePrime, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SPdPtr, HSMHV2SPdBinding, HSMHV2sNodePrime, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SdpPtr, HSMHV2SdpBinding, HSMHV2sNode, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BbpPtr, HSMHV2BbpBinding, HSMHV2bNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPdPtr, HSMHV2BPdBinding, HSMHV2bNodePrime, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DPbpPtr, HSMHV2DPbpBinding, HSMHV2dNodePrime, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DspPtr, HSMHV2DspBinding, HSMHV2dNode, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DPdpPtr, HSMHV2DPdpBinding, HSMHV2dNodePrime, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DBbpPtr, HSMHV2DBbpBinding, HSMHV2dbNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPdbPtr, HSMHV2BPdbBinding, HSMHV2bNodePrime, HSMHV2dbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPgpPtr, HSMHV2BPgpBinding, HSMHV2bNodePrime, HSMHV2gNodePrime) ;

            if (here->HSMHV2subNode > 0)
            { /* 5th substrate node */
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DPsubPtr, HSMHV2DPsubBinding, HSMHV2dNodePrime, HSMHV2subNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DsubPtr, HSMHV2DsubBinding, HSMHV2dNode, HSMHV2subNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SPsubPtr, HSMHV2SPsubBinding, HSMHV2sNodePrime, HSMHV2subNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SsubPtr, HSMHV2SsubBinding, HSMHV2sNode, HSMHV2subNode) ;
            }

            if (here->HSMHV2tempNode > 0)
            { /* self heating */
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2TemptempPtr, HSMHV2TemptempBinding, HSMHV2tempNode, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2TempbpPtr, HSMHV2TempbpBinding, HSMHV2tempNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPtempPtr, HSMHV2BPtempBinding, HSMHV2bNodePrime, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2TempspPtr, HSMHV2TempspBinding, HSMHV2tempNode, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SPtempPtr, HSMHV2SPtempBinding, HSMHV2sNodePrime, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DtempPtr, HSMHV2DtempBinding, HSMHV2dNode, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2TempdPtr, HSMHV2TempdBinding, HSMHV2tempNode, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2TempdpPtr, HSMHV2TempdpBinding, HSMHV2tempNode, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DPtempPtr, HSMHV2DPtempBinding, HSMHV2dNodePrime, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SBtempPtr, HSMHV2SBtempBinding, HSMHV2sbNode, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DBtempPtr, HSMHV2DBtempBinding, HSMHV2dbNode, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2TempgpPtr, HSMHV2TempgpBinding, HSMHV2tempNode, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GPtempPtr, HSMHV2GPtempBinding, HSMHV2gNodePrime, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2StempPtr, HSMHV2StempBinding, HSMHV2sNode, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2TempsPtr, HSMHV2TempsBinding, HSMHV2tempNode, HSMHV2sNode) ;
            }

            if (model->HSMHV2_conqs)
            { /* flat handling of NQS */
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GPqbPtr, HSMHV2GPqbBinding, HSMHV2gNodePrime, HSMHV2qbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QBgpPtr, HSMHV2QBgpBinding, HSMHV2qbNode, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QIbpPtr, HSMHV2QIbpBinding, HSMHV2qiNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QIqiPtr, HSMHV2QIqiBinding, HSMHV2qiNode, HSMHV2qiNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QBbpPtr, HSMHV2QBbpBinding, HSMHV2qbNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2SPqiPtr, HSMHV2SPqiBinding, HSMHV2sNodePrime, HSMHV2qiNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2BPqbPtr, HSMHV2BPqbBinding, HSMHV2bNodePrime, HSMHV2qbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QIspPtr, HSMHV2QIspBinding, HSMHV2qiNode, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QBqbPtr, HSMHV2QBqbBinding, HSMHV2qbNode, HSMHV2qbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QBspPtr, HSMHV2QBspBinding, HSMHV2qbNode, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2DPqiPtr, HSMHV2DPqiBinding, HSMHV2dNodePrime, HSMHV2qiNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QIdpPtr, HSMHV2QIdpBinding, HSMHV2qiNode, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QBdpPtr, HSMHV2QBdpBinding, HSMHV2qbNode, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2GPqiPtr, HSMHV2GPqiBinding, HSMHV2gNodePrime, HSMHV2qiNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QIgpPtr, HSMHV2QIgpBinding, HSMHV2qiNode, HSMHV2gNodePrime) ;

                if (here->HSMHV2tempNode > 0)
                { /* self heating */
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QItempPtr, HSMHV2QItempBinding, HSMHV2qiNode, HSMHV2tempNode) ;
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX (HSMHV2QBtempPtr, HSMHV2QBtempBinding, HSMHV2qbNode, HSMHV2tempNode) ;
                }
            }
        }
    }

    return (OK) ;
}

int
HSMHV2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HSMHV2model *model = (HSMHV2model *)inModel ;
    HSMHV2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HSMHV2 models */
    for ( ; model != NULL ; model = HSMHV2nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = HSMHV2instances(model); here != NULL ; here = HSMHV2nextInstance(here))
        {
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GPsPtr, HSMHV2GPsBinding, HSMHV2gNodePrime, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GPspPtr, HSMHV2GPspBinding, HSMHV2gNodePrime, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SPgpPtr, HSMHV2SPgpBinding, HSMHV2sNodePrime, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DBdPtr, HSMHV2DBdBinding, HSMHV2dbNode, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SsPtr, HSMHV2SsBinding, HSMHV2sNode, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DPsPtr, HSMHV2DPsBinding, HSMHV2dNodePrime, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SsbPtr, HSMHV2SsbBinding, HSMHV2sNode, HSMHV2sbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GPdPtr, HSMHV2GPdBinding, HSMHV2gNodePrime, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPbPtr, HSMHV2BPbBinding, HSMHV2bNodePrime, HSMHV2bNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPspPtr, HSMHV2BPspBinding, HSMHV2bNodePrime, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SPspPtr, HSMHV2SPspBinding, HSMHV2sNodePrime, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GgPtr, HSMHV2GgBinding, HSMHV2gNode, HSMHV2gNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DbpPtr, HSMHV2DbpBinding, HSMHV2dNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SdPtr, HSMHV2SdBinding, HSMHV2sNode, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DdpPtr, HSMHV2DdpBinding, HSMHV2dNode, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DPdPtr, HSMHV2DPdBinding, HSMHV2dNodePrime, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GPbpPtr, HSMHV2GPbpBinding, HSMHV2gNodePrime, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GPdpPtr, HSMHV2GPdpBinding, HSMHV2gNodePrime, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GPgPtr, HSMHV2GPgBinding, HSMHV2gNodePrime, HSMHV2gNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DsPtr, HSMHV2DsBinding, HSMHV2dNode, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SgpPtr, HSMHV2SgpBinding, HSMHV2sNode, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPbpPtr, HSMHV2BPbpBinding, HSMHV2bNodePrime, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SPbpPtr, HSMHV2SPbpBinding, HSMHV2sNodePrime, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPdpPtr, HSMHV2BPdpBinding, HSMHV2bNodePrime, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SPdpPtr, HSMHV2SPdpBinding, HSMHV2sNodePrime, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SBbpPtr, HSMHV2SBbpBinding, HSMHV2sbNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DdPtr, HSMHV2DdBinding, HSMHV2dNode, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SBsPtr, HSMHV2SBsBinding, HSMHV2sbNode, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DPgpPtr, HSMHV2DPgpBinding, HSMHV2dNodePrime, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DBdbPtr, HSMHV2DBdbBinding, HSMHV2dbNode, HSMHV2dbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SspPtr, HSMHV2SspBinding, HSMHV2sNode, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BbPtr, HSMHV2BbBinding, HSMHV2bNode, HSMHV2bNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SPsPtr, HSMHV2SPsBinding, HSMHV2sNodePrime, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GgpPtr, HSMHV2GgpBinding, HSMHV2gNode, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DPspPtr, HSMHV2DPspBinding, HSMHV2dNodePrime, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPsbPtr, HSMHV2BPsbBinding, HSMHV2bNodePrime, HSMHV2sbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DdbPtr, HSMHV2DdbBinding, HSMHV2dNode, HSMHV2dbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SBsbPtr, HSMHV2SBsbBinding, HSMHV2sbNode, HSMHV2sbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DgpPtr, HSMHV2DgpBinding, HSMHV2dNode, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPsPtr, HSMHV2BPsBinding, HSMHV2bNodePrime, HSMHV2sNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SbpPtr, HSMHV2SbpBinding, HSMHV2sNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GPgpPtr, HSMHV2GPgpBinding, HSMHV2gNodePrime, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SPdPtr, HSMHV2SPdBinding, HSMHV2sNodePrime, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SdpPtr, HSMHV2SdpBinding, HSMHV2sNode, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BbpPtr, HSMHV2BbpBinding, HSMHV2bNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPdPtr, HSMHV2BPdBinding, HSMHV2bNodePrime, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DPbpPtr, HSMHV2DPbpBinding, HSMHV2dNodePrime, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DspPtr, HSMHV2DspBinding, HSMHV2dNode, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DPdpPtr, HSMHV2DPdpBinding, HSMHV2dNodePrime, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DBbpPtr, HSMHV2DBbpBinding, HSMHV2dbNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPdbPtr, HSMHV2BPdbBinding, HSMHV2bNodePrime, HSMHV2dbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPgpPtr, HSMHV2BPgpBinding, HSMHV2bNodePrime, HSMHV2gNodePrime) ;

            if (here->HSMHV2subNode > 0)
            { /* 5th substrate node */
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DPsubPtr, HSMHV2DPsubBinding, HSMHV2dNodePrime, HSMHV2subNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DsubPtr, HSMHV2DsubBinding, HSMHV2dNode, HSMHV2subNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SPsubPtr, HSMHV2SPsubBinding, HSMHV2sNodePrime, HSMHV2subNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SsubPtr, HSMHV2SsubBinding, HSMHV2sNode, HSMHV2subNode) ;
            }

            if (here->HSMHV2tempNode > 0)
            { /* self heating */
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2TemptempPtr, HSMHV2TemptempBinding, HSMHV2tempNode, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2TempbpPtr, HSMHV2TempbpBinding, HSMHV2tempNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPtempPtr, HSMHV2BPtempBinding, HSMHV2bNodePrime, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2TempspPtr, HSMHV2TempspBinding, HSMHV2tempNode, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SPtempPtr, HSMHV2SPtempBinding, HSMHV2sNodePrime, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DtempPtr, HSMHV2DtempBinding, HSMHV2dNode, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2TempdPtr, HSMHV2TempdBinding, HSMHV2tempNode, HSMHV2dNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2TempdpPtr, HSMHV2TempdpBinding, HSMHV2tempNode, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DPtempPtr, HSMHV2DPtempBinding, HSMHV2dNodePrime, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SBtempPtr, HSMHV2SBtempBinding, HSMHV2sbNode, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DBtempPtr, HSMHV2DBtempBinding, HSMHV2dbNode, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2TempgpPtr, HSMHV2TempgpBinding, HSMHV2tempNode, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GPtempPtr, HSMHV2GPtempBinding, HSMHV2gNodePrime, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2StempPtr, HSMHV2StempBinding, HSMHV2sNode, HSMHV2tempNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2TempsPtr, HSMHV2TempsBinding, HSMHV2tempNode, HSMHV2sNode) ;
            }

            if (model->HSMHV2_conqs)
            { /* flat handling of NQS */
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GPqbPtr, HSMHV2GPqbBinding, HSMHV2gNodePrime, HSMHV2qbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QBgpPtr, HSMHV2QBgpBinding, HSMHV2qbNode, HSMHV2gNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QIbpPtr, HSMHV2QIbpBinding, HSMHV2qiNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QIqiPtr, HSMHV2QIqiBinding, HSMHV2qiNode, HSMHV2qiNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QBbpPtr, HSMHV2QBbpBinding, HSMHV2qbNode, HSMHV2bNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2SPqiPtr, HSMHV2SPqiBinding, HSMHV2sNodePrime, HSMHV2qiNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2BPqbPtr, HSMHV2BPqbBinding, HSMHV2bNodePrime, HSMHV2qbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QIspPtr, HSMHV2QIspBinding, HSMHV2qiNode, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QBqbPtr, HSMHV2QBqbBinding, HSMHV2qbNode, HSMHV2qbNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QBspPtr, HSMHV2QBspBinding, HSMHV2qbNode, HSMHV2sNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2DPqiPtr, HSMHV2DPqiBinding, HSMHV2dNodePrime, HSMHV2qiNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QIdpPtr, HSMHV2QIdpBinding, HSMHV2qiNode, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QBdpPtr, HSMHV2QBdpBinding, HSMHV2qbNode, HSMHV2dNodePrime) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2GPqiPtr, HSMHV2GPqiBinding, HSMHV2gNodePrime, HSMHV2qiNode) ;
                CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QIgpPtr, HSMHV2QIgpBinding, HSMHV2qiNode, HSMHV2gNodePrime) ;

                if (here->HSMHV2tempNode > 0)
                { /* self heating */
                    CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QItempPtr, HSMHV2QItempBinding, HSMHV2qiNode, HSMHV2tempNode) ;
                    CONVERT_KLU_BINDING_TABLE_TO_REAL (HSMHV2QBtempPtr, HSMHV2QBtempBinding, HSMHV2qbNode, HSMHV2tempNode) ;
                }
            }
        }
    }

    return (OK) ;
}
