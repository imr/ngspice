/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b4soidef.h"
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
    for ( ; model != NULL ; model = model->B4SOInextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B4SOIinstances ; here != NULL ; here = here->B4SOInextInstance)
        {
            if ((model->B4SOIshMod == 1) && (here->B4SOIrth0 != 0.0))
            {
                if ((here-> B4SOItempNode != 0) && (here-> B4SOItempNode != 0))
                {
                    i = here->B4SOITemptempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOITemptempStructPtr = matched ;
                    here->B4SOITemptempPtr = matched->CSC ;
                }

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIdNodePrime != 0))
                {
                    i = here->B4SOITempdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOITempdpStructPtr = matched ;
                    here->B4SOITempdpPtr = matched->CSC ;
                }

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIsNodePrime != 0))
                {
                    i = here->B4SOITempspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOITempspStructPtr = matched ;
                    here->B4SOITempspPtr = matched->CSC ;
                }

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIgNode != 0))
                {
                    i = here->B4SOITempgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOITempgStructPtr = matched ;
                    here->B4SOITempgPtr = matched->CSC ;
                }

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIbNode != 0))
                {
                    i = here->B4SOITempbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOITempbStructPtr = matched ;
                    here->B4SOITempbPtr = matched->CSC ;
                }

                if ((here-> B4SOIgNode != 0) && (here-> B4SOItempNode != 0))
                {
                    i = here->B4SOIGtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGtempStructPtr = matched ;
                    here->B4SOIGtempPtr = matched->CSC ;
                }

                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOItempNode != 0))
                {
                    i = here->B4SOIDPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIDPtempStructPtr = matched ;
                    here->B4SOIDPtempPtr = matched->CSC ;
                }

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOItempNode != 0))
                {
                    i = here->B4SOISPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOISPtempStructPtr = matched ;
                    here->B4SOISPtempPtr = matched->CSC ;
                }

                if ((here-> B4SOIeNode != 0) && (here-> B4SOItempNode != 0))
                {
                    i = here->B4SOIEtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIEtempStructPtr = matched ;
                    here->B4SOIEtempPtr = matched->CSC ;
                }

                if ((here-> B4SOIbNode != 0) && (here-> B4SOItempNode != 0))
                {
                    i = here->B4SOIBtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIBtempStructPtr = matched ;
                    here->B4SOIBtempPtr = matched->CSC ;
                }

                if (here->B4SOIbodyMod == 1)
                {
                    if ((here-> B4SOIpNode != 0) && (here-> B4SOItempNode != 0))
                    {
                        i = here->B4SOIPtempPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->B4SOIPtempStructPtr = matched ;
                        here->B4SOIPtempPtr = matched->CSC ;
                    }

                }
                if (here->B4SOIsoiMod != 0)
                {
                    if ((here-> B4SOItempNode != 0) && (here-> B4SOIeNode != 0))
                    {
                        i = here->B4SOITempePtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->B4SOITempeStructPtr = matched ;
                        here->B4SOITempePtr = matched->CSC ;
                    }

                }
            }
            if (here->B4SOIbodyMod == 2)
            {
            }
            else if (here->B4SOIbodyMod == 1)
            {
                if ((here-> B4SOIbNode != 0) && (here-> B4SOIpNode != 0))
                {
                    i = here->B4SOIBpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIBpStructPtr = matched ;
                    here->B4SOIBpPtr = matched->CSC ;
                }

                if ((here-> B4SOIpNode != 0) && (here-> B4SOIbNode != 0))
                {
                    i = here->B4SOIPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIPbStructPtr = matched ;
                    here->B4SOIPbPtr = matched->CSC ;
                }

                if ((here-> B4SOIpNode != 0) && (here-> B4SOIpNode != 0))
                {
                    i = here->B4SOIPpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIPpStructPtr = matched ;
                    here->B4SOIPpPtr = matched->CSC ;
                }

                if ((here-> B4SOIpNode != 0) && (here-> B4SOIgNode != 0))
                {
                    i = here->B4SOIPgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIPgStructPtr = matched ;
                    here->B4SOIPgPtr = matched->CSC ;
                }

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIpNode != 0))
                {
                    i = here->B4SOIGpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGpStructPtr = matched ;
                    here->B4SOIGpPtr = matched->CSC ;
                }

            }
            if (here->B4SOIrgateMod != 0)
            {
                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNodeExt != 0))
                {
                    i = here->B4SOIGEgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGEgeStructPtr = matched ;
                    here->B4SOIGEgePtr = matched->CSC ;
                }

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNode != 0))
                {
                    i = here->B4SOIGEgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGEgStructPtr = matched ;
                    here->B4SOIGEgPtr = matched->CSC ;
                }

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNodeExt != 0))
                {
                    i = here->B4SOIGgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGgeStructPtr = matched ;
                    here->B4SOIGgePtr = matched->CSC ;
                }

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIdNodePrime != 0))
                {
                    i = here->B4SOIGEdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGEdpStructPtr = matched ;
                    here->B4SOIGEdpPtr = matched->CSC ;
                }

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIsNodePrime != 0))
                {
                    i = here->B4SOIGEspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGEspStructPtr = matched ;
                    here->B4SOIGEspPtr = matched->CSC ;
                }

                if (here->B4SOIsoiMod != 2)
                {
                    if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIbNode != 0))
                    {
                        i = here->B4SOIGEbPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->B4SOIGEbStructPtr = matched ;
                        here->B4SOIGEbPtr = matched->CSC ;
                    }

                }
                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIdNodePrime != 0))
                {
                    i = here->B4SOIGMdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGMdpStructPtr = matched ;
                    here->B4SOIGMdpPtr = matched->CSC ;
                }

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNode != 0))
                {
                    i = here->B4SOIGMgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGMgStructPtr = matched ;
                    here->B4SOIGMgPtr = matched->CSC ;
                }

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNodeMid != 0))
                {
                    i = here->B4SOIGMgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGMgmStructPtr = matched ;
                    here->B4SOIGMgmPtr = matched->CSC ;
                }

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNodeExt != 0))
                {
                    i = here->B4SOIGMgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGMgeStructPtr = matched ;
                    here->B4SOIGMgePtr = matched->CSC ;
                }

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIsNodePrime != 0))
                {
                    i = here->B4SOIGMspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGMspStructPtr = matched ;
                    here->B4SOIGMspPtr = matched->CSC ;
                }

                if (here->B4SOIsoiMod != 2)
                {
                    if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIbNode != 0))
                    {
                        i = here->B4SOIGMbPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->B4SOIGMbStructPtr = matched ;
                        here->B4SOIGMbPtr = matched->CSC ;
                    }

                }
                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIeNode != 0))
                {
                    i = here->B4SOIGMePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGMeStructPtr = matched ;
                    here->B4SOIGMePtr = matched->CSC ;
                }

                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIgNodeMid != 0))
                {
                    i = here->B4SOIDPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIDPgmStructPtr = matched ;
                    here->B4SOIDPgmPtr = matched->CSC ;
                }

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNodeMid != 0))
                {
                    i = here->B4SOIGgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGgmStructPtr = matched ;
                    here->B4SOIGgmPtr = matched->CSC ;
                }

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNodeMid != 0))
                {
                    i = here->B4SOIGEgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGEgmStructPtr = matched ;
                    here->B4SOIGEgmPtr = matched->CSC ;
                }

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIgNodeMid != 0))
                {
                    i = here->B4SOISPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOISPgmStructPtr = matched ;
                    here->B4SOISPgmPtr = matched->CSC ;
                }

                if ((here-> B4SOIeNode != 0) && (here-> B4SOIgNodeMid != 0))
                {
                    i = here->B4SOIEgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIEgmStructPtr = matched ;
                    here->B4SOIEgmPtr = matched->CSC ;
                }

            }
            if (here->B4SOIsoiMod != 2) /* v3.2 */
            {
                if ((here-> B4SOIeNode != 0) && (here-> B4SOIbNode != 0))
                {
                    i = here->B4SOIEbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIEbStructPtr = matched ;
                    here->B4SOIEbPtr = matched->CSC ;
                }

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIbNode != 0))
                {
                    i = here->B4SOIGbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGbStructPtr = matched ;
                    here->B4SOIGbPtr = matched->CSC ;
                }

                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIbNode != 0))
                {
                    i = here->B4SOIDPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIDPbStructPtr = matched ;
                    here->B4SOIDPbPtr = matched->CSC ;
                }

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIbNode != 0))
                {
                    i = here->B4SOISPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOISPbStructPtr = matched ;
                    here->B4SOISPbPtr = matched->CSC ;
                }

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIeNode != 0))
                {
                    i = here->B4SOIBePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIBeStructPtr = matched ;
                    here->B4SOIBePtr = matched->CSC ;
                }

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIgNode != 0))
                {
                    i = here->B4SOIBgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIBgStructPtr = matched ;
                    here->B4SOIBgPtr = matched->CSC ;
                }

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIdNodePrime != 0))
                {
                    i = here->B4SOIBdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIBdpStructPtr = matched ;
                    here->B4SOIBdpPtr = matched->CSC ;
                }

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIsNodePrime != 0))
                {
                    i = here->B4SOIBspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIBspStructPtr = matched ;
                    here->B4SOIBspPtr = matched->CSC ;
                }

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIbNode != 0))
                {
                    i = here->B4SOIBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIBbStructPtr = matched ;
                    here->B4SOIBbPtr = matched->CSC ;
                }

            }
            if ((here-> B4SOIeNode != 0) && (here-> B4SOIgNode != 0))
            {
                i = here->B4SOIEgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIEgStructPtr = matched ;
                here->B4SOIEgPtr = matched->CSC ;
            }

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIdNodePrime != 0))
            {
                i = here->B4SOIEdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIEdpStructPtr = matched ;
                here->B4SOIEdpPtr = matched->CSC ;
            }

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIsNodePrime != 0))
            {
                i = here->B4SOIEspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIEspStructPtr = matched ;
                here->B4SOIEspPtr = matched->CSC ;
            }

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIeNode != 0))
            {
                i = here->B4SOIGePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIGeStructPtr = matched ;
                here->B4SOIGePtr = matched->CSC ;
            }

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIeNode != 0))
            {
                i = here->B4SOIDPePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIDPeStructPtr = matched ;
                here->B4SOIDPePtr = matched->CSC ;
            }

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIeNode != 0))
            {
                i = here->B4SOISPePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOISPeStructPtr = matched ;
                here->B4SOISPePtr = matched->CSC ;
            }

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIbNode != 0))
            {
                i = here->B4SOIEbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIEbStructPtr = matched ;
                here->B4SOIEbPtr = matched->CSC ;
            }

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIeNode != 0))
            {
                i = here->B4SOIEePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIEeStructPtr = matched ;
                here->B4SOIEePtr = matched->CSC ;
            }

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNode != 0))
            {
                i = here->B4SOIGgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIGgStructPtr = matched ;
                here->B4SOIGgPtr = matched->CSC ;
            }

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIdNodePrime != 0))
            {
                i = here->B4SOIGdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIGdpStructPtr = matched ;
                here->B4SOIGdpPtr = matched->CSC ;
            }

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIsNodePrime != 0))
            {
                i = here->B4SOIGspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIGspStructPtr = matched ;
                here->B4SOIGspPtr = matched->CSC ;
            }

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIgNode != 0))
            {
                i = here->B4SOIDPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIDPgStructPtr = matched ;
                here->B4SOIDPgPtr = matched->CSC ;
            }

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdNodePrime != 0))
            {
                i = here->B4SOIDPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIDPdpStructPtr = matched ;
                here->B4SOIDPdpPtr = matched->CSC ;
            }

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIsNodePrime != 0))
            {
                i = here->B4SOIDPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIDPspStructPtr = matched ;
                here->B4SOIDPspPtr = matched->CSC ;
            }

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdNode != 0))
            {
                i = here->B4SOIDPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIDPdStructPtr = matched ;
                here->B4SOIDPdPtr = matched->CSC ;
            }

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIgNode != 0))
            {
                i = here->B4SOISPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOISPgStructPtr = matched ;
                here->B4SOISPgPtr = matched->CSC ;
            }

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIdNodePrime != 0))
            {
                i = here->B4SOISPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOISPdpStructPtr = matched ;
                here->B4SOISPdpPtr = matched->CSC ;
            }

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsNodePrime != 0))
            {
                i = here->B4SOISPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOISPspStructPtr = matched ;
                here->B4SOISPspPtr = matched->CSC ;
            }

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsNode != 0))
            {
                i = here->B4SOISPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOISPsStructPtr = matched ;
                here->B4SOISPsPtr = matched->CSC ;
            }

            if ((here-> B4SOIdNode != 0) && (here-> B4SOIdNode != 0))
            {
                i = here->B4SOIDdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIDdStructPtr = matched ;
                here->B4SOIDdPtr = matched->CSC ;
            }

            if ((here-> B4SOIdNode != 0) && (here-> B4SOIdNodePrime != 0))
            {
                i = here->B4SOIDdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOIDdpStructPtr = matched ;
                here->B4SOIDdpPtr = matched->CSC ;
            }

            if ((here-> B4SOIsNode != 0) && (here-> B4SOIsNode != 0))
            {
                i = here->B4SOISsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOISsStructPtr = matched ;
                here->B4SOISsPtr = matched->CSC ;
            }

            if ((here-> B4SOIsNode != 0) && (here-> B4SOIsNodePrime != 0))
            {
                i = here->B4SOISspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B4SOISspStructPtr = matched ;
                here->B4SOISspPtr = matched->CSC ;
            }

            if (here->B4SOIrbodyMod == 1)
            {
                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdbNode != 0))
                {
                    i = here->B4SOIDPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIDPdbStructPtr = matched ;
                    here->B4SOIDPdbPtr = matched->CSC ;
                }

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsbNode != 0))
                {
                    i = here->B4SOISPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOISPsbStructPtr = matched ;
                    here->B4SOISPsbPtr = matched->CSC ;
                }

                if ((here-> B4SOIdbNode != 0) && (here-> B4SOIdNodePrime != 0))
                {
                    i = here->B4SOIDBdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIDBdpStructPtr = matched ;
                    here->B4SOIDBdpPtr = matched->CSC ;
                }

                if ((here-> B4SOIdbNode != 0) && (here-> B4SOIdbNode != 0))
                {
                    i = here->B4SOIDBdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIDBdbStructPtr = matched ;
                    here->B4SOIDBdbPtr = matched->CSC ;
                }

                if ((here-> B4SOIdbNode != 0) && (here-> B4SOIbNode != 0))
                {
                    i = here->B4SOIDBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIDBbStructPtr = matched ;
                    here->B4SOIDBbPtr = matched->CSC ;
                }

                if ((here-> B4SOIsbNode != 0) && (here-> B4SOIsNodePrime != 0))
                {
                    i = here->B4SOISBspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOISBspStructPtr = matched ;
                    here->B4SOISBspPtr = matched->CSC ;
                }

                if ((here-> B4SOIsbNode != 0) && (here-> B4SOIsbNode != 0))
                {
                    i = here->B4SOISBsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOISBsbStructPtr = matched ;
                    here->B4SOISBsbPtr = matched->CSC ;
                }

                if ((here-> B4SOIsbNode != 0) && (here-> B4SOIbNode != 0))
                {
                    i = here->B4SOISBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOISBbStructPtr = matched ;
                    here->B4SOISBbPtr = matched->CSC ;
                }

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIdbNode != 0))
                {
                    i = here->B4SOIBdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIBdbStructPtr = matched ;
                    here->B4SOIBdbPtr = matched->CSC ;
                }

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIsbNode != 0))
                {
                    i = here->B4SOIBsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIBsbStructPtr = matched ;
                    here->B4SOIBsbPtr = matched->CSC ;
                }

            }
            if (model->B4SOIrdsMod)
            {
                if ((here-> B4SOIdNode != 0) && (here-> B4SOIgNode != 0))
                {
                    i = here->B4SOIDgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIDgStructPtr = matched ;
                    here->B4SOIDgPtr = matched->CSC ;
                }

                if ((here-> B4SOIdNode != 0) && (here-> B4SOIsNodePrime != 0))
                {
                    i = here->B4SOIDspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIDspStructPtr = matched ;
                    here->B4SOIDspPtr = matched->CSC ;
                }

                if ((here-> B4SOIsNode != 0) && (here-> B4SOIdNodePrime != 0))
                {
                    i = here->B4SOISdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOISdpStructPtr = matched ;
                    here->B4SOISdpPtr = matched->CSC ;
                }

                if ((here-> B4SOIsNode != 0) && (here-> B4SOIgNode != 0))
                {
                    i = here->B4SOISgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOISgStructPtr = matched ;
                    here->B4SOISgPtr = matched->CSC ;
                }

                if (model->B4SOIsoiMod != 2)
                {
                    if ((here-> B4SOIdNode != 0) && (here-> B4SOIbNode != 0))
                    {
                        i = here->B4SOIDbPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->B4SOIDbStructPtr = matched ;
                        here->B4SOIDbPtr = matched->CSC ;
                    }

                    if ((here-> B4SOIsNode != 0) && (here-> B4SOIbNode != 0))
                    {
                        i = here->B4SOISbPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->B4SOISbStructPtr = matched ;
                        here->B4SOISbPtr = matched->CSC ;
                    }

                }
            }
            if (here->B4SOIdebugMod != 0)
            {
                if ((here-> B4SOIvbsNode != 0) && (here-> B4SOIvbsNode != 0))
                {
                    i = here->B4SOIVbsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIVbsStructPtr = matched ;
                    here->B4SOIVbsPtr = matched->CSC ;
                }

                if ((here-> B4SOIidsNode != 0) && (here-> B4SOIidsNode != 0))
                {
                    i = here->B4SOIIdsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIIdsStructPtr = matched ;
                    here->B4SOIIdsPtr = matched->CSC ;
                }

                if ((here-> B4SOIicNode != 0) && (here-> B4SOIicNode != 0))
                {
                    i = here->B4SOIIcPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIIcStructPtr = matched ;
                    here->B4SOIIcPtr = matched->CSC ;
                }

                if ((here-> B4SOIibsNode != 0) && (here-> B4SOIibsNode != 0))
                {
                    i = here->B4SOIIbsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIIbsStructPtr = matched ;
                    here->B4SOIIbsPtr = matched->CSC ;
                }

                if ((here-> B4SOIibdNode != 0) && (here-> B4SOIibdNode != 0))
                {
                    i = here->B4SOIIbdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIIbdStructPtr = matched ;
                    here->B4SOIIbdPtr = matched->CSC ;
                }

                if ((here-> B4SOIiiiNode != 0) && (here-> B4SOIiiiNode != 0))
                {
                    i = here->B4SOIIiiPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIIiiStructPtr = matched ;
                    here->B4SOIIiiPtr = matched->CSC ;
                }

                if ((here-> B4SOIigNode != 0) && (here-> B4SOIigNode != 0))
                {
                    i = here->B4SOIIgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIIgStructPtr = matched ;
                    here->B4SOIIgPtr = matched->CSC ;
                }

                if ((here-> B4SOIgiggNode != 0) && (here-> B4SOIgiggNode != 0))
                {
                    i = here->B4SOIGiggPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGiggStructPtr = matched ;
                    here->B4SOIGiggPtr = matched->CSC ;
                }

                if ((here-> B4SOIgigdNode != 0) && (here-> B4SOIgigdNode != 0))
                {
                    i = here->B4SOIGigdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGigdStructPtr = matched ;
                    here->B4SOIGigdPtr = matched->CSC ;
                }

                if ((here-> B4SOIgigbNode != 0) && (here-> B4SOIgigbNode != 0))
                {
                    i = here->B4SOIGigbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIGigbStructPtr = matched ;
                    here->B4SOIGigbPtr = matched->CSC ;
                }

                if ((here-> B4SOIigidlNode != 0) && (here-> B4SOIigidlNode != 0))
                {
                    i = here->B4SOIIgidlPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIIgidlStructPtr = matched ;
                    here->B4SOIIgidlPtr = matched->CSC ;
                }

                if ((here-> B4SOIitunNode != 0) && (here-> B4SOIitunNode != 0))
                {
                    i = here->B4SOIItunPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIItunStructPtr = matched ;
                    here->B4SOIItunPtr = matched->CSC ;
                }

                if ((here-> B4SOIibpNode != 0) && (here-> B4SOIibpNode != 0))
                {
                    i = here->B4SOIIbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIIbpStructPtr = matched ;
                    here->B4SOIIbpPtr = matched->CSC ;
                }

                if ((here-> B4SOIcbbNode != 0) && (here-> B4SOIcbbNode != 0))
                {
                    i = here->B4SOICbbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOICbbStructPtr = matched ;
                    here->B4SOICbbPtr = matched->CSC ;
                }

                if ((here-> B4SOIcbdNode != 0) && (here-> B4SOIcbdNode != 0))
                {
                    i = here->B4SOICbdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOICbdStructPtr = matched ;
                    here->B4SOICbdPtr = matched->CSC ;
                }

                if ((here-> B4SOIcbgNode != 0) && (here-> B4SOIcbgNode != 0))
                {
                    i = here->B4SOICbgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOICbgStructPtr = matched ;
                    here->B4SOICbgPtr = matched->CSC ;
                }

                if ((here-> B4SOIqbfNode != 0) && (here-> B4SOIqbfNode != 0))
                {
                    i = here->B4SOIQbfPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIQbfStructPtr = matched ;
                    here->B4SOIQbfPtr = matched->CSC ;
                }

                if ((here-> B4SOIqjsNode != 0) && (here-> B4SOIqjsNode != 0))
                {
                    i = here->B4SOIQjsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIQjsStructPtr = matched ;
                    here->B4SOIQjsPtr = matched->CSC ;
                }

                if ((here-> B4SOIqjdNode != 0) && (here-> B4SOIqjdNode != 0))
                {
                    i = here->B4SOIQjdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B4SOIQjdStructPtr = matched ;
                    here->B4SOIQjdPtr = matched->CSC ;
                }

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
    for ( ; model != NULL ; model = model->B4SOInextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B4SOIinstances ; here != NULL ; here = here->B4SOInextInstance)
        {
            if ((model->B4SOIshMod == 1) && (here->B4SOIrth0 != 0.0))
            {
                if ((here-> B4SOItempNode != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOITemptempPtr = here->B4SOITemptempStructPtr->CSC_Complex ;

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOITempdpPtr = here->B4SOITempdpStructPtr->CSC_Complex ;

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOITempspPtr = here->B4SOITempspStructPtr->CSC_Complex ;

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOITempgPtr = here->B4SOITempgStructPtr->CSC_Complex ;

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOITempbPtr = here->B4SOITempbStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNode != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOIGtempPtr = here->B4SOIGtempStructPtr->CSC_Complex ;

                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOIDPtempPtr = here->B4SOIDPtempStructPtr->CSC_Complex ;

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOISPtempPtr = here->B4SOISPtempStructPtr->CSC_Complex ;

                if ((here-> B4SOIeNode != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOIEtempPtr = here->B4SOIEtempStructPtr->CSC_Complex ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOIBtempPtr = here->B4SOIBtempStructPtr->CSC_Complex ;

                if (here->B4SOIbodyMod == 1)
                {
                    if ((here-> B4SOIpNode != 0) && (here-> B4SOItempNode != 0))
                        here->B4SOIPtempPtr = here->B4SOIPtempStructPtr->CSC_Complex ;

                }
                if (here->B4SOIsoiMod != 0)
                {
                    if ((here-> B4SOItempNode != 0) && (here-> B4SOIeNode != 0))
                        here->B4SOITempePtr = here->B4SOITempeStructPtr->CSC_Complex ;

                }
            }
            if (here->B4SOIbodyMod == 2)
            {
            }
            else if (here->B4SOIbodyMod == 1)
            {
                if ((here-> B4SOIbNode != 0) && (here-> B4SOIpNode != 0))
                    here->B4SOIBpPtr = here->B4SOIBpStructPtr->CSC_Complex ;

                if ((here-> B4SOIpNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIPbPtr = here->B4SOIPbStructPtr->CSC_Complex ;

                if ((here-> B4SOIpNode != 0) && (here-> B4SOIpNode != 0))
                    here->B4SOIPpPtr = here->B4SOIPpStructPtr->CSC_Complex ;

                if ((here-> B4SOIpNode != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOIPgPtr = here->B4SOIPgStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIpNode != 0))
                    here->B4SOIGpPtr = here->B4SOIGpStructPtr->CSC_Complex ;

            }
            if (here->B4SOIrgateMod != 0)
            {
                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNodeExt != 0))
                    here->B4SOIGEgePtr = here->B4SOIGEgeStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOIGEgPtr = here->B4SOIGEgStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNodeExt != 0))
                    here->B4SOIGgePtr = here->B4SOIGgeStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOIGEdpPtr = here->B4SOIGEdpStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOIGEspPtr = here->B4SOIGEspStructPtr->CSC_Complex ;

                if (here->B4SOIsoiMod != 2)
                {
                    if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIbNode != 0))
                        here->B4SOIGEbPtr = here->B4SOIGEbStructPtr->CSC_Complex ;

                }
                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOIGMdpPtr = here->B4SOIGMdpStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOIGMgPtr = here->B4SOIGMgStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOIGMgmPtr = here->B4SOIGMgmStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNodeExt != 0))
                    here->B4SOIGMgePtr = here->B4SOIGMgeStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOIGMspPtr = here->B4SOIGMspStructPtr->CSC_Complex ;

                if (here->B4SOIsoiMod != 2)
                {
                    if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIbNode != 0))
                        here->B4SOIGMbPtr = here->B4SOIGMbStructPtr->CSC_Complex ;

                }
                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIeNode != 0))
                    here->B4SOIGMePtr = here->B4SOIGMeStructPtr->CSC_Complex ;

                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOIDPgmPtr = here->B4SOIDPgmStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOIGgmPtr = here->B4SOIGgmStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOIGEgmPtr = here->B4SOIGEgmStructPtr->CSC_Complex ;

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOISPgmPtr = here->B4SOISPgmStructPtr->CSC_Complex ;

                if ((here-> B4SOIeNode != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOIEgmPtr = here->B4SOIEgmStructPtr->CSC_Complex ;

            }
            if (here->B4SOIsoiMod != 2) /* v3.2 */
            {
                if ((here-> B4SOIeNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIEbPtr = here->B4SOIEbStructPtr->CSC_Complex ;

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIGbPtr = here->B4SOIGbStructPtr->CSC_Complex ;

                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIDPbPtr = here->B4SOIDPbStructPtr->CSC_Complex ;

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOISPbPtr = here->B4SOISPbStructPtr->CSC_Complex ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIeNode != 0))
                    here->B4SOIBePtr = here->B4SOIBeStructPtr->CSC_Complex ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOIBgPtr = here->B4SOIBgStructPtr->CSC_Complex ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOIBdpPtr = here->B4SOIBdpStructPtr->CSC_Complex ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOIBspPtr = here->B4SOIBspStructPtr->CSC_Complex ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIBbPtr = here->B4SOIBbStructPtr->CSC_Complex ;

            }
            if ((here-> B4SOIeNode != 0) && (here-> B4SOIgNode != 0))
                here->B4SOIEgPtr = here->B4SOIEgStructPtr->CSC_Complex ;

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIdNodePrime != 0))
                here->B4SOIEdpPtr = here->B4SOIEdpStructPtr->CSC_Complex ;

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIsNodePrime != 0))
                here->B4SOIEspPtr = here->B4SOIEspStructPtr->CSC_Complex ;

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIeNode != 0))
                here->B4SOIGePtr = here->B4SOIGeStructPtr->CSC_Complex ;

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIeNode != 0))
                here->B4SOIDPePtr = here->B4SOIDPeStructPtr->CSC_Complex ;

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIeNode != 0))
                here->B4SOISPePtr = here->B4SOISPeStructPtr->CSC_Complex ;

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIbNode != 0))
                here->B4SOIEbPtr = here->B4SOIEbStructPtr->CSC_Complex ;

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIeNode != 0))
                here->B4SOIEePtr = here->B4SOIEeStructPtr->CSC_Complex ;

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNode != 0))
                here->B4SOIGgPtr = here->B4SOIGgStructPtr->CSC_Complex ;

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIdNodePrime != 0))
                here->B4SOIGdpPtr = here->B4SOIGdpStructPtr->CSC_Complex ;

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIsNodePrime != 0))
                here->B4SOIGspPtr = here->B4SOIGspStructPtr->CSC_Complex ;

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIgNode != 0))
                here->B4SOIDPgPtr = here->B4SOIDPgStructPtr->CSC_Complex ;

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdNodePrime != 0))
                here->B4SOIDPdpPtr = here->B4SOIDPdpStructPtr->CSC_Complex ;

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIsNodePrime != 0))
                here->B4SOIDPspPtr = here->B4SOIDPspStructPtr->CSC_Complex ;

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdNode != 0))
                here->B4SOIDPdPtr = here->B4SOIDPdStructPtr->CSC_Complex ;

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIgNode != 0))
                here->B4SOISPgPtr = here->B4SOISPgStructPtr->CSC_Complex ;

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIdNodePrime != 0))
                here->B4SOISPdpPtr = here->B4SOISPdpStructPtr->CSC_Complex ;

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsNodePrime != 0))
                here->B4SOISPspPtr = here->B4SOISPspStructPtr->CSC_Complex ;

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsNode != 0))
                here->B4SOISPsPtr = here->B4SOISPsStructPtr->CSC_Complex ;

            if ((here-> B4SOIdNode != 0) && (here-> B4SOIdNode != 0))
                here->B4SOIDdPtr = here->B4SOIDdStructPtr->CSC_Complex ;

            if ((here-> B4SOIdNode != 0) && (here-> B4SOIdNodePrime != 0))
                here->B4SOIDdpPtr = here->B4SOIDdpStructPtr->CSC_Complex ;

            if ((here-> B4SOIsNode != 0) && (here-> B4SOIsNode != 0))
                here->B4SOISsPtr = here->B4SOISsStructPtr->CSC_Complex ;

            if ((here-> B4SOIsNode != 0) && (here-> B4SOIsNodePrime != 0))
                here->B4SOISspPtr = here->B4SOISspStructPtr->CSC_Complex ;

            if (here->B4SOIrbodyMod == 1)
            {
                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdbNode != 0))
                    here->B4SOIDPdbPtr = here->B4SOIDPdbStructPtr->CSC_Complex ;

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsbNode != 0))
                    here->B4SOISPsbPtr = here->B4SOISPsbStructPtr->CSC_Complex ;

                if ((here-> B4SOIdbNode != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOIDBdpPtr = here->B4SOIDBdpStructPtr->CSC_Complex ;

                if ((here-> B4SOIdbNode != 0) && (here-> B4SOIdbNode != 0))
                    here->B4SOIDBdbPtr = here->B4SOIDBdbStructPtr->CSC_Complex ;

                if ((here-> B4SOIdbNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIDBbPtr = here->B4SOIDBbStructPtr->CSC_Complex ;

                if ((here-> B4SOIsbNode != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOISBspPtr = here->B4SOISBspStructPtr->CSC_Complex ;

                if ((here-> B4SOIsbNode != 0) && (here-> B4SOIsbNode != 0))
                    here->B4SOISBsbPtr = here->B4SOISBsbStructPtr->CSC_Complex ;

                if ((here-> B4SOIsbNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOISBbPtr = here->B4SOISBbStructPtr->CSC_Complex ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIdbNode != 0))
                    here->B4SOIBdbPtr = here->B4SOIBdbStructPtr->CSC_Complex ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIsbNode != 0))
                    here->B4SOIBsbPtr = here->B4SOIBsbStructPtr->CSC_Complex ;

            }
            if (model->B4SOIrdsMod)
            {
                if ((here-> B4SOIdNode != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOIDgPtr = here->B4SOIDgStructPtr->CSC_Complex ;

                if ((here-> B4SOIdNode != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOIDspPtr = here->B4SOIDspStructPtr->CSC_Complex ;

                if ((here-> B4SOIsNode != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOISdpPtr = here->B4SOISdpStructPtr->CSC_Complex ;

                if ((here-> B4SOIsNode != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOISgPtr = here->B4SOISgStructPtr->CSC_Complex ;

                if (model->B4SOIsoiMod != 2)
                {
                    if ((here-> B4SOIdNode != 0) && (here-> B4SOIbNode != 0))
                        here->B4SOIDbPtr = here->B4SOIDbStructPtr->CSC_Complex ;

                    if ((here-> B4SOIsNode != 0) && (here-> B4SOIbNode != 0))
                        here->B4SOISbPtr = here->B4SOISbStructPtr->CSC_Complex ;

                }
            }
            if (here->B4SOIdebugMod != 0)
            {
                if ((here-> B4SOIvbsNode != 0) && (here-> B4SOIvbsNode != 0))
                    here->B4SOIVbsPtr = here->B4SOIVbsStructPtr->CSC_Complex ;

                if ((here-> B4SOIidsNode != 0) && (here-> B4SOIidsNode != 0))
                    here->B4SOIIdsPtr = here->B4SOIIdsStructPtr->CSC_Complex ;

                if ((here-> B4SOIicNode != 0) && (here-> B4SOIicNode != 0))
                    here->B4SOIIcPtr = here->B4SOIIcStructPtr->CSC_Complex ;

                if ((here-> B4SOIibsNode != 0) && (here-> B4SOIibsNode != 0))
                    here->B4SOIIbsPtr = here->B4SOIIbsStructPtr->CSC_Complex ;

                if ((here-> B4SOIibdNode != 0) && (here-> B4SOIibdNode != 0))
                    here->B4SOIIbdPtr = here->B4SOIIbdStructPtr->CSC_Complex ;

                if ((here-> B4SOIiiiNode != 0) && (here-> B4SOIiiiNode != 0))
                    here->B4SOIIiiPtr = here->B4SOIIiiStructPtr->CSC_Complex ;

                if ((here-> B4SOIigNode != 0) && (here-> B4SOIigNode != 0))
                    here->B4SOIIgPtr = here->B4SOIIgStructPtr->CSC_Complex ;

                if ((here-> B4SOIgiggNode != 0) && (here-> B4SOIgiggNode != 0))
                    here->B4SOIGiggPtr = here->B4SOIGiggStructPtr->CSC_Complex ;

                if ((here-> B4SOIgigdNode != 0) && (here-> B4SOIgigdNode != 0))
                    here->B4SOIGigdPtr = here->B4SOIGigdStructPtr->CSC_Complex ;

                if ((here-> B4SOIgigbNode != 0) && (here-> B4SOIgigbNode != 0))
                    here->B4SOIGigbPtr = here->B4SOIGigbStructPtr->CSC_Complex ;

                if ((here-> B4SOIigidlNode != 0) && (here-> B4SOIigidlNode != 0))
                    here->B4SOIIgidlPtr = here->B4SOIIgidlStructPtr->CSC_Complex ;

                if ((here-> B4SOIitunNode != 0) && (here-> B4SOIitunNode != 0))
                    here->B4SOIItunPtr = here->B4SOIItunStructPtr->CSC_Complex ;

                if ((here-> B4SOIibpNode != 0) && (here-> B4SOIibpNode != 0))
                    here->B4SOIIbpPtr = here->B4SOIIbpStructPtr->CSC_Complex ;

                if ((here-> B4SOIcbbNode != 0) && (here-> B4SOIcbbNode != 0))
                    here->B4SOICbbPtr = here->B4SOICbbStructPtr->CSC_Complex ;

                if ((here-> B4SOIcbdNode != 0) && (here-> B4SOIcbdNode != 0))
                    here->B4SOICbdPtr = here->B4SOICbdStructPtr->CSC_Complex ;

                if ((here-> B4SOIcbgNode != 0) && (here-> B4SOIcbgNode != 0))
                    here->B4SOICbgPtr = here->B4SOICbgStructPtr->CSC_Complex ;

                if ((here-> B4SOIqbfNode != 0) && (here-> B4SOIqbfNode != 0))
                    here->B4SOIQbfPtr = here->B4SOIQbfStructPtr->CSC_Complex ;

                if ((here-> B4SOIqjsNode != 0) && (here-> B4SOIqjsNode != 0))
                    here->B4SOIQjsPtr = here->B4SOIQjsStructPtr->CSC_Complex ;

                if ((here-> B4SOIqjdNode != 0) && (here-> B4SOIqjdNode != 0))
                    here->B4SOIQjdPtr = here->B4SOIQjdStructPtr->CSC_Complex ;

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
    for ( ; model != NULL ; model = model->B4SOInextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B4SOIinstances ; here != NULL ; here = here->B4SOInextInstance)
        {
            if ((model->B4SOIshMod == 1) && (here->B4SOIrth0 != 0.0))
            {
                if ((here-> B4SOItempNode != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOITemptempPtr = here->B4SOITemptempStructPtr->CSC ;

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOITempdpPtr = here->B4SOITempdpStructPtr->CSC ;

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOITempspPtr = here->B4SOITempspStructPtr->CSC ;

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOITempgPtr = here->B4SOITempgStructPtr->CSC ;

                if ((here-> B4SOItempNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOITempbPtr = here->B4SOITempbStructPtr->CSC ;

                if ((here-> B4SOIgNode != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOIGtempPtr = here->B4SOIGtempStructPtr->CSC ;

                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOIDPtempPtr = here->B4SOIDPtempStructPtr->CSC ;

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOISPtempPtr = here->B4SOISPtempStructPtr->CSC ;

                if ((here-> B4SOIeNode != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOIEtempPtr = here->B4SOIEtempStructPtr->CSC ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOItempNode != 0))
                    here->B4SOIBtempPtr = here->B4SOIBtempStructPtr->CSC ;

                if (here->B4SOIbodyMod == 1)
                {
                    if ((here-> B4SOIpNode != 0) && (here-> B4SOItempNode != 0))
                        here->B4SOIPtempPtr = here->B4SOIPtempStructPtr->CSC ;

                }
                if (here->B4SOIsoiMod != 0)
                {
                    if ((here-> B4SOItempNode != 0) && (here-> B4SOIeNode != 0))
                        here->B4SOITempePtr = here->B4SOITempeStructPtr->CSC ;

                }
            }
            if (here->B4SOIbodyMod == 2)
            {
            }
            else if (here->B4SOIbodyMod == 1)
            {
                if ((here-> B4SOIbNode != 0) && (here-> B4SOIpNode != 0))
                    here->B4SOIBpPtr = here->B4SOIBpStructPtr->CSC ;

                if ((here-> B4SOIpNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIPbPtr = here->B4SOIPbStructPtr->CSC ;

                if ((here-> B4SOIpNode != 0) && (here-> B4SOIpNode != 0))
                    here->B4SOIPpPtr = here->B4SOIPpStructPtr->CSC ;

                if ((here-> B4SOIpNode != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOIPgPtr = here->B4SOIPgStructPtr->CSC ;

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIpNode != 0))
                    here->B4SOIGpPtr = here->B4SOIGpStructPtr->CSC ;

            }
            if (here->B4SOIrgateMod != 0)
            {
                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNodeExt != 0))
                    here->B4SOIGEgePtr = here->B4SOIGEgeStructPtr->CSC ;

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOIGEgPtr = here->B4SOIGEgStructPtr->CSC ;

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNodeExt != 0))
                    here->B4SOIGgePtr = here->B4SOIGgeStructPtr->CSC ;

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOIGEdpPtr = here->B4SOIGEdpStructPtr->CSC ;

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOIGEspPtr = here->B4SOIGEspStructPtr->CSC ;

                if (here->B4SOIsoiMod != 2)
                {
                    if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIbNode != 0))
                        here->B4SOIGEbPtr = here->B4SOIGEbStructPtr->CSC ;

                }
                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOIGMdpPtr = here->B4SOIGMdpStructPtr->CSC ;

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOIGMgPtr = here->B4SOIGMgStructPtr->CSC ;

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOIGMgmPtr = here->B4SOIGMgmStructPtr->CSC ;

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNodeExt != 0))
                    here->B4SOIGMgePtr = here->B4SOIGMgeStructPtr->CSC ;

                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOIGMspPtr = here->B4SOIGMspStructPtr->CSC ;

                if (here->B4SOIsoiMod != 2)
                {
                    if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIbNode != 0))
                        here->B4SOIGMbPtr = here->B4SOIGMbStructPtr->CSC ;

                }
                if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIeNode != 0))
                    here->B4SOIGMePtr = here->B4SOIGMeStructPtr->CSC ;

                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOIDPgmPtr = here->B4SOIDPgmStructPtr->CSC ;

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOIGgmPtr = here->B4SOIGgmStructPtr->CSC ;

                if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOIGEgmPtr = here->B4SOIGEgmStructPtr->CSC ;

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOISPgmPtr = here->B4SOISPgmStructPtr->CSC ;

                if ((here-> B4SOIeNode != 0) && (here-> B4SOIgNodeMid != 0))
                    here->B4SOIEgmPtr = here->B4SOIEgmStructPtr->CSC ;

            }
            if (here->B4SOIsoiMod != 2) /* v3.2 */
            {
                if ((here-> B4SOIeNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIEbPtr = here->B4SOIEbStructPtr->CSC ;

                if ((here-> B4SOIgNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIGbPtr = here->B4SOIGbStructPtr->CSC ;

                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIDPbPtr = here->B4SOIDPbStructPtr->CSC ;

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOISPbPtr = here->B4SOISPbStructPtr->CSC ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIeNode != 0))
                    here->B4SOIBePtr = here->B4SOIBeStructPtr->CSC ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOIBgPtr = here->B4SOIBgStructPtr->CSC ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOIBdpPtr = here->B4SOIBdpStructPtr->CSC ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOIBspPtr = here->B4SOIBspStructPtr->CSC ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIBbPtr = here->B4SOIBbStructPtr->CSC ;

            }
            if ((here-> B4SOIeNode != 0) && (here-> B4SOIgNode != 0))
                here->B4SOIEgPtr = here->B4SOIEgStructPtr->CSC ;

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIdNodePrime != 0))
                here->B4SOIEdpPtr = here->B4SOIEdpStructPtr->CSC ;

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIsNodePrime != 0))
                here->B4SOIEspPtr = here->B4SOIEspStructPtr->CSC ;

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIeNode != 0))
                here->B4SOIGePtr = here->B4SOIGeStructPtr->CSC ;

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIeNode != 0))
                here->B4SOIDPePtr = here->B4SOIDPeStructPtr->CSC ;

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIeNode != 0))
                here->B4SOISPePtr = here->B4SOISPeStructPtr->CSC ;

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIbNode != 0))
                here->B4SOIEbPtr = here->B4SOIEbStructPtr->CSC ;

            if ((here-> B4SOIeNode != 0) && (here-> B4SOIeNode != 0))
                here->B4SOIEePtr = here->B4SOIEeStructPtr->CSC ;

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNode != 0))
                here->B4SOIGgPtr = here->B4SOIGgStructPtr->CSC ;

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIdNodePrime != 0))
                here->B4SOIGdpPtr = here->B4SOIGdpStructPtr->CSC ;

            if ((here-> B4SOIgNode != 0) && (here-> B4SOIsNodePrime != 0))
                here->B4SOIGspPtr = here->B4SOIGspStructPtr->CSC ;

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIgNode != 0))
                here->B4SOIDPgPtr = here->B4SOIDPgStructPtr->CSC ;

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdNodePrime != 0))
                here->B4SOIDPdpPtr = here->B4SOIDPdpStructPtr->CSC ;

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIsNodePrime != 0))
                here->B4SOIDPspPtr = here->B4SOIDPspStructPtr->CSC ;

            if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdNode != 0))
                here->B4SOIDPdPtr = here->B4SOIDPdStructPtr->CSC ;

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIgNode != 0))
                here->B4SOISPgPtr = here->B4SOISPgStructPtr->CSC ;

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIdNodePrime != 0))
                here->B4SOISPdpPtr = here->B4SOISPdpStructPtr->CSC ;

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsNodePrime != 0))
                here->B4SOISPspPtr = here->B4SOISPspStructPtr->CSC ;

            if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsNode != 0))
                here->B4SOISPsPtr = here->B4SOISPsStructPtr->CSC ;

            if ((here-> B4SOIdNode != 0) && (here-> B4SOIdNode != 0))
                here->B4SOIDdPtr = here->B4SOIDdStructPtr->CSC ;

            if ((here-> B4SOIdNode != 0) && (here-> B4SOIdNodePrime != 0))
                here->B4SOIDdpPtr = here->B4SOIDdpStructPtr->CSC ;

            if ((here-> B4SOIsNode != 0) && (here-> B4SOIsNode != 0))
                here->B4SOISsPtr = here->B4SOISsStructPtr->CSC ;

            if ((here-> B4SOIsNode != 0) && (here-> B4SOIsNodePrime != 0))
                here->B4SOISspPtr = here->B4SOISspStructPtr->CSC ;

            if (here->B4SOIrbodyMod == 1)
            {
                if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdbNode != 0))
                    here->B4SOIDPdbPtr = here->B4SOIDPdbStructPtr->CSC ;

                if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsbNode != 0))
                    here->B4SOISPsbPtr = here->B4SOISPsbStructPtr->CSC ;

                if ((here-> B4SOIdbNode != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOIDBdpPtr = here->B4SOIDBdpStructPtr->CSC ;

                if ((here-> B4SOIdbNode != 0) && (here-> B4SOIdbNode != 0))
                    here->B4SOIDBdbPtr = here->B4SOIDBdbStructPtr->CSC ;

                if ((here-> B4SOIdbNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOIDBbPtr = here->B4SOIDBbStructPtr->CSC ;

                if ((here-> B4SOIsbNode != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOISBspPtr = here->B4SOISBspStructPtr->CSC ;

                if ((here-> B4SOIsbNode != 0) && (here-> B4SOIsbNode != 0))
                    here->B4SOISBsbPtr = here->B4SOISBsbStructPtr->CSC ;

                if ((here-> B4SOIsbNode != 0) && (here-> B4SOIbNode != 0))
                    here->B4SOISBbPtr = here->B4SOISBbStructPtr->CSC ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIdbNode != 0))
                    here->B4SOIBdbPtr = here->B4SOIBdbStructPtr->CSC ;

                if ((here-> B4SOIbNode != 0) && (here-> B4SOIsbNode != 0))
                    here->B4SOIBsbPtr = here->B4SOIBsbStructPtr->CSC ;

            }
            if (model->B4SOIrdsMod)
            {
                if ((here-> B4SOIdNode != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOIDgPtr = here->B4SOIDgStructPtr->CSC ;

                if ((here-> B4SOIdNode != 0) && (here-> B4SOIsNodePrime != 0))
                    here->B4SOIDspPtr = here->B4SOIDspStructPtr->CSC ;

                if ((here-> B4SOIsNode != 0) && (here-> B4SOIdNodePrime != 0))
                    here->B4SOISdpPtr = here->B4SOISdpStructPtr->CSC ;

                if ((here-> B4SOIsNode != 0) && (here-> B4SOIgNode != 0))
                    here->B4SOISgPtr = here->B4SOISgStructPtr->CSC ;

                if (model->B4SOIsoiMod != 2)
                {
                    if ((here-> B4SOIdNode != 0) && (here-> B4SOIbNode != 0))
                        here->B4SOIDbPtr = here->B4SOIDbStructPtr->CSC ;

                    if ((here-> B4SOIsNode != 0) && (here-> B4SOIbNode != 0))
                        here->B4SOISbPtr = here->B4SOISbStructPtr->CSC ;

                }
            }
            if (here->B4SOIdebugMod != 0)
            {
                if ((here-> B4SOIvbsNode != 0) && (here-> B4SOIvbsNode != 0))
                    here->B4SOIVbsPtr = here->B4SOIVbsStructPtr->CSC ;

                if ((here-> B4SOIidsNode != 0) && (here-> B4SOIidsNode != 0))
                    here->B4SOIIdsPtr = here->B4SOIIdsStructPtr->CSC ;

                if ((here-> B4SOIicNode != 0) && (here-> B4SOIicNode != 0))
                    here->B4SOIIcPtr = here->B4SOIIcStructPtr->CSC ;

                if ((here-> B4SOIibsNode != 0) && (here-> B4SOIibsNode != 0))
                    here->B4SOIIbsPtr = here->B4SOIIbsStructPtr->CSC ;

                if ((here-> B4SOIibdNode != 0) && (here-> B4SOIibdNode != 0))
                    here->B4SOIIbdPtr = here->B4SOIIbdStructPtr->CSC ;

                if ((here-> B4SOIiiiNode != 0) && (here-> B4SOIiiiNode != 0))
                    here->B4SOIIiiPtr = here->B4SOIIiiStructPtr->CSC ;

                if ((here-> B4SOIigNode != 0) && (here-> B4SOIigNode != 0))
                    here->B4SOIIgPtr = here->B4SOIIgStructPtr->CSC ;

                if ((here-> B4SOIgiggNode != 0) && (here-> B4SOIgiggNode != 0))
                    here->B4SOIGiggPtr = here->B4SOIGiggStructPtr->CSC ;

                if ((here-> B4SOIgigdNode != 0) && (here-> B4SOIgigdNode != 0))
                    here->B4SOIGigdPtr = here->B4SOIGigdStructPtr->CSC ;

                if ((here-> B4SOIgigbNode != 0) && (here-> B4SOIgigbNode != 0))
                    here->B4SOIGigbPtr = here->B4SOIGigbStructPtr->CSC ;

                if ((here-> B4SOIigidlNode != 0) && (here-> B4SOIigidlNode != 0))
                    here->B4SOIIgidlPtr = here->B4SOIIgidlStructPtr->CSC ;

                if ((here-> B4SOIitunNode != 0) && (here-> B4SOIitunNode != 0))
                    here->B4SOIItunPtr = here->B4SOIItunStructPtr->CSC ;

                if ((here-> B4SOIibpNode != 0) && (here-> B4SOIibpNode != 0))
                    here->B4SOIIbpPtr = here->B4SOIIbpStructPtr->CSC ;

                if ((here-> B4SOIcbbNode != 0) && (here-> B4SOIcbbNode != 0))
                    here->B4SOICbbPtr = here->B4SOICbbStructPtr->CSC ;

                if ((here-> B4SOIcbdNode != 0) && (here-> B4SOIcbdNode != 0))
                    here->B4SOICbdPtr = here->B4SOICbdStructPtr->CSC ;

                if ((here-> B4SOIcbgNode != 0) && (here-> B4SOIcbgNode != 0))
                    here->B4SOICbgPtr = here->B4SOICbgStructPtr->CSC ;

                if ((here-> B4SOIqbfNode != 0) && (here-> B4SOIqbfNode != 0))
                    here->B4SOIQbfPtr = here->B4SOIQbfStructPtr->CSC ;

                if ((here-> B4SOIqjsNode != 0) && (here-> B4SOIqjsNode != 0))
                    here->B4SOIQjsPtr = here->B4SOIQjsStructPtr->CSC ;

                if ((here-> B4SOIqjdNode != 0) && (here-> B4SOIqjdNode != 0))
                    here->B4SOIQjdPtr = here->B4SOIQjdStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
