/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
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
    for ( ; model != NULL ; model = model->B3SOIDDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIDDinstances ; here != NULL ; here = here->B3SOIDDnextInstance)
        {
            if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0 != 0.0))
            {
                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDtempNode != 0))
                {
                    i = here->B3SOIDDTemptempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDTemptempStructPtr = matched ;
                    here->B3SOIDDTemptempPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                {
                    i = here->B3SOIDDTempdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDTempdpStructPtr = matched ;
                    here->B3SOIDDTempdpPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                {
                    i = here->B3SOIDDTempspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDTempspStructPtr = matched ;
                    here->B3SOIDDTempspPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDgNode != 0))
                {
                    i = here->B3SOIDDTempgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDTempgStructPtr = matched ;
                    here->B3SOIDDTempgPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDbNode != 0))
                {
                    i = here->B3SOIDDTempbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDTempbStructPtr = matched ;
                    here->B3SOIDDTempbPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDeNode != 0))
                {
                    i = here->B3SOIDDTempePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDTempeStructPtr = matched ;
                    here->B3SOIDDTempePtr = matched->CSC ;
                }

                if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDtempNode != 0))
                {
                    i = here->B3SOIDDGtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDGtempStructPtr = matched ;
                    here->B3SOIDDGtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDtempNode != 0))
                {
                    i = here->B3SOIDDDPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDDPtempStructPtr = matched ;
                    here->B3SOIDDDPtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDtempNode != 0))
                {
                    i = here->B3SOIDDSPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDSPtempStructPtr = matched ;
                    here->B3SOIDDSPtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDtempNode != 0))
                {
                    i = here->B3SOIDDEtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDEtempStructPtr = matched ;
                    here->B3SOIDDEtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDtempNode != 0))
                {
                    i = here->B3SOIDDBtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDBtempStructPtr = matched ;
                    here->B3SOIDDBtempPtr = matched->CSC ;
                }

                if (here->B3SOIDDbodyMod == 1)
                {
                    if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDtempNode != 0))
                    {
                        i = here->B3SOIDDPtempPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->B3SOIDDPtempStructPtr = matched ;
                        here->B3SOIDDPtempPtr = matched->CSC ;
                    }

                }
            }
            if (here->B3SOIDDbodyMod == 2)
            {
            }
            else if (here->B3SOIDDbodyMod == 1)
            {
                if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDpNode != 0))
                {
                    i = here->B3SOIDDBpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDBpStructPtr = matched ;
                    here->B3SOIDDBpPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDbNode != 0))
                {
                    i = here->B3SOIDDPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDPbStructPtr = matched ;
                    here->B3SOIDDPbPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDpNode != 0))
                {
                    i = here->B3SOIDDPpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDPpStructPtr = matched ;
                    here->B3SOIDDPpPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDgNode != 0))
                {
                    i = here->B3SOIDDPgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDPgStructPtr = matched ;
                    here->B3SOIDDPgPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                {
                    i = here->B3SOIDDPdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDPdpStructPtr = matched ;
                    here->B3SOIDDPdpPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                {
                    i = here->B3SOIDDPspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDPspStructPtr = matched ;
                    here->B3SOIDDPspPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDeNode != 0))
                {
                    i = here->B3SOIDDPePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDPeStructPtr = matched ;
                    here->B3SOIDDPePtr = matched->CSC ;
                }

            }
            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDgNode != 0))
            {
                i = here->B3SOIDDEgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDEgStructPtr = matched ;
                here->B3SOIDDEgPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDdNodePrime != 0))
            {
                i = here->B3SOIDDEdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDEdpStructPtr = matched ;
                here->B3SOIDDEdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDsNodePrime != 0))
            {
                i = here->B3SOIDDEspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDEspStructPtr = matched ;
                here->B3SOIDDEspPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDeNode != 0))
            {
                i = here->B3SOIDDGePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDGeStructPtr = matched ;
                here->B3SOIDDGePtr = matched->CSC ;
            }

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDeNode != 0))
            {
                i = here->B3SOIDDDPePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDDPeStructPtr = matched ;
                here->B3SOIDDDPePtr = matched->CSC ;
            }

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDeNode != 0))
            {
                i = here->B3SOIDDSPePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDSPeStructPtr = matched ;
                here->B3SOIDDSPePtr = matched->CSC ;
            }

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDbNode != 0))
            {
                i = here->B3SOIDDEbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDEbStructPtr = matched ;
                here->B3SOIDDEbPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDbNode != 0))
            {
                i = here->B3SOIDDGbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDGbStructPtr = matched ;
                here->B3SOIDDGbPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDbNode != 0))
            {
                i = here->B3SOIDDDPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDDPbStructPtr = matched ;
                here->B3SOIDDDPbPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDbNode != 0))
            {
                i = here->B3SOIDDSPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDSPbStructPtr = matched ;
                here->B3SOIDDSPbPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDeNode != 0))
            {
                i = here->B3SOIDDBePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDBeStructPtr = matched ;
                here->B3SOIDDBePtr = matched->CSC ;
            }

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDgNode != 0))
            {
                i = here->B3SOIDDBgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDBgStructPtr = matched ;
                here->B3SOIDDBgPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDdNodePrime != 0))
            {
                i = here->B3SOIDDBdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDBdpStructPtr = matched ;
                here->B3SOIDDBdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDsNodePrime != 0))
            {
                i = here->B3SOIDDBspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDBspStructPtr = matched ;
                here->B3SOIDDBspPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDbNode != 0))
            {
                i = here->B3SOIDDBbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDBbStructPtr = matched ;
                here->B3SOIDDBbPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDeNode != 0))
            {
                i = here->B3SOIDDEePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDEeStructPtr = matched ;
                here->B3SOIDDEePtr = matched->CSC ;
            }

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDgNode != 0))
            {
                i = here->B3SOIDDGgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDGgStructPtr = matched ;
                here->B3SOIDDGgPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDdNodePrime != 0))
            {
                i = here->B3SOIDDGdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDGdpStructPtr = matched ;
                here->B3SOIDDGdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDsNodePrime != 0))
            {
                i = here->B3SOIDDGspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDGspStructPtr = matched ;
                here->B3SOIDDGspPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDgNode != 0))
            {
                i = here->B3SOIDDDPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDDPgStructPtr = matched ;
                here->B3SOIDDDPgPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDdNodePrime != 0))
            {
                i = here->B3SOIDDDPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDDPdpStructPtr = matched ;
                here->B3SOIDDDPdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDsNodePrime != 0))
            {
                i = here->B3SOIDDDPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDDPspStructPtr = matched ;
                here->B3SOIDDDPspPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDdNode != 0))
            {
                i = here->B3SOIDDDPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDDPdStructPtr = matched ;
                here->B3SOIDDDPdPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDgNode != 0))
            {
                i = here->B3SOIDDSPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDSPgStructPtr = matched ;
                here->B3SOIDDSPgPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDdNodePrime != 0))
            {
                i = here->B3SOIDDSPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDSPdpStructPtr = matched ;
                here->B3SOIDDSPdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDsNodePrime != 0))
            {
                i = here->B3SOIDDSPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDSPspStructPtr = matched ;
                here->B3SOIDDSPspPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDsNode != 0))
            {
                i = here->B3SOIDDSPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDSPsStructPtr = matched ;
                here->B3SOIDDSPsPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDdNode != 0) && (here-> B3SOIDDdNode != 0))
            {
                i = here->B3SOIDDDdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDDdStructPtr = matched ;
                here->B3SOIDDDdPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDdNode != 0) && (here-> B3SOIDDdNodePrime != 0))
            {
                i = here->B3SOIDDDdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDDdpStructPtr = matched ;
                here->B3SOIDDDdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDsNode != 0) && (here-> B3SOIDDsNode != 0))
            {
                i = here->B3SOIDDSsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDSsStructPtr = matched ;
                here->B3SOIDDSsPtr = matched->CSC ;
            }

            if ((here-> B3SOIDDsNode != 0) && (here-> B3SOIDDsNodePrime != 0))
            {
                i = here->B3SOIDDSspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIDDSspStructPtr = matched ;
                here->B3SOIDDSspPtr = matched->CSC ;
            }

            if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
            {
                if ((here-> B3SOIDDvbsNode != 0) && (here-> B3SOIDDvbsNode != 0))
                {
                    i = here->B3SOIDDVbsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDVbsStructPtr = matched ;
                    here->B3SOIDDVbsPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDidsNode != 0) && (here-> B3SOIDDidsNode != 0))
                {
                    i = here->B3SOIDDIdsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDIdsStructPtr = matched ;
                    here->B3SOIDDIdsPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDicNode != 0) && (here-> B3SOIDDicNode != 0))
                {
                    i = here->B3SOIDDIcPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDIcStructPtr = matched ;
                    here->B3SOIDDIcPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDibsNode != 0) && (here-> B3SOIDDibsNode != 0))
                {
                    i = here->B3SOIDDIbsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDIbsStructPtr = matched ;
                    here->B3SOIDDIbsPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDibdNode != 0) && (here-> B3SOIDDibdNode != 0))
                {
                    i = here->B3SOIDDIbdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDIbdStructPtr = matched ;
                    here->B3SOIDDIbdPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDiiiNode != 0) && (here-> B3SOIDDiiiNode != 0))
                {
                    i = here->B3SOIDDIiiPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDIiiStructPtr = matched ;
                    here->B3SOIDDIiiPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDigidlNode != 0) && (here-> B3SOIDDigidlNode != 0))
                {
                    i = here->B3SOIDDIgidlPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDIgidlStructPtr = matched ;
                    here->B3SOIDDIgidlPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDitunNode != 0) && (here-> B3SOIDDitunNode != 0))
                {
                    i = here->B3SOIDDItunPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDItunStructPtr = matched ;
                    here->B3SOIDDItunPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDibpNode != 0) && (here-> B3SOIDDibpNode != 0))
                {
                    i = here->B3SOIDDIbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDIbpStructPtr = matched ;
                    here->B3SOIDDIbpPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDabeffNode != 0) && (here-> B3SOIDDabeffNode != 0))
                {
                    i = here->B3SOIDDAbeffPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDAbeffStructPtr = matched ;
                    here->B3SOIDDAbeffPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDvbs0effNode != 0) && (here-> B3SOIDDvbs0effNode != 0))
                {
                    i = here->B3SOIDDVbs0effPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDVbs0effStructPtr = matched ;
                    here->B3SOIDDVbs0effPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDvbseffNode != 0) && (here-> B3SOIDDvbseffNode != 0))
                {
                    i = here->B3SOIDDVbseffPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDVbseffStructPtr = matched ;
                    here->B3SOIDDVbseffPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDxcNode != 0) && (here-> B3SOIDDxcNode != 0))
                {
                    i = here->B3SOIDDXcPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDXcStructPtr = matched ;
                    here->B3SOIDDXcPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDcbbNode != 0) && (here-> B3SOIDDcbbNode != 0))
                {
                    i = here->B3SOIDDCbbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDCbbStructPtr = matched ;
                    here->B3SOIDDCbbPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDcbdNode != 0) && (here-> B3SOIDDcbdNode != 0))
                {
                    i = here->B3SOIDDCbdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDCbdStructPtr = matched ;
                    here->B3SOIDDCbdPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDcbgNode != 0) && (here-> B3SOIDDcbgNode != 0))
                {
                    i = here->B3SOIDDCbgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDCbgStructPtr = matched ;
                    here->B3SOIDDCbgPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDqbNode != 0) && (here-> B3SOIDDqbNode != 0))
                {
                    i = here->B3SOIDDqbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDqbStructPtr = matched ;
                    here->B3SOIDDqbPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDqbfNode != 0) && (here-> B3SOIDDqbfNode != 0))
                {
                    i = here->B3SOIDDQbfPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDQbfStructPtr = matched ;
                    here->B3SOIDDQbfPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDqjsNode != 0) && (here-> B3SOIDDqjsNode != 0))
                {
                    i = here->B3SOIDDQjsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDQjsStructPtr = matched ;
                    here->B3SOIDDQjsPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDqjdNode != 0) && (here-> B3SOIDDqjdNode != 0))
                {
                    i = here->B3SOIDDQjdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDQjdStructPtr = matched ;
                    here->B3SOIDDQjdPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDgmNode != 0) && (here-> B3SOIDDgmNode != 0))
                {
                    i = here->B3SOIDDGmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDGmStructPtr = matched ;
                    here->B3SOIDDGmPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDgmbsNode != 0) && (here-> B3SOIDDgmbsNode != 0))
                {
                    i = here->B3SOIDDGmbsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDGmbsStructPtr = matched ;
                    here->B3SOIDDGmbsPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDgdsNode != 0) && (here-> B3SOIDDgdsNode != 0))
                {
                    i = here->B3SOIDDGdsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDGdsStructPtr = matched ;
                    here->B3SOIDDGdsPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDgmeNode != 0) && (here-> B3SOIDDgmeNode != 0))
                {
                    i = here->B3SOIDDGmePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDGmeStructPtr = matched ;
                    here->B3SOIDDGmePtr = matched->CSC ;
                }

                if ((here-> B3SOIDDvbs0teffNode != 0) && (here-> B3SOIDDvbs0teffNode != 0))
                {
                    i = here->B3SOIDDVbs0teffPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDVbs0teffStructPtr = matched ;
                    here->B3SOIDDVbs0teffPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDvthNode != 0) && (here-> B3SOIDDvthNode != 0))
                {
                    i = here->B3SOIDDVthPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDVthStructPtr = matched ;
                    here->B3SOIDDVthPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDvgsteffNode != 0) && (here-> B3SOIDDvgsteffNode != 0))
                {
                    i = here->B3SOIDDVgsteffPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDVgsteffStructPtr = matched ;
                    here->B3SOIDDVgsteffPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDxcsatNode != 0) && (here-> B3SOIDDxcsatNode != 0))
                {
                    i = here->B3SOIDDXcsatPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDXcsatStructPtr = matched ;
                    here->B3SOIDDXcsatPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDvcscvNode != 0) && (here-> B3SOIDDvcscvNode != 0))
                {
                    i = here->B3SOIDDVcscvPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDVcscvStructPtr = matched ;
                    here->B3SOIDDVcscvPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDvdscvNode != 0) && (here-> B3SOIDDvdscvNode != 0))
                {
                    i = here->B3SOIDDVdscvPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDVdscvStructPtr = matched ;
                    here->B3SOIDDVdscvPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDcbeNode != 0) && (here-> B3SOIDDcbeNode != 0))
                {
                    i = here->B3SOIDDCbePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDCbeStructPtr = matched ;
                    here->B3SOIDDCbePtr = matched->CSC ;
                }

                if ((here-> B3SOIDDdum1Node != 0) && (here-> B3SOIDDdum1Node != 0))
                {
                    i = here->B3SOIDDDum1Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDDum1StructPtr = matched ;
                    here->B3SOIDDDum1Ptr = matched->CSC ;
                }

                if ((here-> B3SOIDDdum2Node != 0) && (here-> B3SOIDDdum2Node != 0))
                {
                    i = here->B3SOIDDDum2Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDDum2StructPtr = matched ;
                    here->B3SOIDDDum2Ptr = matched->CSC ;
                }

                if ((here-> B3SOIDDdum3Node != 0) && (here-> B3SOIDDdum3Node != 0))
                {
                    i = here->B3SOIDDDum3Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDDum3StructPtr = matched ;
                    here->B3SOIDDDum3Ptr = matched->CSC ;
                }

                if ((here-> B3SOIDDdum4Node != 0) && (here-> B3SOIDDdum4Node != 0))
                {
                    i = here->B3SOIDDDum4Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDDum4StructPtr = matched ;
                    here->B3SOIDDDum4Ptr = matched->CSC ;
                }

                if ((here-> B3SOIDDdum5Node != 0) && (here-> B3SOIDDdum5Node != 0))
                {
                    i = here->B3SOIDDDum5Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDDum5StructPtr = matched ;
                    here->B3SOIDDDum5Ptr = matched->CSC ;
                }

                if ((here-> B3SOIDDqaccNode != 0) && (here-> B3SOIDDqaccNode != 0))
                {
                    i = here->B3SOIDDQaccPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDQaccStructPtr = matched ;
                    here->B3SOIDDQaccPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDqsub0Node != 0) && (here-> B3SOIDDqsub0Node != 0))
                {
                    i = here->B3SOIDDQsub0Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDQsub0StructPtr = matched ;
                    here->B3SOIDDQsub0Ptr = matched->CSC ;
                }

                if ((here-> B3SOIDDqsubs1Node != 0) && (here-> B3SOIDDqsubs1Node != 0))
                {
                    i = here->B3SOIDDQsubs1Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDQsubs1StructPtr = matched ;
                    here->B3SOIDDQsubs1Ptr = matched->CSC ;
                }

                if ((here-> B3SOIDDqsubs2Node != 0) && (here-> B3SOIDDqsubs2Node != 0))
                {
                    i = here->B3SOIDDQsubs2Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDQsubs2StructPtr = matched ;
                    here->B3SOIDDQsubs2Ptr = matched->CSC ;
                }

                if ((here-> B3SOIDDqeNode != 0) && (here-> B3SOIDDqeNode != 0))
                {
                    i = here->B3SOIDDqePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDqeStructPtr = matched ;
                    here->B3SOIDDqePtr = matched->CSC ;
                }

                if ((here-> B3SOIDDqdNode != 0) && (here-> B3SOIDDqdNode != 0))
                {
                    i = here->B3SOIDDqdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDqdStructPtr = matched ;
                    here->B3SOIDDqdPtr = matched->CSC ;
                }

                if ((here-> B3SOIDDqgNode != 0) && (here-> B3SOIDDqgNode != 0))
                {
                    i = here->B3SOIDDqgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIDDqgStructPtr = matched ;
                    here->B3SOIDDqgPtr = matched->CSC ;
                }

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
    for ( ; model != NULL ; model = model->B3SOIDDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIDDinstances ; here != NULL ; here = here->B3SOIDDnextInstance)
        {
            if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0 != 0.0))
            {
                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDTemptempPtr = here->B3SOIDDTemptempStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                    here->B3SOIDDTempdpPtr = here->B3SOIDDTempdpStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                    here->B3SOIDDTempspPtr = here->B3SOIDDTempspStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDgNode != 0))
                    here->B3SOIDDTempgPtr = here->B3SOIDDTempgStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDbNode != 0))
                    here->B3SOIDDTempbPtr = here->B3SOIDDTempbStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDeNode != 0))
                    here->B3SOIDDTempePtr = here->B3SOIDDTempeStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDGtempPtr = here->B3SOIDDGtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDDPtempPtr = here->B3SOIDDDPtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDSPtempPtr = here->B3SOIDDSPtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDEtempPtr = here->B3SOIDDEtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDBtempPtr = here->B3SOIDDBtempStructPtr->CSC_Complex ;

                if (here->B3SOIDDbodyMod == 1)
                {
                    if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDtempNode != 0))
                        here->B3SOIDDPtempPtr = here->B3SOIDDPtempStructPtr->CSC_Complex ;

                }
            }
            if (here->B3SOIDDbodyMod == 2)
            {
            }
            else if (here->B3SOIDDbodyMod == 1)
            {
                if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDpNode != 0))
                    here->B3SOIDDBpPtr = here->B3SOIDDBpStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDbNode != 0))
                    here->B3SOIDDPbPtr = here->B3SOIDDPbStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDpNode != 0))
                    here->B3SOIDDPpPtr = here->B3SOIDDPpStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDgNode != 0))
                    here->B3SOIDDPgPtr = here->B3SOIDDPgStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                    here->B3SOIDDPdpPtr = here->B3SOIDDPdpStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                    here->B3SOIDDPspPtr = here->B3SOIDDPspStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDeNode != 0))
                    here->B3SOIDDPePtr = here->B3SOIDDPeStructPtr->CSC_Complex ;

            }
            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDgNode != 0))
                here->B3SOIDDEgPtr = here->B3SOIDDEgStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDEdpPtr = here->B3SOIDDEdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDEspPtr = here->B3SOIDDEspStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDeNode != 0))
                here->B3SOIDDGePtr = here->B3SOIDDGeStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDeNode != 0))
                here->B3SOIDDDPePtr = here->B3SOIDDDPeStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDeNode != 0))
                here->B3SOIDDSPePtr = here->B3SOIDDSPeStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDbNode != 0))
                here->B3SOIDDEbPtr = here->B3SOIDDEbStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDbNode != 0))
                here->B3SOIDDGbPtr = here->B3SOIDDGbStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDbNode != 0))
                here->B3SOIDDDPbPtr = here->B3SOIDDDPbStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDbNode != 0))
                here->B3SOIDDSPbPtr = here->B3SOIDDSPbStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDeNode != 0))
                here->B3SOIDDBePtr = here->B3SOIDDBeStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDgNode != 0))
                here->B3SOIDDBgPtr = here->B3SOIDDBgStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDBdpPtr = here->B3SOIDDBdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDBspPtr = here->B3SOIDDBspStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDbNode != 0))
                here->B3SOIDDBbPtr = here->B3SOIDDBbStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDeNode != 0))
                here->B3SOIDDEePtr = here->B3SOIDDEeStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDgNode != 0))
                here->B3SOIDDGgPtr = here->B3SOIDDGgStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDGdpPtr = here->B3SOIDDGdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDGspPtr = here->B3SOIDDGspStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDgNode != 0))
                here->B3SOIDDDPgPtr = here->B3SOIDDDPgStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDDPdpPtr = here->B3SOIDDDPdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDDPspPtr = here->B3SOIDDDPspStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDdNode != 0))
                here->B3SOIDDDPdPtr = here->B3SOIDDDPdStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDgNode != 0))
                here->B3SOIDDSPgPtr = here->B3SOIDDSPgStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDSPdpPtr = here->B3SOIDDSPdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDSPspPtr = here->B3SOIDDSPspStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDsNode != 0))
                here->B3SOIDDSPsPtr = here->B3SOIDDSPsStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDdNode != 0) && (here-> B3SOIDDdNode != 0))
                here->B3SOIDDDdPtr = here->B3SOIDDDdStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDdNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDDdpPtr = here->B3SOIDDDdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDsNode != 0) && (here-> B3SOIDDsNode != 0))
                here->B3SOIDDSsPtr = here->B3SOIDDSsStructPtr->CSC_Complex ;

            if ((here-> B3SOIDDsNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDSspPtr = here->B3SOIDDSspStructPtr->CSC_Complex ;

            if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
            {
                if ((here-> B3SOIDDvbsNode != 0) && (here-> B3SOIDDvbsNode != 0))
                    here->B3SOIDDVbsPtr = here->B3SOIDDVbsStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDidsNode != 0) && (here-> B3SOIDDidsNode != 0))
                    here->B3SOIDDIdsPtr = here->B3SOIDDIdsStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDicNode != 0) && (here-> B3SOIDDicNode != 0))
                    here->B3SOIDDIcPtr = here->B3SOIDDIcStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDibsNode != 0) && (here-> B3SOIDDibsNode != 0))
                    here->B3SOIDDIbsPtr = here->B3SOIDDIbsStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDibdNode != 0) && (here-> B3SOIDDibdNode != 0))
                    here->B3SOIDDIbdPtr = here->B3SOIDDIbdStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDiiiNode != 0) && (here-> B3SOIDDiiiNode != 0))
                    here->B3SOIDDIiiPtr = here->B3SOIDDIiiStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDigidlNode != 0) && (here-> B3SOIDDigidlNode != 0))
                    here->B3SOIDDIgidlPtr = here->B3SOIDDIgidlStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDitunNode != 0) && (here-> B3SOIDDitunNode != 0))
                    here->B3SOIDDItunPtr = here->B3SOIDDItunStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDibpNode != 0) && (here-> B3SOIDDibpNode != 0))
                    here->B3SOIDDIbpPtr = here->B3SOIDDIbpStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDabeffNode != 0) && (here-> B3SOIDDabeffNode != 0))
                    here->B3SOIDDAbeffPtr = here->B3SOIDDAbeffStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDvbs0effNode != 0) && (here-> B3SOIDDvbs0effNode != 0))
                    here->B3SOIDDVbs0effPtr = here->B3SOIDDVbs0effStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDvbseffNode != 0) && (here-> B3SOIDDvbseffNode != 0))
                    here->B3SOIDDVbseffPtr = here->B3SOIDDVbseffStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDxcNode != 0) && (here-> B3SOIDDxcNode != 0))
                    here->B3SOIDDXcPtr = here->B3SOIDDXcStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDcbbNode != 0) && (here-> B3SOIDDcbbNode != 0))
                    here->B3SOIDDCbbPtr = here->B3SOIDDCbbStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDcbdNode != 0) && (here-> B3SOIDDcbdNode != 0))
                    here->B3SOIDDCbdPtr = here->B3SOIDDCbdStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDcbgNode != 0) && (here-> B3SOIDDcbgNode != 0))
                    here->B3SOIDDCbgPtr = here->B3SOIDDCbgStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqbNode != 0) && (here-> B3SOIDDqbNode != 0))
                    here->B3SOIDDqbPtr = here->B3SOIDDqbStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqbfNode != 0) && (here-> B3SOIDDqbfNode != 0))
                    here->B3SOIDDQbfPtr = here->B3SOIDDQbfStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqjsNode != 0) && (here-> B3SOIDDqjsNode != 0))
                    here->B3SOIDDQjsPtr = here->B3SOIDDQjsStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqjdNode != 0) && (here-> B3SOIDDqjdNode != 0))
                    here->B3SOIDDQjdPtr = here->B3SOIDDQjdStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDgmNode != 0) && (here-> B3SOIDDgmNode != 0))
                    here->B3SOIDDGmPtr = here->B3SOIDDGmStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDgmbsNode != 0) && (here-> B3SOIDDgmbsNode != 0))
                    here->B3SOIDDGmbsPtr = here->B3SOIDDGmbsStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDgdsNode != 0) && (here-> B3SOIDDgdsNode != 0))
                    here->B3SOIDDGdsPtr = here->B3SOIDDGdsStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDgmeNode != 0) && (here-> B3SOIDDgmeNode != 0))
                    here->B3SOIDDGmePtr = here->B3SOIDDGmeStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDvbs0teffNode != 0) && (here-> B3SOIDDvbs0teffNode != 0))
                    here->B3SOIDDVbs0teffPtr = here->B3SOIDDVbs0teffStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDvthNode != 0) && (here-> B3SOIDDvthNode != 0))
                    here->B3SOIDDVthPtr = here->B3SOIDDVthStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDvgsteffNode != 0) && (here-> B3SOIDDvgsteffNode != 0))
                    here->B3SOIDDVgsteffPtr = here->B3SOIDDVgsteffStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDxcsatNode != 0) && (here-> B3SOIDDxcsatNode != 0))
                    here->B3SOIDDXcsatPtr = here->B3SOIDDXcsatStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDvcscvNode != 0) && (here-> B3SOIDDvcscvNode != 0))
                    here->B3SOIDDVcscvPtr = here->B3SOIDDVcscvStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDvdscvNode != 0) && (here-> B3SOIDDvdscvNode != 0))
                    here->B3SOIDDVdscvPtr = here->B3SOIDDVdscvStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDcbeNode != 0) && (here-> B3SOIDDcbeNode != 0))
                    here->B3SOIDDCbePtr = here->B3SOIDDCbeStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDdum1Node != 0) && (here-> B3SOIDDdum1Node != 0))
                    here->B3SOIDDDum1Ptr = here->B3SOIDDDum1StructPtr->CSC_Complex ;

                if ((here-> B3SOIDDdum2Node != 0) && (here-> B3SOIDDdum2Node != 0))
                    here->B3SOIDDDum2Ptr = here->B3SOIDDDum2StructPtr->CSC_Complex ;

                if ((here-> B3SOIDDdum3Node != 0) && (here-> B3SOIDDdum3Node != 0))
                    here->B3SOIDDDum3Ptr = here->B3SOIDDDum3StructPtr->CSC_Complex ;

                if ((here-> B3SOIDDdum4Node != 0) && (here-> B3SOIDDdum4Node != 0))
                    here->B3SOIDDDum4Ptr = here->B3SOIDDDum4StructPtr->CSC_Complex ;

                if ((here-> B3SOIDDdum5Node != 0) && (here-> B3SOIDDdum5Node != 0))
                    here->B3SOIDDDum5Ptr = here->B3SOIDDDum5StructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqaccNode != 0) && (here-> B3SOIDDqaccNode != 0))
                    here->B3SOIDDQaccPtr = here->B3SOIDDQaccStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqsub0Node != 0) && (here-> B3SOIDDqsub0Node != 0))
                    here->B3SOIDDQsub0Ptr = here->B3SOIDDQsub0StructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqsubs1Node != 0) && (here-> B3SOIDDqsubs1Node != 0))
                    here->B3SOIDDQsubs1Ptr = here->B3SOIDDQsubs1StructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqsubs2Node != 0) && (here-> B3SOIDDqsubs2Node != 0))
                    here->B3SOIDDQsubs2Ptr = here->B3SOIDDQsubs2StructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqeNode != 0) && (here-> B3SOIDDqeNode != 0))
                    here->B3SOIDDqePtr = here->B3SOIDDqeStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqdNode != 0) && (here-> B3SOIDDqdNode != 0))
                    here->B3SOIDDqdPtr = here->B3SOIDDqdStructPtr->CSC_Complex ;

                if ((here-> B3SOIDDqgNode != 0) && (here-> B3SOIDDqgNode != 0))
                    here->B3SOIDDqgPtr = here->B3SOIDDqgStructPtr->CSC_Complex ;

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
    for ( ; model != NULL ; model = model->B3SOIDDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIDDinstances ; here != NULL ; here = here->B3SOIDDnextInstance)
        {
            if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0 != 0.0))
            {
                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDTemptempPtr = here->B3SOIDDTemptempStructPtr->CSC ;

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                    here->B3SOIDDTempdpPtr = here->B3SOIDDTempdpStructPtr->CSC ;

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                    here->B3SOIDDTempspPtr = here->B3SOIDDTempspStructPtr->CSC ;

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDgNode != 0))
                    here->B3SOIDDTempgPtr = here->B3SOIDDTempgStructPtr->CSC ;

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDbNode != 0))
                    here->B3SOIDDTempbPtr = here->B3SOIDDTempbStructPtr->CSC ;

                if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDeNode != 0))
                    here->B3SOIDDTempePtr = here->B3SOIDDTempeStructPtr->CSC ;

                if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDGtempPtr = here->B3SOIDDGtempStructPtr->CSC ;

                if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDDPtempPtr = here->B3SOIDDDPtempStructPtr->CSC ;

                if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDSPtempPtr = here->B3SOIDDSPtempStructPtr->CSC ;

                if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDEtempPtr = here->B3SOIDDEtempStructPtr->CSC ;

                if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDtempNode != 0))
                    here->B3SOIDDBtempPtr = here->B3SOIDDBtempStructPtr->CSC ;

                if (here->B3SOIDDbodyMod == 1)
                {
                    if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDtempNode != 0))
                        here->B3SOIDDPtempPtr = here->B3SOIDDPtempStructPtr->CSC ;

                }
            }
            if (here->B3SOIDDbodyMod == 2)
            {
            }
            else if (here->B3SOIDDbodyMod == 1)
            {
                if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDpNode != 0))
                    here->B3SOIDDBpPtr = here->B3SOIDDBpStructPtr->CSC ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDbNode != 0))
                    here->B3SOIDDPbPtr = here->B3SOIDDPbStructPtr->CSC ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDpNode != 0))
                    here->B3SOIDDPpPtr = here->B3SOIDDPpStructPtr->CSC ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDgNode != 0))
                    here->B3SOIDDPgPtr = here->B3SOIDDPgStructPtr->CSC ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                    here->B3SOIDDPdpPtr = here->B3SOIDDPdpStructPtr->CSC ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                    here->B3SOIDDPspPtr = here->B3SOIDDPspStructPtr->CSC ;

                if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDeNode != 0))
                    here->B3SOIDDPePtr = here->B3SOIDDPeStructPtr->CSC ;

            }
            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDgNode != 0))
                here->B3SOIDDEgPtr = here->B3SOIDDEgStructPtr->CSC ;

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDEdpPtr = here->B3SOIDDEdpStructPtr->CSC ;

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDEspPtr = here->B3SOIDDEspStructPtr->CSC ;

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDeNode != 0))
                here->B3SOIDDGePtr = here->B3SOIDDGeStructPtr->CSC ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDeNode != 0))
                here->B3SOIDDDPePtr = here->B3SOIDDDPeStructPtr->CSC ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDeNode != 0))
                here->B3SOIDDSPePtr = here->B3SOIDDSPeStructPtr->CSC ;

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDbNode != 0))
                here->B3SOIDDEbPtr = here->B3SOIDDEbStructPtr->CSC ;

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDbNode != 0))
                here->B3SOIDDGbPtr = here->B3SOIDDGbStructPtr->CSC ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDbNode != 0))
                here->B3SOIDDDPbPtr = here->B3SOIDDDPbStructPtr->CSC ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDbNode != 0))
                here->B3SOIDDSPbPtr = here->B3SOIDDSPbStructPtr->CSC ;

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDeNode != 0))
                here->B3SOIDDBePtr = here->B3SOIDDBeStructPtr->CSC ;

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDgNode != 0))
                here->B3SOIDDBgPtr = here->B3SOIDDBgStructPtr->CSC ;

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDBdpPtr = here->B3SOIDDBdpStructPtr->CSC ;

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDBspPtr = here->B3SOIDDBspStructPtr->CSC ;

            if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDbNode != 0))
                here->B3SOIDDBbPtr = here->B3SOIDDBbStructPtr->CSC ;

            if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDeNode != 0))
                here->B3SOIDDEePtr = here->B3SOIDDEeStructPtr->CSC ;

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDgNode != 0))
                here->B3SOIDDGgPtr = here->B3SOIDDGgStructPtr->CSC ;

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDGdpPtr = here->B3SOIDDGdpStructPtr->CSC ;

            if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDGspPtr = here->B3SOIDDGspStructPtr->CSC ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDgNode != 0))
                here->B3SOIDDDPgPtr = here->B3SOIDDDPgStructPtr->CSC ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDDPdpPtr = here->B3SOIDDDPdpStructPtr->CSC ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDDPspPtr = here->B3SOIDDDPspStructPtr->CSC ;

            if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDdNode != 0))
                here->B3SOIDDDPdPtr = here->B3SOIDDDPdStructPtr->CSC ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDgNode != 0))
                here->B3SOIDDSPgPtr = here->B3SOIDDSPgStructPtr->CSC ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDSPdpPtr = here->B3SOIDDSPdpStructPtr->CSC ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDSPspPtr = here->B3SOIDDSPspStructPtr->CSC ;

            if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDsNode != 0))
                here->B3SOIDDSPsPtr = here->B3SOIDDSPsStructPtr->CSC ;

            if ((here-> B3SOIDDdNode != 0) && (here-> B3SOIDDdNode != 0))
                here->B3SOIDDDdPtr = here->B3SOIDDDdStructPtr->CSC ;

            if ((here-> B3SOIDDdNode != 0) && (here-> B3SOIDDdNodePrime != 0))
                here->B3SOIDDDdpPtr = here->B3SOIDDDdpStructPtr->CSC ;

            if ((here-> B3SOIDDsNode != 0) && (here-> B3SOIDDsNode != 0))
                here->B3SOIDDSsPtr = here->B3SOIDDSsStructPtr->CSC ;

            if ((here-> B3SOIDDsNode != 0) && (here-> B3SOIDDsNodePrime != 0))
                here->B3SOIDDSspPtr = here->B3SOIDDSspStructPtr->CSC ;

            if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
            {
                if ((here-> B3SOIDDvbsNode != 0) && (here-> B3SOIDDvbsNode != 0))
                    here->B3SOIDDVbsPtr = here->B3SOIDDVbsStructPtr->CSC ;

                if ((here-> B3SOIDDidsNode != 0) && (here-> B3SOIDDidsNode != 0))
                    here->B3SOIDDIdsPtr = here->B3SOIDDIdsStructPtr->CSC ;

                if ((here-> B3SOIDDicNode != 0) && (here-> B3SOIDDicNode != 0))
                    here->B3SOIDDIcPtr = here->B3SOIDDIcStructPtr->CSC ;

                if ((here-> B3SOIDDibsNode != 0) && (here-> B3SOIDDibsNode != 0))
                    here->B3SOIDDIbsPtr = here->B3SOIDDIbsStructPtr->CSC ;

                if ((here-> B3SOIDDibdNode != 0) && (here-> B3SOIDDibdNode != 0))
                    here->B3SOIDDIbdPtr = here->B3SOIDDIbdStructPtr->CSC ;

                if ((here-> B3SOIDDiiiNode != 0) && (here-> B3SOIDDiiiNode != 0))
                    here->B3SOIDDIiiPtr = here->B3SOIDDIiiStructPtr->CSC ;

                if ((here-> B3SOIDDigidlNode != 0) && (here-> B3SOIDDigidlNode != 0))
                    here->B3SOIDDIgidlPtr = here->B3SOIDDIgidlStructPtr->CSC ;

                if ((here-> B3SOIDDitunNode != 0) && (here-> B3SOIDDitunNode != 0))
                    here->B3SOIDDItunPtr = here->B3SOIDDItunStructPtr->CSC ;

                if ((here-> B3SOIDDibpNode != 0) && (here-> B3SOIDDibpNode != 0))
                    here->B3SOIDDIbpPtr = here->B3SOIDDIbpStructPtr->CSC ;

                if ((here-> B3SOIDDabeffNode != 0) && (here-> B3SOIDDabeffNode != 0))
                    here->B3SOIDDAbeffPtr = here->B3SOIDDAbeffStructPtr->CSC ;

                if ((here-> B3SOIDDvbs0effNode != 0) && (here-> B3SOIDDvbs0effNode != 0))
                    here->B3SOIDDVbs0effPtr = here->B3SOIDDVbs0effStructPtr->CSC ;

                if ((here-> B3SOIDDvbseffNode != 0) && (here-> B3SOIDDvbseffNode != 0))
                    here->B3SOIDDVbseffPtr = here->B3SOIDDVbseffStructPtr->CSC ;

                if ((here-> B3SOIDDxcNode != 0) && (here-> B3SOIDDxcNode != 0))
                    here->B3SOIDDXcPtr = here->B3SOIDDXcStructPtr->CSC ;

                if ((here-> B3SOIDDcbbNode != 0) && (here-> B3SOIDDcbbNode != 0))
                    here->B3SOIDDCbbPtr = here->B3SOIDDCbbStructPtr->CSC ;

                if ((here-> B3SOIDDcbdNode != 0) && (here-> B3SOIDDcbdNode != 0))
                    here->B3SOIDDCbdPtr = here->B3SOIDDCbdStructPtr->CSC ;

                if ((here-> B3SOIDDcbgNode != 0) && (here-> B3SOIDDcbgNode != 0))
                    here->B3SOIDDCbgPtr = here->B3SOIDDCbgStructPtr->CSC ;

                if ((here-> B3SOIDDqbNode != 0) && (here-> B3SOIDDqbNode != 0))
                    here->B3SOIDDqbPtr = here->B3SOIDDqbStructPtr->CSC ;

                if ((here-> B3SOIDDqbfNode != 0) && (here-> B3SOIDDqbfNode != 0))
                    here->B3SOIDDQbfPtr = here->B3SOIDDQbfStructPtr->CSC ;

                if ((here-> B3SOIDDqjsNode != 0) && (here-> B3SOIDDqjsNode != 0))
                    here->B3SOIDDQjsPtr = here->B3SOIDDQjsStructPtr->CSC ;

                if ((here-> B3SOIDDqjdNode != 0) && (here-> B3SOIDDqjdNode != 0))
                    here->B3SOIDDQjdPtr = here->B3SOIDDQjdStructPtr->CSC ;

                if ((here-> B3SOIDDgmNode != 0) && (here-> B3SOIDDgmNode != 0))
                    here->B3SOIDDGmPtr = here->B3SOIDDGmStructPtr->CSC ;

                if ((here-> B3SOIDDgmbsNode != 0) && (here-> B3SOIDDgmbsNode != 0))
                    here->B3SOIDDGmbsPtr = here->B3SOIDDGmbsStructPtr->CSC ;

                if ((here-> B3SOIDDgdsNode != 0) && (here-> B3SOIDDgdsNode != 0))
                    here->B3SOIDDGdsPtr = here->B3SOIDDGdsStructPtr->CSC ;

                if ((here-> B3SOIDDgmeNode != 0) && (here-> B3SOIDDgmeNode != 0))
                    here->B3SOIDDGmePtr = here->B3SOIDDGmeStructPtr->CSC ;

                if ((here-> B3SOIDDvbs0teffNode != 0) && (here-> B3SOIDDvbs0teffNode != 0))
                    here->B3SOIDDVbs0teffPtr = here->B3SOIDDVbs0teffStructPtr->CSC ;

                if ((here-> B3SOIDDvthNode != 0) && (here-> B3SOIDDvthNode != 0))
                    here->B3SOIDDVthPtr = here->B3SOIDDVthStructPtr->CSC ;

                if ((here-> B3SOIDDvgsteffNode != 0) && (here-> B3SOIDDvgsteffNode != 0))
                    here->B3SOIDDVgsteffPtr = here->B3SOIDDVgsteffStructPtr->CSC ;

                if ((here-> B3SOIDDxcsatNode != 0) && (here-> B3SOIDDxcsatNode != 0))
                    here->B3SOIDDXcsatPtr = here->B3SOIDDXcsatStructPtr->CSC ;

                if ((here-> B3SOIDDvcscvNode != 0) && (here-> B3SOIDDvcscvNode != 0))
                    here->B3SOIDDVcscvPtr = here->B3SOIDDVcscvStructPtr->CSC ;

                if ((here-> B3SOIDDvdscvNode != 0) && (here-> B3SOIDDvdscvNode != 0))
                    here->B3SOIDDVdscvPtr = here->B3SOIDDVdscvStructPtr->CSC ;

                if ((here-> B3SOIDDcbeNode != 0) && (here-> B3SOIDDcbeNode != 0))
                    here->B3SOIDDCbePtr = here->B3SOIDDCbeStructPtr->CSC ;

                if ((here-> B3SOIDDdum1Node != 0) && (here-> B3SOIDDdum1Node != 0))
                    here->B3SOIDDDum1Ptr = here->B3SOIDDDum1StructPtr->CSC ;

                if ((here-> B3SOIDDdum2Node != 0) && (here-> B3SOIDDdum2Node != 0))
                    here->B3SOIDDDum2Ptr = here->B3SOIDDDum2StructPtr->CSC ;

                if ((here-> B3SOIDDdum3Node != 0) && (here-> B3SOIDDdum3Node != 0))
                    here->B3SOIDDDum3Ptr = here->B3SOIDDDum3StructPtr->CSC ;

                if ((here-> B3SOIDDdum4Node != 0) && (here-> B3SOIDDdum4Node != 0))
                    here->B3SOIDDDum4Ptr = here->B3SOIDDDum4StructPtr->CSC ;

                if ((here-> B3SOIDDdum5Node != 0) && (here-> B3SOIDDdum5Node != 0))
                    here->B3SOIDDDum5Ptr = here->B3SOIDDDum5StructPtr->CSC ;

                if ((here-> B3SOIDDqaccNode != 0) && (here-> B3SOIDDqaccNode != 0))
                    here->B3SOIDDQaccPtr = here->B3SOIDDQaccStructPtr->CSC ;

                if ((here-> B3SOIDDqsub0Node != 0) && (here-> B3SOIDDqsub0Node != 0))
                    here->B3SOIDDQsub0Ptr = here->B3SOIDDQsub0StructPtr->CSC ;

                if ((here-> B3SOIDDqsubs1Node != 0) && (here-> B3SOIDDqsubs1Node != 0))
                    here->B3SOIDDQsubs1Ptr = here->B3SOIDDQsubs1StructPtr->CSC ;

                if ((here-> B3SOIDDqsubs2Node != 0) && (here-> B3SOIDDqsubs2Node != 0))
                    here->B3SOIDDQsubs2Ptr = here->B3SOIDDQsubs2StructPtr->CSC ;

                if ((here-> B3SOIDDqeNode != 0) && (here-> B3SOIDDqeNode != 0))
                    here->B3SOIDDqePtr = here->B3SOIDDqeStructPtr->CSC ;

                if ((here-> B3SOIDDqdNode != 0) && (here-> B3SOIDDqdNode != 0))
                    here->B3SOIDDqdPtr = here->B3SOIDDqdStructPtr->CSC ;

                if ((here-> B3SOIDDqgNode != 0) && (here-> B3SOIDDqgNode != 0))
                    here->B3SOIDDqgPtr = here->B3SOIDDqgStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
