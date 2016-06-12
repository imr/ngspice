/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soifddef.h"
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
    for ( ; model != NULL ; model = model->B3SOIFDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIFDinstances ; here != NULL ; here = here->B3SOIFDnextInstance)
        {
            if ((model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0 != 0.0))
            {
                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDtempNode != 0))
                {
                    i = here->B3SOIFDTemptempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDTemptempStructPtr = matched ;
                    here->B3SOIFDTemptempPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                {
                    i = here->B3SOIFDTempdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDTempdpStructPtr = matched ;
                    here->B3SOIFDTempdpPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                {
                    i = here->B3SOIFDTempspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDTempspStructPtr = matched ;
                    here->B3SOIFDTempspPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDgNode != 0))
                {
                    i = here->B3SOIFDTempgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDTempgStructPtr = matched ;
                    here->B3SOIFDTempgPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDbNode != 0))
                {
                    i = here->B3SOIFDTempbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDTempbStructPtr = matched ;
                    here->B3SOIFDTempbPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDeNode != 0))
                {
                    i = here->B3SOIFDTempePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDTempeStructPtr = matched ;
                    here->B3SOIFDTempePtr = matched->CSC ;
                }

                if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDtempNode != 0))
                {
                    i = here->B3SOIFDGtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDGtempStructPtr = matched ;
                    here->B3SOIFDGtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDtempNode != 0))
                {
                    i = here->B3SOIFDDPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDDPtempStructPtr = matched ;
                    here->B3SOIFDDPtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDtempNode != 0))
                {
                    i = here->B3SOIFDSPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDSPtempStructPtr = matched ;
                    here->B3SOIFDSPtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDtempNode != 0))
                {
                    i = here->B3SOIFDEtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDEtempStructPtr = matched ;
                    here->B3SOIFDEtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDbNode != 0) && (here-> B3SOIFDtempNode != 0))
                {
                    i = here->B3SOIFDBtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDBtempStructPtr = matched ;
                    here->B3SOIFDBtempPtr = matched->CSC ;
                }

                if (here->B3SOIFDbodyMod == 1)
                {
                    if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDtempNode != 0))
                    {
                        i = here->B3SOIFDPtempPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->B3SOIFDPtempStructPtr = matched ;
                        here->B3SOIFDPtempPtr = matched->CSC ;
                    }

                }
            }
            if (here->B3SOIFDbodyMod == 2)
            {
            }
            else if (here->B3SOIFDbodyMod == 1)
            {
                if ((here-> B3SOIFDbNode != 0) && (here-> B3SOIFDpNode != 0))
                {
                    i = here->B3SOIFDBpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDBpStructPtr = matched ;
                    here->B3SOIFDBpPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDbNode != 0))
                {
                    i = here->B3SOIFDPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDPbStructPtr = matched ;
                    here->B3SOIFDPbPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDpNode != 0))
                {
                    i = here->B3SOIFDPpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDPpStructPtr = matched ;
                    here->B3SOIFDPpPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDgNode != 0))
                {
                    i = here->B3SOIFDPgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDPgStructPtr = matched ;
                    here->B3SOIFDPgPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                {
                    i = here->B3SOIFDPdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDPdpStructPtr = matched ;
                    here->B3SOIFDPdpPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                {
                    i = here->B3SOIFDPspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDPspStructPtr = matched ;
                    here->B3SOIFDPspPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDeNode != 0))
                {
                    i = here->B3SOIFDPePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDPeStructPtr = matched ;
                    here->B3SOIFDPePtr = matched->CSC ;
                }

            }
            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDgNode != 0))
            {
                i = here->B3SOIFDEgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDEgStructPtr = matched ;
                here->B3SOIFDEgPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDdNodePrime != 0))
            {
                i = here->B3SOIFDEdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDEdpStructPtr = matched ;
                here->B3SOIFDEdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDsNodePrime != 0))
            {
                i = here->B3SOIFDEspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDEspStructPtr = matched ;
                here->B3SOIFDEspPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDeNode != 0))
            {
                i = here->B3SOIFDGePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDGeStructPtr = matched ;
                here->B3SOIFDGePtr = matched->CSC ;
            }

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDeNode != 0))
            {
                i = here->B3SOIFDDPePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDDPeStructPtr = matched ;
                here->B3SOIFDDPePtr = matched->CSC ;
            }

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDeNode != 0))
            {
                i = here->B3SOIFDSPePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDSPeStructPtr = matched ;
                here->B3SOIFDSPePtr = matched->CSC ;
            }

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDbNode != 0))
            {
                i = here->B3SOIFDEbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDEbStructPtr = matched ;
                here->B3SOIFDEbPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDeNode != 0))
            {
                i = here->B3SOIFDEePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDEeStructPtr = matched ;
                here->B3SOIFDEePtr = matched->CSC ;
            }

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDgNode != 0))
            {
                i = here->B3SOIFDGgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDGgStructPtr = matched ;
                here->B3SOIFDGgPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDdNodePrime != 0))
            {
                i = here->B3SOIFDGdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDGdpStructPtr = matched ;
                here->B3SOIFDGdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDsNodePrime != 0))
            {
                i = here->B3SOIFDGspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDGspStructPtr = matched ;
                here->B3SOIFDGspPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDgNode != 0))
            {
                i = here->B3SOIFDDPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDDPgStructPtr = matched ;
                here->B3SOIFDDPgPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDdNodePrime != 0))
            {
                i = here->B3SOIFDDPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDDPdpStructPtr = matched ;
                here->B3SOIFDDPdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDsNodePrime != 0))
            {
                i = here->B3SOIFDDPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDDPspStructPtr = matched ;
                here->B3SOIFDDPspPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDdNode != 0))
            {
                i = here->B3SOIFDDPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDDPdStructPtr = matched ;
                here->B3SOIFDDPdPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDgNode != 0))
            {
                i = here->B3SOIFDSPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDSPgStructPtr = matched ;
                here->B3SOIFDSPgPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDdNodePrime != 0))
            {
                i = here->B3SOIFDSPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDSPdpStructPtr = matched ;
                here->B3SOIFDSPdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDsNodePrime != 0))
            {
                i = here->B3SOIFDSPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDSPspStructPtr = matched ;
                here->B3SOIFDSPspPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDsNode != 0))
            {
                i = here->B3SOIFDSPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDSPsStructPtr = matched ;
                here->B3SOIFDSPsPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDdNode != 0) && (here-> B3SOIFDdNode != 0))
            {
                i = here->B3SOIFDDdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDDdStructPtr = matched ;
                here->B3SOIFDDdPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDdNode != 0) && (here-> B3SOIFDdNodePrime != 0))
            {
                i = here->B3SOIFDDdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDDdpStructPtr = matched ;
                here->B3SOIFDDdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDsNode != 0) && (here-> B3SOIFDsNode != 0))
            {
                i = here->B3SOIFDSsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDSsStructPtr = matched ;
                here->B3SOIFDSsPtr = matched->CSC ;
            }

            if ((here-> B3SOIFDsNode != 0) && (here-> B3SOIFDsNodePrime != 0))
            {
                i = here->B3SOIFDSspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIFDSspStructPtr = matched ;
                here->B3SOIFDSspPtr = matched->CSC ;
            }

            if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1))
            {
                if ((here-> B3SOIFDvbsNode != 0) && (here-> B3SOIFDvbsNode != 0))
                {
                    i = here->B3SOIFDVbsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDVbsStructPtr = matched ;
                    here->B3SOIFDVbsPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDidsNode != 0) && (here-> B3SOIFDidsNode != 0))
                {
                    i = here->B3SOIFDIdsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDIdsStructPtr = matched ;
                    here->B3SOIFDIdsPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDicNode != 0) && (here-> B3SOIFDicNode != 0))
                {
                    i = here->B3SOIFDIcPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDIcStructPtr = matched ;
                    here->B3SOIFDIcPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDibsNode != 0) && (here-> B3SOIFDibsNode != 0))
                {
                    i = here->B3SOIFDIbsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDIbsStructPtr = matched ;
                    here->B3SOIFDIbsPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDibdNode != 0) && (here-> B3SOIFDibdNode != 0))
                {
                    i = here->B3SOIFDIbdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDIbdStructPtr = matched ;
                    here->B3SOIFDIbdPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDiiiNode != 0) && (here-> B3SOIFDiiiNode != 0))
                {
                    i = here->B3SOIFDIiiPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDIiiStructPtr = matched ;
                    here->B3SOIFDIiiPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDigidlNode != 0) && (here-> B3SOIFDigidlNode != 0))
                {
                    i = here->B3SOIFDIgidlPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDIgidlStructPtr = matched ;
                    here->B3SOIFDIgidlPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDitunNode != 0) && (here-> B3SOIFDitunNode != 0))
                {
                    i = here->B3SOIFDItunPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDItunStructPtr = matched ;
                    here->B3SOIFDItunPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDibpNode != 0) && (here-> B3SOIFDibpNode != 0))
                {
                    i = here->B3SOIFDIbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDIbpStructPtr = matched ;
                    here->B3SOIFDIbpPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDabeffNode != 0) && (here-> B3SOIFDabeffNode != 0))
                {
                    i = here->B3SOIFDAbeffPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDAbeffStructPtr = matched ;
                    here->B3SOIFDAbeffPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDvbs0effNode != 0) && (here-> B3SOIFDvbs0effNode != 0))
                {
                    i = here->B3SOIFDVbs0effPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDVbs0effStructPtr = matched ;
                    here->B3SOIFDVbs0effPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDvbseffNode != 0) && (here-> B3SOIFDvbseffNode != 0))
                {
                    i = here->B3SOIFDVbseffPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDVbseffStructPtr = matched ;
                    here->B3SOIFDVbseffPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDxcNode != 0) && (here-> B3SOIFDxcNode != 0))
                {
                    i = here->B3SOIFDXcPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDXcStructPtr = matched ;
                    here->B3SOIFDXcPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDcbbNode != 0) && (here-> B3SOIFDcbbNode != 0))
                {
                    i = here->B3SOIFDCbbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDCbbStructPtr = matched ;
                    here->B3SOIFDCbbPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDcbdNode != 0) && (here-> B3SOIFDcbdNode != 0))
                {
                    i = here->B3SOIFDCbdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDCbdStructPtr = matched ;
                    here->B3SOIFDCbdPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDcbgNode != 0) && (here-> B3SOIFDcbgNode != 0))
                {
                    i = here->B3SOIFDCbgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDCbgStructPtr = matched ;
                    here->B3SOIFDCbgPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDqbNode != 0) && (here-> B3SOIFDqbNode != 0))
                {
                    i = here->B3SOIFDqbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDqbStructPtr = matched ;
                    here->B3SOIFDqbPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDqbfNode != 0) && (here-> B3SOIFDqbfNode != 0))
                {
                    i = here->B3SOIFDQbfPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDQbfStructPtr = matched ;
                    here->B3SOIFDQbfPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDqjsNode != 0) && (here-> B3SOIFDqjsNode != 0))
                {
                    i = here->B3SOIFDQjsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDQjsStructPtr = matched ;
                    here->B3SOIFDQjsPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDqjdNode != 0) && (here-> B3SOIFDqjdNode != 0))
                {
                    i = here->B3SOIFDQjdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDQjdStructPtr = matched ;
                    here->B3SOIFDQjdPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDgmNode != 0) && (here-> B3SOIFDgmNode != 0))
                {
                    i = here->B3SOIFDGmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDGmStructPtr = matched ;
                    here->B3SOIFDGmPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDgmbsNode != 0) && (here-> B3SOIFDgmbsNode != 0))
                {
                    i = here->B3SOIFDGmbsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDGmbsStructPtr = matched ;
                    here->B3SOIFDGmbsPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDgdsNode != 0) && (here-> B3SOIFDgdsNode != 0))
                {
                    i = here->B3SOIFDGdsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDGdsStructPtr = matched ;
                    here->B3SOIFDGdsPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDgmeNode != 0) && (here-> B3SOIFDgmeNode != 0))
                {
                    i = here->B3SOIFDGmePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDGmeStructPtr = matched ;
                    here->B3SOIFDGmePtr = matched->CSC ;
                }

                if ((here-> B3SOIFDvbs0teffNode != 0) && (here-> B3SOIFDvbs0teffNode != 0))
                {
                    i = here->B3SOIFDVbs0teffPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDVbs0teffStructPtr = matched ;
                    here->B3SOIFDVbs0teffPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDvthNode != 0) && (here-> B3SOIFDvthNode != 0))
                {
                    i = here->B3SOIFDVthPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDVthStructPtr = matched ;
                    here->B3SOIFDVthPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDvgsteffNode != 0) && (here-> B3SOIFDvgsteffNode != 0))
                {
                    i = here->B3SOIFDVgsteffPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDVgsteffStructPtr = matched ;
                    here->B3SOIFDVgsteffPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDxcsatNode != 0) && (here-> B3SOIFDxcsatNode != 0))
                {
                    i = here->B3SOIFDXcsatPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDXcsatStructPtr = matched ;
                    here->B3SOIFDXcsatPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDvcscvNode != 0) && (here-> B3SOIFDvcscvNode != 0))
                {
                    i = here->B3SOIFDVcscvPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDVcscvStructPtr = matched ;
                    here->B3SOIFDVcscvPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDvdscvNode != 0) && (here-> B3SOIFDvdscvNode != 0))
                {
                    i = here->B3SOIFDVdscvPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDVdscvStructPtr = matched ;
                    here->B3SOIFDVdscvPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDcbeNode != 0) && (here-> B3SOIFDcbeNode != 0))
                {
                    i = here->B3SOIFDCbePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDCbeStructPtr = matched ;
                    here->B3SOIFDCbePtr = matched->CSC ;
                }

                if ((here-> B3SOIFDdum1Node != 0) && (here-> B3SOIFDdum1Node != 0))
                {
                    i = here->B3SOIFDDum1Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDDum1StructPtr = matched ;
                    here->B3SOIFDDum1Ptr = matched->CSC ;
                }

                if ((here-> B3SOIFDdum2Node != 0) && (here-> B3SOIFDdum2Node != 0))
                {
                    i = here->B3SOIFDDum2Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDDum2StructPtr = matched ;
                    here->B3SOIFDDum2Ptr = matched->CSC ;
                }

                if ((here-> B3SOIFDdum3Node != 0) && (here-> B3SOIFDdum3Node != 0))
                {
                    i = here->B3SOIFDDum3Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDDum3StructPtr = matched ;
                    here->B3SOIFDDum3Ptr = matched->CSC ;
                }

                if ((here-> B3SOIFDdum4Node != 0) && (here-> B3SOIFDdum4Node != 0))
                {
                    i = here->B3SOIFDDum4Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDDum4StructPtr = matched ;
                    here->B3SOIFDDum4Ptr = matched->CSC ;
                }

                if ((here-> B3SOIFDdum5Node != 0) && (here-> B3SOIFDdum5Node != 0))
                {
                    i = here->B3SOIFDDum5Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDDum5StructPtr = matched ;
                    here->B3SOIFDDum5Ptr = matched->CSC ;
                }

                if ((here-> B3SOIFDqaccNode != 0) && (here-> B3SOIFDqaccNode != 0))
                {
                    i = here->B3SOIFDQaccPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDQaccStructPtr = matched ;
                    here->B3SOIFDQaccPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDqsub0Node != 0) && (here-> B3SOIFDqsub0Node != 0))
                {
                    i = here->B3SOIFDQsub0Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDQsub0StructPtr = matched ;
                    here->B3SOIFDQsub0Ptr = matched->CSC ;
                }

                if ((here-> B3SOIFDqsubs1Node != 0) && (here-> B3SOIFDqsubs1Node != 0))
                {
                    i = here->B3SOIFDQsubs1Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDQsubs1StructPtr = matched ;
                    here->B3SOIFDQsubs1Ptr = matched->CSC ;
                }

                if ((here-> B3SOIFDqsubs2Node != 0) && (here-> B3SOIFDqsubs2Node != 0))
                {
                    i = here->B3SOIFDQsubs2Ptr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDQsubs2StructPtr = matched ;
                    here->B3SOIFDQsubs2Ptr = matched->CSC ;
                }

                if ((here-> B3SOIFDqeNode != 0) && (here-> B3SOIFDqeNode != 0))
                {
                    i = here->B3SOIFDqePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDqeStructPtr = matched ;
                    here->B3SOIFDqePtr = matched->CSC ;
                }

                if ((here-> B3SOIFDqdNode != 0) && (here-> B3SOIFDqdNode != 0))
                {
                    i = here->B3SOIFDqdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDqdStructPtr = matched ;
                    here->B3SOIFDqdPtr = matched->CSC ;
                }

                if ((here-> B3SOIFDqgNode != 0) && (here-> B3SOIFDqgNode != 0))
                {
                    i = here->B3SOIFDqgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIFDqgStructPtr = matched ;
                    here->B3SOIFDqgPtr = matched->CSC ;
                }

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
    for ( ; model != NULL ; model = model->B3SOIFDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIFDinstances ; here != NULL ; here = here->B3SOIFDnextInstance)
        {
            if ((model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0 != 0.0))
            {
                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDTemptempPtr = here->B3SOIFDTemptempStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                    here->B3SOIFDTempdpPtr = here->B3SOIFDTempdpStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                    here->B3SOIFDTempspPtr = here->B3SOIFDTempspStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDgNode != 0))
                    here->B3SOIFDTempgPtr = here->B3SOIFDTempgStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDbNode != 0))
                    here->B3SOIFDTempbPtr = here->B3SOIFDTempbStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDeNode != 0))
                    here->B3SOIFDTempePtr = here->B3SOIFDTempeStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDGtempPtr = here->B3SOIFDGtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDDPtempPtr = here->B3SOIFDDPtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDSPtempPtr = here->B3SOIFDSPtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDEtempPtr = here->B3SOIFDEtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDbNode != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDBtempPtr = here->B3SOIFDBtempStructPtr->CSC_Complex ;

                if (here->B3SOIFDbodyMod == 1)
                {
                    if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDtempNode != 0))
                        here->B3SOIFDPtempPtr = here->B3SOIFDPtempStructPtr->CSC_Complex ;

                }
            }
            if (here->B3SOIFDbodyMod == 2)
            {
            }
            else if (here->B3SOIFDbodyMod == 1)
            {
                if ((here-> B3SOIFDbNode != 0) && (here-> B3SOIFDpNode != 0))
                    here->B3SOIFDBpPtr = here->B3SOIFDBpStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDbNode != 0))
                    here->B3SOIFDPbPtr = here->B3SOIFDPbStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDpNode != 0))
                    here->B3SOIFDPpPtr = here->B3SOIFDPpStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDgNode != 0))
                    here->B3SOIFDPgPtr = here->B3SOIFDPgStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                    here->B3SOIFDPdpPtr = here->B3SOIFDPdpStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                    here->B3SOIFDPspPtr = here->B3SOIFDPspStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDeNode != 0))
                    here->B3SOIFDPePtr = here->B3SOIFDPeStructPtr->CSC_Complex ;

            }
            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDgNode != 0))
                here->B3SOIFDEgPtr = here->B3SOIFDEgStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                here->B3SOIFDEdpPtr = here->B3SOIFDEdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                here->B3SOIFDEspPtr = here->B3SOIFDEspStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDeNode != 0))
                here->B3SOIFDGePtr = here->B3SOIFDGeStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDeNode != 0))
                here->B3SOIFDDPePtr = here->B3SOIFDDPeStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDeNode != 0))
                here->B3SOIFDSPePtr = here->B3SOIFDSPeStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDbNode != 0))
                here->B3SOIFDEbPtr = here->B3SOIFDEbStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDeNode != 0))
                here->B3SOIFDEePtr = here->B3SOIFDEeStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDgNode != 0))
                here->B3SOIFDGgPtr = here->B3SOIFDGgStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                here->B3SOIFDGdpPtr = here->B3SOIFDGdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                here->B3SOIFDGspPtr = here->B3SOIFDGspStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDgNode != 0))
                here->B3SOIFDDPgPtr = here->B3SOIFDDPgStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDdNodePrime != 0))
                here->B3SOIFDDPdpPtr = here->B3SOIFDDPdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDsNodePrime != 0))
                here->B3SOIFDDPspPtr = here->B3SOIFDDPspStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDdNode != 0))
                here->B3SOIFDDPdPtr = here->B3SOIFDDPdStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDgNode != 0))
                here->B3SOIFDSPgPtr = here->B3SOIFDSPgStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDdNodePrime != 0))
                here->B3SOIFDSPdpPtr = here->B3SOIFDSPdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDsNodePrime != 0))
                here->B3SOIFDSPspPtr = here->B3SOIFDSPspStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDsNode != 0))
                here->B3SOIFDSPsPtr = here->B3SOIFDSPsStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDdNode != 0) && (here-> B3SOIFDdNode != 0))
                here->B3SOIFDDdPtr = here->B3SOIFDDdStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDdNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                here->B3SOIFDDdpPtr = here->B3SOIFDDdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDsNode != 0) && (here-> B3SOIFDsNode != 0))
                here->B3SOIFDSsPtr = here->B3SOIFDSsStructPtr->CSC_Complex ;

            if ((here-> B3SOIFDsNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                here->B3SOIFDSspPtr = here->B3SOIFDSspStructPtr->CSC_Complex ;

            if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1))
            {
                if ((here-> B3SOIFDvbsNode != 0) && (here-> B3SOIFDvbsNode != 0))
                    here->B3SOIFDVbsPtr = here->B3SOIFDVbsStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDidsNode != 0) && (here-> B3SOIFDidsNode != 0))
                    here->B3SOIFDIdsPtr = here->B3SOIFDIdsStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDicNode != 0) && (here-> B3SOIFDicNode != 0))
                    here->B3SOIFDIcPtr = here->B3SOIFDIcStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDibsNode != 0) && (here-> B3SOIFDibsNode != 0))
                    here->B3SOIFDIbsPtr = here->B3SOIFDIbsStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDibdNode != 0) && (here-> B3SOIFDibdNode != 0))
                    here->B3SOIFDIbdPtr = here->B3SOIFDIbdStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDiiiNode != 0) && (here-> B3SOIFDiiiNode != 0))
                    here->B3SOIFDIiiPtr = here->B3SOIFDIiiStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDigidlNode != 0) && (here-> B3SOIFDigidlNode != 0))
                    here->B3SOIFDIgidlPtr = here->B3SOIFDIgidlStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDitunNode != 0) && (here-> B3SOIFDitunNode != 0))
                    here->B3SOIFDItunPtr = here->B3SOIFDItunStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDibpNode != 0) && (here-> B3SOIFDibpNode != 0))
                    here->B3SOIFDIbpPtr = here->B3SOIFDIbpStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDabeffNode != 0) && (here-> B3SOIFDabeffNode != 0))
                    here->B3SOIFDAbeffPtr = here->B3SOIFDAbeffStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDvbs0effNode != 0) && (here-> B3SOIFDvbs0effNode != 0))
                    here->B3SOIFDVbs0effPtr = here->B3SOIFDVbs0effStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDvbseffNode != 0) && (here-> B3SOIFDvbseffNode != 0))
                    here->B3SOIFDVbseffPtr = here->B3SOIFDVbseffStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDxcNode != 0) && (here-> B3SOIFDxcNode != 0))
                    here->B3SOIFDXcPtr = here->B3SOIFDXcStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDcbbNode != 0) && (here-> B3SOIFDcbbNode != 0))
                    here->B3SOIFDCbbPtr = here->B3SOIFDCbbStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDcbdNode != 0) && (here-> B3SOIFDcbdNode != 0))
                    here->B3SOIFDCbdPtr = here->B3SOIFDCbdStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDcbgNode != 0) && (here-> B3SOIFDcbgNode != 0))
                    here->B3SOIFDCbgPtr = here->B3SOIFDCbgStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqbNode != 0) && (here-> B3SOIFDqbNode != 0))
                    here->B3SOIFDqbPtr = here->B3SOIFDqbStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqbfNode != 0) && (here-> B3SOIFDqbfNode != 0))
                    here->B3SOIFDQbfPtr = here->B3SOIFDQbfStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqjsNode != 0) && (here-> B3SOIFDqjsNode != 0))
                    here->B3SOIFDQjsPtr = here->B3SOIFDQjsStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqjdNode != 0) && (here-> B3SOIFDqjdNode != 0))
                    here->B3SOIFDQjdPtr = here->B3SOIFDQjdStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDgmNode != 0) && (here-> B3SOIFDgmNode != 0))
                    here->B3SOIFDGmPtr = here->B3SOIFDGmStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDgmbsNode != 0) && (here-> B3SOIFDgmbsNode != 0))
                    here->B3SOIFDGmbsPtr = here->B3SOIFDGmbsStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDgdsNode != 0) && (here-> B3SOIFDgdsNode != 0))
                    here->B3SOIFDGdsPtr = here->B3SOIFDGdsStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDgmeNode != 0) && (here-> B3SOIFDgmeNode != 0))
                    here->B3SOIFDGmePtr = here->B3SOIFDGmeStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDvbs0teffNode != 0) && (here-> B3SOIFDvbs0teffNode != 0))
                    here->B3SOIFDVbs0teffPtr = here->B3SOIFDVbs0teffStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDvthNode != 0) && (here-> B3SOIFDvthNode != 0))
                    here->B3SOIFDVthPtr = here->B3SOIFDVthStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDvgsteffNode != 0) && (here-> B3SOIFDvgsteffNode != 0))
                    here->B3SOIFDVgsteffPtr = here->B3SOIFDVgsteffStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDxcsatNode != 0) && (here-> B3SOIFDxcsatNode != 0))
                    here->B3SOIFDXcsatPtr = here->B3SOIFDXcsatStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDvcscvNode != 0) && (here-> B3SOIFDvcscvNode != 0))
                    here->B3SOIFDVcscvPtr = here->B3SOIFDVcscvStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDvdscvNode != 0) && (here-> B3SOIFDvdscvNode != 0))
                    here->B3SOIFDVdscvPtr = here->B3SOIFDVdscvStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDcbeNode != 0) && (here-> B3SOIFDcbeNode != 0))
                    here->B3SOIFDCbePtr = here->B3SOIFDCbeStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDdum1Node != 0) && (here-> B3SOIFDdum1Node != 0))
                    here->B3SOIFDDum1Ptr = here->B3SOIFDDum1StructPtr->CSC_Complex ;

                if ((here-> B3SOIFDdum2Node != 0) && (here-> B3SOIFDdum2Node != 0))
                    here->B3SOIFDDum2Ptr = here->B3SOIFDDum2StructPtr->CSC_Complex ;

                if ((here-> B3SOIFDdum3Node != 0) && (here-> B3SOIFDdum3Node != 0))
                    here->B3SOIFDDum3Ptr = here->B3SOIFDDum3StructPtr->CSC_Complex ;

                if ((here-> B3SOIFDdum4Node != 0) && (here-> B3SOIFDdum4Node != 0))
                    here->B3SOIFDDum4Ptr = here->B3SOIFDDum4StructPtr->CSC_Complex ;

                if ((here-> B3SOIFDdum5Node != 0) && (here-> B3SOIFDdum5Node != 0))
                    here->B3SOIFDDum5Ptr = here->B3SOIFDDum5StructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqaccNode != 0) && (here-> B3SOIFDqaccNode != 0))
                    here->B3SOIFDQaccPtr = here->B3SOIFDQaccStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqsub0Node != 0) && (here-> B3SOIFDqsub0Node != 0))
                    here->B3SOIFDQsub0Ptr = here->B3SOIFDQsub0StructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqsubs1Node != 0) && (here-> B3SOIFDqsubs1Node != 0))
                    here->B3SOIFDQsubs1Ptr = here->B3SOIFDQsubs1StructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqsubs2Node != 0) && (here-> B3SOIFDqsubs2Node != 0))
                    here->B3SOIFDQsubs2Ptr = here->B3SOIFDQsubs2StructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqeNode != 0) && (here-> B3SOIFDqeNode != 0))
                    here->B3SOIFDqePtr = here->B3SOIFDqeStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqdNode != 0) && (here-> B3SOIFDqdNode != 0))
                    here->B3SOIFDqdPtr = here->B3SOIFDqdStructPtr->CSC_Complex ;

                if ((here-> B3SOIFDqgNode != 0) && (here-> B3SOIFDqgNode != 0))
                    here->B3SOIFDqgPtr = here->B3SOIFDqgStructPtr->CSC_Complex ;

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
    for ( ; model != NULL ; model = model->B3SOIFDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIFDinstances ; here != NULL ; here = here->B3SOIFDnextInstance)
        {
            if ((model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0 != 0.0))
            {
                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDTemptempPtr = here->B3SOIFDTemptempStructPtr->CSC ;

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                    here->B3SOIFDTempdpPtr = here->B3SOIFDTempdpStructPtr->CSC ;

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                    here->B3SOIFDTempspPtr = here->B3SOIFDTempspStructPtr->CSC ;

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDgNode != 0))
                    here->B3SOIFDTempgPtr = here->B3SOIFDTempgStructPtr->CSC ;

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDbNode != 0))
                    here->B3SOIFDTempbPtr = here->B3SOIFDTempbStructPtr->CSC ;

                if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDeNode != 0))
                    here->B3SOIFDTempePtr = here->B3SOIFDTempeStructPtr->CSC ;

                if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDGtempPtr = here->B3SOIFDGtempStructPtr->CSC ;

                if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDDPtempPtr = here->B3SOIFDDPtempStructPtr->CSC ;

                if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDSPtempPtr = here->B3SOIFDSPtempStructPtr->CSC ;

                if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDEtempPtr = here->B3SOIFDEtempStructPtr->CSC ;

                if ((here-> B3SOIFDbNode != 0) && (here-> B3SOIFDtempNode != 0))
                    here->B3SOIFDBtempPtr = here->B3SOIFDBtempStructPtr->CSC ;

                if (here->B3SOIFDbodyMod == 1)
                {
                    if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDtempNode != 0))
                        here->B3SOIFDPtempPtr = here->B3SOIFDPtempStructPtr->CSC ;

                }
            }
            if (here->B3SOIFDbodyMod == 2)
            {
            }
            else if (here->B3SOIFDbodyMod == 1)
            {
                if ((here-> B3SOIFDbNode != 0) && (here-> B3SOIFDpNode != 0))
                    here->B3SOIFDBpPtr = here->B3SOIFDBpStructPtr->CSC ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDbNode != 0))
                    here->B3SOIFDPbPtr = here->B3SOIFDPbStructPtr->CSC ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDpNode != 0))
                    here->B3SOIFDPpPtr = here->B3SOIFDPpStructPtr->CSC ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDgNode != 0))
                    here->B3SOIFDPgPtr = here->B3SOIFDPgStructPtr->CSC ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                    here->B3SOIFDPdpPtr = here->B3SOIFDPdpStructPtr->CSC ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                    here->B3SOIFDPspPtr = here->B3SOIFDPspStructPtr->CSC ;

                if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDeNode != 0))
                    here->B3SOIFDPePtr = here->B3SOIFDPeStructPtr->CSC ;

            }
            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDgNode != 0))
                here->B3SOIFDEgPtr = here->B3SOIFDEgStructPtr->CSC ;

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                here->B3SOIFDEdpPtr = here->B3SOIFDEdpStructPtr->CSC ;

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                here->B3SOIFDEspPtr = here->B3SOIFDEspStructPtr->CSC ;

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDeNode != 0))
                here->B3SOIFDGePtr = here->B3SOIFDGeStructPtr->CSC ;

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDeNode != 0))
                here->B3SOIFDDPePtr = here->B3SOIFDDPeStructPtr->CSC ;

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDeNode != 0))
                here->B3SOIFDSPePtr = here->B3SOIFDSPeStructPtr->CSC ;

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDbNode != 0))
                here->B3SOIFDEbPtr = here->B3SOIFDEbStructPtr->CSC ;

            if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDeNode != 0))
                here->B3SOIFDEePtr = here->B3SOIFDEeStructPtr->CSC ;

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDgNode != 0))
                here->B3SOIFDGgPtr = here->B3SOIFDGgStructPtr->CSC ;

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                here->B3SOIFDGdpPtr = here->B3SOIFDGdpStructPtr->CSC ;

            if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                here->B3SOIFDGspPtr = here->B3SOIFDGspStructPtr->CSC ;

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDgNode != 0))
                here->B3SOIFDDPgPtr = here->B3SOIFDDPgStructPtr->CSC ;

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDdNodePrime != 0))
                here->B3SOIFDDPdpPtr = here->B3SOIFDDPdpStructPtr->CSC ;

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDsNodePrime != 0))
                here->B3SOIFDDPspPtr = here->B3SOIFDDPspStructPtr->CSC ;

            if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDdNode != 0))
                here->B3SOIFDDPdPtr = here->B3SOIFDDPdStructPtr->CSC ;

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDgNode != 0))
                here->B3SOIFDSPgPtr = here->B3SOIFDSPgStructPtr->CSC ;

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDdNodePrime != 0))
                here->B3SOIFDSPdpPtr = here->B3SOIFDSPdpStructPtr->CSC ;

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDsNodePrime != 0))
                here->B3SOIFDSPspPtr = here->B3SOIFDSPspStructPtr->CSC ;

            if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDsNode != 0))
                here->B3SOIFDSPsPtr = here->B3SOIFDSPsStructPtr->CSC ;

            if ((here-> B3SOIFDdNode != 0) && (here-> B3SOIFDdNode != 0))
                here->B3SOIFDDdPtr = here->B3SOIFDDdStructPtr->CSC ;

            if ((here-> B3SOIFDdNode != 0) && (here-> B3SOIFDdNodePrime != 0))
                here->B3SOIFDDdpPtr = here->B3SOIFDDdpStructPtr->CSC ;

            if ((here-> B3SOIFDsNode != 0) && (here-> B3SOIFDsNode != 0))
                here->B3SOIFDSsPtr = here->B3SOIFDSsStructPtr->CSC ;

            if ((here-> B3SOIFDsNode != 0) && (here-> B3SOIFDsNodePrime != 0))
                here->B3SOIFDSspPtr = here->B3SOIFDSspStructPtr->CSC ;

            if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1))
            {
                if ((here-> B3SOIFDvbsNode != 0) && (here-> B3SOIFDvbsNode != 0))
                    here->B3SOIFDVbsPtr = here->B3SOIFDVbsStructPtr->CSC ;

                if ((here-> B3SOIFDidsNode != 0) && (here-> B3SOIFDidsNode != 0))
                    here->B3SOIFDIdsPtr = here->B3SOIFDIdsStructPtr->CSC ;

                if ((here-> B3SOIFDicNode != 0) && (here-> B3SOIFDicNode != 0))
                    here->B3SOIFDIcPtr = here->B3SOIFDIcStructPtr->CSC ;

                if ((here-> B3SOIFDibsNode != 0) && (here-> B3SOIFDibsNode != 0))
                    here->B3SOIFDIbsPtr = here->B3SOIFDIbsStructPtr->CSC ;

                if ((here-> B3SOIFDibdNode != 0) && (here-> B3SOIFDibdNode != 0))
                    here->B3SOIFDIbdPtr = here->B3SOIFDIbdStructPtr->CSC ;

                if ((here-> B3SOIFDiiiNode != 0) && (here-> B3SOIFDiiiNode != 0))
                    here->B3SOIFDIiiPtr = here->B3SOIFDIiiStructPtr->CSC ;

                if ((here-> B3SOIFDigidlNode != 0) && (here-> B3SOIFDigidlNode != 0))
                    here->B3SOIFDIgidlPtr = here->B3SOIFDIgidlStructPtr->CSC ;

                if ((here-> B3SOIFDitunNode != 0) && (here-> B3SOIFDitunNode != 0))
                    here->B3SOIFDItunPtr = here->B3SOIFDItunStructPtr->CSC ;

                if ((here-> B3SOIFDibpNode != 0) && (here-> B3SOIFDibpNode != 0))
                    here->B3SOIFDIbpPtr = here->B3SOIFDIbpStructPtr->CSC ;

                if ((here-> B3SOIFDabeffNode != 0) && (here-> B3SOIFDabeffNode != 0))
                    here->B3SOIFDAbeffPtr = here->B3SOIFDAbeffStructPtr->CSC ;

                if ((here-> B3SOIFDvbs0effNode != 0) && (here-> B3SOIFDvbs0effNode != 0))
                    here->B3SOIFDVbs0effPtr = here->B3SOIFDVbs0effStructPtr->CSC ;

                if ((here-> B3SOIFDvbseffNode != 0) && (here-> B3SOIFDvbseffNode != 0))
                    here->B3SOIFDVbseffPtr = here->B3SOIFDVbseffStructPtr->CSC ;

                if ((here-> B3SOIFDxcNode != 0) && (here-> B3SOIFDxcNode != 0))
                    here->B3SOIFDXcPtr = here->B3SOIFDXcStructPtr->CSC ;

                if ((here-> B3SOIFDcbbNode != 0) && (here-> B3SOIFDcbbNode != 0))
                    here->B3SOIFDCbbPtr = here->B3SOIFDCbbStructPtr->CSC ;

                if ((here-> B3SOIFDcbdNode != 0) && (here-> B3SOIFDcbdNode != 0))
                    here->B3SOIFDCbdPtr = here->B3SOIFDCbdStructPtr->CSC ;

                if ((here-> B3SOIFDcbgNode != 0) && (here-> B3SOIFDcbgNode != 0))
                    here->B3SOIFDCbgPtr = here->B3SOIFDCbgStructPtr->CSC ;

                if ((here-> B3SOIFDqbNode != 0) && (here-> B3SOIFDqbNode != 0))
                    here->B3SOIFDqbPtr = here->B3SOIFDqbStructPtr->CSC ;

                if ((here-> B3SOIFDqbfNode != 0) && (here-> B3SOIFDqbfNode != 0))
                    here->B3SOIFDQbfPtr = here->B3SOIFDQbfStructPtr->CSC ;

                if ((here-> B3SOIFDqjsNode != 0) && (here-> B3SOIFDqjsNode != 0))
                    here->B3SOIFDQjsPtr = here->B3SOIFDQjsStructPtr->CSC ;

                if ((here-> B3SOIFDqjdNode != 0) && (here-> B3SOIFDqjdNode != 0))
                    here->B3SOIFDQjdPtr = here->B3SOIFDQjdStructPtr->CSC ;

                if ((here-> B3SOIFDgmNode != 0) && (here-> B3SOIFDgmNode != 0))
                    here->B3SOIFDGmPtr = here->B3SOIFDGmStructPtr->CSC ;

                if ((here-> B3SOIFDgmbsNode != 0) && (here-> B3SOIFDgmbsNode != 0))
                    here->B3SOIFDGmbsPtr = here->B3SOIFDGmbsStructPtr->CSC ;

                if ((here-> B3SOIFDgdsNode != 0) && (here-> B3SOIFDgdsNode != 0))
                    here->B3SOIFDGdsPtr = here->B3SOIFDGdsStructPtr->CSC ;

                if ((here-> B3SOIFDgmeNode != 0) && (here-> B3SOIFDgmeNode != 0))
                    here->B3SOIFDGmePtr = here->B3SOIFDGmeStructPtr->CSC ;

                if ((here-> B3SOIFDvbs0teffNode != 0) && (here-> B3SOIFDvbs0teffNode != 0))
                    here->B3SOIFDVbs0teffPtr = here->B3SOIFDVbs0teffStructPtr->CSC ;

                if ((here-> B3SOIFDvthNode != 0) && (here-> B3SOIFDvthNode != 0))
                    here->B3SOIFDVthPtr = here->B3SOIFDVthStructPtr->CSC ;

                if ((here-> B3SOIFDvgsteffNode != 0) && (here-> B3SOIFDvgsteffNode != 0))
                    here->B3SOIFDVgsteffPtr = here->B3SOIFDVgsteffStructPtr->CSC ;

                if ((here-> B3SOIFDxcsatNode != 0) && (here-> B3SOIFDxcsatNode != 0))
                    here->B3SOIFDXcsatPtr = here->B3SOIFDXcsatStructPtr->CSC ;

                if ((here-> B3SOIFDvcscvNode != 0) && (here-> B3SOIFDvcscvNode != 0))
                    here->B3SOIFDVcscvPtr = here->B3SOIFDVcscvStructPtr->CSC ;

                if ((here-> B3SOIFDvdscvNode != 0) && (here-> B3SOIFDvdscvNode != 0))
                    here->B3SOIFDVdscvPtr = here->B3SOIFDVdscvStructPtr->CSC ;

                if ((here-> B3SOIFDcbeNode != 0) && (here-> B3SOIFDcbeNode != 0))
                    here->B3SOIFDCbePtr = here->B3SOIFDCbeStructPtr->CSC ;

                if ((here-> B3SOIFDdum1Node != 0) && (here-> B3SOIFDdum1Node != 0))
                    here->B3SOIFDDum1Ptr = here->B3SOIFDDum1StructPtr->CSC ;

                if ((here-> B3SOIFDdum2Node != 0) && (here-> B3SOIFDdum2Node != 0))
                    here->B3SOIFDDum2Ptr = here->B3SOIFDDum2StructPtr->CSC ;

                if ((here-> B3SOIFDdum3Node != 0) && (here-> B3SOIFDdum3Node != 0))
                    here->B3SOIFDDum3Ptr = here->B3SOIFDDum3StructPtr->CSC ;

                if ((here-> B3SOIFDdum4Node != 0) && (here-> B3SOIFDdum4Node != 0))
                    here->B3SOIFDDum4Ptr = here->B3SOIFDDum4StructPtr->CSC ;

                if ((here-> B3SOIFDdum5Node != 0) && (here-> B3SOIFDdum5Node != 0))
                    here->B3SOIFDDum5Ptr = here->B3SOIFDDum5StructPtr->CSC ;

                if ((here-> B3SOIFDqaccNode != 0) && (here-> B3SOIFDqaccNode != 0))
                    here->B3SOIFDQaccPtr = here->B3SOIFDQaccStructPtr->CSC ;

                if ((here-> B3SOIFDqsub0Node != 0) && (here-> B3SOIFDqsub0Node != 0))
                    here->B3SOIFDQsub0Ptr = here->B3SOIFDQsub0StructPtr->CSC ;

                if ((here-> B3SOIFDqsubs1Node != 0) && (here-> B3SOIFDqsubs1Node != 0))
                    here->B3SOIFDQsubs1Ptr = here->B3SOIFDQsubs1StructPtr->CSC ;

                if ((here-> B3SOIFDqsubs2Node != 0) && (here-> B3SOIFDqsubs2Node != 0))
                    here->B3SOIFDQsubs2Ptr = here->B3SOIFDQsubs2StructPtr->CSC ;

                if ((here-> B3SOIFDqeNode != 0) && (here-> B3SOIFDqeNode != 0))
                    here->B3SOIFDqePtr = here->B3SOIFDqeStructPtr->CSC ;

                if ((here-> B3SOIFDqdNode != 0) && (here-> B3SOIFDqdNode != 0))
                    here->B3SOIFDqdPtr = here->B3SOIFDqdStructPtr->CSC ;

                if ((here-> B3SOIFDqgNode != 0) && (here-> B3SOIFDqgNode != 0))
                    here->B3SOIFDqgPtr = here->B3SOIFDqgStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
