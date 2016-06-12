/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
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
    for ( ; model != NULL ; model = model->B3SOIPDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIPDinstances ; here != NULL ; here = here->B3SOIPDnextInstance)
        {
            if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0 != 0.0))
            {
                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDtempNode != 0))
                {
                    i = here->B3SOIPDTemptempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDTemptempStructPtr = matched ;
                    here->B3SOIPDTemptempPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                {
                    i = here->B3SOIPDTempdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDTempdpStructPtr = matched ;
                    here->B3SOIPDTempdpPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                {
                    i = here->B3SOIPDTempspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDTempspStructPtr = matched ;
                    here->B3SOIPDTempspPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDgNode != 0))
                {
                    i = here->B3SOIPDTempgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDTempgStructPtr = matched ;
                    here->B3SOIPDTempgPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDbNode != 0))
                {
                    i = here->B3SOIPDTempbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDTempbStructPtr = matched ;
                    here->B3SOIPDTempbPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDtempNode != 0))
                {
                    i = here->B3SOIPDGtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDGtempStructPtr = matched ;
                    here->B3SOIPDGtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDtempNode != 0))
                {
                    i = here->B3SOIPDDPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDDPtempStructPtr = matched ;
                    here->B3SOIPDDPtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDtempNode != 0))
                {
                    i = here->B3SOIPDSPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDSPtempStructPtr = matched ;
                    here->B3SOIPDSPtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDtempNode != 0))
                {
                    i = here->B3SOIPDEtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDEtempStructPtr = matched ;
                    here->B3SOIPDEtempPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDtempNode != 0))
                {
                    i = here->B3SOIPDBtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDBtempStructPtr = matched ;
                    here->B3SOIPDBtempPtr = matched->CSC ;
                }

                if (here->B3SOIPDbodyMod == 1)
                {
                    if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDtempNode != 0))
                    {
                        i = here->B3SOIPDPtempPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->B3SOIPDPtempStructPtr = matched ;
                        here->B3SOIPDPtempPtr = matched->CSC ;
                    }

                }
            }
            if (here->B3SOIPDbodyMod == 2)
            {
            }
            else if (here->B3SOIPDbodyMod == 1)
            {
                if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDpNode != 0))
                {
                    i = here->B3SOIPDBpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDBpStructPtr = matched ;
                    here->B3SOIPDBpPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDbNode != 0))
                {
                    i = here->B3SOIPDPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDPbStructPtr = matched ;
                    here->B3SOIPDPbPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDpNode != 0))
                {
                    i = here->B3SOIPDPpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDPpStructPtr = matched ;
                    here->B3SOIPDPpPtr = matched->CSC ;
                }

            }
            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDbNode != 0))
            {
                i = here->B3SOIPDEbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDEbStructPtr = matched ;
                here->B3SOIPDEbPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDbNode != 0))
            {
                i = here->B3SOIPDGbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDGbStructPtr = matched ;
                here->B3SOIPDGbPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDbNode != 0))
            {
                i = here->B3SOIPDDPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDDPbStructPtr = matched ;
                here->B3SOIPDDPbPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDbNode != 0))
            {
                i = here->B3SOIPDSPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDSPbStructPtr = matched ;
                here->B3SOIPDSPbPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDeNode != 0))
            {
                i = here->B3SOIPDBePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDBeStructPtr = matched ;
                here->B3SOIPDBePtr = matched->CSC ;
            }

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDgNode != 0))
            {
                i = here->B3SOIPDBgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDBgStructPtr = matched ;
                here->B3SOIPDBgPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDdNodePrime != 0))
            {
                i = here->B3SOIPDBdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDBdpStructPtr = matched ;
                here->B3SOIPDBdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDsNodePrime != 0))
            {
                i = here->B3SOIPDBspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDBspStructPtr = matched ;
                here->B3SOIPDBspPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDbNode != 0))
            {
                i = here->B3SOIPDBbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDBbStructPtr = matched ;
                here->B3SOIPDBbPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDgNode != 0))
            {
                i = here->B3SOIPDEgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDEgStructPtr = matched ;
                here->B3SOIPDEgPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDdNodePrime != 0))
            {
                i = here->B3SOIPDEdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDEdpStructPtr = matched ;
                here->B3SOIPDEdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDsNodePrime != 0))
            {
                i = here->B3SOIPDEspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDEspStructPtr = matched ;
                here->B3SOIPDEspPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDeNode != 0))
            {
                i = here->B3SOIPDGePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDGeStructPtr = matched ;
                here->B3SOIPDGePtr = matched->CSC ;
            }

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDeNode != 0))
            {
                i = here->B3SOIPDDPePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDDPeStructPtr = matched ;
                here->B3SOIPDDPePtr = matched->CSC ;
            }

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDeNode != 0))
            {
                i = here->B3SOIPDSPePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDSPeStructPtr = matched ;
                here->B3SOIPDSPePtr = matched->CSC ;
            }

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDeNode != 0))
            {
                i = here->B3SOIPDEePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDEeStructPtr = matched ;
                here->B3SOIPDEePtr = matched->CSC ;
            }

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDgNode != 0))
            {
                i = here->B3SOIPDGgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDGgStructPtr = matched ;
                here->B3SOIPDGgPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDdNodePrime != 0))
            {
                i = here->B3SOIPDGdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDGdpStructPtr = matched ;
                here->B3SOIPDGdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDsNodePrime != 0))
            {
                i = here->B3SOIPDGspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDGspStructPtr = matched ;
                here->B3SOIPDGspPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDgNode != 0))
            {
                i = here->B3SOIPDDPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDDPgStructPtr = matched ;
                here->B3SOIPDDPgPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDdNodePrime != 0))
            {
                i = here->B3SOIPDDPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDDPdpStructPtr = matched ;
                here->B3SOIPDDPdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDsNodePrime != 0))
            {
                i = here->B3SOIPDDPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDDPspStructPtr = matched ;
                here->B3SOIPDDPspPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDdNode != 0))
            {
                i = here->B3SOIPDDPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDDPdStructPtr = matched ;
                here->B3SOIPDDPdPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDgNode != 0))
            {
                i = here->B3SOIPDSPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDSPgStructPtr = matched ;
                here->B3SOIPDSPgPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDdNodePrime != 0))
            {
                i = here->B3SOIPDSPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDSPdpStructPtr = matched ;
                here->B3SOIPDSPdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDsNodePrime != 0))
            {
                i = here->B3SOIPDSPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDSPspStructPtr = matched ;
                here->B3SOIPDSPspPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDsNode != 0))
            {
                i = here->B3SOIPDSPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDSPsStructPtr = matched ;
                here->B3SOIPDSPsPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDdNode != 0) && (here-> B3SOIPDdNode != 0))
            {
                i = here->B3SOIPDDdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDDdStructPtr = matched ;
                here->B3SOIPDDdPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDdNode != 0) && (here-> B3SOIPDdNodePrime != 0))
            {
                i = here->B3SOIPDDdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDDdpStructPtr = matched ;
                here->B3SOIPDDdpPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDsNode != 0) && (here-> B3SOIPDsNode != 0))
            {
                i = here->B3SOIPDSsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDSsStructPtr = matched ;
                here->B3SOIPDSsPtr = matched->CSC ;
            }

            if ((here-> B3SOIPDsNode != 0) && (here-> B3SOIPDsNodePrime != 0))
            {
                i = here->B3SOIPDSspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B3SOIPDSspStructPtr = matched ;
                here->B3SOIPDSspPtr = matched->CSC ;
            }

            if (here->B3SOIPDdebugMod != 0)
            {
                if ((here-> B3SOIPDvbsNode != 0) && (here-> B3SOIPDvbsNode != 0))
                {
                    i = here->B3SOIPDVbsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDVbsStructPtr = matched ;
                    here->B3SOIPDVbsPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDidsNode != 0) && (here-> B3SOIPDidsNode != 0))
                {
                    i = here->B3SOIPDIdsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDIdsStructPtr = matched ;
                    here->B3SOIPDIdsPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDicNode != 0) && (here-> B3SOIPDicNode != 0))
                {
                    i = here->B3SOIPDIcPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDIcStructPtr = matched ;
                    here->B3SOIPDIcPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDibsNode != 0) && (here-> B3SOIPDibsNode != 0))
                {
                    i = here->B3SOIPDIbsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDIbsStructPtr = matched ;
                    here->B3SOIPDIbsPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDibdNode != 0) && (here-> B3SOIPDibdNode != 0))
                {
                    i = here->B3SOIPDIbdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDIbdStructPtr = matched ;
                    here->B3SOIPDIbdPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDiiiNode != 0) && (here-> B3SOIPDiiiNode != 0))
                {
                    i = here->B3SOIPDIiiPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDIiiStructPtr = matched ;
                    here->B3SOIPDIiiPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDigNode != 0) && (here-> B3SOIPDigNode != 0))
                {
                    i = here->B3SOIPDIgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDIgStructPtr = matched ;
                    here->B3SOIPDIgPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDgiggNode != 0) && (here-> B3SOIPDgiggNode != 0))
                {
                    i = here->B3SOIPDGiggPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDGiggStructPtr = matched ;
                    here->B3SOIPDGiggPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDgigdNode != 0) && (here-> B3SOIPDgigdNode != 0))
                {
                    i = here->B3SOIPDGigdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDGigdStructPtr = matched ;
                    here->B3SOIPDGigdPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDgigbNode != 0) && (here-> B3SOIPDgigbNode != 0))
                {
                    i = here->B3SOIPDGigbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDGigbStructPtr = matched ;
                    here->B3SOIPDGigbPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDigidlNode != 0) && (here-> B3SOIPDigidlNode != 0))
                {
                    i = here->B3SOIPDIgidlPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDIgidlStructPtr = matched ;
                    here->B3SOIPDIgidlPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDitunNode != 0) && (here-> B3SOIPDitunNode != 0))
                {
                    i = here->B3SOIPDItunPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDItunStructPtr = matched ;
                    here->B3SOIPDItunPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDibpNode != 0) && (here-> B3SOIPDibpNode != 0))
                {
                    i = here->B3SOIPDIbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDIbpStructPtr = matched ;
                    here->B3SOIPDIbpPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDcbbNode != 0) && (here-> B3SOIPDcbbNode != 0))
                {
                    i = here->B3SOIPDCbbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDCbbStructPtr = matched ;
                    here->B3SOIPDCbbPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDcbdNode != 0) && (here-> B3SOIPDcbdNode != 0))
                {
                    i = here->B3SOIPDCbdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDCbdStructPtr = matched ;
                    here->B3SOIPDCbdPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDcbgNode != 0) && (here-> B3SOIPDcbgNode != 0))
                {
                    i = here->B3SOIPDCbgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDCbgStructPtr = matched ;
                    here->B3SOIPDCbgPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDqbfNode != 0) && (here-> B3SOIPDqbfNode != 0))
                {
                    i = here->B3SOIPDQbfPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDQbfStructPtr = matched ;
                    here->B3SOIPDQbfPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDqjsNode != 0) && (here-> B3SOIPDqjsNode != 0))
                {
                    i = here->B3SOIPDQjsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDQjsStructPtr = matched ;
                    here->B3SOIPDQjsPtr = matched->CSC ;
                }

                if ((here-> B3SOIPDqjdNode != 0) && (here-> B3SOIPDqjdNode != 0))
                {
                    i = here->B3SOIPDQjdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->B3SOIPDQjdStructPtr = matched ;
                    here->B3SOIPDQjdPtr = matched->CSC ;
                }

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
    for ( ; model != NULL ; model = model->B3SOIPDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIPDinstances ; here != NULL ; here = here->B3SOIPDnextInstance)
        {
            if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0 != 0.0))
            {
                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDTemptempPtr = here->B3SOIPDTemptempStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                    here->B3SOIPDTempdpPtr = here->B3SOIPDTempdpStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                    here->B3SOIPDTempspPtr = here->B3SOIPDTempspStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDgNode != 0))
                    here->B3SOIPDTempgPtr = here->B3SOIPDTempgStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDbNode != 0))
                    here->B3SOIPDTempbPtr = here->B3SOIPDTempbStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDGtempPtr = here->B3SOIPDGtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDDPtempPtr = here->B3SOIPDDPtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDSPtempPtr = here->B3SOIPDSPtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDEtempPtr = here->B3SOIPDEtempStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDBtempPtr = here->B3SOIPDBtempStructPtr->CSC_Complex ;

                if (here->B3SOIPDbodyMod == 1)
                {
                    if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDtempNode != 0))
                        here->B3SOIPDPtempPtr = here->B3SOIPDPtempStructPtr->CSC_Complex ;

                }
            }
            if (here->B3SOIPDbodyMod == 2)
            {
            }
            else if (here->B3SOIPDbodyMod == 1)
            {
                if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDpNode != 0))
                    here->B3SOIPDBpPtr = here->B3SOIPDBpStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDbNode != 0))
                    here->B3SOIPDPbPtr = here->B3SOIPDPbStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDpNode != 0))
                    here->B3SOIPDPpPtr = here->B3SOIPDPpStructPtr->CSC_Complex ;

            }
            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDbNode != 0))
                here->B3SOIPDEbPtr = here->B3SOIPDEbStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDbNode != 0))
                here->B3SOIPDGbPtr = here->B3SOIPDGbStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDbNode != 0))
                here->B3SOIPDDPbPtr = here->B3SOIPDDPbStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDbNode != 0))
                here->B3SOIPDSPbPtr = here->B3SOIPDSPbStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDeNode != 0))
                here->B3SOIPDBePtr = here->B3SOIPDBeStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDgNode != 0))
                here->B3SOIPDBgPtr = here->B3SOIPDBgStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDBdpPtr = here->B3SOIPDBdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDBspPtr = here->B3SOIPDBspStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDbNode != 0))
                here->B3SOIPDBbPtr = here->B3SOIPDBbStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDgNode != 0))
                here->B3SOIPDEgPtr = here->B3SOIPDEgStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDEdpPtr = here->B3SOIPDEdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDEspPtr = here->B3SOIPDEspStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDeNode != 0))
                here->B3SOIPDGePtr = here->B3SOIPDGeStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDeNode != 0))
                here->B3SOIPDDPePtr = here->B3SOIPDDPeStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDeNode != 0))
                here->B3SOIPDSPePtr = here->B3SOIPDSPeStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDeNode != 0))
                here->B3SOIPDEePtr = here->B3SOIPDEeStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDgNode != 0))
                here->B3SOIPDGgPtr = here->B3SOIPDGgStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDGdpPtr = here->B3SOIPDGdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDGspPtr = here->B3SOIPDGspStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDgNode != 0))
                here->B3SOIPDDPgPtr = here->B3SOIPDDPgStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDDPdpPtr = here->B3SOIPDDPdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDDPspPtr = here->B3SOIPDDPspStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDdNode != 0))
                here->B3SOIPDDPdPtr = here->B3SOIPDDPdStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDgNode != 0))
                here->B3SOIPDSPgPtr = here->B3SOIPDSPgStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDSPdpPtr = here->B3SOIPDSPdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDSPspPtr = here->B3SOIPDSPspStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDsNode != 0))
                here->B3SOIPDSPsPtr = here->B3SOIPDSPsStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDdNode != 0) && (here-> B3SOIPDdNode != 0))
                here->B3SOIPDDdPtr = here->B3SOIPDDdStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDdNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDDdpPtr = here->B3SOIPDDdpStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDsNode != 0) && (here-> B3SOIPDsNode != 0))
                here->B3SOIPDSsPtr = here->B3SOIPDSsStructPtr->CSC_Complex ;

            if ((here-> B3SOIPDsNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDSspPtr = here->B3SOIPDSspStructPtr->CSC_Complex ;

            if (here->B3SOIPDdebugMod != 0)
            {
                if ((here-> B3SOIPDvbsNode != 0) && (here-> B3SOIPDvbsNode != 0))
                    here->B3SOIPDVbsPtr = here->B3SOIPDVbsStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDidsNode != 0) && (here-> B3SOIPDidsNode != 0))
                    here->B3SOIPDIdsPtr = here->B3SOIPDIdsStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDicNode != 0) && (here-> B3SOIPDicNode != 0))
                    here->B3SOIPDIcPtr = here->B3SOIPDIcStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDibsNode != 0) && (here-> B3SOIPDibsNode != 0))
                    here->B3SOIPDIbsPtr = here->B3SOIPDIbsStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDibdNode != 0) && (here-> B3SOIPDibdNode != 0))
                    here->B3SOIPDIbdPtr = here->B3SOIPDIbdStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDiiiNode != 0) && (here-> B3SOIPDiiiNode != 0))
                    here->B3SOIPDIiiPtr = here->B3SOIPDIiiStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDigNode != 0) && (here-> B3SOIPDigNode != 0))
                    here->B3SOIPDIgPtr = here->B3SOIPDIgStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDgiggNode != 0) && (here-> B3SOIPDgiggNode != 0))
                    here->B3SOIPDGiggPtr = here->B3SOIPDGiggStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDgigdNode != 0) && (here-> B3SOIPDgigdNode != 0))
                    here->B3SOIPDGigdPtr = here->B3SOIPDGigdStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDgigbNode != 0) && (here-> B3SOIPDgigbNode != 0))
                    here->B3SOIPDGigbPtr = here->B3SOIPDGigbStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDigidlNode != 0) && (here-> B3SOIPDigidlNode != 0))
                    here->B3SOIPDIgidlPtr = here->B3SOIPDIgidlStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDitunNode != 0) && (here-> B3SOIPDitunNode != 0))
                    here->B3SOIPDItunPtr = here->B3SOIPDItunStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDibpNode != 0) && (here-> B3SOIPDibpNode != 0))
                    here->B3SOIPDIbpPtr = here->B3SOIPDIbpStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDcbbNode != 0) && (here-> B3SOIPDcbbNode != 0))
                    here->B3SOIPDCbbPtr = here->B3SOIPDCbbStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDcbdNode != 0) && (here-> B3SOIPDcbdNode != 0))
                    here->B3SOIPDCbdPtr = here->B3SOIPDCbdStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDcbgNode != 0) && (here-> B3SOIPDcbgNode != 0))
                    here->B3SOIPDCbgPtr = here->B3SOIPDCbgStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDqbfNode != 0) && (here-> B3SOIPDqbfNode != 0))
                    here->B3SOIPDQbfPtr = here->B3SOIPDQbfStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDqjsNode != 0) && (here-> B3SOIPDqjsNode != 0))
                    here->B3SOIPDQjsPtr = here->B3SOIPDQjsStructPtr->CSC_Complex ;

                if ((here-> B3SOIPDqjdNode != 0) && (here-> B3SOIPDqjdNode != 0))
                    here->B3SOIPDQjdPtr = here->B3SOIPDQjdStructPtr->CSC_Complex ;

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
    for ( ; model != NULL ; model = model->B3SOIPDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIPDinstances ; here != NULL ; here = here->B3SOIPDnextInstance)
        {
            if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0 != 0.0))
            {
                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDTemptempPtr = here->B3SOIPDTemptempStructPtr->CSC ;

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                    here->B3SOIPDTempdpPtr = here->B3SOIPDTempdpStructPtr->CSC ;

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                    here->B3SOIPDTempspPtr = here->B3SOIPDTempspStructPtr->CSC ;

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDgNode != 0))
                    here->B3SOIPDTempgPtr = here->B3SOIPDTempgStructPtr->CSC ;

                if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDbNode != 0))
                    here->B3SOIPDTempbPtr = here->B3SOIPDTempbStructPtr->CSC ;

                if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDGtempPtr = here->B3SOIPDGtempStructPtr->CSC ;

                if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDDPtempPtr = here->B3SOIPDDPtempStructPtr->CSC ;

                if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDSPtempPtr = here->B3SOIPDSPtempStructPtr->CSC ;

                if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDEtempPtr = here->B3SOIPDEtempStructPtr->CSC ;

                if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDtempNode != 0))
                    here->B3SOIPDBtempPtr = here->B3SOIPDBtempStructPtr->CSC ;

                if (here->B3SOIPDbodyMod == 1)
                {
                    if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDtempNode != 0))
                        here->B3SOIPDPtempPtr = here->B3SOIPDPtempStructPtr->CSC ;

                }
            }
            if (here->B3SOIPDbodyMod == 2)
            {
            }
            else if (here->B3SOIPDbodyMod == 1)
            {
                if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDpNode != 0))
                    here->B3SOIPDBpPtr = here->B3SOIPDBpStructPtr->CSC ;

                if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDbNode != 0))
                    here->B3SOIPDPbPtr = here->B3SOIPDPbStructPtr->CSC ;

                if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDpNode != 0))
                    here->B3SOIPDPpPtr = here->B3SOIPDPpStructPtr->CSC ;

            }
            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDbNode != 0))
                here->B3SOIPDEbPtr = here->B3SOIPDEbStructPtr->CSC ;

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDbNode != 0))
                here->B3SOIPDGbPtr = here->B3SOIPDGbStructPtr->CSC ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDbNode != 0))
                here->B3SOIPDDPbPtr = here->B3SOIPDDPbStructPtr->CSC ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDbNode != 0))
                here->B3SOIPDSPbPtr = here->B3SOIPDSPbStructPtr->CSC ;

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDeNode != 0))
                here->B3SOIPDBePtr = here->B3SOIPDBeStructPtr->CSC ;

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDgNode != 0))
                here->B3SOIPDBgPtr = here->B3SOIPDBgStructPtr->CSC ;

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDBdpPtr = here->B3SOIPDBdpStructPtr->CSC ;

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDBspPtr = here->B3SOIPDBspStructPtr->CSC ;

            if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDbNode != 0))
                here->B3SOIPDBbPtr = here->B3SOIPDBbStructPtr->CSC ;

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDgNode != 0))
                here->B3SOIPDEgPtr = here->B3SOIPDEgStructPtr->CSC ;

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDEdpPtr = here->B3SOIPDEdpStructPtr->CSC ;

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDEspPtr = here->B3SOIPDEspStructPtr->CSC ;

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDeNode != 0))
                here->B3SOIPDGePtr = here->B3SOIPDGeStructPtr->CSC ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDeNode != 0))
                here->B3SOIPDDPePtr = here->B3SOIPDDPeStructPtr->CSC ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDeNode != 0))
                here->B3SOIPDSPePtr = here->B3SOIPDSPeStructPtr->CSC ;

            if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDeNode != 0))
                here->B3SOIPDEePtr = here->B3SOIPDEeStructPtr->CSC ;

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDgNode != 0))
                here->B3SOIPDGgPtr = here->B3SOIPDGgStructPtr->CSC ;

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDGdpPtr = here->B3SOIPDGdpStructPtr->CSC ;

            if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDGspPtr = here->B3SOIPDGspStructPtr->CSC ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDgNode != 0))
                here->B3SOIPDDPgPtr = here->B3SOIPDDPgStructPtr->CSC ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDDPdpPtr = here->B3SOIPDDPdpStructPtr->CSC ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDDPspPtr = here->B3SOIPDDPspStructPtr->CSC ;

            if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDdNode != 0))
                here->B3SOIPDDPdPtr = here->B3SOIPDDPdStructPtr->CSC ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDgNode != 0))
                here->B3SOIPDSPgPtr = here->B3SOIPDSPgStructPtr->CSC ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDSPdpPtr = here->B3SOIPDSPdpStructPtr->CSC ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDSPspPtr = here->B3SOIPDSPspStructPtr->CSC ;

            if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDsNode != 0))
                here->B3SOIPDSPsPtr = here->B3SOIPDSPsStructPtr->CSC ;

            if ((here-> B3SOIPDdNode != 0) && (here-> B3SOIPDdNode != 0))
                here->B3SOIPDDdPtr = here->B3SOIPDDdStructPtr->CSC ;

            if ((here-> B3SOIPDdNode != 0) && (here-> B3SOIPDdNodePrime != 0))
                here->B3SOIPDDdpPtr = here->B3SOIPDDdpStructPtr->CSC ;

            if ((here-> B3SOIPDsNode != 0) && (here-> B3SOIPDsNode != 0))
                here->B3SOIPDSsPtr = here->B3SOIPDSsStructPtr->CSC ;

            if ((here-> B3SOIPDsNode != 0) && (here-> B3SOIPDsNodePrime != 0))
                here->B3SOIPDSspPtr = here->B3SOIPDSspStructPtr->CSC ;

            if (here->B3SOIPDdebugMod != 0)
            {
                if ((here-> B3SOIPDvbsNode != 0) && (here-> B3SOIPDvbsNode != 0))
                    here->B3SOIPDVbsPtr = here->B3SOIPDVbsStructPtr->CSC ;

                if ((here-> B3SOIPDidsNode != 0) && (here-> B3SOIPDidsNode != 0))
                    here->B3SOIPDIdsPtr = here->B3SOIPDIdsStructPtr->CSC ;

                if ((here-> B3SOIPDicNode != 0) && (here-> B3SOIPDicNode != 0))
                    here->B3SOIPDIcPtr = here->B3SOIPDIcStructPtr->CSC ;

                if ((here-> B3SOIPDibsNode != 0) && (here-> B3SOIPDibsNode != 0))
                    here->B3SOIPDIbsPtr = here->B3SOIPDIbsStructPtr->CSC ;

                if ((here-> B3SOIPDibdNode != 0) && (here-> B3SOIPDibdNode != 0))
                    here->B3SOIPDIbdPtr = here->B3SOIPDIbdStructPtr->CSC ;

                if ((here-> B3SOIPDiiiNode != 0) && (here-> B3SOIPDiiiNode != 0))
                    here->B3SOIPDIiiPtr = here->B3SOIPDIiiStructPtr->CSC ;

                if ((here-> B3SOIPDigNode != 0) && (here-> B3SOIPDigNode != 0))
                    here->B3SOIPDIgPtr = here->B3SOIPDIgStructPtr->CSC ;

                if ((here-> B3SOIPDgiggNode != 0) && (here-> B3SOIPDgiggNode != 0))
                    here->B3SOIPDGiggPtr = here->B3SOIPDGiggStructPtr->CSC ;

                if ((here-> B3SOIPDgigdNode != 0) && (here-> B3SOIPDgigdNode != 0))
                    here->B3SOIPDGigdPtr = here->B3SOIPDGigdStructPtr->CSC ;

                if ((here-> B3SOIPDgigbNode != 0) && (here-> B3SOIPDgigbNode != 0))
                    here->B3SOIPDGigbPtr = here->B3SOIPDGigbStructPtr->CSC ;

                if ((here-> B3SOIPDigidlNode != 0) && (here-> B3SOIPDigidlNode != 0))
                    here->B3SOIPDIgidlPtr = here->B3SOIPDIgidlStructPtr->CSC ;

                if ((here-> B3SOIPDitunNode != 0) && (here-> B3SOIPDitunNode != 0))
                    here->B3SOIPDItunPtr = here->B3SOIPDItunStructPtr->CSC ;

                if ((here-> B3SOIPDibpNode != 0) && (here-> B3SOIPDibpNode != 0))
                    here->B3SOIPDIbpPtr = here->B3SOIPDIbpStructPtr->CSC ;

                if ((here-> B3SOIPDcbbNode != 0) && (here-> B3SOIPDcbbNode != 0))
                    here->B3SOIPDCbbPtr = here->B3SOIPDCbbStructPtr->CSC ;

                if ((here-> B3SOIPDcbdNode != 0) && (here-> B3SOIPDcbdNode != 0))
                    here->B3SOIPDCbdPtr = here->B3SOIPDCbdStructPtr->CSC ;

                if ((here-> B3SOIPDcbgNode != 0) && (here-> B3SOIPDcbgNode != 0))
                    here->B3SOIPDCbgPtr = here->B3SOIPDCbgStructPtr->CSC ;

                if ((here-> B3SOIPDqbfNode != 0) && (here-> B3SOIPDqbfNode != 0))
                    here->B3SOIPDQbfPtr = here->B3SOIPDQbfStructPtr->CSC ;

                if ((here-> B3SOIPDqjsNode != 0) && (here-> B3SOIPDqjsNode != 0))
                    here->B3SOIPDQjsPtr = here->B3SOIPDQjsStructPtr->CSC ;

                if ((here-> B3SOIPDqjdNode != 0) && (here-> B3SOIPDqjdNode != 0))
                    here->B3SOIPDQjdPtr = here->B3SOIPDQjdStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
