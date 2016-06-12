/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhvdef.h"
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
    for ( ; model != NULL ; model = model->HSMHVnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSMHVinstances ; here != NULL ; here = here->HSMHVnextInstance)
        {
            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
            {
                i = here->HSMHVDPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDPbpStructPtr = matched ;
                here->HSMHVDPbpPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
            {
                i = here->HSMHVSPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSPbpStructPtr = matched ;
                here->HSMHVSPbpPtr = matched->CSC ;
            }

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
            {
                i = here->HSMHVGPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVGPbpStructPtr = matched ;
                here->HSMHVGPbpPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdNode != 0))
            {
                i = here->HSMHVBPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBPdStructPtr = matched ;
                here->HSMHVBPdPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsNode != 0))
            {
                i = here->HSMHVBPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBPsStructPtr = matched ;
                here->HSMHVBPsPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
            {
                i = here->HSMHVBPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBPdpStructPtr = matched ;
                here->HSMHVBPdpPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
            {
                i = here->HSMHVBPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBPspStructPtr = matched ;
                here->HSMHVBPspPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
            {
                i = here->HSMHVBPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBPgpStructPtr = matched ;
                here->HSMHVBPgpPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
            {
                i = here->HSMHVBPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBPbpStructPtr = matched ;
                here->HSMHVBPbpPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVdNode != 0))
            {
                i = here->HSMHVDdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDdStructPtr = matched ;
                here->HSMHVDdPtr = matched->CSC ;
            }

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
            {
                i = here->HSMHVGPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVGPgpStructPtr = matched ;
                here->HSMHVGPgpPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVsNode != 0))
            {
                i = here->HSMHVSsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSsStructPtr = matched ;
                here->HSMHVSsPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
            {
                i = here->HSMHVDPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDPdpStructPtr = matched ;
                here->HSMHVDPdpPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
            {
                i = here->HSMHVSPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSPspStructPtr = matched ;
                here->HSMHVSPspPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVdNodePrime != 0))
            {
                i = here->HSMHVDdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDdpStructPtr = matched ;
                here->HSMHVDdpPtr = matched->CSC ;
            }

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
            {
                i = here->HSMHVGPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVGPdpStructPtr = matched ;
                here->HSMHVGPdpPtr = matched->CSC ;
            }

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
            {
                i = here->HSMHVGPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVGPspStructPtr = matched ;
                here->HSMHVGPspPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVsNodePrime != 0))
            {
                i = here->HSMHVSspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSspStructPtr = matched ;
                here->HSMHVSspPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
            {
                i = here->HSMHVDPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDPspStructPtr = matched ;
                here->HSMHVDPspPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVdNode != 0))
            {
                i = here->HSMHVDPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDPdStructPtr = matched ;
                here->HSMHVDPdPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
            {
                i = here->HSMHVDPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDPgpStructPtr = matched ;
                here->HSMHVDPgpPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
            {
                i = here->HSMHVSPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSPgpStructPtr = matched ;
                here->HSMHVSPgpPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsNode != 0))
            {
                i = here->HSMHVSPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSPsStructPtr = matched ;
                here->HSMHVSPsPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
            {
                i = here->HSMHVSPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSPdpStructPtr = matched ;
                here->HSMHVSPdpPtr = matched->CSC ;
            }

            if ((here-> HSMHVgNode != 0) && (here-> HSMHVgNode != 0))
            {
                i = here->HSMHVGgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVGgStructPtr = matched ;
                here->HSMHVGgPtr = matched->CSC ;
            }

            if ((here-> HSMHVgNode != 0) && (here-> HSMHVgNodePrime != 0))
            {
                i = here->HSMHVGgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVGgpStructPtr = matched ;
                here->HSMHVGgpPtr = matched->CSC ;
            }

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVgNode != 0))
            {
                i = here->HSMHVGPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVGPgStructPtr = matched ;
                here->HSMHVGPgPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVdbNode != 0))
            {
                i = here->HSMHVDdbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDdbStructPtr = matched ;
                here->HSMHVDdbPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVsbNode != 0))
            {
                i = here->HSMHVSsbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSsbStructPtr = matched ;
                here->HSMHVSsbPtr = matched->CSC ;
            }

            if ((here-> HSMHVdbNode != 0) && (here-> HSMHVdNode != 0))
            {
                i = here->HSMHVDBdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDBdStructPtr = matched ;
                here->HSMHVDBdPtr = matched->CSC ;
            }

            if ((here-> HSMHVdbNode != 0) && (here-> HSMHVdbNode != 0))
            {
                i = here->HSMHVDBdbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDBdbStructPtr = matched ;
                here->HSMHVDBdbPtr = matched->CSC ;
            }

            if ((here-> HSMHVdbNode != 0) && (here-> HSMHVbNodePrime != 0))
            {
                i = here->HSMHVDBbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDBbpStructPtr = matched ;
                here->HSMHVDBbpPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdbNode != 0))
            {
                i = here->HSMHVBPdbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBPdbStructPtr = matched ;
                here->HSMHVBPdbPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVbNode != 0))
            {
                i = here->HSMHVBPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBPbStructPtr = matched ;
                here->HSMHVBPbPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsbNode != 0))
            {
                i = here->HSMHVBPsbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBPsbStructPtr = matched ;
                here->HSMHVBPsbPtr = matched->CSC ;
            }

            if ((here-> HSMHVsbNode != 0) && (here-> HSMHVsNode != 0))
            {
                i = here->HSMHVSBsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSBsStructPtr = matched ;
                here->HSMHVSBsPtr = matched->CSC ;
            }

            if ((here-> HSMHVsbNode != 0) && (here-> HSMHVbNodePrime != 0))
            {
                i = here->HSMHVSBbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSBbpStructPtr = matched ;
                here->HSMHVSBbpPtr = matched->CSC ;
            }

            if ((here-> HSMHVsbNode != 0) && (here-> HSMHVsbNode != 0))
            {
                i = here->HSMHVSBsbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSBsbStructPtr = matched ;
                here->HSMHVSBsbPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNode != 0) && (here-> HSMHVbNodePrime != 0))
            {
                i = here->HSMHVBbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBbpStructPtr = matched ;
                here->HSMHVBbpPtr = matched->CSC ;
            }

            if ((here-> HSMHVbNode != 0) && (here-> HSMHVbNode != 0))
            {
                i = here->HSMHVBbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVBbStructPtr = matched ;
                here->HSMHVBbPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVgNodePrime != 0))
            {
                i = here->HSMHVDgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDgpStructPtr = matched ;
                here->HSMHVDgpPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVsNode != 0))
            {
                i = here->HSMHVDsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDsStructPtr = matched ;
                here->HSMHVDsPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVbNodePrime != 0))
            {
                i = here->HSMHVDbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDbpStructPtr = matched ;
                here->HSMHVDbpPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVsNodePrime != 0))
            {
                i = here->HSMHVDspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDspStructPtr = matched ;
                here->HSMHVDspPtr = matched->CSC ;
            }

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsNode != 0))
            {
                i = here->HSMHVDPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVDPsStructPtr = matched ;
                here->HSMHVDPsPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVgNodePrime != 0))
            {
                i = here->HSMHVSgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSgpStructPtr = matched ;
                here->HSMHVSgpPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVdNode != 0))
            {
                i = here->HSMHVSdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSdStructPtr = matched ;
                here->HSMHVSdPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVbNodePrime != 0))
            {
                i = here->HSMHVSbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSbpStructPtr = matched ;
                here->HSMHVSbpPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVdNodePrime != 0))
            {
                i = here->HSMHVSdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSdpStructPtr = matched ;
                here->HSMHVSdpPtr = matched->CSC ;
            }

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVdNode != 0))
            {
                i = here->HSMHVSPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVSPdStructPtr = matched ;
                here->HSMHVSPdPtr = matched->CSC ;
            }

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVdNode != 0))
            {
                i = here->HSMHVGPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVGPdStructPtr = matched ;
                here->HSMHVGPdPtr = matched->CSC ;
            }

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVsNode != 0))
            {
                i = here->HSMHVGPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSMHVGPsStructPtr = matched ;
                here->HSMHVGPsPtr = matched->CSC ;
            }

            if (here->HSMHVsubNode > 0)
            {
                if ((here-> HSMHVdNode != 0) && (here-> HSMHVsubNode != 0))
                {
                    i = here->HSMHVDsubPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVDsubStructPtr = matched ;
                    here->HSMHVDsubPtr = matched->CSC ;
                }

                if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsubNode != 0))
                {
                    i = here->HSMHVDPsubPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVDPsubStructPtr = matched ;
                    here->HSMHVDPsubPtr = matched->CSC ;
                }

                if ((here-> HSMHVsNode != 0) && (here-> HSMHVsubNode != 0))
                {
                    i = here->HSMHVSsubPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVSsubStructPtr = matched ;
                    here->HSMHVSsubPtr = matched->CSC ;
                }

                if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsubNode != 0))
                {
                    i = here->HSMHVSPsubPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVSPsubStructPtr = matched ;
                    here->HSMHVSPsubPtr = matched->CSC ;
                }

            }
            if (here->HSMHV_coselfheat > 0)
            {
                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVtempNode != 0))
                {
                    i = here->HSMHVTemptempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVTemptempStructPtr = matched ;
                    here->HSMHVTemptempPtr = matched->CSC ;
                }

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVdNode != 0))
                {
                    i = here->HSMHVTempdPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVTempdStructPtr = matched ;
                    here->HSMHVTempdPtr = matched->CSC ;
                }

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVdNodePrime != 0))
                {
                    i = here->HSMHVTempdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVTempdpStructPtr = matched ;
                    here->HSMHVTempdpPtr = matched->CSC ;
                }

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVsNode != 0))
                {
                    i = here->HSMHVTempsPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVTempsStructPtr = matched ;
                    here->HSMHVTempsPtr = matched->CSC ;
                }

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVsNodePrime != 0))
                {
                    i = here->HSMHVTempspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVTempspStructPtr = matched ;
                    here->HSMHVTempspPtr = matched->CSC ;
                }

                if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVtempNode != 0))
                {
                    i = here->HSMHVDPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVDPtempStructPtr = matched ;
                    here->HSMHVDPtempPtr = matched->CSC ;
                }

                if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVtempNode != 0))
                {
                    i = here->HSMHVSPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVSPtempStructPtr = matched ;
                    here->HSMHVSPtempPtr = matched->CSC ;
                }

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVgNodePrime != 0))
                {
                    i = here->HSMHVTempgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVTempgpStructPtr = matched ;
                    here->HSMHVTempgpPtr = matched->CSC ;
                }

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVbNodePrime != 0))
                {
                    i = here->HSMHVTempbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVTempbpStructPtr = matched ;
                    here->HSMHVTempbpPtr = matched->CSC ;
                }

                if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVtempNode != 0))
                {
                    i = here->HSMHVGPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVGPtempStructPtr = matched ;
                    here->HSMHVGPtempPtr = matched->CSC ;
                }

                if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVtempNode != 0))
                {
                    i = here->HSMHVBPtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVBPtempStructPtr = matched ;
                    here->HSMHVBPtempPtr = matched->CSC ;
                }

                if ((here-> HSMHVdbNode != 0) && (here-> HSMHVtempNode != 0))
                {
                    i = here->HSMHVDBtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVDBtempStructPtr = matched ;
                    here->HSMHVDBtempPtr = matched->CSC ;
                }

                if ((here-> HSMHVsbNode != 0) && (here-> HSMHVtempNode != 0))
                {
                    i = here->HSMHVSBtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVSBtempStructPtr = matched ;
                    here->HSMHVSBtempPtr = matched->CSC ;
                }

                if ((here-> HSMHVdNode != 0) && (here-> HSMHVtempNode != 0))
                {
                    i = here->HSMHVDtempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVDtempStructPtr = matched ;
                    here->HSMHVDtempPtr = matched->CSC ;
                }

                if ((here-> HSMHVsNode != 0) && (here-> HSMHVtempNode != 0))
                {
                    i = here->HSMHVStempPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVStempStructPtr = matched ;
                    here->HSMHVStempPtr = matched->CSC ;
                }

            }
            if (model->HSMHV_conqs)
            {
                if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVqiNode != 0))
                {
                    i = here->HSMHVDPqiPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVDPqiStructPtr = matched ;
                    here->HSMHVDPqiPtr = matched->CSC ;
                }

                if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVqiNode != 0))
                {
                    i = here->HSMHVGPqiPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVGPqiStructPtr = matched ;
                    here->HSMHVGPqiPtr = matched->CSC ;
                }

                if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVqbNode != 0))
                {
                    i = here->HSMHVGPqbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVGPqbStructPtr = matched ;
                    here->HSMHVGPqbPtr = matched->CSC ;
                }

                if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVqiNode != 0))
                {
                    i = here->HSMHVSPqiPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVSPqiStructPtr = matched ;
                    here->HSMHVSPqiPtr = matched->CSC ;
                }

                if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVqbNode != 0))
                {
                    i = here->HSMHVBPqbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVBPqbStructPtr = matched ;
                    here->HSMHVBPqbPtr = matched->CSC ;
                }

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVdNodePrime != 0))
                {
                    i = here->HSMHVQIdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVQIdpStructPtr = matched ;
                    here->HSMHVQIdpPtr = matched->CSC ;
                }

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVgNodePrime != 0))
                {
                    i = here->HSMHVQIgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVQIgpStructPtr = matched ;
                    here->HSMHVQIgpPtr = matched->CSC ;
                }

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVsNodePrime != 0))
                {
                    i = here->HSMHVQIspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVQIspStructPtr = matched ;
                    here->HSMHVQIspPtr = matched->CSC ;
                }

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVbNodePrime != 0))
                {
                    i = here->HSMHVQIbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVQIbpStructPtr = matched ;
                    here->HSMHVQIbpPtr = matched->CSC ;
                }

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVqiNode != 0))
                {
                    i = here->HSMHVQIqiPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVQIqiStructPtr = matched ;
                    here->HSMHVQIqiPtr = matched->CSC ;
                }

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVdNodePrime != 0))
                {
                    i = here->HSMHVQBdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVQBdpStructPtr = matched ;
                    here->HSMHVQBdpPtr = matched->CSC ;
                }

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVgNodePrime != 0))
                {
                    i = here->HSMHVQBgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVQBgpStructPtr = matched ;
                    here->HSMHVQBgpPtr = matched->CSC ;
                }

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVsNodePrime != 0))
                {
                    i = here->HSMHVQBspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVQBspStructPtr = matched ;
                    here->HSMHVQBspPtr = matched->CSC ;
                }

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVbNodePrime != 0))
                {
                    i = here->HSMHVQBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVQBbpStructPtr = matched ;
                    here->HSMHVQBbpPtr = matched->CSC ;
                }

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVqbNode != 0))
                {
                    i = here->HSMHVQBqbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSMHVQBqbStructPtr = matched ;
                    here->HSMHVQBqbPtr = matched->CSC ;
                }

                if (here->HSMHV_coselfheat > 0)
                {
                    if ((here-> HSMHVqiNode != 0) && (here-> HSMHVtempNode != 0))
                    {
                        i = here->HSMHVQItempPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->HSMHVQItempStructPtr = matched ;
                        here->HSMHVQItempPtr = matched->CSC ;
                    }

                    if ((here-> HSMHVqbNode != 0) && (here-> HSMHVtempNode != 0))
                    {
                        i = here->HSMHVQBtempPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->HSMHVQBtempStructPtr = matched ;
                        here->HSMHVQBtempPtr = matched->CSC ;
                    }

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
    for ( ; model != NULL ; model = model->HSMHVnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSMHVinstances ; here != NULL ; here = here->HSMHVnextInstance)
        {
            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVDPbpPtr = here->HSMHVDPbpStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVSPbpPtr = here->HSMHVSPbpStructPtr->CSC_Complex ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVGPbpPtr = here->HSMHVGPbpStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVBPdPtr = here->HSMHVBPdStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVBPsPtr = here->HSMHVBPsStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVBPdpPtr = here->HSMHVBPdpStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVBPspPtr = here->HSMHVBPspStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVBPgpPtr = here->HSMHVBPgpStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVBPbpPtr = here->HSMHVBPbpStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVDdPtr = here->HSMHVDdStructPtr->CSC_Complex ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVGPgpPtr = here->HSMHVGPgpStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVSsPtr = here->HSMHVSsStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVDPdpPtr = here->HSMHVDPdpStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVSPspPtr = here->HSMHVSPspStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVDdpPtr = here->HSMHVDdpStructPtr->CSC_Complex ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVGPdpPtr = here->HSMHVGPdpStructPtr->CSC_Complex ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVGPspPtr = here->HSMHVGPspStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVSspPtr = here->HSMHVSspStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVDPspPtr = here->HSMHVDPspStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVDPdPtr = here->HSMHVDPdStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVDPgpPtr = here->HSMHVDPgpStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVSPgpPtr = here->HSMHVSPgpStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVSPsPtr = here->HSMHVSPsStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVSPdpPtr = here->HSMHVSPdpStructPtr->CSC_Complex ;

            if ((here-> HSMHVgNode != 0) && (here-> HSMHVgNode != 0))
                here->HSMHVGgPtr = here->HSMHVGgStructPtr->CSC_Complex ;

            if ((here-> HSMHVgNode != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVGgpPtr = here->HSMHVGgpStructPtr->CSC_Complex ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVgNode != 0))
                here->HSMHVGPgPtr = here->HSMHVGPgStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVdbNode != 0))
                here->HSMHVDdbPtr = here->HSMHVDdbStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVsbNode != 0))
                here->HSMHVSsbPtr = here->HSMHVSsbStructPtr->CSC_Complex ;

            if ((here-> HSMHVdbNode != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVDBdPtr = here->HSMHVDBdStructPtr->CSC_Complex ;

            if ((here-> HSMHVdbNode != 0) && (here-> HSMHVdbNode != 0))
                here->HSMHVDBdbPtr = here->HSMHVDBdbStructPtr->CSC_Complex ;

            if ((here-> HSMHVdbNode != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVDBbpPtr = here->HSMHVDBbpStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdbNode != 0))
                here->HSMHVBPdbPtr = here->HSMHVBPdbStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVbNode != 0))
                here->HSMHVBPbPtr = here->HSMHVBPbStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsbNode != 0))
                here->HSMHVBPsbPtr = here->HSMHVBPsbStructPtr->CSC_Complex ;

            if ((here-> HSMHVsbNode != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVSBsPtr = here->HSMHVSBsStructPtr->CSC_Complex ;

            if ((here-> HSMHVsbNode != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVSBbpPtr = here->HSMHVSBbpStructPtr->CSC_Complex ;

            if ((here-> HSMHVsbNode != 0) && (here-> HSMHVsbNode != 0))
                here->HSMHVSBsbPtr = here->HSMHVSBsbStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNode != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVBbpPtr = here->HSMHVBbpStructPtr->CSC_Complex ;

            if ((here-> HSMHVbNode != 0) && (here-> HSMHVbNode != 0))
                here->HSMHVBbPtr = here->HSMHVBbStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVDgpPtr = here->HSMHVDgpStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVDsPtr = here->HSMHVDsStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVDbpPtr = here->HSMHVDbpStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVDspPtr = here->HSMHVDspStructPtr->CSC_Complex ;

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVDPsPtr = here->HSMHVDPsStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVSgpPtr = here->HSMHVSgpStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVSdPtr = here->HSMHVSdStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVSbpPtr = here->HSMHVSbpStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVSdpPtr = here->HSMHVSdpStructPtr->CSC_Complex ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVSPdPtr = here->HSMHVSPdStructPtr->CSC_Complex ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVGPdPtr = here->HSMHVGPdStructPtr->CSC_Complex ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVGPsPtr = here->HSMHVGPsStructPtr->CSC_Complex ;

            if (here->HSMHVsubNode > 0)
            {
                if ((here-> HSMHVdNode != 0) && (here-> HSMHVsubNode != 0))
                    here->HSMHVDsubPtr = here->HSMHVDsubStructPtr->CSC_Complex ;

                if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsubNode != 0))
                    here->HSMHVDPsubPtr = here->HSMHVDPsubStructPtr->CSC_Complex ;

                if ((here-> HSMHVsNode != 0) && (here-> HSMHVsubNode != 0))
                    here->HSMHVSsubPtr = here->HSMHVSsubStructPtr->CSC_Complex ;

                if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsubNode != 0))
                    here->HSMHVSPsubPtr = here->HSMHVSPsubStructPtr->CSC_Complex ;

            }
            if (here->HSMHV_coselfheat > 0)
            {
                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVTemptempPtr = here->HSMHVTemptempStructPtr->CSC_Complex ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVdNode != 0))
                    here->HSMHVTempdPtr = here->HSMHVTempdStructPtr->CSC_Complex ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVdNodePrime != 0))
                    here->HSMHVTempdpPtr = here->HSMHVTempdpStructPtr->CSC_Complex ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVsNode != 0))
                    here->HSMHVTempsPtr = here->HSMHVTempsStructPtr->CSC_Complex ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVsNodePrime != 0))
                    here->HSMHVTempspPtr = here->HSMHVTempspStructPtr->CSC_Complex ;

                if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVDPtempPtr = here->HSMHVDPtempStructPtr->CSC_Complex ;

                if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVSPtempPtr = here->HSMHVSPtempStructPtr->CSC_Complex ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVgNodePrime != 0))
                    here->HSMHVTempgpPtr = here->HSMHVTempgpStructPtr->CSC_Complex ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVbNodePrime != 0))
                    here->HSMHVTempbpPtr = here->HSMHVTempbpStructPtr->CSC_Complex ;

                if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVGPtempPtr = here->HSMHVGPtempStructPtr->CSC_Complex ;

                if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVBPtempPtr = here->HSMHVBPtempStructPtr->CSC_Complex ;

                if ((here-> HSMHVdbNode != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVDBtempPtr = here->HSMHVDBtempStructPtr->CSC_Complex ;

                if ((here-> HSMHVsbNode != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVSBtempPtr = here->HSMHVSBtempStructPtr->CSC_Complex ;

                if ((here-> HSMHVdNode != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVDtempPtr = here->HSMHVDtempStructPtr->CSC_Complex ;

                if ((here-> HSMHVsNode != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVStempPtr = here->HSMHVStempStructPtr->CSC_Complex ;

            }
            if (model->HSMHV_conqs)
            {
                if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVqiNode != 0))
                    here->HSMHVDPqiPtr = here->HSMHVDPqiStructPtr->CSC_Complex ;

                if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVqiNode != 0))
                    here->HSMHVGPqiPtr = here->HSMHVGPqiStructPtr->CSC_Complex ;

                if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVqbNode != 0))
                    here->HSMHVGPqbPtr = here->HSMHVGPqbStructPtr->CSC_Complex ;

                if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVqiNode != 0))
                    here->HSMHVSPqiPtr = here->HSMHVSPqiStructPtr->CSC_Complex ;

                if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVqbNode != 0))
                    here->HSMHVBPqbPtr = here->HSMHVBPqbStructPtr->CSC_Complex ;

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVdNodePrime != 0))
                    here->HSMHVQIdpPtr = here->HSMHVQIdpStructPtr->CSC_Complex ;

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVgNodePrime != 0))
                    here->HSMHVQIgpPtr = here->HSMHVQIgpStructPtr->CSC_Complex ;

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVsNodePrime != 0))
                    here->HSMHVQIspPtr = here->HSMHVQIspStructPtr->CSC_Complex ;

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVbNodePrime != 0))
                    here->HSMHVQIbpPtr = here->HSMHVQIbpStructPtr->CSC_Complex ;

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVqiNode != 0))
                    here->HSMHVQIqiPtr = here->HSMHVQIqiStructPtr->CSC_Complex ;

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVdNodePrime != 0))
                    here->HSMHVQBdpPtr = here->HSMHVQBdpStructPtr->CSC_Complex ;

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVgNodePrime != 0))
                    here->HSMHVQBgpPtr = here->HSMHVQBgpStructPtr->CSC_Complex ;

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVsNodePrime != 0))
                    here->HSMHVQBspPtr = here->HSMHVQBspStructPtr->CSC_Complex ;

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVbNodePrime != 0))
                    here->HSMHVQBbpPtr = here->HSMHVQBbpStructPtr->CSC_Complex ;

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVqbNode != 0))
                    here->HSMHVQBqbPtr = here->HSMHVQBqbStructPtr->CSC_Complex ;

                if (here->HSMHV_coselfheat > 0)
                {
                    if ((here-> HSMHVqiNode != 0) && (here-> HSMHVtempNode != 0))
                        here->HSMHVQItempPtr = here->HSMHVQItempStructPtr->CSC_Complex ;

                    if ((here-> HSMHVqbNode != 0) && (here-> HSMHVtempNode != 0))
                        here->HSMHVQBtempPtr = here->HSMHVQBtempStructPtr->CSC_Complex ;

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
    for ( ; model != NULL ; model = model->HSMHVnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSMHVinstances ; here != NULL ; here = here->HSMHVnextInstance)
        {
            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVDPbpPtr = here->HSMHVDPbpStructPtr->CSC ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVSPbpPtr = here->HSMHVSPbpStructPtr->CSC ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVGPbpPtr = here->HSMHVGPbpStructPtr->CSC ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVBPdPtr = here->HSMHVBPdStructPtr->CSC ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVBPsPtr = here->HSMHVBPsStructPtr->CSC ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVBPdpPtr = here->HSMHVBPdpStructPtr->CSC ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVBPspPtr = here->HSMHVBPspStructPtr->CSC ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVBPgpPtr = here->HSMHVBPgpStructPtr->CSC ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVBPbpPtr = here->HSMHVBPbpStructPtr->CSC ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVDdPtr = here->HSMHVDdStructPtr->CSC ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVGPgpPtr = here->HSMHVGPgpStructPtr->CSC ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVSsPtr = here->HSMHVSsStructPtr->CSC ;

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVDPdpPtr = here->HSMHVDPdpStructPtr->CSC ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVSPspPtr = here->HSMHVSPspStructPtr->CSC ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVDdpPtr = here->HSMHVDdpStructPtr->CSC ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVGPdpPtr = here->HSMHVGPdpStructPtr->CSC ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVGPspPtr = here->HSMHVGPspStructPtr->CSC ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVSspPtr = here->HSMHVSspStructPtr->CSC ;

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVDPspPtr = here->HSMHVDPspStructPtr->CSC ;

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVDPdPtr = here->HSMHVDPdStructPtr->CSC ;

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVDPgpPtr = here->HSMHVDPgpStructPtr->CSC ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVSPgpPtr = here->HSMHVSPgpStructPtr->CSC ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVSPsPtr = here->HSMHVSPsStructPtr->CSC ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVSPdpPtr = here->HSMHVSPdpStructPtr->CSC ;

            if ((here-> HSMHVgNode != 0) && (here-> HSMHVgNode != 0))
                here->HSMHVGgPtr = here->HSMHVGgStructPtr->CSC ;

            if ((here-> HSMHVgNode != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVGgpPtr = here->HSMHVGgpStructPtr->CSC ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVgNode != 0))
                here->HSMHVGPgPtr = here->HSMHVGPgStructPtr->CSC ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVdbNode != 0))
                here->HSMHVDdbPtr = here->HSMHVDdbStructPtr->CSC ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVsbNode != 0))
                here->HSMHVSsbPtr = here->HSMHVSsbStructPtr->CSC ;

            if ((here-> HSMHVdbNode != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVDBdPtr = here->HSMHVDBdStructPtr->CSC ;

            if ((here-> HSMHVdbNode != 0) && (here-> HSMHVdbNode != 0))
                here->HSMHVDBdbPtr = here->HSMHVDBdbStructPtr->CSC ;

            if ((here-> HSMHVdbNode != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVDBbpPtr = here->HSMHVDBbpStructPtr->CSC ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdbNode != 0))
                here->HSMHVBPdbPtr = here->HSMHVBPdbStructPtr->CSC ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVbNode != 0))
                here->HSMHVBPbPtr = here->HSMHVBPbStructPtr->CSC ;

            if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsbNode != 0))
                here->HSMHVBPsbPtr = here->HSMHVBPsbStructPtr->CSC ;

            if ((here-> HSMHVsbNode != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVSBsPtr = here->HSMHVSBsStructPtr->CSC ;

            if ((here-> HSMHVsbNode != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVSBbpPtr = here->HSMHVSBbpStructPtr->CSC ;

            if ((here-> HSMHVsbNode != 0) && (here-> HSMHVsbNode != 0))
                here->HSMHVSBsbPtr = here->HSMHVSBsbStructPtr->CSC ;

            if ((here-> HSMHVbNode != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVBbpPtr = here->HSMHVBbpStructPtr->CSC ;

            if ((here-> HSMHVbNode != 0) && (here-> HSMHVbNode != 0))
                here->HSMHVBbPtr = here->HSMHVBbStructPtr->CSC ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVDgpPtr = here->HSMHVDgpStructPtr->CSC ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVDsPtr = here->HSMHVDsStructPtr->CSC ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVDbpPtr = here->HSMHVDbpStructPtr->CSC ;

            if ((here-> HSMHVdNode != 0) && (here-> HSMHVsNodePrime != 0))
                here->HSMHVDspPtr = here->HSMHVDspStructPtr->CSC ;

            if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVDPsPtr = here->HSMHVDPsStructPtr->CSC ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVgNodePrime != 0))
                here->HSMHVSgpPtr = here->HSMHVSgpStructPtr->CSC ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVSdPtr = here->HSMHVSdStructPtr->CSC ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVbNodePrime != 0))
                here->HSMHVSbpPtr = here->HSMHVSbpStructPtr->CSC ;

            if ((here-> HSMHVsNode != 0) && (here-> HSMHVdNodePrime != 0))
                here->HSMHVSdpPtr = here->HSMHVSdpStructPtr->CSC ;

            if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVSPdPtr = here->HSMHVSPdStructPtr->CSC ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVdNode != 0))
                here->HSMHVGPdPtr = here->HSMHVGPdStructPtr->CSC ;

            if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVsNode != 0))
                here->HSMHVGPsPtr = here->HSMHVGPsStructPtr->CSC ;

            if (here->HSMHVsubNode > 0)
            {
                if ((here-> HSMHVdNode != 0) && (here-> HSMHVsubNode != 0))
                    here->HSMHVDsubPtr = here->HSMHVDsubStructPtr->CSC ;

                if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsubNode != 0))
                    here->HSMHVDPsubPtr = here->HSMHVDPsubStructPtr->CSC ;

                if ((here-> HSMHVsNode != 0) && (here-> HSMHVsubNode != 0))
                    here->HSMHVSsubPtr = here->HSMHVSsubStructPtr->CSC ;

                if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsubNode != 0))
                    here->HSMHVSPsubPtr = here->HSMHVSPsubStructPtr->CSC ;

            }
            if (here->HSMHV_coselfheat > 0)
            {
                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVTemptempPtr = here->HSMHVTemptempStructPtr->CSC ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVdNode != 0))
                    here->HSMHVTempdPtr = here->HSMHVTempdStructPtr->CSC ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVdNodePrime != 0))
                    here->HSMHVTempdpPtr = here->HSMHVTempdpStructPtr->CSC ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVsNode != 0))
                    here->HSMHVTempsPtr = here->HSMHVTempsStructPtr->CSC ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVsNodePrime != 0))
                    here->HSMHVTempspPtr = here->HSMHVTempspStructPtr->CSC ;

                if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVDPtempPtr = here->HSMHVDPtempStructPtr->CSC ;

                if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVSPtempPtr = here->HSMHVSPtempStructPtr->CSC ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVgNodePrime != 0))
                    here->HSMHVTempgpPtr = here->HSMHVTempgpStructPtr->CSC ;

                if ((here-> HSMHVtempNode != 0) && (here-> HSMHVbNodePrime != 0))
                    here->HSMHVTempbpPtr = here->HSMHVTempbpStructPtr->CSC ;

                if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVGPtempPtr = here->HSMHVGPtempStructPtr->CSC ;

                if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVBPtempPtr = here->HSMHVBPtempStructPtr->CSC ;

                if ((here-> HSMHVdbNode != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVDBtempPtr = here->HSMHVDBtempStructPtr->CSC ;

                if ((here-> HSMHVsbNode != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVSBtempPtr = here->HSMHVSBtempStructPtr->CSC ;

                if ((here-> HSMHVdNode != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVDtempPtr = here->HSMHVDtempStructPtr->CSC ;

                if ((here-> HSMHVsNode != 0) && (here-> HSMHVtempNode != 0))
                    here->HSMHVStempPtr = here->HSMHVStempStructPtr->CSC ;

            }
            if (model->HSMHV_conqs)
            {
                if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVqiNode != 0))
                    here->HSMHVDPqiPtr = here->HSMHVDPqiStructPtr->CSC ;

                if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVqiNode != 0))
                    here->HSMHVGPqiPtr = here->HSMHVGPqiStructPtr->CSC ;

                if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVqbNode != 0))
                    here->HSMHVGPqbPtr = here->HSMHVGPqbStructPtr->CSC ;

                if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVqiNode != 0))
                    here->HSMHVSPqiPtr = here->HSMHVSPqiStructPtr->CSC ;

                if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVqbNode != 0))
                    here->HSMHVBPqbPtr = here->HSMHVBPqbStructPtr->CSC ;

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVdNodePrime != 0))
                    here->HSMHVQIdpPtr = here->HSMHVQIdpStructPtr->CSC ;

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVgNodePrime != 0))
                    here->HSMHVQIgpPtr = here->HSMHVQIgpStructPtr->CSC ;

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVsNodePrime != 0))
                    here->HSMHVQIspPtr = here->HSMHVQIspStructPtr->CSC ;

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVbNodePrime != 0))
                    here->HSMHVQIbpPtr = here->HSMHVQIbpStructPtr->CSC ;

                if ((here-> HSMHVqiNode != 0) && (here-> HSMHVqiNode != 0))
                    here->HSMHVQIqiPtr = here->HSMHVQIqiStructPtr->CSC ;

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVdNodePrime != 0))
                    here->HSMHVQBdpPtr = here->HSMHVQBdpStructPtr->CSC ;

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVgNodePrime != 0))
                    here->HSMHVQBgpPtr = here->HSMHVQBgpStructPtr->CSC ;

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVsNodePrime != 0))
                    here->HSMHVQBspPtr = here->HSMHVQBspStructPtr->CSC ;

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVbNodePrime != 0))
                    here->HSMHVQBbpPtr = here->HSMHVQBbpStructPtr->CSC ;

                if ((here-> HSMHVqbNode != 0) && (here-> HSMHVqbNode != 0))
                    here->HSMHVQBqbPtr = here->HSMHVQBqbStructPtr->CSC ;

                if (here->HSMHV_coselfheat > 0)
                {
                    if ((here-> HSMHVqiNode != 0) && (here-> HSMHVtempNode != 0))
                        here->HSMHVQItempPtr = here->HSMHVQItempStructPtr->CSC ;

                    if ((here-> HSMHVqbNode != 0) && (here-> HSMHVtempNode != 0))
                        here->HSMHVQBtempPtr = here->HSMHVQBtempStructPtr->CSC ;

                }
            }
        }
    }

    return (OK) ;
}
