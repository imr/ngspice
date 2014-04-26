/*
 * Copyright (c) 2014, NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation and/or
 *    other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v7def.h"
#include "ngspice/sperror.h"

#define TopologyMatrixInsert(Ptr, instance_ID, offset, Value, global_ID) \
    ckt->CKTtopologyMatrixCOOi [global_ID] = (int)(here->Ptr - basePtr) ; \
    ckt->CKTtopologyMatrixCOOj [global_ID] = model->PositionVector [instance_ID] + offset ; \
    ckt->CKTtopologyMatrixCOOx [global_ID] = Value ;

#define TopologyMatrixInsertRHS(offset, instance_ID, offsetRHS, Value, global_ID) \
    ckt->CKTtopologyMatrixCOOiRHS [global_ID] = here->offset ; \
    ckt->CKTtopologyMatrixCOOjRHS [global_ID] = model->PositionVectorRHS [instance_ID] + offsetRHS ; \
    ckt->CKTtopologyMatrixCOOxRHS [global_ID] = Value ;

int
BSIM4v7topology (GENmodel *inModel, CKTcircuit *ckt, int *i, int *j)
{
    BSIM4v7model *model = (BSIM4v7model *)inModel ;
    BSIM4v7instance *here ;
    int k, total_offset, total_offsetRHS ;
    double *basePtr ;
    basePtr = ckt->CKTmatrix->CKTkluAx ;

    /*  loop through all the capacitor models */
    for ( ; model != NULL ; model = BSIM4v7nextModel(model))
    {
        k = 0 ;

        /* loop through all the instances of the model */
        for (here = BSIM4v7instances(model); here != NULL ; here = BSIM4v7nextInstance(here))
        {
            total_offset = 0 ;
            total_offsetRHS = 0 ;

            /* For the Matrix */
            if (here->BSIM4v7rgateMod == 1)
            {
                /* m * geltd */
                if ((here->BSIM4v7gNodeExt != 0) && (here->BSIM4v7gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GEgePtr, k, 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * geltd */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPgePtr, k, 0, -1, *i) ;
                    (*i)++ ;
                }

                /* m * geltd */
                if ((here->BSIM4v7gNodeExt != 0) && (here->BSIM4v7gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GEgpPtr, k, 0, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcggb + geltd - ggtg + gIgtotg) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPgpPtr, k, 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgdb - ggtd + gIgtotd) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPdpPtr, k, 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgsb - ggts + gIgtots) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPspPtr, k, 3, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgbb - ggtb + gIgtotb) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPbpPtr, k, 4, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 5 ;
            }
            else if (here->BSIM4v7rgateMod == 2)
            {
                /* m * gcrg */
                if ((here->BSIM4v7gNodeExt != 0) && (here->BSIM4v7gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GEgePtr, k, 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrgg */
                if ((here->BSIM4v7gNodeExt != 0) && (here->BSIM4v7gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GEgpPtr, k, 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrgd */
                if ((here->BSIM4v7gNodeExt != 0) && (here->BSIM4v7dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GEdpPtr, k, 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrgs */
                if ((here->BSIM4v7gNodeExt != 0) && (here->BSIM4v7sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GEspPtr, k, 3, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrgb */
                if ((here->BSIM4v7gNodeExt != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GEbpPtr, k, 4, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrg */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPgePtr, k, 0, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcggb  - gcrgg - ggtg + gIgtotg) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPgpPtr, k, 5, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgdb - gcrgd - ggtd + gIgtotd) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPdpPtr, k, 6, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgsb - gcrgs - ggts + gIgtots) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPspPtr, k, 7, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgbb - gcrgb - ggtb + gIgtotb) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPbpPtr, k, 8, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 9 ;
            }
            else if (here->BSIM4v7rgateMod == 3)
            {
                /* m * geltd */
                if ((here->BSIM4v7gNodeExt != 0) && (here->BSIM4v7gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GEgePtr, k, 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * geltd */
                if ((here->BSIM4v7gNodeExt != 0) && (here->BSIM4v7gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GEgmPtr, k, 0, -1, *i) ;
                    (*i)++ ;
                }

                /* m * geltd */
                if ((here->BSIM4v7gNodeMid != 0) && (here->BSIM4v7gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GMgePtr, k, 0, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (geltd + gcrg + gcgmgmb) */
                if ((here->BSIM4v7gNodeMid != 0) && (here->BSIM4v7gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GMgmPtr, k, 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcrgd + gcgmdb) */
                if ((here->BSIM4v7gNodeMid != 0) && (here->BSIM4v7dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GMdpPtr, k, 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrgg */
                if ((here->BSIM4v7gNodeMid != 0) && (here->BSIM4v7gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GMgpPtr, k, 3, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcrgs + gcgmsb) */
                if ((here->BSIM4v7gNodeMid != 0) && (here->BSIM4v7sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GMspPtr, k, 4, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcrgb + gcgmbb) */
                if ((here->BSIM4v7gNodeMid != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GMbpPtr, k, 5, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcdgmb */
                if ((here->BSIM4v7dNodePrime != 0) && (here->BSIM4v7gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4v7DPgmPtr, k, 6, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrg */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPgmPtr, k, 7, -1, *i) ;
                    (*i)++ ;
                }

                /* m * gcsgmb */
                if ((here->BSIM4v7sNodePrime != 0) && (here->BSIM4v7gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4v7SPgmPtr, k, 8, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcbgmb */
                if ((here->BSIM4v7bNodePrime != 0) && (here->BSIM4v7gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4v7BPgmPtr, k, 9, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcggb - gcrgg - ggtg + gIgtotg) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPgpPtr, k, 10, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgdb - gcrgd - ggtd + gIgtotd) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPdpPtr, k, 11, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgsb - gcrgs - ggts + gIgtots) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPspPtr, k, 12, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgbb - gcrgb - ggtb + gIgtotb) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPbpPtr, k, 13, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 14 ;
            } else {
                /* m * (gcggb - ggtg + gIgtotg) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPgpPtr, k, 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgdb - ggtd + gIgtotd) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPdpPtr, k, 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgsb - ggts + gIgtots) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPspPtr, k, 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgbb - ggtb + gIgtotb) */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPbpPtr, k, 3, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 4 ;
            }


            if (model->BSIM4v7rdsMod)
            {
                /* m * gdtotg */
                if ((here->BSIM4v7dNode != 0) && (here->BSIM4v7gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7DgpPtr, k, total_offset + 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gdtots */
                if ((here->BSIM4v7dNode != 0) && (here->BSIM4v7sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7DspPtr, k, total_offset + 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gdtotb */
                if ((here->BSIM4v7dNode != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7DbpPtr, k, total_offset + 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gstotd */
                if ((here->BSIM4v7sNode != 0) && (here->BSIM4v7dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7SdpPtr, k, total_offset + 3, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gstotg */
                if ((here->BSIM4v7sNode != 0) && (here->BSIM4v7gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7SgpPtr, k, total_offset + 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gstotb */
                if ((here->BSIM4v7sNode != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7SbpPtr, k, total_offset + 2, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 4 ;
            }


            /* m * (gdpr + here->BSIM4v7gds + here->BSIM4v7gbd + T1 * ddxpart_dVd -
               gdtotd + RevSum + gcddb + gbdpdp + dxpart * ggtd - gIdtotd) + m * ggidld */
            if ((here->BSIM4v7dNodePrime != 0) && (here->BSIM4v7dNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7DPdpPtr, k, total_offset + 0, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gdpr + gdtot) */
            if ((here->BSIM4v7dNodePrime != 0) && (here->BSIM4v7dNode != 0))
            {
                TopologyMatrixInsert (BSIM4v7DPdPtr, k, total_offset + 1, -1, *i) ;
                (*i)++ ;
            }

            /* m * (Gm + gcdgb - gdtotg + gbdpg - gIdtotg + dxpart * ggtg + T1 * ddxpart_dVg) + m * ggidlg */
            if ((here->BSIM4v7dNodePrime != 0) && (here->BSIM4v7gNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7DPgpPtr, k, total_offset + 2, 1, *i) ;
                (*i)++ ;
            }

            /* m * (here->BSIM4v7gds + gdtots - dxpart * ggts + gIdtots -
               T1 * ddxpart_dVs + FwdSum - gcdsb - gbdpsp) + m * (ggidlg + ggidld + ggidlb) */
            if ((here->BSIM4v7dNodePrime != 0) && (here->BSIM4v7sNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7DPspPtr, k, total_offset + 3, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gjbd + gdtotb - Gmbs - gcdbb - gbdpb + gIdtotb - T1 * ddxpart_dVb - dxpart * ggtb) - m * ggidlb */
            if ((here->BSIM4v7dNodePrime != 0) && (here->BSIM4v7bNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7DPbpPtr, k, total_offset + 4, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gdpr - gdtotd) */
            if ((here->BSIM4v7dNode != 0) && (here->BSIM4v7dNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7DdpPtr, k, total_offset + 5, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gdpr + gdtot) */
            if ((here->BSIM4v7dNode != 0) && (here->BSIM4v7dNode != 0))
            {
                TopologyMatrixInsert (BSIM4v7DdPtr, k, total_offset + 1, 1, *i) ;
                (*i)++ ;
            }

            /* m * (here->BSIM4v7gds + gstotd + RevSum - gcsdb - gbspdp -
               T1 * dsxpart_dVd - sxpart * ggtd + gIstotd) + m * (ggisls + ggislg + ggislb) */
            if ((here->BSIM4v7sNodePrime != 0) && (here->BSIM4v7dNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7SPdpPtr, k, total_offset + 6, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gcsgb - Gm - gstotg + gbspg + sxpart * ggtg + T1 * dsxpart_dVg - gIstotg) + m * ggislg */
            if ((here->BSIM4v7sNodePrime != 0) && (here->BSIM4v7gNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7SPgpPtr, k, total_offset + 7, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gspr + here->BSIM4v7gds + here->BSIM4v7gbs + T1 * dsxpart_dVs -
               gstots + FwdSum + gcssb + gbspsp + sxpart * ggts - gIstots) + m * ggisls */
            if ((here->BSIM4v7sNodePrime != 0) && (here->BSIM4v7sNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7SPspPtr, k, total_offset + 8, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gspr + gstot) */
            if ((here->BSIM4v7sNodePrime != 0) && (here->BSIM4v7sNode != 0))
            {
                TopologyMatrixInsert (BSIM4v7SPsPtr, k, total_offset + 9, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gjbs + gstotb + Gmbs - gcsbb - gbspb - sxpart * ggtb - T1 * dsxpart_dVb + gIstotb) - m * ggislb */
            if ((here->BSIM4v7sNodePrime != 0) && (here->BSIM4v7bNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7SPbpPtr, k, total_offset + 10, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gspr - gstots) */
            if ((here->BSIM4v7sNode != 0) && (here->BSIM4v7sNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7SspPtr, k, total_offset + 11, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gspr + gstot) */
            if ((here->BSIM4v7sNode != 0) && (here->BSIM4v7sNode != 0))
            {
                TopologyMatrixInsert (BSIM4v7SsPtr, k, total_offset + 9, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gcbdb - gjbd + gbbdp - gIbtotd) - m * ggidld + m * (ggislg + ggisls + ggislb) */
            if ((here->BSIM4v7bNodePrime != 0) && (here->BSIM4v7dNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7BPdpPtr, k, total_offset + 12, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gcbgb - here->BSIM4v7gbgs - gIbtotg) - m * ggidlg - m * ggislg */
            if ((here->BSIM4v7bNodePrime != 0) && (here->BSIM4v7gNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7BPgpPtr, k, total_offset + 13, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gcbsb - gjbs + gbbsp - gIbtots) + m * (ggidlg + ggidld + ggidlb) - m * ggisls */
            if ((here->BSIM4v7bNodePrime != 0) && (here->BSIM4v7sNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7BPspPtr, k, total_offset + 14, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gjbd + gjbs + gcbbb - here->BSIM4v7gbbs - gIbtotb) - m * ggidlb - m * ggislb */
            if ((here->BSIM4v7bNodePrime != 0) && (here->BSIM4v7bNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4v7BPbpPtr, k, total_offset + 15, 1, *i) ;
                (*i)++ ;
            }

            total_offset += 16 ;

            /* stamp gidl included above */
            /* stamp gisl included above */


            if (here->BSIM4v7rbodyMod)
            {
                /* m * (gcdbdb - here->BSIM4v7gbd) */
                if ((here->BSIM4v7dNodePrime != 0) && (here->BSIM4v7dbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7DPdbPtr, k, total_offset + 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (here->BSIM4v7gbs - gcsbsb) */
                if ((here->BSIM4v7sNodePrime != 0) && (here->BSIM4v7sbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7SPsbPtr, k, total_offset + 1, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcdbdb - here->BSIM4v7gbd) */
                if ((here->BSIM4v7dbNode != 0) && (here->BSIM4v7dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7DBdpPtr, k, total_offset + 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (here->BSIM4v7gbd - gcdbdb + here->BSIM4v7grbpd + here->BSIM4v7grbdb) */
                if ((here->BSIM4v7dbNode != 0) && (here->BSIM4v7dbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7DBdbPtr, k, total_offset + 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7grbpd */
                if ((here->BSIM4v7dbNode != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7DBbpPtr, k, total_offset + 3, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7grbdb */
                if ((here->BSIM4v7dbNode != 0) && (here->BSIM4v7bNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7DBbPtr, k, total_offset + 4, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7grbpd */
                if ((here->BSIM4v7bNodePrime != 0) && (here->BSIM4v7dbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7BPdbPtr, k, total_offset + 3, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7grbpb */
                if ((here->BSIM4v7bNodePrime != 0) && (here->BSIM4v7bNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7BPbPtr, k, total_offset + 5, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7grbps */
                if ((here->BSIM4v7bNodePrime != 0) && (here->BSIM4v7sbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7BPsbPtr, k, total_offset + 6, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (here->BSIM4v7grbpd + here->BSIM4v7grbps  + here->BSIM4v7grbpb) */
                if ((here->BSIM4v7bNodePrime != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7BPbpPtr, k, total_offset + 7, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcsbsb - here->BSIM4v7gbs) */
                if ((here->BSIM4v7sbNode != 0) && (here->BSIM4v7sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7SBspPtr, k, total_offset + 8, 1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7grbps */
                if ((here->BSIM4v7sbNode != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7SBbpPtr, k, total_offset + 6, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7grbsb */
                if ((here->BSIM4v7sbNode != 0) && (here->BSIM4v7bNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7SBbPtr, k, total_offset + 9, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (here->BSIM4v7gbs - gcsbsb + here->BSIM4v7grbps + here->BSIM4v7grbsb) */
                if ((here->BSIM4v7sbNode != 0) && (here->BSIM4v7sbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7SBsbPtr, k, total_offset + 10, 1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7grbdb */
                if ((here->BSIM4v7bNode != 0) && (here->BSIM4v7dbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7BdbPtr, k, total_offset + 4, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7grbpb */
                if ((here->BSIM4v7bNode != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7BbpPtr, k, total_offset + 5, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7grbsb */
                if ((here->BSIM4v7bNode != 0) && (here->BSIM4v7sbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7BsbPtr, k, total_offset + 9, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (here->BSIM4v7grbsb + here->BSIM4v7grbdb + here->BSIM4v7grbpb) */
                if ((here->BSIM4v7bNode != 0) && (here->BSIM4v7bNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7BbPtr, k, total_offset + 11, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 12 ;
            }


            if (here->BSIM4v7trnqsMod)
            {
                /* m * (gqdef + here->BSIM4v7gtau) */
                if ((here->BSIM4v7qNode != 0) && (here->BSIM4v7qNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7QqPtr, k, total_offset + 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (ggtg - gcqgb) */
                if ((here->BSIM4v7qNode != 0) && (here->BSIM4v7gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7QgpPtr, k, total_offset + 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (ggtd - gcqdb) */
                if ((here->BSIM4v7qNode != 0) && (here->BSIM4v7dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7QdpPtr, k, total_offset + 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (ggts - gcqsb) */
                if ((here->BSIM4v7qNode != 0) && (here->BSIM4v7sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7QspPtr, k, total_offset + 3, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (ggtb - gcqbb) */
                if ((here->BSIM4v7qNode != 0) && (here->BSIM4v7bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4v7QbpPtr, k, total_offset + 4, 1, *i) ;
                    (*i)++ ;
                }

                /* m * dxpart * here->BSIM4v7gtau */
                if ((here->BSIM4v7dNodePrime != 0) && (here->BSIM4v7qNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7DPqPtr, k, total_offset + 5, 1, *i) ;
                    (*i)++ ;
                }

                /* m * sxpart * here->BSIM4v7gtau */
                if ((here->BSIM4v7sNodePrime != 0) && (here->BSIM4v7qNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7SPqPtr, k, total_offset + 6, 1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4v7gtau */
                if ((here->BSIM4v7gNodePrime != 0) && (here->BSIM4v7qNode != 0))
                {
                    TopologyMatrixInsert (BSIM4v7GPqPtr, k, total_offset + 7, -1, *i) ;
                    (*i)++ ;
                }
            }



            /* For the RHS */
            /* m * (ceqjd - ceqbd + ceqgdtot - ceqdrn - ceqqd + Idtoteq) */
            if (here->BSIM4v7dNodePrime != 0)
            {
                TopologyMatrixInsertRHS (BSIM4v7dNodePrime, k, total_offsetRHS + 0, 1, *j) ;
                (*j)++ ;
            }

            /* m * (ceqqg - ceqgcrg + Igtoteq) */
            if (here->BSIM4v7gNodePrime != 0)
            {
                TopologyMatrixInsertRHS (BSIM4v7gNodePrime, k, total_offsetRHS + 1, -1, *j) ;
                (*j)++ ;
            }

            total_offsetRHS += 2 ;


            if (here->BSIM4v7rgateMod == 2)
            {
                /* m * ceqgcrg */
                if (here->BSIM4v7gNodeExt != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7gNodeExt, k, total_offsetRHS + 0, -1, *j) ;
                    (*j)++ ;
                }

                total_offsetRHS += 1 ;
            }
            else if (here->BSIM4v7rgateMod == 3)
            {
                /* m * (ceqqgmid + ceqgcrg) */
                if (here->BSIM4v7gNodeMid != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7gNodeMid, k, total_offsetRHS + 0, -1, *j) ;
                    (*j)++ ;
                }

                total_offsetRHS += 1 ;
            }


            if (!here->BSIM4v7rbodyMod)
            {
                /* m * (ceqbd + ceqbs - ceqjd - ceqjs - ceqqb + Ibtoteq) */
                if (here->BSIM4v7bNodePrime != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7bNodePrime, k, total_offsetRHS + 0, 1, *j) ;
                    (*j)++ ;
                }

                /* m * (ceqdrn - ceqbs + ceqjs + ceqqg + ceqqb + ceqqd + ceqqgmid - ceqgstot + Istoteq) */
                if (here->BSIM4v7sNodePrime != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7sNodePrime, k, total_offsetRHS + 1, 1, *j) ;
                    (*j)++ ;
                }

                total_offsetRHS += 2 ;

            } else {
                /* m * (ceqjd + ceqqjd) */
                if (here->BSIM4v7dbNode != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7dbNode, k, total_offsetRHS + 0, -1, *j) ;
                    (*j)++ ;
                }

                /* m * (ceqbd + ceqbs - ceqqb + Ibtoteq) */
                if (here->BSIM4v7bNodePrime != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7bNodePrime, k, total_offsetRHS + 1, 1, *j) ;
                    (*j)++ ;
                }

                /* m * (ceqjs + ceqqjs) */
                if (here->BSIM4v7sbNode != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7sbNode, k, total_offsetRHS + 2, -1, *j) ;
                    (*j)++ ;
                }

                /* m * (ceqdrn - ceqbs + ceqjs + ceqqd + ceqqg + ceqqb +
                   ceqqjd + ceqqjs + ceqqgmid - ceqgstot + Istoteq) */
                if (here->BSIM4v7sNodePrime != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7sNodePrime, k, total_offsetRHS + 3, 1, *j) ;
                    (*j)++ ;
                }

                total_offsetRHS += 4 ;
            }


            if (model->BSIM4v7rdsMod)
            {
                /* m * ceqgdtot */
                if (here->BSIM4v7dNode != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7dNode, k, total_offsetRHS + 0, -1, *j) ;
                    (*j)++ ;
                }

                /* m * ceqgstot */
                if (here->BSIM4v7sNode != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7sNode, k, total_offsetRHS + 1, 1, *j) ;
                    (*j)++ ;
                }

                total_offsetRHS += 2 ;

            }


            if (here->BSIM4v7trnqsMod)
            {
                /* m * (cqcheq - cqdef) */
                if (here->BSIM4v7qNode != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4v7qNode, k, total_offsetRHS + 0, 1, *j) ;
                    (*j)++ ;
                }
            }

            k++ ;
        }
    }

    return (OK) ;
}
