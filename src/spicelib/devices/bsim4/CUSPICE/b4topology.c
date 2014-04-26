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
#include "bsim4def.h"
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
BSIM4topology (GENmodel *inModel, CKTcircuit *ckt, int *i, int *j)
{
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;
    int k, total_offset, total_offsetRHS ;
    double *basePtr ;
    basePtr = ckt->CKTmatrix->CKTkluAx ;

    /*  loop through all the capacitor models */
    for ( ; model != NULL ; model = model->BSIM4nextModel)
    {
        k = 0 ;

        /* loop through all the instances of the model */
        for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
        {
            total_offset = 0 ;
            total_offsetRHS = 0 ;

            /* For the Matrix */
            if (here->BSIM4rgateMod == 1)
            {
                /* m * geltd */
                if ((here->BSIM4gNodeExt != 0) && (here->BSIM4gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4GEgePtr, k, 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * geltd */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4GPgePtr, k, 0, -1, *i) ;
                    (*i)++ ;
                }

                /* m * geltd */
                if ((here->BSIM4gNodeExt != 0) && (here->BSIM4gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GEgpPtr, k, 0, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcggb + geltd - ggtg + gIgtotg) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPgpPtr, k, 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgdb - ggtd + gIgtotd) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPdpPtr, k, 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgsb - ggts + gIgtots) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPspPtr, k, 3, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgbb - ggtb + gIgtotb) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPbpPtr, k, 4, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 5 ;
            }
            else if (here->BSIM4rgateMod == 2)        
            {
                /* m * gcrg */
                if ((here->BSIM4gNodeExt != 0) && (here->BSIM4gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4GEgePtr, k, 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrgg */
                if ((here->BSIM4gNodeExt != 0) && (here->BSIM4gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GEgpPtr, k, 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrgd */
                if ((here->BSIM4gNodeExt != 0) && (here->BSIM4dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GEdpPtr, k, 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrgs */
                if ((here->BSIM4gNodeExt != 0) && (here->BSIM4sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GEspPtr, k, 3, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrgb */
                if ((here->BSIM4gNodeExt != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GEbpPtr, k, 4, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrg */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4GPgePtr, k, 0, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcggb  - gcrgg - ggtg + gIgtotg) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPgpPtr, k, 5, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgdb - gcrgd - ggtd + gIgtotd) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPdpPtr, k, 6, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgsb - gcrgs - ggts + gIgtots) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPspPtr, k, 7, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgbb - gcrgb - ggtb + gIgtotb) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPbpPtr, k, 8, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 9 ;
            }
            else if (here->BSIM4rgateMod == 3)
            {
                /* m * geltd */
                if ((here->BSIM4gNodeExt != 0) && (here->BSIM4gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4GEgePtr, k, 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * geltd */
                if ((here->BSIM4gNodeExt != 0) && (here->BSIM4gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4GEgmPtr, k, 0, -1, *i) ;
                    (*i)++ ;
                }

                /* m * geltd */
                if ((here->BSIM4gNodeMid != 0) && (here->BSIM4gNodeExt != 0))
                {
                    TopologyMatrixInsert (BSIM4GMgePtr, k, 0, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (geltd + gcrg + gcgmgmb) */
                if ((here->BSIM4gNodeMid != 0) && (here->BSIM4gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4GMgmPtr, k, 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcrgd + gcgmdb) */
                if ((here->BSIM4gNodeMid != 0) && (here->BSIM4dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GMdpPtr, k, 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrgg */
                if ((here->BSIM4gNodeMid != 0) && (here->BSIM4gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GMgpPtr, k, 3, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcrgs + gcgmsb) */
                if ((here->BSIM4gNodeMid != 0) && (here->BSIM4sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GMspPtr, k, 4, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcrgb + gcgmbb) */
                if ((here->BSIM4gNodeMid != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GMbpPtr, k, 5, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcdgmb */
                if ((here->BSIM4dNodePrime != 0) && (here->BSIM4gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4DPgmPtr, k, 6, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcrg */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4GPgmPtr, k, 7, -1, *i) ;
                    (*i)++ ;
                }

                /* m * gcsgmb */
                if ((here->BSIM4sNodePrime != 0) && (here->BSIM4gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4SPgmPtr, k, 8, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gcbgmb */
                if ((here->BSIM4bNodePrime != 0) && (here->BSIM4gNodeMid != 0))
                {
                    TopologyMatrixInsert (BSIM4BPgmPtr, k, 9, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcggb - gcrgg - ggtg + gIgtotg) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPgpPtr, k, 10, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgdb - gcrgd - ggtd + gIgtotd) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPdpPtr, k, 11, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgsb - gcrgs - ggts + gIgtots) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPspPtr, k, 12, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgbb - gcrgb - ggtb + gIgtotb) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPbpPtr, k, 13, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 14 ;
            } else {
                /* m * (gcggb - ggtg + gIgtotg) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPgpPtr, k, 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgdb - ggtd + gIgtotd) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPdpPtr, k, 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgsb - ggts + gIgtots) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPspPtr, k, 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcgbb - ggtb + gIgtotb) */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4GPbpPtr, k, 3, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 4 ;
            }


            if (model->BSIM4rdsMod)
            {
                /* m * gdtotg */
                if ((here->BSIM4dNode != 0) && (here->BSIM4gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4DgpPtr, k, total_offset + 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gdtots */
                if ((here->BSIM4dNode != 0) && (here->BSIM4sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4DspPtr, k, total_offset + 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gdtotb */
                if ((here->BSIM4dNode != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4DbpPtr, k, total_offset + 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gstotd */
                if ((here->BSIM4sNode != 0) && (here->BSIM4dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4SdpPtr, k, total_offset + 3, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gstotg */
                if ((here->BSIM4sNode != 0) && (here->BSIM4gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4SgpPtr, k, total_offset + 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * gstotb */
                if ((here->BSIM4sNode != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4SbpPtr, k, total_offset + 2, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 4 ;
            }


            /* m * (gdpr + here->BSIM4gds + here->BSIM4gbd + T1 * ddxpart_dVd -
               gdtotd + RevSum + gcddb + gbdpdp + dxpart * ggtd - gIdtotd) + m * ggidld */
            if ((here->BSIM4dNodePrime != 0) && (here->BSIM4dNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4DPdpPtr, k, total_offset + 0, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gdpr + gdtot) */
            if ((here->BSIM4dNodePrime != 0) && (here->BSIM4dNode != 0))
            {
                TopologyMatrixInsert (BSIM4DPdPtr, k, total_offset + 1, -1, *i) ;
                (*i)++ ;
            }

            /* m * (Gm + gcdgb - gdtotg + gbdpg - gIdtotg + dxpart * ggtg + T1 * ddxpart_dVg) + m * ggidlg */
            if ((here->BSIM4dNodePrime != 0) && (here->BSIM4gNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4DPgpPtr, k, total_offset + 2, 1, *i) ;
                (*i)++ ;
            }

            /* m * (here->BSIM4gds + gdtots - dxpart * ggts + gIdtots -
               T1 * ddxpart_dVs + FwdSum - gcdsb - gbdpsp) + m * (ggidlg + ggidld + ggidlb) */
            if ((here->BSIM4dNodePrime != 0) && (here->BSIM4sNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4DPspPtr, k, total_offset + 3, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gjbd + gdtotb - Gmbs - gcdbb - gbdpb + gIdtotb - T1 * ddxpart_dVb - dxpart * ggtb) - m * ggidlb */
            if ((here->BSIM4dNodePrime != 0) && (here->BSIM4bNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4DPbpPtr, k, total_offset + 4, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gdpr - gdtotd) */
            if ((here->BSIM4dNode != 0) && (here->BSIM4dNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4DdpPtr, k, total_offset + 5, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gdpr + gdtot) */
            if ((here->BSIM4dNode != 0) && (here->BSIM4dNode != 0))
            {
                TopologyMatrixInsert (BSIM4DdPtr, k, total_offset + 1, 1, *i) ;
                (*i)++ ;
            }

            /* m * (here->BSIM4gds + gstotd + RevSum - gcsdb - gbspdp -
               T1 * dsxpart_dVd - sxpart * ggtd + gIstotd) + m * (ggisls + ggislg + ggislb) */
            if ((here->BSIM4sNodePrime != 0) && (here->BSIM4dNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4SPdpPtr, k, total_offset + 6, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gcsgb - Gm - gstotg + gbspg + sxpart * ggtg + T1 * dsxpart_dVg - gIstotg) + m * ggislg */
            if ((here->BSIM4sNodePrime != 0) && (here->BSIM4gNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4SPgpPtr, k, total_offset + 7, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gspr + here->BSIM4gds + here->BSIM4gbs + T1 * dsxpart_dVs -
               gstots + FwdSum + gcssb + gbspsp + sxpart * ggts - gIstots) + m * ggisls */
            if ((here->BSIM4sNodePrime != 0) && (here->BSIM4sNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4SPspPtr, k, total_offset + 8, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gspr + gstot) */
            if ((here->BSIM4sNodePrime != 0) && (here->BSIM4sNode != 0))
            {
                TopologyMatrixInsert (BSIM4SPsPtr, k, total_offset + 9, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gjbs + gstotb + Gmbs - gcsbb - gbspb - sxpart * ggtb - T1 * dsxpart_dVb + gIstotb) - m * ggislb */
            if ((here->BSIM4sNodePrime != 0) && (here->BSIM4bNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4SPbpPtr, k, total_offset + 10, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gspr - gstots) */
            if ((here->BSIM4sNode != 0) && (here->BSIM4sNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4SspPtr, k, total_offset + 11, -1, *i) ;
                (*i)++ ;
            }

            /* m * (gspr + gstot) */
            if ((here->BSIM4sNode != 0) && (here->BSIM4sNode != 0))
            {
                TopologyMatrixInsert (BSIM4SsPtr, k, total_offset + 9, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gcbdb - gjbd + gbbdp - gIbtotd) - m * ggidld + m * (ggislg + ggisls + ggislb) */
            if ((here->BSIM4bNodePrime != 0) && (here->BSIM4dNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4BPdpPtr, k, total_offset + 12, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gcbgb - here->BSIM4gbgs - gIbtotg) - m * ggidlg - m * ggislg */
            if ((here->BSIM4bNodePrime != 0) && (here->BSIM4gNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4BPgpPtr, k, total_offset + 13, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gcbsb - gjbs + gbbsp - gIbtots) + m * (ggidlg + ggidld + ggidlb) - m * ggisls */
            if ((here->BSIM4bNodePrime != 0) && (here->BSIM4sNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4BPspPtr, k, total_offset + 14, 1, *i) ;
                (*i)++ ;
            }

            /* m * (gjbd + gjbs + gcbbb - here->BSIM4gbbs - gIbtotb) - m * ggidlb - m * ggislb */
            if ((here->BSIM4bNodePrime != 0) && (here->BSIM4bNodePrime != 0))
            {
                TopologyMatrixInsert (BSIM4BPbpPtr, k, total_offset + 15, 1, *i) ;
                (*i)++ ;
            }

            total_offset += 16 ;

            /* stamp gidl included above */
            /* stamp gisl included above */


            if (here->BSIM4rbodyMod)
            {
                /* m * (gcdbdb - here->BSIM4gbd) */
                if ((here->BSIM4dNodePrime != 0) && (here->BSIM4dbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4DPdbPtr, k, total_offset + 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (here->BSIM4gbs - gcsbsb) */
                if ((here->BSIM4sNodePrime != 0) && (here->BSIM4sbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4SPsbPtr, k, total_offset + 1, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcdbdb - here->BSIM4gbd) */
                if ((here->BSIM4dbNode != 0) && (here->BSIM4dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4DBdpPtr, k, total_offset + 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (here->BSIM4gbd - gcdbdb + here->BSIM4grbpd + here->BSIM4grbdb) */
                if ((here->BSIM4dbNode != 0) && (here->BSIM4dbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4DBdbPtr, k, total_offset + 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4grbpd */
                if ((here->BSIM4dbNode != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4DBbpPtr, k, total_offset + 3, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4grbdb */
                if ((here->BSIM4dbNode != 0) && (here->BSIM4bNode != 0))
                {
                    TopologyMatrixInsert (BSIM4DBbPtr, k, total_offset + 4, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4grbpd */
                if ((here->BSIM4bNodePrime != 0) && (here->BSIM4dbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4BPdbPtr, k, total_offset + 3, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4grbpb */
                if ((here->BSIM4bNodePrime != 0) && (here->BSIM4bNode != 0))
                {
                    TopologyMatrixInsert (BSIM4BPbPtr, k, total_offset + 5, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4grbps */
                if ((here->BSIM4bNodePrime != 0) && (here->BSIM4sbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4BPsbPtr, k, total_offset + 6, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (here->BSIM4grbpd + here->BSIM4grbps  + here->BSIM4grbpb) */
                if ((here->BSIM4bNodePrime != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4BPbpPtr, k, total_offset + 7, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (gcsbsb - here->BSIM4gbs) */
                if ((here->BSIM4sbNode != 0) && (here->BSIM4sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4SBspPtr, k, total_offset + 8, 1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4grbps */
                if ((here->BSIM4sbNode != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4SBbpPtr, k, total_offset + 6, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4grbsb */
                if ((here->BSIM4sbNode != 0) && (here->BSIM4bNode != 0))
                {
                    TopologyMatrixInsert (BSIM4SBbPtr, k, total_offset + 9, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (here->BSIM4gbs - gcsbsb + here->BSIM4grbps + here->BSIM4grbsb) */
                if ((here->BSIM4sbNode != 0) && (here->BSIM4sbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4SBsbPtr, k, total_offset + 10, 1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4grbdb */
                if ((here->BSIM4bNode != 0) && (here->BSIM4dbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4BdbPtr, k, total_offset + 4, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4grbpb */
                if ((here->BSIM4bNode != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4BbpPtr, k, total_offset + 5, -1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4grbsb */
                if ((here->BSIM4bNode != 0) && (here->BSIM4sbNode != 0))
                {
                    TopologyMatrixInsert (BSIM4BsbPtr, k, total_offset + 9, -1, *i) ;
                    (*i)++ ;
                }

                /* m * (here->BSIM4grbsb + here->BSIM4grbdb + here->BSIM4grbpb) */
                if ((here->BSIM4bNode != 0) && (here->BSIM4bNode != 0))
                {
                    TopologyMatrixInsert (BSIM4BbPtr, k, total_offset + 11, 1, *i) ;
                    (*i)++ ;
                }

                total_offset += 12 ;
            }


            if (here->BSIM4trnqsMod)
            {
                /* m * (gqdef + here->BSIM4gtau) */
                if ((here->BSIM4qNode != 0) && (here->BSIM4qNode != 0))
                {
                    TopologyMatrixInsert (BSIM4QqPtr, k, total_offset + 0, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (ggtg - gcqgb) */
                if ((here->BSIM4qNode != 0) && (here->BSIM4gNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4QgpPtr, k, total_offset + 1, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (ggtd - gcqdb) */
                if ((here->BSIM4qNode != 0) && (here->BSIM4dNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4QdpPtr, k, total_offset + 2, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (ggts - gcqsb) */
                if ((here->BSIM4qNode != 0) && (here->BSIM4sNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4QspPtr, k, total_offset + 3, 1, *i) ;
                    (*i)++ ;
                }

                /* m * (ggtb - gcqbb) */
                if ((here->BSIM4qNode != 0) && (here->BSIM4bNodePrime != 0))
                {
                    TopologyMatrixInsert (BSIM4QbpPtr, k, total_offset + 4, 1, *i) ;
                    (*i)++ ;
                }

                /* m * dxpart * here->BSIM4gtau */
                if ((here->BSIM4dNodePrime != 0) && (here->BSIM4qNode != 0))
                {
                    TopologyMatrixInsert (BSIM4DPqPtr, k, total_offset + 5, 1, *i) ;
                    (*i)++ ;
                }

                /* m * sxpart * here->BSIM4gtau */
                if ((here->BSIM4sNodePrime != 0) && (here->BSIM4qNode != 0))
                {
                    TopologyMatrixInsert (BSIM4SPqPtr, k, total_offset + 6, 1, *i) ;
                    (*i)++ ;
                }

                /* m * here->BSIM4gtau */
                if ((here->BSIM4gNodePrime != 0) && (here->BSIM4qNode != 0))
                {
                    TopologyMatrixInsert (BSIM4GPqPtr, k, total_offset + 7, -1, *i) ;
                    (*i)++ ;
                }
            }



            /* For the RHS */
            /* m * (ceqjd - ceqbd + ceqgdtot - ceqdrn - ceqqd + Idtoteq) */
            if (here->BSIM4dNodePrime != 0)
            {
                TopologyMatrixInsertRHS (BSIM4dNodePrime, k, total_offsetRHS + 0, 1, *j) ;
                (*j)++ ;
            }

            /* m * (ceqqg - ceqgcrg + Igtoteq) */
            if (here->BSIM4gNodePrime != 0)
            {
                TopologyMatrixInsertRHS (BSIM4gNodePrime, k, total_offsetRHS + 1, -1, *j) ;
                (*j)++ ;
            }

            total_offsetRHS += 2 ;


            if (here->BSIM4rgateMod == 2)
            {
                /* m * ceqgcrg */
                if (here->BSIM4gNodeExt != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4gNodeExt, k, total_offsetRHS + 0, -1, *j) ;
                    (*j)++ ;
                }

                total_offsetRHS += 1 ;
            }
            else if (here->BSIM4rgateMod == 3)
            {
                /* m * (ceqqgmid + ceqgcrg) */
                if (here->BSIM4gNodeMid != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4gNodeMid, k, total_offsetRHS + 0, -1, *j) ;
                    (*j)++ ;
                }

                total_offsetRHS += 1 ;
            }


            if (!here->BSIM4rbodyMod)
            {
                /* m * (ceqbd + ceqbs - ceqjd - ceqjs - ceqqb + Ibtoteq) */
                if (here->BSIM4bNodePrime != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4bNodePrime, k, total_offsetRHS + 0, 1, *j) ;
                    (*j)++ ;
                }

                /* m * (ceqdrn - ceqbs + ceqjs + ceqqg + ceqqb + ceqqd + ceqqgmid - ceqgstot + Istoteq) */
                if (here->BSIM4sNodePrime != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4sNodePrime, k, total_offsetRHS + 1, 1, *j) ;
                    (*j)++ ;
                }

                total_offsetRHS += 2 ;

            } else {
                /* m * (ceqjd + ceqqjd) */
                if (here->BSIM4dbNode != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4dbNode, k, total_offsetRHS + 0, -1, *j) ;
                    (*j)++ ;
                }

                /* m * (ceqbd + ceqbs - ceqqb + Ibtoteq) */
                if (here->BSIM4bNodePrime != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4bNodePrime, k, total_offsetRHS + 1, 1, *j) ;
                    (*j)++ ;
                }

                /* m * (ceqjs + ceqqjs) */
                if (here->BSIM4sbNode != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4sbNode, k, total_offsetRHS + 2, -1, *j) ;
                    (*j)++ ;
                }

                /* m * (ceqdrn - ceqbs + ceqjs + ceqqd + ceqqg + ceqqb +
                   ceqqjd + ceqqjs + ceqqgmid - ceqgstot + Istoteq) */
                if (here->BSIM4sNodePrime != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4sNodePrime, k, total_offsetRHS + 3, 1, *j) ;
                    (*j)++ ;
                }

                total_offsetRHS += 4 ;
            }


            if (model->BSIM4rdsMod)
            {
                /* m * ceqgdtot */
                if (here->BSIM4dNode != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4dNode, k, total_offsetRHS + 0, -1, *j) ;
                    (*j)++ ;
                }

                /* m * ceqgstot */
                if (here->BSIM4sNode != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4sNode, k, total_offsetRHS + 1, 1, *j) ;
                    (*j)++ ;
                }

                total_offsetRHS += 2 ;

            }


            if (here->BSIM4trnqsMod)
            {
                /* m * (cqcheq - cqdef) */
                if (here->BSIM4qNode != 0)
                {
                    TopologyMatrixInsertRHS (BSIM4qNode, k, total_offsetRHS + 0, 1, *j) ;
                    (*j)++ ;
                }
            }

            k++ ;
        }
    }

    return (OK) ;
}
