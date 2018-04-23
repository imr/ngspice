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
#include "isrcdefs.h"
#include "ngspice/sperror.h"

#define TopologyMatrixInsertRHS(offset, instance_ID, offsetRHS, Value, global_ID) \
    ckt->CKTtopologyMatrixCOOiRHS [global_ID] = here->offset ; \
    ckt->CKTtopologyMatrixCOOjRHS [global_ID] = model->PositionVectorRHS [instance_ID] + offsetRHS ; \
    ckt->CKTtopologyMatrixCOOxRHS [global_ID] = Value ;

int
ISRCtopology (GENmodel *inModel, CKTcircuit *ckt, int *i, int *j)
{
    NG_IGNORE (i) ;

    ISRCmodel *model = (ISRCmodel *)inModel ;
    ISRCinstance *here ;
    int k ;

    /*  loop through all the voltage source models */
    for ( ; model != NULL ; model = ISRCnextModel(model))
    {
        k = 0 ;

        /* loop through all the instances of the model */
        for (here = ISRCinstances(model); here != NULL ; here = ISRCnextInstance(here))
        {
            if (here->ISRCposNode != 0)
            {
                TopologyMatrixInsertRHS (ISRCposNode, k, 0, 1, *j) ;
                (*j)++ ;
            }

            if (here->ISRCnegNode != 0)
            {
                TopologyMatrixInsertRHS (ISRCnegNode, k, 0, -1, *j) ;
                (*j)++ ;
            }

            k++ ;
        }
    }

    return (OK) ;
}
