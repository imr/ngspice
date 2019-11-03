/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "swdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define TSTALLOC(ptr, first, second)                                    \
    do {                                                                \
        if ((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL) \
            return E_NOMEM;                                             \
    } while (0)


int
SWsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    SWmodel *model = (SWmodel *) inModel;
    SWinstance *here;

    for (; model; model = SWnextModel(model)) {

        /* Default Value Processing for Switch Model */
        if (!model->SWthreshGiven)
            model->SWvThreshold = 0;
        if (!model->SWhystGiven)
            model->SWvHysteresis = 0;
        if (!model->SWonGiven)  {
            model->SWonConduct = SW_ON_CONDUCTANCE;
            model->SWonResistance = 1.0 / model->SWonConduct;
        }
        if (!model->SWoffGiven)  {
            model->SWoffConduct = SW_OFF_CONDUCTANCE;
            model->SWoffResistance = 1.0 / model->SWoffConduct;
        }

        for (here = SWinstances(model); here; here = SWnextInstance(here)) {

            here->SWstate = *states;
            *states += SW_NUM_STATES;

            /* Default Value Processing for Switch Instance */
            /* none */

            TSTALLOC(SWposPosPtr, SWposNode, SWposNode);
            TSTALLOC(SWposNegPtr, SWposNode, SWnegNode);
            TSTALLOC(SWnegPosPtr, SWnegNode, SWposNode);
            TSTALLOC(SWnegNegPtr, SWnegNode, SWnegNode);
        }
    }

    return OK;
}
