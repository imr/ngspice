/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cswdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define TSTALLOC(ptr, first, second)                                    \
    do {                                                                \
        if ((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL) \
            return E_NOMEM;                                             \
    } while (0)


int
CSWsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    CSWmodel *model = (CSWmodel *) inModel;
    CSWinstance *here;

    for (; model; model = CSWnextModel(model)) {

        /* Default Value Processing for Switch Model */
        if (!model->CSWthreshGiven)
            model->CSWiThreshold = 0;
        if (!model->CSWhystGiven)
            model->CSWiHysteresis = 0;
        if (!model->CSWonGiven)  {
            model->CSWonConduct = CSW_ON_CONDUCTANCE;
            model->CSWonResistance = 1.0 / model->CSWonConduct;
        }
        if (!model->CSWoffGiven)  {
            model->CSWoffConduct = CSW_OFF_CONDUCTANCE;
            model->CSWoffResistance = 1.0 / model->CSWoffConduct;
        }

        for (here = CSWinstances(model); here; here = CSWnextInstance(here)) {

            /* Default Value Processing for Switch Instance */
            here->CSWstate = *states;
            *states += CSW_NUM_STATES;

            here->CSWcontBranch = CKTfndBranch(ckt, here->CSWcontName);
            if (here->CSWcontBranch == 0) {
                SPfrontEnd->IFerrorf(ERR_FATAL,
                                     "%s: unknown controlling source %s", here->CSWname, here->CSWcontName);
                return E_BADPARM;
            }

            TSTALLOC(CSWposPosPtr, CSWposNode, CSWposNode);
            TSTALLOC(CSWposNegPtr, CSWposNode, CSWnegNode);
            TSTALLOC(CSWnegPosPtr, CSWnegNode, CSWposNode);
            TSTALLOC(CSWnegNegPtr, CSWnegNode, CSWnegNode);
        }
    }

    return OK;
}
