/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
**********/

/*
 * This routine performs truncation error calculations for
 * HICUMs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hicum2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
HICUMtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    HICUMmodel *model = (HICUMmodel*)inModel;
    HICUMinstance *here;

    for( ; model != NULL; model = HICUMnextModel(model)) {
        for(here=HICUMinstances(model);here!=NULL;
            here = HICUMnextInstance(here)){

            CKTterr(here->HICUMqrbi,     ckt, timeStep);
            CKTterr(here->HICUMqjei,     ckt, timeStep);
            CKTterr(here->HICUMqf,    ckt, timeStep);
            CKTterr(here->HICUMqjci,     ckt, timeStep);
            CKTterr(here->HICUMqr,     ckt, timeStep);
            CKTterr(here->HICUMqjep,     ckt, timeStep);
            CKTterr(here->HICUMqjcx0_i,  ckt, timeStep);
            CKTterr(here->HICUMqjcx0_ii, ckt, timeStep);
            CKTterr(here->HICUMqdsu,     ckt, timeStep);
            CKTterr(here->HICUMqjs,      ckt, timeStep);
            CKTterr(here->HICUMqscp,     ckt, timeStep);
        }
    }
    return(OK);
}
