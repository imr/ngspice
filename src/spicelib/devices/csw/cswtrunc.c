/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#include "cswdefs.h"


int
CSWtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    CSWmodel *model = (CSWmodel *) inModel;
    CSWinstance *here;

    double   lastChange, maxChange, maxStep, ref;

    for (; model; model = CSWnextModel(model))
        for (here = CSWinstances(model); here; here = CSWnextInstance(here)) {
            lastChange =
                ckt->CKTstate0[here->CSWctrlvalue] -
                ckt->CKTstate1[here->CSWctrlvalue];
            if (ckt->CKTstate0[here->CSWswitchstate] == 0) {
                ref = model->CSWiThreshold + model->CSWiHysteresis;
                if (ckt->CKTstate0[here->CSWctrlvalue] < ref && lastChange > 0) {
                    maxChange =
                        (ref - ckt->CKTstate0[here->CSWctrlvalue]) * 0.75
                        + 0.00005;
                    maxStep = maxChange / lastChange * ckt->CKTdeltaOld[0];
                    if (*timeStep > maxStep)
                        *timeStep = maxStep;
                }
            } else {
                ref = model->CSWiThreshold - model->CSWiHysteresis;
                if (ckt->CKTstate0[here->CSWctrlvalue] > ref && lastChange < 0) {
                    maxChange =
                        (ref - ckt->CKTstate0[here->CSWctrlvalue]) * 0.75
                        - 0.00005;
                    maxStep = maxChange / lastChange * ckt->CKTdeltaOld[0];
                    if (*timeStep > maxStep)
                        *timeStep = maxStep;
                }
            }
        }

    return OK;
}
