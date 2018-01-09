/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#include "cswdefs.h"

int
CSWtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    CSWmodel *model = (CSWmodel*)inModel;
    CSWinstance *here;

    double   lastChange, maxChange, maxStep, ref;

    for( ; model!= NULL; model = CSWnextModel(model)) {
        for(here = CSWinstances(model) ; here != NULL ;
                here = here->CSWnextInstance) {
            lastChange = *(ckt->CKTstate0+(here->CSWstate+1)) -
                          *(ckt->CKTstate1+(here->CSWstate+1));
            if (*(ckt->CKTstate0+(here->CSWstate))==0) {
              ref = (model->CSWiThreshold + model->CSWiHysteresis);
              if ((*(ckt->CKTstate0+(here->CSWstate+1))<ref) && (lastChange>0)) {
                   maxChange = (ref - *(ckt->CKTstate0+(here->CSWstate+1))) *
                                0.75 + 0.00005;
                   maxStep = maxChange/lastChange * ckt->CKTdeltaOld[0];
                   if (*timeStep > maxStep) { *timeStep = maxStep; }
              }
            } else {
              ref = (model->CSWiThreshold - model->CSWiHysteresis);
              if ((*(ckt->CKTstate0+(here->CSWstate+1))>ref) && (lastChange<0)) {
                   maxChange = (ref - *(ckt->CKTstate0+(here->CSWstate+1))) *
                                0.75 - 0.00005;
                   maxStep = maxChange/lastChange * ckt->CKTdeltaOld[0];
                   if (*timeStep > maxStep) { *timeStep = maxStep; }
              }
            }
        }
    }
    return(OK);
}
