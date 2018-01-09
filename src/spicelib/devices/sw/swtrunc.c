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
#include "swdefs.h"

int
SWtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    SWmodel *model = (SWmodel*)inModel;
    SWinstance *here;
    double   lastChange, maxChange, maxStep, ref;

    for( ; model!= NULL; model = SWnextModel(model)) {
        for(here = SWinstances(model) ; here != NULL ;
                here = here->SWnextInstance) {
            lastChange = *(ckt->CKTstate0+(here->SWstate+1)) -
                          *(ckt->CKTstate1+(here->SWstate+1));
            if (*(ckt->CKTstate0+(here->SWstate))==0) {
              ref = (model->SWvThreshold + model->SWvHysteresis);
              if ((*(ckt->CKTstate0+(here->SWstate+1))<ref) && (lastChange>0)) {
                   maxChange = (ref - *(ckt->CKTstate0+(here->SWstate+1))) *
                                0.75 + 0.05;
                   maxStep = maxChange/lastChange * ckt->CKTdeltaOld[0];
                   if (*timeStep > maxStep) { *timeStep = maxStep; }
              }
            } else {
              ref = (model->SWvThreshold - model->SWvHysteresis);
              if ((*(ckt->CKTstate0+(here->SWstate+1))>ref) && (lastChange<0)) {
                   maxChange = (ref - *(ckt->CKTstate0+(here->SWstate+1))) *
                                0.75 - 0.05;
                   maxStep = maxChange/lastChange * ckt->CKTdeltaOld[0];
                   if (*timeStep > maxStep) { *timeStep = maxStep; }
              }
            }
        }
    }
    return(OK);
}
