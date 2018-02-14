/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mesdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
MEStemp(GENmodel *inModel, CKTcircuit *ckt)
        /* load the mes structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    MESmodel *model = (MESmodel*)inModel;
    double xfc, temp;

    NG_IGNORE(ckt);

    /*  loop through all the diode models */
    for( ; model != NULL; model = MESnextModel(model)) {

        if(model->MESdrainResist != 0) {
            model->MESdrainConduct = 1/model->MESdrainResist;
        } else {
            model->MESdrainConduct = 0;
        }
        if(model->MESsourceResist != 0) {
            model->MESsourceConduct = 1/model->MESsourceResist;
        } else {
            model->MESsourceConduct = 0;
        }

        model->MESdepletionCap = model->MESdepletionCapCoeff *
                model->MESgatePotential;
        xfc = (1 - model->MESdepletionCapCoeff);
        temp = sqrt(xfc);
        model->MESf1 = model->MESgatePotential * (1 - temp)/(1-.5);
        model->MESf2 = temp * temp * temp;
        model->MESf3 = 1 - model->MESdepletionCapCoeff * (1 + .5);
        model->MESvcrit = CONSTvt0 * log(CONSTvt0/
                (CONSTroot2 * model->MESgateSatCurrent));

    }
    return(OK);
}
